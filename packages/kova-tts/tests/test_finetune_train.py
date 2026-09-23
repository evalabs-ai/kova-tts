"""Run orchestration, plus an end-to-end smoke test against the real checkpoint.

The CPU tests here cover the things that go wrong between configs and the Trainer: run
directory numbering, command-line overrides, and wandb staying off. The one GPU test actually
trains, because nothing short of that catches a mismatch between the collated batch and what
the model's forward expects.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from kova_tts import paths, prompt
from kova_tts.finetune import config as config_mod
from kova_tts.finetune import merge, train


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    return tmp_path


def make_config(workspace, body=""):
    path = workspace / "voice.yaml"
    path.write_text(f"dataset: data/train.jsonl\n{body}", encoding="utf-8")
    return config_mod.load_config(path)


class TestRunDirectory:
    def test_first_run_is_numbered_one(self, workspace):
        run_dir = train._prepare_run_dir(make_config(workspace, "voice: my_voice\n"))
        assert run_dir.parent == workspace / "runs"
        assert run_dir.name.startswith("ft_001_my_voice_")
        assert run_dir.is_dir()

    def test_numbering_continues_past_existing_runs(self, workspace):
        config = make_config(workspace, "voice: my_voice\n")
        (workspace / "runs").mkdir()
        (workspace / "runs" / "ft_007_other_2026-01-01_00-00-00").mkdir()
        (workspace / "runs" / "not-a-run").mkdir()

        assert train._prepare_run_dir(config).name.startswith("ft_008_my_voice_")

    def test_the_voice_defaults_to_the_corpus_name(self, workspace):
        assert "_train_" in train._prepare_run_dir(make_config(workspace)).name

    def test_an_explicit_run_name_is_used_verbatim(self, workspace):
        config = make_config(workspace, "run_name: my-experiment\n")
        assert train._prepare_run_dir(config).name == "my-experiment"

    def test_the_name_carries_no_machine_specific_path(self, workspace):
        name = train._prepare_run_dir(make_config(workspace, "voice: my_voice\n")).name
        assert "/" not in name and "mnt" not in name


class TestOverrides:
    def parse(self, argv):
        return train.build_parser().parse_args(argv)

    def test_scalar_overrides_replace_config_values(self, workspace):
        config = make_config(workspace, "voice: my_voice\nepochs: 4\n")
        argv = ["--config", "x", "--epochs", "1", "--lr", "3e-5", "--voice", "other_voice"]
        args = self.parse(argv)
        updated = train._apply_overrides(config, args)

        assert updated.epochs == 1.0
        assert updated.lr == pytest.approx(3e-5)
        assert updated.voice_name == "other_voice"
        assert updated.dataset == config.dataset

    def test_path_overrides_resolve_against_the_cwd(self, workspace, monkeypatch):
        monkeypatch.chdir(workspace)
        config = make_config(workspace)
        args = self.parse(["--config", "x", "--dataset", "data/train.jsonl"])
        assert train._apply_overrides(config, args).dataset == workspace / "data" / "train.jsonl"

    def test_no_wandb_wins_over_the_config(self, workspace):
        config = make_config(workspace, "wandb:\n  enabled: true\n  project: p\n")
        args = self.parse(["--config", "x", "--no-wandb"])
        assert not train._apply_overrides(config, args).wandb.enabled

    def test_no_overrides_leaves_the_config_untouched(self, workspace):
        config = make_config(workspace)
        assert train._apply_overrides(config, self.parse(["--config", "x"])) is config

    def test_main_reports_a_bad_config_without_a_traceback(self, workspace, capsys):
        bad = workspace / "bad.yaml"
        bad.write_text("dataset: absent.jsonl\n", encoding="utf-8")
        assert train.main(["--config", str(bad)]) == 2
        assert "error:" in capsys.readouterr().err


class TestWandb:
    def test_stays_off_and_names_no_project_by_default(self, workspace, monkeypatch):
        monkeypatch.delenv("WANDB_PROJECT", raising=False)
        train._configure_wandb(make_config(workspace))
        assert "WANDB_PROJECT" not in os.environ
        assert os.environ.get("WANDB_DISABLED") == "true"

    def test_uses_the_configured_project_when_enabled(self, workspace, monkeypatch):
        monkeypatch.delenv("WANDB_DISABLED", raising=False)
        config = make_config(
            workspace, "wandb:\n  enabled: true\n  project: my-voices\n  tags: [lora, my_voice]\n"
        )
        train._configure_wandb(config)
        assert os.environ["WANDB_PROJECT"] == "my-voices"
        assert os.environ["WANDB_TAGS"] == "lora,my_voice"

    def test_the_project_name_can_only_come_from_the_config(self):
        """No default project: an open-source run must not report into someone else's space."""
        source = Path(train.__file__).read_text(encoding="utf-8")
        assignments = [
            line for line in source.splitlines() if "WANDB_PROJECT" in line and "=" in line
        ]
        assert assignments
        assert all("config.wandb.project" in line for line in assignments), assignments


class TestMergeHelpers:
    def test_base_model_is_read_from_the_adapter(self, tmp_path):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": "kova-ai/kova-tts-1", "r": 64}),
            encoding="utf-8",
        )
        assert merge.base_model_of(adapter) == "kova-ai/kova-tts-1"

    def test_a_directory_that_is_not_an_adapter_says_so(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a saved peft adapter"):
            merge.base_model_of(tmp_path)

    def test_an_unknown_dtype_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="dtype must be one of"):
            merge.merge_adapter(tmp_path, tmp_path / "out", dtype="int4")


# ------------------------------------------------------------------------ end-to-end smoke


def freest_gpu() -> str:
    """Index of the GPU with the most free memory, so a job running elsewhere is left alone.

    The index is nvidia-smi's, which is PCI-bus order. CUDA numbers devices FASTEST_FIRST by
    default, so this only selects the card it names if the child is also told to order by bus
    -- see ``CUDA_DEVICE_ORDER`` where this is used.
    """
    output = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    rows = [line.split(",") for line in output.strip().splitlines()]
    return max(rows, key=lambda row: int(row[1]))[0].strip()


#: Driver for the smoke test. It runs in a subprocess so that ``CUDA_VISIBLE_DEVICES`` is
#: honoured (torch caches the device list at first use, and an earlier test in the same session
#: may already have initialised CUDA on the other card) and so that the training run's GPU
#: memory is handed back the moment it finishes.
_DRIVER = """
import json, sys
from transformers import TrainerCallback
from kova_tts.finetune import config as config_mod, train

class Recorder(TrainerCallback):
    def __init__(self):
        self.losses = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            self.losses.append(float(logs["loss"]))

recorder = Recorder()
adapter = train.run(config_mod.load_config(sys.argv[1]), callbacks=[recorder])
with open(sys.argv[2], "w") as handle:
    json.dump({"losses": recorder.losses, "adapter": str(adapter)}, handle)
"""


@pytest.mark.gpu
@pytest.mark.weights
def test_end_to_end_training_lowers_the_loss_and_writes_a_loadable_adapter(tmp_path, capsys):
    """~15 optimizer steps against the real checkpoint, on synthetic audio codes.

    Synthetic codes are the point: the model has never seen this mapping from text to codes, so
    a working training loop has to visibly drive the loss down within a handful of steps. A run
    that trains nothing -- masked-out labels, a detached adapter, gradient checkpointing eating
    the graph -- shows up here as a flat curve.
    """
    import torch

    # Not just "an accelerator": this test picks its card through nvidia-smi and pins the
    # child process with CUDA_VISIBLE_DEVICES, neither of which means anything anywhere else.
    # Finetuning on Apple Silicon is a separate question -- see docs/apple-silicon.md -- and
    # a machine without CUDA should skip here rather than die inside subprocess.
    if not torch.cuda.is_available():
        pytest.skip("finetuning needs a CUDA device")

    rng = random.Random(0)
    lines = [
        {
            "text": prompt.training_example(
                f"Line number {index} of the smoke test corpus.",
                [rng.randrange(8192) for _ in range(64)],
            )
        }
        for index in range(10)
    ]
    corpus = tmp_path / "train.jsonl"
    corpus.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")

    config_file = tmp_path / "voice.yaml"
    config_file.write_text(
        "\n".join(
            [
                "dataset: train.jsonl",
                "voice: smoke",
                "output_dir: runs",
                "val_split: 0",
                "max_length: 256",
                "epochs: 3",
                "batch_size: 2",
                "grad_accum: 1",
                "lr: 0.002",
                "warmup_ratio: 0",
                "logging_steps: 1",
                "save_strategy: 'no'",
                "lora:",
                "  r: 16",
                "  alpha: 32",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Sanity-check the config before spending a subprocess and a model load on it.
    config_mod.load_config(config_file)

    driver = tmp_path / "driver.py"
    driver.write_text(_DRIVER, encoding="utf-8")
    result_file = tmp_path / "result.json"
    completed = subprocess.run(
        [sys.executable, str(driver), str(config_file), str(result_file)],
        env={
            **os.environ,
            # Both, together: CUDA_VISIBLE_DEVICES indexes CUDA's own enumeration, which is
            # FASTEST_FIRST unless told otherwise, while freest_gpu() reports nvidia-smi's
            # bus order. Setting only one of these picks the wrong card on a mixed-GPU box.
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": freest_gpu(),
        },
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-4000:]

    result = json.loads(result_file.read_text(encoding="utf-8"))
    losses = result["losses"]
    adapter = Path(result["adapter"])

    assert len(losses) >= 12, losses
    mean_first = sum(losses[:3]) / 3
    mean_last = sum(losses[-3:]) / 3
    with capsys.disabled():
        print(f"\nsmoke-test loss curve: {[round(value, 3) for value in losses]}")
        print(f"first three mean {mean_first:.3f} -> last three mean {mean_last:.3f}")
    assert mean_last < mean_first, losses

    # The run is reproducible from what it left behind.
    assert (adapter.parent / "config.yaml").is_file()
    assert (adapter / "adapter_config.json").is_file()
    assert (adapter / "adapter_model.safetensors").is_file()

    # And peft can load it back onto the base model.
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(paths.model_path(), dtype=torch.bfloat16)
    loaded = PeftModel.from_pretrained(base, str(adapter))
    trained = [
        parameter
        for name, parameter in loaded.named_parameters()
        if "lora_B" in name and parameter.abs().sum().item() > 0
    ]
    assert trained, "every lora_B is still zero, so nothing was learned"

    tokenizer = AutoTokenizer.from_pretrained(str(adapter))
    assert tokenizer.convert_tokens_to_ids("<|speech_end|>") == 136450
