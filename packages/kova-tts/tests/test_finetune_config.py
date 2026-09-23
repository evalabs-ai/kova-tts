"""YAML config loading and validation.

The point of validation here is that a mistake costs a GPU-hour to discover otherwise, so every
test below asserts on the *message* as well as the failure: a rejection that does not say what
to fix is barely better than a silent default.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

import kova_tts.finetune
from kova_tts.finetune.config import (
    ConfigError,
    FinetuneConfig,
    from_dict,
    load_config,
    snapshot,
)
from kova_tts.finetune.loss import EndingWeight

#: The example config shipped inside the package, which a stranger is expected to copy.
EXAMPLE_CONFIG = Path(kova_tts.finetune.__file__).parent / "configs" / "example_voice.yaml"


@pytest.fixture
def workspace(tmp_path):
    """A directory holding a corpus, so configs that reference it validate."""
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    (tmp_path / "data" / "val.jsonl").write_text('{"text": "x"}\n', encoding="utf-8")
    return tmp_path


def write_config(directory, body):
    path = directory / "voice.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class TestMinimalConfig:
    def test_one_line_is_enough(self, workspace):
        config = load_config(write_config(workspace, "dataset: data/train.jsonl\n"))

        assert config.dataset == workspace / "data" / "train.jsonl"
        assert config.voice_name == "train"
        assert config.output_dir == workspace / "runs"

    def test_defaults_are_the_proven_hyperparameters(self, workspace):
        config = load_config(write_config(workspace, "dataset: data/train.jsonl\n"))

        assert (config.lora.r, config.lora.alpha, config.lora.dropout) == (64, 64, 0.0)
        assert config.lora.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")
        assert (config.batch_size, config.grad_accum, config.max_length) == (2, 4, 4096)
        assert config.lr == pytest.approx(1.5e-4)
        assert config.lr_scheduler_type == "cosine"
        assert (config.warmup_ratio, config.weight_decay) == (0.01, 0.01)
        assert config.bf16 and config.gradient_checkpointing
        assert config.ending_weight == EndingWeight()
        assert not config.wandb.enabled and config.wandb.project is None

    def test_voice_names_the_run(self, workspace):
        body = "dataset: data/train.jsonl\nvoice: my_voice\n"
        config = load_config(write_config(workspace, body))
        assert config.voice_name == "my_voice"


class TestPathResolution:
    def test_relative_paths_resolve_against_the_config_file(self, workspace):
        nested = workspace / "configs"
        nested.mkdir()
        config = load_config(write_config(nested, "dataset: ../data/train.jsonl\n"))
        assert config.dataset == workspace / "data" / "train.jsonl"

    def test_absolute_paths_are_left_alone(self, workspace):
        target = workspace / "data" / "train.jsonl"
        config = load_config(write_config(workspace, f"dataset: {target}\n"))
        assert config.dataset == target

    def test_a_hub_id_is_not_treated_as_a_path(self, workspace):
        config = load_config(
            write_config(workspace, "dataset: data/train.jsonl\nmodel: kova-ai/kova-tts-1\n")
        )
        assert config.model == "kova-ai/kova-tts-1"

    def test_an_explicitly_relative_model_path_is_resolved(self, workspace):
        config = load_config(
            write_config(workspace, "dataset: data/train.jsonl\nmodel: ./checkpoints/base\n")
        )
        assert config.model == str(workspace / "checkpoints" / "base")


class TestValidation:
    def load(self, workspace, body):
        return load_config(write_config(workspace, body))

    def test_a_missing_dataset_key_is_named(self, workspace):
        with pytest.raises(ConfigError, match="'dataset' is required"):
            self.load(workspace, "voice: my_voice\n")

    def test_a_dataset_that_does_not_exist_explains_path_resolution(self, workspace):
        with pytest.raises(ConfigError, match="resolved relative"):
            self.load(workspace, "dataset: data/absent.jsonl\n")

    def test_a_typo_suggests_the_real_key(self, workspace):
        with pytest.raises(ConfigError, match="Did you mean 'max_length'"):
            self.load(workspace, "dataset: data/train.jsonl\nmax_lenght: 2048\n")

    def test_an_unknown_key_lists_the_valid_ones(self, workspace):
        with pytest.raises(ConfigError, match="Valid keys here:.*grad_accum"):
            self.load(workspace, "dataset: data/train.jsonl\nzzz_unlikely: 1\n")

    def test_a_typo_inside_a_section_is_scoped_to_that_section(self, workspace):
        with pytest.raises(ConfigError, match="lora.alpha"):
            self.load(workspace, "dataset: data/train.jsonl\nlora:\n  alpah: 32\n")

    @pytest.mark.parametrize(
        ("body", "message"),
        [
            ("val_split: 1.5", "fraction of the corpus, not a percentage"),
            ("val_split: -0.1", "val_split must be in"),
            ("epochs: 0", "epochs must be > 0"),
            ("batch_size: 0", "batch_size must be >= 1"),
            ("grad_accum: 0", "grad_accum must be >= 1"),
            ("lr: 0", "lr must be in"),
            ("lr: 5", "default is 1.5e-4"),
            ("max_length: 4", "80 audio tokens per second"),
            ("warmup_ratio: 2", "warmup_ratio must be in"),
            ("weight_decay: -1", "weight_decay must be >= 0"),
            ("save_strategy: sometimes", "one of no/epoch/steps"),
            ("logging_steps: 0", "logging_steps must be >= 1"),
            ("lora:\n  r: 0", "lora.r must be >= 1"),
            ("lora:\n  dropout: 1.0", "lora.dropout must be in"),
            ("lora:\n  target_modules: []", "no trainable weights"),
            ("ending_weight:\n  ramp_tokens: -3", "ramp_tokens must be >= 0"),
            ("ending_weight:\n  ramp_max: 0.5", "opposite of the intent"),
            ("ending_weight:\n  eos_token_weight: 0", "eos_token_weight must be >= 1.0"),
            ("wandb:\n  enabled: true", "wandb.project is unset"),
        ],
    )
    def test_bad_values_are_rejected_with_an_actionable_message(self, workspace, body, message):
        with pytest.raises(ConfigError, match=message):
            self.load(workspace, f"dataset: data/train.jsonl\n{body}\n")

    def test_a_string_where_a_list_belongs_shows_the_yaml_form(self, workspace):
        with pytest.raises(ConfigError, match=r"\[q_proj, k_proj"):
            self.load(workspace, "dataset: data/train.jsonl\nlora:\n  target_modules: q_proj\n")

    def test_a_section_that_is_not_a_mapping_is_rejected(self, workspace):
        with pytest.raises(ConfigError, match="'lora' must be a mapping"):
            self.load(workspace, "dataset: data/train.jsonl\nlora: 64\n")

    def test_an_init_adapter_without_a_config_file_is_rejected(self, workspace):
        (workspace / "adapter").mkdir()
        with pytest.raises(ConfigError, match="adapter_config.json"):
            self.load(workspace, "dataset: data/train.jsonl\ninit_from_adapter: adapter\n")

    def test_a_missing_file_is_reported_before_anything_is_parsed(self, workspace):
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config(workspace / "nope.yaml")

    def test_an_empty_file_suggests_the_minimum(self, workspace):
        with pytest.raises(ConfigError, match="A minimal config needs"):
            self.load(workspace, "\n")

    def test_broken_yaml_names_the_file(self, workspace):
        with pytest.raises(ConfigError, match="not valid YAML"):
            self.load(workspace, "dataset: [unclosed\n")

    def test_a_scalar_document_is_rejected(self, workspace):
        with pytest.raises(ConfigError, match="must contain a mapping"):
            self.load(workspace, "just-a-string\n")


class TestCoercion:
    def test_scientific_notation_becomes_a_float(self, workspace):
        config = load_config(write_config(workspace, "dataset: data/train.jsonl\nlr: 2e-5\n"))
        assert isinstance(config.lr, float)
        assert config.lr == pytest.approx(2e-5)

    def test_an_integer_epoch_count_becomes_a_float(self, workspace):
        config = load_config(write_config(workspace, "dataset: data/train.jsonl\nepochs: 3\n"))
        assert config.epochs == pytest.approx(3.0)

    def test_a_string_where_a_bool_belongs_is_rejected(self, workspace):
        with pytest.raises(ConfigError, match="must be true or false"):
            load_config(write_config(workspace, "dataset: data/train.jsonl\nbf16: yes please\n"))

    def test_lists_become_tuples_so_the_config_stays_hashable_and_frozen(self, workspace):
        config = load_config(
            write_config(
                workspace,
                "dataset: data/train.jsonl\nlora:\n  target_modules: [q_proj, v_proj]\n",
            )
        )
        assert config.lora.target_modules == ("q_proj", "v_proj")


class TestSnapshot:
    def test_resolved_config_round_trips_through_the_run_directory(self, workspace):
        config = load_config(
            write_config(workspace, "dataset: data/train.jsonl\nvoice: my_voice\nepochs: 2\n")
        )
        written = snapshot(config, workspace / "runs" / "ft_001")

        assert written.name == "config.yaml"
        reloaded = load_config(written)
        assert reloaded == config

    def test_snapshot_records_defaults_that_were_never_written_down(self, workspace):
        import yaml

        config = load_config(write_config(workspace, "dataset: data/train.jsonl\n"))
        written = snapshot(config, workspace / "runs" / "ft_001")
        data = yaml.safe_load(written.read_text(encoding="utf-8"))

        assert data["lora"]["r"] == 64
        assert data["ending_weight"] == {
            "enabled": True,
            "ramp_tokens": 10,
            "ramp_max": 4.5,
            "eos_token_weight": 3.0,
        }
        assert data["dataset"] == str(workspace / "data" / "train.jsonl")


class TestProgrammaticUse:
    def test_from_dict_does_not_validate(self, tmp_path):
        config = from_dict({"dataset": "nowhere.jsonl"}, base_dir=tmp_path)
        assert config.dataset == tmp_path / "nowhere.jsonl"
        with pytest.raises(ConfigError):
            config.validate()

    def test_the_config_is_frozen_and_replaceable(self, workspace):
        config = load_config(write_config(workspace, "dataset: data/train.jsonl\n"))
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.epochs = 9
        assert dataclasses.replace(config, epochs=9).epochs == 9

    def test_the_shipped_example_config_is_valid_apart_from_its_placeholder_paths(self, workspace):
        assert EXAMPLE_CONFIG.is_file()

        # Repoint the placeholder corpus at the fixture, then validate the rest as written.
        body = EXAMPLE_CONFIG.read_text(encoding="utf-8").replace(
            "dataset: data/train.jsonl", f"dataset: {workspace / 'data' / 'train.jsonl'}"
        )
        config = load_config(write_config(workspace, body))
        assert isinstance(config, FinetuneConfig)
        assert config.voice_name == "my_voice"

    def test_no_absolute_machine_paths_are_committed_in_the_example(self):
        assert "/mnt/" not in EXAMPLE_CONFIG.read_text(encoding="utf-8")
