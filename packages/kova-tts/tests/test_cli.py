"""The command line: dispatch, argument handling, and the import laziness ``paths`` depends on.

Everything here drives :func:`kova_tts.cli.main` directly with an argv list, which is the same
path the console script takes. The entry points the CLI dispatches to are replaced with fakes
installed into ``sys.modules``: that is exactly how :func:`kova_tts.cli._entry_point` resolves
them, so a test can assert what a subcommand *calls* without loading a model, and can make an
installed module look absent to check what the CLI says when an extra is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import types
from pathlib import Path

import numpy as np
import pytest

from kova_tts import cli

# --------------------------------------------------------------------------------- environment


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch, tmp_path):
    """A configured, entirely local environment: no .env, no Hub, no downloads."""
    monkeypatch.setenv("KOVA_DISABLE_DOTENV", "1")
    model = tmp_path / "model"
    model.mkdir()
    codec = tmp_path / "codec.pt"
    codec.write_bytes(b"")
    loras = tmp_path / "loras"
    (loras / "voice_a").mkdir(parents=True)
    (loras / "voice_a" / "adapter_config.json").write_text("{}")
    monkeypatch.setenv("KOVA_MODEL_PATH", str(model))
    monkeypatch.setenv("KOVA_CODEC_PATH", str(codec))
    monkeypatch.setenv("KOVA_WAVLM_PATH", str(model))
    monkeypatch.setenv("KOVA_LORA_DIR", str(loras))
    from kova_tts import paths

    paths.reset_dotenv_cache()
    yield
    paths.reset_dotenv_cache()


def fake_module(name: str, monkeypatch, **attributes) -> types.ModuleType:
    """Install a stand-in module, the way an entry point's target is resolved."""
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def missing_module(name: str, monkeypatch) -> None:
    """Make `name` un-importable, whether or not it exists on this machine.

    ``None`` in ``sys.modules`` is what the import system uses to mean "this is not available",
    so the test result does not depend on which extras the machine running it installed.
    """
    monkeypatch.setitem(sys.modules, name, None)


class FakeTTS:
    """Stands in for :class:`~kova_tts.engine.tts.KovaTTS`, recording what it was asked for."""

    sample_rate = 32_000
    calls: list[tuple] = []

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    @classmethod
    def from_pretrained(cls, model=None, **kwargs):
        instance = cls(model=model, **kwargs)
        cls.calls.append(("load", model, kwargs))
        return instance

    def clone(self, audio, transcript=None, *, name=None):
        type(self).calls.append(("clone", audio, transcript))
        return types.SimpleNamespace(name="cloned", is_clone=True, ref_text=transcript or "heard")

    def generate(self, text, voice=None, *, params=None, seed=None):
        type(self).calls.append(("generate", text, voice, params, seed))
        return np.zeros(self.sample_rate, dtype=np.float32)

    def stream(self, text, voice=None, *, params=None, seed=None):
        type(self).calls.append(("stream", text, voice, params, seed))
        half = np.zeros(self.sample_rate // 2, dtype=np.float32)
        yield types.SimpleNamespace(samples=half, is_final=False)
        yield types.SimpleNamespace(samples=half, is_final=True)

    def save(self, wav, path, sample_rate=None):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"RIFF fake wav")
        type(self).calls.append(("save", target, wav.size))
        return target


@pytest.fixture
def engine(monkeypatch) -> list[tuple]:
    """A fake engine module, and the list of calls made against it."""
    FakeTTS.calls = []
    fake_module("kova_tts.engine.tts", monkeypatch, KovaTTS=FakeTTS)
    return FakeTTS.calls


@pytest.fixture
def reference(tmp_path):
    """A reference clip for --clone-audio. Only its existence is checked; the engine is fake."""
    clip = tmp_path / "ref.wav"
    clip.write_bytes(b"RIFF")
    return clip


# --------------------------------------------------------------------------------------- help


class TestHelp:
    """``--help`` works for every subcommand, including the ones that forward it onwards."""

    @pytest.mark.parametrize(
        "argv",
        [
            ["--help"],
            ["paths", "--help"],
            ["generate", "--help"],
            ["prepare-data", "--help"],
            ["download", "--help"],
        ],
    )
    def test_help_exits_zero(self, argv, capsys):
        with pytest.raises(SystemExit) as caught:
            cli.main(argv)
        assert caught.value.code == 0
        assert "usage:" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["finetune", "merge", "serve"])
    def test_forwarded_help_is_the_real_parser(self, command, monkeypatch, capsys):
        """The forwarding subcommands hand ``--help`` to the parser that owns those flags."""
        printed: list[list[str]] = []

        def fake_main(argv):
            printed.append(argv)
            print(f"usage: kova-tts {command} ...")
            return 0

        module = {
            "finetune": "kova_tts.finetune.train",
            "merge": "kova_tts.finetune.merge",
            "serve": "kova_tts.server.app",
        }
        fake_module(module[command], monkeypatch, main=fake_main)

        assert cli.main([command, "--help"]) == 0
        assert printed == [["--help"]]
        assert f"kova-tts {command}" in capsys.readouterr().out

    def test_version(self, capsys):
        from kova_tts import __version__

        with pytest.raises(SystemExit) as caught:
            cli.main(["--version"])
        assert caught.value.code == 0
        assert __version__ in capsys.readouterr().out

    def test_no_command_is_an_error(self):
        with pytest.raises(SystemExit) as caught:
            cli.main([])
        assert caught.value.code == 2

    def test_unknown_command_is_an_error(self):
        with pytest.raises(SystemExit) as caught:
            cli.main(["nonesuch"])
        assert caught.value.code == 2

    def test_unrecognized_flag_is_rejected(self, capsys):
        """A typo on a normal subcommand is an error, not something quietly forwarded."""
        with pytest.raises(SystemExit) as caught:
            cli.main(["generate", "hello", "--temperture", "0.5"])
        assert caught.value.code == 2
        assert "unrecognized arguments" in capsys.readouterr().err


# -------------------------------------------------------------------------------------- paths


class TestPaths:
    def test_reports_every_artifact(self, capsys):
        assert cli.main(["paths"]) == 0
        out = capsys.readouterr().out
        for label in ("config", "model", "wavlm", "codec", "loras", "voices"):
            assert label in out
        assert "voice_a" in out

    def test_missing_artifact_is_reported_not_raised(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("KOVA_MODEL_PATH", str(tmp_path / "gone"))
        assert cli.main(["paths"]) == 1
        assert "ERROR" in capsys.readouterr().out


class TestLaziness:
    """``kova-tts paths`` must not pay for the model stack.

    Run in a subprocess because the assertion is about ``sys.modules``, and by the time this
    file runs another test module may already have imported half of torch.
    """

    SCRIPT = textwrap.dedent(
        """
        import sys
        import kova_tts                       # whatever the package itself costs
        before = set(sys.modules)
        from kova_tts.cli import main
        assert main(["paths"]) == 0, "paths failed"
        added = set(sys.modules) - before
        heavy = {"transformers", "peft", "gradio", "faster_whisper", "fastapi", "datasets"}
        print("ADDED", " ".join(sorted(m for m in added if m.split(".")[0] in heavy | {"torch"})))
        print("PRESENT", " ".join(sorted(m for m in heavy if m in sys.modules)))
        """
    )

    def run(self, tmp_path) -> tuple[str, float]:
        env = dict(os.environ)
        env["KOVA_DISABLE_DOTENV"] = "1"
        model = tmp_path / "model"
        model.mkdir(exist_ok=True)
        codec = tmp_path / "codec.pt"
        codec.write_bytes(b"")
        env.update(
            KOVA_MODEL_PATH=str(model),
            KOVA_CODEC_PATH=str(codec),
            KOVA_WAVLM_PATH=str(model),
            KOVA_LORA_DIR=str(model),
        )
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT], env=env, capture_output=True, text=True
        )
        elapsed = time.perf_counter() - started
        assert result.returncode == 0, result.stderr
        return result.stdout, elapsed

    def test_imports_nothing_heavy(self, tmp_path):
        stdout, _ = self.run(tmp_path)
        added = stdout.split("ADDED", 1)[1].split("\n", 1)[0].split()
        present = stdout.split("PRESENT", 1)[1].split("\n", 1)[0].split()
        # Nothing heavy is present at all, and the CLI adds nothing on top of the package
        # import -- torch included, which is what kova_codec's lazy __init__ buys.
        assert present == []
        assert added == []

    def test_stays_fast(self, tmp_path):
        _, elapsed = self.run(tmp_path)
        # Generous on purpose: the bound is here to catch a model stack creeping into the import
        # path (seconds), not to police a hundred milliseconds on a busy machine.
        assert elapsed < 8.0


# ----------------------------------------------------------------------------------- generate


class TestGenerate:
    def test_writes_a_file_and_reports_it(self, engine, tmp_path, capsys):
        out = tmp_path / "out.wav"
        assert cli.main(["generate", "Hello there.", "--out", str(out)]) == 0
        assert ("generate", "Hello there.", None, None, None) in engine
        assert out.is_file()
        assert str(out) in capsys.readouterr().out

    def test_text_file(self, engine, tmp_path):
        script = tmp_path / "line.txt"
        script.write_text("  From a file.\n", encoding="utf-8")
        assert (
            cli.main(["generate", "--text-file", str(script), "-o", str(tmp_path / "a.wav")]) == 0
        )
        assert ("generate", "From a file.", None, None, None) in engine

    def test_stdin(self, engine, tmp_path, monkeypatch):
        monkeypatch.setattr("sys.stdin", __import__("io").StringIO("Piped in."))
        assert cli.main(["generate", "--text-file", "-", "-o", str(tmp_path / "a.wav")]) == 0
        assert ("generate", "Piped in.", None, None, None) in engine

    def test_missing_text_file(self, engine, tmp_path, capsys):
        code = cli.main(["generate", "--text-file", str(tmp_path / "nope.txt")])
        assert code == 2
        assert "not found" in capsys.readouterr().err

    def test_empty_text(self, engine, capsys):
        assert cli.main(["generate", "   "]) == 2
        assert "Nothing to speak" in capsys.readouterr().err

    def test_text_twice(self, engine, tmp_path, capsys):
        script = tmp_path / "line.txt"
        script.write_text("text", encoding="utf-8")
        assert cli.main(["generate", "spoken", "--text-file", str(script)]) == 2
        assert "not both" in capsys.readouterr().err

    def test_voice_is_passed_through(self, engine, tmp_path):
        out = tmp_path / "a.wav"
        assert cli.main(["generate", "Hi.", "--voice", "voice_a", "-o", str(out)]) == 0
        assert ("generate", "Hi.", "voice_a", None, None) in engine

    def test_artifact_flags_reach_the_engine(self, engine, tmp_path):
        assert (
            cli.main(
                [
                    "generate",
                    "Hi.",
                    "-o",
                    str(tmp_path / "a.wav"),
                    "--model",
                    "some/repo",
                    "--device",
                    "cpu",
                    "--lora-dir",
                    str(tmp_path),
                ]
            )
            == 0
        )
        _, model, kwargs = engine[0]
        assert model == "some/repo"
        assert kwargs["device"] == "cpu"
        assert kwargs["lora_root"] == str(tmp_path)
        assert callable(kwargs["transcriber"])

    def test_seed_and_sampling_overrides(self, engine, tmp_path):
        argv = [
            "generate",
            "Hi.",
            "-o",
            str(tmp_path / "a.wav"),
            "--seed",
            "7",
            "--temperature",
            "0.5",
            "--top-k",
            "12",
        ]
        assert cli.main(argv) == 0
        call = next(c for c in engine if c[0] == "generate")
        params, seed = call[3], call[4]
        assert seed == 7
        assert (params.temperature, params.top_k) == (0.5, 12)
        # Untouched knobs keep the plain-TTS preset rather than the dataclass defaults.
        from kova_tts import TTS_SAMPLING

        assert params.top_p == TTS_SAMPLING.top_p

    def test_no_overrides_leaves_the_preset_alone(self, engine, tmp_path):
        assert cli.main(["generate", "Hi.", "-o", str(tmp_path / "a.wav")]) == 0
        assert next(c for c in engine if c[0] == "generate")[3] is None

    def test_invalid_sampling_value_is_rejected_before_loading(self, engine, tmp_path, capsys):
        assert cli.main(["generate", "Hi.", "-o", str(tmp_path / "a.wav"), "--top-p", "3"]) == 2
        assert "top_p" in capsys.readouterr().err
        assert engine == []  # the model was never loaded

    def test_stream_reports_first_audio(self, engine, tmp_path, capsys):
        out = tmp_path / "a.wav"
        assert cli.main(["generate", "Hi.", "-o", str(out), "--stream"]) == 0
        assert any(call[0] == "stream" for call in engine)
        assert "first audio" in capsys.readouterr().out
        assert ("save", out, FakeTTS.sample_rate) in engine

    def test_empty_audio_is_an_error(self, engine, tmp_path, monkeypatch, capsys):
        monkeypatch.setattr(
            FakeTTS, "generate", lambda *a, **k: np.zeros(0, dtype=np.float32), raising=True
        )
        assert cli.main(["generate", "Hi.", "-o", str(tmp_path / "a.wav")]) == 2
        assert "no audio" in capsys.readouterr().err

    def test_unloadable_checkpoint_is_a_message_not_a_traceback(
        self, engine, monkeypatch, capsys, tmp_path
    ):
        """What transformers raises for a checkpoint it cannot find is an OSError."""

        def refuse(model=None, **kwargs):
            raise OSError("some/repo is not a local folder and is not a valid model identifier")

        monkeypatch.setattr(FakeTTS, "from_pretrained", refuse, raising=True)
        assert cli.main(["generate", "Hi.", "-o", str(tmp_path / "a.wav")]) == 2
        assert "not a valid model identifier" in capsys.readouterr().err

    def test_engine_import_failure_is_a_message(self, monkeypatch, capsys, tmp_path):
        missing_module("kova_tts.engine.tts", monkeypatch)
        assert cli.main(["generate", "Hi.", "-o", str(tmp_path / "a.wav")]) == 2
        assert "TTS engine is not available" in capsys.readouterr().err


class TestCloning:
    def test_clone_with_a_transcript(self, engine, tmp_path, reference):
        argv = [
            "generate",
            "Say something new.",
            "-o",
            str(tmp_path / "a.wav"),
            "--clone-audio",
            str(reference),
            "--clone-text",
            "what the clip says",
        ]
        assert cli.main(argv) == 0
        assert ("clone", str(reference), "what the clip says") in engine
        # A cloned voice gets the cloning preset when it is overridden, not the TTS one.
        assert next(c for c in engine if c[0] == "generate")[2].is_clone

    def test_clone_without_a_transcript_asks_the_transcriber(
        self, engine, tmp_path, reference, capsys
    ):
        argv = [
            "generate",
            "Say something new.",
            "-o",
            str(tmp_path / "a.wav"),
            "--clone-audio",
            str(reference),
        ]
        assert cli.main(argv) == 0
        assert ("clone", str(reference), None) in engine
        assert "reference heard as" in capsys.readouterr().err

    def test_cloning_preset_is_the_override_base(self, engine, tmp_path, reference):
        argv = [
            "generate",
            "Hi.",
            "-o",
            str(tmp_path / "a.wav"),
            "--clone-audio",
            str(reference),
            "--clone-text",
            "x",
            "--temperature",
            "0.8",
        ]
        assert cli.main(argv) == 0
        from kova_tts import CLONE_SAMPLING

        params = next(c for c in engine if c[0] == "generate")[3]
        assert (params.temperature, params.top_k) == (0.8, CLONE_SAMPLING.top_k)

    def test_missing_reference_file(self, engine, tmp_path, capsys):
        code = cli.main(["generate", "Hi.", "--clone-audio", str(tmp_path / "nope.wav")])
        assert code == 2
        assert "Reference audio not found" in capsys.readouterr().err
        assert engine == []

    def test_voice_and_clone_are_exclusive(self, engine, reference, capsys):
        code = cli.main(["generate", "Hi.", "--voice", "voice_a", "--clone-audio", str(reference)])
        assert code == 2
        assert "not both" in capsys.readouterr().err

    def test_clone_text_without_audio(self, engine, capsys):
        assert cli.main(["generate", "Hi.", "--clone-text", "orphan"]) == 2
        assert "--clone-audio" in capsys.readouterr().err


class TestAsrSeam:
    """The ``callable(path) -> str`` that lets the engine clone without a transcript."""

    def test_transcribes_through_the_data_extra(self, monkeypatch, tmp_path, capsys):
        clip = tmp_path / "ref.wav"
        clip.write_bytes(b"RIFF")
        seen: list = []

        class Fake:
            def transcribe(self, wav, sample_rate=32_000):
                seen.append(wav)
                return "the reference transcript"

        monkeypatch.setattr("kova_tts.audio.load_audio", lambda path, rate=32_000: np.zeros(4))
        monkeypatch.setattr("kova_tts.data.asr.load_transcriber", lambda **kw: Fake())

        transcribe = cli.asr_transcriber("small", language="en")
        assert transcribe(str(clip)) == "the reference transcript"
        assert len(seen) == 1
        assert "transcribing" in capsys.readouterr().err

    def test_model_is_built_once(self, monkeypatch, tmp_path):
        builds: list[dict] = []

        class Fake:
            def transcribe(self, wav, sample_rate=32_000):
                return "x"

        def build(**options):
            builds.append(options)
            return Fake()

        monkeypatch.setattr("kova_tts.audio.load_audio", lambda path, rate=32_000: np.zeros(4))
        monkeypatch.setattr("kova_tts.data.asr.load_transcriber", build)

        transcribe = cli.asr_transcriber(device="cpu")
        transcribe("a.wav")
        transcribe("b.wav")
        assert builds == [{"device": "cpu"}]

    def test_missing_extra_names_the_extra(self, monkeypatch):
        from kova_tts.data.asr import MissingDependency

        def refuse(**options):
            raise MissingDependency("Transcription needs faster-whisper: pip install ...")

        monkeypatch.setattr("kova_tts.data.asr.load_transcriber", refuse)
        with pytest.raises(cli.CommandError, match="faster-whisper"):
            cli.asr_transcriber()("a.wav")

    def test_the_engine_reports_it_as_a_command_error(
        self, engine, monkeypatch, tmp_path, reference, capsys
    ):
        """A missing ``data`` extra surfaces from inside clone() as one sentence, not a stack."""

        def refuse(self, audio, transcript=None, *, name=None):
            raise cli.CommandError("Transcription needs faster-whisper: pip install ...")

        monkeypatch.setattr(FakeTTS, "clone", refuse, raising=True)
        argv = ["generate", "Hi.", "-o", str(tmp_path / "a.wav"), "--clone-audio", str(reference)]
        assert cli.main(argv) == 2
        assert "faster-whisper" in capsys.readouterr().err


# ------------------------------------------------------------------------- forwarded commands


class TestPrepareData:
    def test_dispatches_with_parsed_arguments(self, monkeypatch, tmp_path):
        captured: list = []
        fake_module(
            "kova_tts.data.prepare",
            monkeypatch,
            run=lambda args: captured.append(args) or 0,
            # ``build_parser`` grafts the real flags before dispatch, so the stand-in has to
            # offer them too; borrowing the real function keeps the two in step.
            add_arguments=_real_prepare_arguments(),
        )
        assert cli.main(["prepare-data", str(tmp_path), "--val-split", "0.1", "--dry-run"]) == 0
        (args,) = captured
        assert args.source == tmp_path
        assert (args.val_split, args.dry_run) == (0.1, True)

    def test_flags_are_not_restated(self):
        """The subparser hosts the pipeline's own flags rather than a copy of them."""
        from kova_tts.data.prepare import build_parser

        parser = cli.build_parser(["prepare-data"])
        action = parser._subparsers._group_actions[0]  # noqa: SLF001 - argparse has no public API
        hosted = {a.dest for a in action.choices["prepare-data"]._actions}
        assert {a.dest for a in build_parser()._actions} <= hosted

    def test_bad_source_is_reported(self, tmp_path, capsys):
        assert cli.main(["prepare-data", str(tmp_path / "nowhere"), "--dry-run"]) == 2
        assert "error:" in capsys.readouterr().err


def _real_prepare_arguments():
    from kova_tts.data.prepare import add_arguments

    return add_arguments


class TestFinetuneAndMerge:
    @pytest.mark.parametrize(
        ("command", "module", "argv"),
        [
            ("finetune", "kova_tts.finetune.train", ["--config", "run.yaml", "--epochs", "2"]),
            ("merge", "kova_tts.finetune.merge", ["adapter", "out", "--dtype", "float16"]),
        ],
    )
    def test_arguments_are_forwarded_verbatim(self, command, module, argv, monkeypatch):
        seen: list[list[str]] = []
        fake_module(module, monkeypatch, main=lambda forwarded: seen.append(forwarded) or 0)
        assert cli.main([command, *argv]) == 0
        assert seen == [argv]

    def test_exit_code_comes_from_the_target(self, monkeypatch):
        fake_module("kova_tts.finetune.train", monkeypatch, main=lambda argv: 2)
        assert cli.main(["finetune", "--config", "missing.yaml"]) == 2

    def test_missing_extra_names_it(self, monkeypatch, capsys):
        missing_module("kova_tts.finetune.train", monkeypatch)
        assert cli.main(["finetune", "--config", "x.yaml"]) == 2
        assert "kova-tts[finetune]" in capsys.readouterr().err


class TestServe:
    """``serve`` forwards to :mod:`kova_tts.server.app`, which lives behind the server extra."""

    def test_missing_module_names_the_extra(self, monkeypatch, capsys):
        missing_module("kova_tts.server.app", monkeypatch)
        assert cli.main(["serve"]) == 2
        error = capsys.readouterr().err
        assert "server is not available" in error
        assert "kova-tts[server]" in error

    def test_forwards_its_arguments(self, monkeypatch):
        seen: list[list[str]] = []
        fake_module("kova_tts.server.app", monkeypatch, main=lambda argv: seen.append(argv) or 0)
        assert cli.main(["serve", "--host", "0.0.0.0", "--port", "8080"]) == 0
        assert seen == [["--host", "0.0.0.0", "--port", "8080"]]

    def test_a_target_returning_nothing_still_exits_zero(self, monkeypatch):
        fake_module("kova_tts.server.app", monkeypatch, main=lambda argv: None)
        assert cli.main(["serve"]) == 0

    def test_module_without_main(self, monkeypatch, capsys):
        fake_module("kova_tts.server.app", monkeypatch)
        assert cli.main(["serve"]) == 2
        assert "no main()" in capsys.readouterr().err


class TestDemo:
    """The demo ships in ``apps/``, not in the package, so it is loaded from the checkout."""

    def test_missing_app_explains_where_it_looked(self, monkeypatch, tmp_path, capsys):
        monkeypatch.setattr(cli, "_demo_app_path", lambda: tmp_path / "apps" / "demo" / "app.py")
        assert cli.main(["demo"]) == 2
        error = capsys.readouterr().err
        assert "demo app was not found" in error
        assert "app.py" in error

    def test_loads_and_forwards(self, monkeypatch, tmp_path):
        app = tmp_path / "app.py"
        marker = tmp_path / "argv.txt"
        app.write_text(
            textwrap.dedent(
                f"""
                def main(argv=None):
                    open({str(marker)!r}, "w").write(" ".join(argv or []))
                    return 0
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(cli, "_demo_app_path", lambda: app)
        assert cli.main(["demo", "--share"]) == 0
        assert marker.read_text() == "--share"

    def test_missing_extra_names_it(self, monkeypatch, tmp_path, capsys):
        app = tmp_path / "app.py"
        app.write_text("import gradio_not_installed\n", encoding="utf-8")
        monkeypatch.setattr(cli, "_demo_app_path", lambda: app)
        assert cli.main(["demo"]) == 2
        assert "kova-tts[demo]" in capsys.readouterr().err

    def test_help_is_forwarded(self, monkeypatch, tmp_path, capsys):
        """``kova-tts demo --help`` prints the demo's own options, not a paraphrase."""
        app = tmp_path / "app.py"
        app.write_text(
            "def main(argv=None):\n    print('usage: kova-tts demo ...')\n    return 0\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(cli, "_demo_app_path", lambda: app)
        assert cli.main(["demo", "--help"]) == 0
        assert "usage: kova-tts demo" in capsys.readouterr().out

    def test_app_path_is_the_checkout(self):
        """The default location tracks this file, so a source checkout finds the app."""
        assert cli._demo_app_path().parts[-3:] == ("apps", "demo", "app.py")


# ----------------------------------------------------------------------------------- download


class TestDownload:
    def test_fetches_model_and_codec(self, monkeypatch, tmp_path, capsys):
        snapshot = tmp_path / "snapshot"
        snapshot.mkdir()
        (snapshot / "model.safetensors").write_bytes(b"x" * 2048)
        codec = tmp_path / "codec.pt"
        codec.write_bytes(b"y" * 1024)
        asked: list[tuple] = []

        monkeypatch.delenv("KOVA_CODEC_PATH")
        monkeypatch.delenv("KOVA_MODEL_PATH")
        fake_module(
            "huggingface_hub",
            monkeypatch,
            snapshot_download=lambda repo, **kw: asked.append(("snapshot", repo)) or str(snapshot),
            hf_hub_download=lambda repo, filename, **kw: (
                asked.append((repo, filename)) or str(codec)
            ),
        )
        assert cli.main(["download"]) == 0
        out = capsys.readouterr().out
        assert "model" in out and "codec" in out and "total" in out
        assert ("snapshot", "kova-ai/kova-tts-1b") in asked

    def test_local_artifacts_are_skipped(self, monkeypatch, capsys):
        fake_module(
            "huggingface_hub",
            monkeypatch,
            snapshot_download=lambda *a, **k: pytest.fail("should not download"),
            hf_hub_download=lambda *a, **k: pytest.fail("should not download"),
        )
        assert cli.main(["download"]) == 0
        out = capsys.readouterr().out
        assert out.count("already local") == 2
        assert "0 B downloaded" in out

    def test_wavlm_is_opt_in(self, monkeypatch, capsys):
        monkeypatch.delenv("KOVA_WAVLM_PATH")
        seen: list[str] = []
        fake_module(
            "huggingface_hub",
            monkeypatch,
            snapshot_download=lambda repo, **kw: seen.append(repo) or "/tmp",
            hf_hub_download=lambda *a, **k: "/tmp/codec.pt",
        )
        assert cli.main(["download"]) == 0
        assert seen == []
        assert cli.main(["download", "--wavlm"]) == 0
        assert seen == ["microsoft/wavlm-large"]

    def test_hub_failure_is_actionable(self, monkeypatch, capsys):
        monkeypatch.delenv("KOVA_MODEL_PATH")

        def explode(*args, **kwargs):
            raise OSError("404 Client Error: Repository Not Found")

        fake_module(
            "huggingface_hub", monkeypatch, snapshot_download=explode, hf_hub_download=explode
        )
        assert cli.main(["download"]) == 2
        error = capsys.readouterr().err
        assert "Could not download" in error
        assert "KOVA_HUB_REPO" in error
        assert "kova-tts paths" in error

    def test_script_wrapper_shares_the_parser(self, tmp_path):
        """``scripts/download_weights.py`` hosts the same flags rather than its own."""
        import argparse

        parser = cli.add_download_arguments(argparse.ArgumentParser())
        args = parser.parse_args(["--wavlm", "--repo", "some/repo"])
        assert (args.wavlm, args.repo) == (True, "some/repo")


# ---------------------------------------------------------------------------- with real weights


@pytest.mark.gpu
@pytest.mark.weights
class TestRealGeneration:
    def test_generate_writes_a_plausible_wav(self, tmp_path, monkeypatch):
        import soundfile as sf

        monkeypatch.delenv("KOVA_DISABLE_DOTENV", raising=False)
        for var in ("KOVA_MODEL_PATH", "KOVA_CODEC_PATH", "KOVA_WAVLM_PATH", "KOVA_LORA_DIR"):
            monkeypatch.delenv(var, raising=False)
        from kova_tts import paths

        paths.reset_dotenv_cache()
        paths.load_dotenv(force=True)

        out = tmp_path / "spoken.wav"
        text = "The kettle boiled while the rain kept up against the window."
        code = cli.main(
            ["generate", text, "--out", str(out), "--seed", "1234", "--device", _free_device()]
        )
        assert code == 0

        wav, rate = sf.read(str(out), dtype="float32")
        assert rate == 32_000
        # Roughly a syllable every 200 ms: anything far outside that is a broken generation,
        # not a slow reader.
        assert 2.0 < wav.size / rate < 20.0
        assert float(np.abs(wav).max()) > 0.05


def _free_device() -> str:
    """The CUDA device with the most memory free, so a parallel test run is not fought over."""
    try:
        import torch

        # torch.cuda.mem_get_info, not nvidia-smi: CUDA defaults to FASTEST_FIRST device
        # ordering, so nvidia-smi's GPU 0 is not necessarily torch's cuda:0. Indexing one tool
        # by the other's order silently selects the wrong card.
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return f"cuda:{max(range(len(free)), key=free.__getitem__)}" if free else "cuda"
    except Exception:  # noqa: BLE001 - any failure just means "let torch choose"
        return "cuda"
