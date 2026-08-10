"""Picking a decode loop: the artifact decides, and an explicit choice can only refuse one."""

from __future__ import annotations

import json

import pytest

from kova_tts import paths
from kova_tts.engine import backends


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """No ``.env`` on disk gets to decide what these tests resolve."""
    monkeypatch.setenv(paths.ENV_DISABLE_DOTENV, "1")
    for var in (paths.ENV_MODEL, paths.ENV_HUB_REPO):
        monkeypatch.delenv(var, raising=False)
    paths.reset_dotenv_cache()
    yield
    paths.reset_dotenv_cache()


def model_dir(tmp_path, name: str, config: dict) -> str:
    directory = tmp_path / name
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return str(directory)


@pytest.fixture
def plain(tmp_path):
    """A bf16 transformers checkpoint: full vocabulary, tied head."""
    return model_dir(tmp_path, "bf16", {"model_type": "llama", "vocab_size": 136576})


@pytest.fixture
def converted(tmp_path):
    """An Apple conversion: quantized, with the head geometry recorded."""
    return model_dir(
        tmp_path,
        "mlx",
        {
            "model_type": "llama",
            "vocab_size": 136576,
            "head_vocab_size": 8195,
            "head_vocab_offset": 128256,
            "quantization": {"group_size": 128, "bits": 4},
        },
    )


class TestIsMLXArtifact:
    def test_a_converted_directory_is_recognised(self, converted):
        assert backends.is_mlx_artifact(converted)

    def test_a_plain_checkpoint_is_not(self, plain):
        assert not backends.is_mlx_artifact(plain)

    def test_a_hub_repo_id_is_not(self):
        # Answering would mean downloading a config to decide which backend does the
        # downloading. A repo id is assumed unconverted.
        assert not backends.is_mlx_artifact("kova-ai/kova-tts-1b")

    def test_an_unreadable_config_is_not(self, tmp_path):
        directory = tmp_path / "broken"
        directory.mkdir()
        (directory / "config.json").write_text("{not json", encoding="utf-8")
        # Reported by the loader, with the loader's message; all this owes us is a backend.
        assert not backends.is_mlx_artifact(directory)


class TestResolve:
    def test_a_plain_checkpoint_resolves_to_torch(self, plain):
        assert backends.resolve(plain) == "torch"

    def test_torch_can_be_asked_for_explicitly(self, plain):
        assert backends.resolve(plain, "torch") == "torch"

    def test_auto_is_the_same_as_no_request(self, plain):
        assert backends.resolve(plain, "auto") == backends.resolve(plain)

    def test_a_converted_checkpoint_will_not_run_under_torch(self, converted):
        with pytest.raises(ValueError, match="converted for the MLX backend"):
            backends.resolve(converted, "torch")

    def test_an_unknown_backend_is_rejected(self, plain):
        with pytest.raises(ValueError, match="backend must be one of"):
            backends.resolve(plain, "tensorrt")

    def test_the_model_argument_wins_over_the_environment(self, plain, converted, monkeypatch):
        monkeypatch.setenv(paths.ENV_MODEL, converted)
        assert backends.resolve(plain) == "torch"

    @pytest.mark.skipif(not backends.mlx_available(), reason="needs mlx")
    def test_a_converted_checkpoint_resolves_to_mlx(self, converted):
        assert backends.resolve(converted) == "mlx"

    @pytest.mark.skipif(backends.mlx_available(), reason="needs a machine without mlx")
    def test_without_mlx_a_converted_checkpoint_says_so(self, converted):
        with pytest.raises(ImportError, match="mlx is not installed"):
            backends.resolve(converted)

    @pytest.mark.skipif(backends.mlx_available(), reason="needs a machine without mlx")
    def test_asking_for_mlx_without_mlx_says_so(self, plain):
        with pytest.raises(ImportError, match="Apple Silicon only"):
            backends.resolve(plain, "mlx")
