"""Artifact resolution: explicit argument > KOVA_* env var > Hugging Face Hub."""

from __future__ import annotations

import pytest

from kova_tts import paths


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Run against a clean environment with no .env on disk influencing the result."""
    monkeypatch.setenv(paths.ENV_DISABLE_DOTENV, "1")
    for var in (
        paths.ENV_MODEL,
        paths.ENV_CODEC,
        paths.ENV_WAVLM,
        paths.ENV_LORA_DIR,
        paths.ENV_HUB_REPO,
    ):
        monkeypatch.delenv(var, raising=False)
    paths.reset_dotenv_cache()
    yield
    paths.reset_dotenv_cache()


@pytest.fixture
def model_dir(tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    (d / "config.json").write_text("{}")
    return d


@pytest.fixture
def lora_root(tmp_path):
    root = tmp_path / "loras"
    for name in ("alpha", "bravo"):
        (root / name).mkdir(parents=True)
        (root / name / "adapter_config.json").write_text("{}")
    (root / "not-a-lora").mkdir()
    return root


class TestResolutionOrder:
    def test_falls_back_to_hub_when_unset(self):
        assert paths.model_path() == paths.DEFAULT_HUB_REPO
        assert paths.wavlm_path() == paths.DEFAULT_WAVLM_REPO
        assert paths.lora_dir() is None

    def test_env_var_is_used(self, monkeypatch, model_dir):
        monkeypatch.setenv(paths.ENV_MODEL, str(model_dir))
        assert paths.model_path() == str(model_dir)

    def test_explicit_argument_beats_env_var(self, monkeypatch, model_dir, tmp_path):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.setenv(paths.ENV_MODEL, str(model_dir))
        assert paths.model_path(other) == str(other)

    def test_hub_repo_override(self, monkeypatch):
        monkeypatch.setenv(paths.ENV_HUB_REPO, "someone/fork")
        assert paths.model_path() == "someone/fork"

    def test_home_relative_path_is_expanded(self, monkeypatch, model_dir):
        monkeypatch.setenv("HOME", str(model_dir.parent))
        monkeypatch.setenv(paths.ENV_MODEL, "~/model")
        assert paths.model_path() == str(model_dir)


class TestValidation:
    def test_missing_directory_raises_rather_than_downloading(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.ENV_MODEL, str(tmp_path / "nope"))
        with pytest.raises(paths.MissingArtifact, match="Model directory not found"):
            paths.model_path()

    def test_error_names_the_env_var_to_fix(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.ENV_CODEC, str(tmp_path / "nope.pt"))
        with pytest.raises(paths.MissingArtifact, match=paths.ENV_CODEC):
            paths.codec_path()

    def test_file_where_a_directory_is_expected(self, monkeypatch, tmp_path):
        stray = tmp_path / "model.txt"
        stray.write_text("")
        monkeypatch.setenv(paths.ENV_MODEL, str(stray))
        with pytest.raises(paths.MissingArtifact, match="must be a directory"):
            paths.model_path()

    def test_directory_where_a_file_is_expected(self, monkeypatch, tmp_path):
        monkeypatch.setenv(paths.ENV_CODEC, str(tmp_path))
        with pytest.raises(paths.MissingArtifact, match="must be a file"):
            paths.codec_path()

    def test_empty_env_var_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv(paths.ENV_MODEL, "   ")
        assert paths.model_path() == paths.DEFAULT_HUB_REPO

    def test_hub_id_is_not_checked_on_disk(self, monkeypatch):
        monkeypatch.setenv(paths.ENV_MODEL, "someone/some-model")
        assert paths.model_path() == "someone/some-model"


class TestLoras:
    def test_lists_only_directories_with_an_adapter_config(self, monkeypatch, lora_root):
        monkeypatch.setenv(paths.ENV_LORA_DIR, str(lora_root))
        assert paths.available_loras() == ["alpha", "bravo"]

    def test_resolves_a_named_adapter(self, monkeypatch, lora_root):
        monkeypatch.setenv(paths.ENV_LORA_DIR, str(lora_root))
        assert paths.lora_path("alpha") == lora_root / "alpha"

    def test_unknown_name_lists_what_is_available(self, monkeypatch, lora_root):
        monkeypatch.setenv(paths.ENV_LORA_DIR, str(lora_root))
        with pytest.raises(paths.MissingArtifact, match="Available: alpha, bravo"):
            paths.lora_path("nobody")

    def test_no_lora_dir_configured(self):
        assert paths.available_loras() == []
        with pytest.raises(paths.MissingArtifact, match=paths.ENV_LORA_DIR):
            paths.lora_path("alpha")


class TestDotenv:
    def test_reads_nearest_dotenv(self, monkeypatch, tmp_path, model_dir):
        (tmp_path / ".env").write_text(f"{paths.ENV_MODEL}={model_dir}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(paths.ENV_DISABLE_DOTENV, raising=False)
        paths.reset_dotenv_cache()
        assert paths.model_path() == str(model_dir)

    def test_real_env_wins_over_dotenv(self, monkeypatch, tmp_path, model_dir):
        (tmp_path / ".env").write_text(f"{paths.ENV_MODEL}={tmp_path / 'from-dotenv'}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv(paths.ENV_DISABLE_DOTENV, raising=False)
        monkeypatch.setenv(paths.ENV_MODEL, str(model_dir))
        paths.reset_dotenv_cache()
        assert paths.model_path() == str(model_dir)

    def test_disable_flag_skips_the_file(self, monkeypatch, tmp_path, model_dir):
        (tmp_path / ".env").write_text(f"{paths.ENV_MODEL}={model_dir}\n")
        monkeypatch.chdir(tmp_path)
        paths.reset_dotenv_cache()
        assert paths.model_path() == paths.DEFAULT_HUB_REPO

    @pytest.mark.parametrize("value", ["0", "false", "FALSE", "no", "off", "", "  "])
    def test_a_falsy_disable_flag_still_reads_the_file(
        self, monkeypatch, tmp_path, model_dir, value
    ):
        """``KOVA_DISABLE_DOTENV=0`` means "do not disable" -- the opposite of truthiness."""
        (tmp_path / ".env").write_text(f"{paths.ENV_MODEL}={model_dir}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV_DISABLE_DOTENV, value)
        paths.reset_dotenv_cache()
        assert paths.model_path() == str(model_dir)

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
    def test_a_truthy_disable_flag_skips_the_file(self, monkeypatch, tmp_path, model_dir, value):
        (tmp_path / ".env").write_text(f"{paths.ENV_MODEL}={model_dir}\n")
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv(paths.ENV_DISABLE_DOTENV, value)
        paths.reset_dotenv_cache()
        assert paths.model_path() == paths.DEFAULT_HUB_REPO
