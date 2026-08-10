"""Voice resolution, and the error messages it produces when a name does not resolve."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kova_codec.constants import SAMPLE_RATE
from kova_tts import paths, voices
from kova_tts.engine.types import Voice


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """No .env and no configured LoRA directory unless a test asks for one."""
    monkeypatch.setenv(paths.ENV_DISABLE_DOTENV, "1")
    monkeypatch.delenv(paths.ENV_LORA_DIR, raising=False)
    paths.reset_dotenv_cache()
    yield
    paths.reset_dotenv_cache()


@pytest.fixture
def lora_root(tmp_path):
    root = tmp_path / "loras"
    for name in ("alpha", "bravo"):
        (root / name).mkdir(parents=True)
        (root / name / "adapter_config.json").write_text("{}")
    (root / "not-an-adapter").mkdir()
    return root


class FakeCodec:
    """Encodes to one code per 400 samples, which is what the real codec's rate works out to."""

    device = torch.device("cpu")
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        self.seen: np.ndarray | None = None

    def encode(self, wav):
        self.seen = np.asarray(wav, dtype=np.float32)
        return torch.arange(self.seen.size // 400, dtype=torch.long) % 8192


class TestRegistry:
    def test_lists_only_directories_holding_an_adapter(self, lora_root):
        assert voices.VoiceRegistry(lora_root).names() == ["alpha", "bravo"]

    def test_reads_the_environment_when_no_root_is_given(self, lora_root, monkeypatch):
        monkeypatch.setenv(paths.ENV_LORA_DIR, str(lora_root))
        assert voices.available() == ["alpha", "bravo"]

    def test_is_empty_without_a_configured_directory(self):
        assert voices.available() == []
        assert len(voices.VoiceRegistry.load()) == 0

    def test_get_returns_a_lora_voice(self, lora_root):
        voice = voices.VoiceRegistry(lora_root).get("alpha")
        assert voice.name == "alpha" and voice.lora_path == lora_root / "alpha"
        assert not voice.is_clone

    def test_membership_and_iteration(self, lora_root):
        registry = voices.VoiceRegistry(lora_root)
        assert "alpha" in registry and "nobody" not in registry
        assert list(registry) == ["alpha", "bravo"]

    def test_describe_names_the_directory_it_searched(self, lora_root):
        assert str(lora_root) in voices.VoiceRegistry(lora_root).describe()

    def test_describe_says_when_nothing_is_configured(self):
        assert paths.ENV_LORA_DIR in voices.VoiceRegistry.load().describe()

    def test_describe_says_when_the_directory_is_empty(self, tmp_path):
        assert "no adapters in" in voices.VoiceRegistry(tmp_path).describe()


class TestResolve:
    def test_none_stays_none(self):
        assert voices.resolve(None) is None

    def test_a_voice_passes_through_untouched(self):
        voice = Voice(name="cloned", ref_codes=(1, 2, 3), ref_text="hello")
        assert voices.resolve(voice) is voice

    def test_a_name_resolves_through_the_registry(self, lora_root):
        assert voices.resolve("bravo", root=lora_root).lora_path == lora_root / "bravo"

    def test_an_unknown_name_lists_what_is_available(self, lora_root):
        with pytest.raises(paths.MissingArtifact, match="alpha, bravo"):
            voices.resolve("nobody", root=lora_root)

    def test_an_audio_path_is_told_to_clone_first(self, lora_root):
        with pytest.raises(paths.MissingArtifact, match="Clone it first"):
            voices.resolve("reference.wav", root=lora_root)

    def test_no_configured_directory_names_the_variable_to_set(self):
        with pytest.raises(paths.MissingArtifact, match=paths.ENV_LORA_DIR):
            voices.resolve("alpha")

    def test_a_wrong_type_says_what_a_voice_is(self):
        with pytest.raises(TypeError, match="name, a Voice, or None"):
            voices.resolve(42)


class TestFromAudio:
    def wav(self, seconds: float = 3.0) -> np.ndarray:
        t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
        return (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def test_encodes_a_waveform_into_a_cloneable_voice(self):
        codec = FakeCodec()
        voice = voices.from_audio(self.wav(), "Hello there.", codec=codec, name="ref")
        assert voice.is_clone and voice.ref_text == "Hello there."
        assert len(voice.ref_codes) == 3 * SAMPLE_RATE // 400
        assert voice.ref_seconds == pytest.approx(3.0)

    def test_normalises_loudness_before_encoding(self):
        """Clips are normalised to -23 LUFS so every reference encodes at one level."""
        codec = FakeCodec()
        quiet = self.wav() * 0.01
        voices.from_audio(quiet, "Hello there.", codec=codec)
        assert np.abs(codec.seen).max() > np.abs(quiet).max() * 5

    def test_reads_a_file_when_given_a_path(self, tmp_path):
        from kova_tts import audio as audio_io

        path = audio_io.save_wav(tmp_path / "ref.wav", self.wav())
        voice = voices.from_audio(path, "Hello there.", codec=FakeCodec())
        assert voice.name == "ref"

    def test_a_missing_transcript_says_what_to_pass(self):
        with pytest.raises(ValueError, match="transcript="):
            voices.from_audio(self.wav(), "", codec=FakeCodec())

    def test_too_short_a_clip_is_refused(self):
        with pytest.raises(ValueError, match="at least"):
            voices.from_audio(self.wav(0.5), "Hi.", codec=FakeCodec())

    def test_a_long_clip_is_trimmed(self):
        codec = FakeCodec()
        voice = voices.from_audio(self.wav(30.0), "Hello there.", codec=codec, max_seconds=5.0)
        assert voice.ref_seconds == pytest.approx(5.0)

    def test_silence_encoding_to_nothing_is_reported(self):
        class Empty(FakeCodec):
            def encode(self, wav):
                return torch.zeros(0, dtype=torch.long)

        with pytest.raises(ValueError, match="probably silent"):
            voices.from_audio(self.wav(), "Hello.", codec=Empty())
