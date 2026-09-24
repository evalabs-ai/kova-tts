"""Stage-boundary dataclasses: validation and the two sampling presets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kova_codec.constants import OUTPUT_SAMPLE_RATE, TOKEN_RATE
from kova_tts.engine.types import (
    CLONE_SAMPLING,
    TTS_SAMPLING,
    AudioFrame,
    SamplingParams,
    Voice,
)


class TestVoice:
    def test_lora_only_voice(self, tmp_path):
        voice = Voice(name="alpha", lora_path=tmp_path / "alpha")
        assert not voice.is_clone
        assert voice.ref_seconds == 0.0

    def test_reference_audio_voice(self):
        voice = Voice(name="ref", ref_codes=(1, 2, 3), ref_text="Hello there.")
        assert voice.is_clone
        assert voice.ref_seconds == pytest.approx(3 / TOKEN_RATE)

    def test_string_lora_path_becomes_a_path(self):
        assert isinstance(Voice(name="t", lora_path="/tmp/t").lora_path, Path)

    def test_ref_codes_are_stored_as_a_tuple(self):
        assert Voice(name="r", ref_codes=[1, 2], ref_text="hi").ref_codes == (1, 2)

    def test_reference_seconds_match_the_token_rate(self):
        voice = Voice(name="r", ref_codes=tuple(range(TOKEN_RATE)), ref_text="hi")
        assert voice.ref_seconds == pytest.approx(1.0)

    def test_a_voice_must_change_something(self):
        with pytest.raises(ValueError, match="neither a LoRA adapter nor reference audio"):
            Voice(name="empty")

    def test_reference_audio_requires_its_transcript(self):
        with pytest.raises(ValueError, match="no ref_text"):
            Voice(name="r", ref_codes=(1, 2))

    def test_rejects_out_of_range_codes(self):
        with pytest.raises(ValueError, match="outside the codebook"):
            Voice(name="r", ref_codes=(8192,), ref_text="hi")

    def test_requires_a_name(self, tmp_path):
        with pytest.raises(ValueError, match="needs a name"):
            Voice(name="  ", lora_path=tmp_path)

    def test_is_hashable_so_it_can_key_a_cache(self, tmp_path):
        assert len({Voice(name="t", lora_path=tmp_path), Voice(name="t", lora_path=tmp_path)}) == 1


class TestSamplingParams:
    def test_tts_preset(self):
        assert (TTS_SAMPLING.temperature, TTS_SAMPLING.top_p) == (1.1, 0.9)
        assert (TTS_SAMPLING.top_k, TTS_SAMPLING.repetition_penalty) == (75, 1.1)
        assert TTS_SAMPLING.max_tokens == 2048

    def test_clone_preset_differs_only_in_the_token_budget(self):
        assert CLONE_SAMPLING.max_tokens == 3500
        assert CLONE_SAMPLING.replace(max_tokens=TTS_SAMPLING.max_tokens) == TTS_SAMPLING

    def test_defaults_are_the_tts_preset(self):
        assert SamplingParams() == TTS_SAMPLING

    def test_classmethods_return_the_presets(self):
        assert SamplingParams.for_tts() == TTS_SAMPLING
        assert SamplingParams.for_cloning() == CLONE_SAMPLING

    def test_overrides_leave_the_preset_untouched(self):
        tweaked = SamplingParams.for_cloning(temperature=0.7)
        assert tweaked.temperature == 0.7
        assert tweaked.top_k == CLONE_SAMPLING.top_k
        assert CLONE_SAMPLING.temperature == 1.1

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("temperature", 0.0),
            ("top_p", 0.0),
            ("top_p", 1.5),
            ("top_k", -1),
            ("repetition_penalty", 0.0),
            ("max_tokens", 0),
        ],
    )
    def test_rejects_invalid_values(self, field, value):
        with pytest.raises(ValueError, match=field):
            SamplingParams(**{field: value})

    def test_overrides_are_validated_too(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams.for_tts(max_tokens=-5)


class TestAudioFrame:
    def test_defaults_to_the_codec_sample_rate(self):
        assert AudioFrame(np.zeros(4, dtype=np.float32)).sample_rate == OUTPUT_SAMPLE_RATE

    def test_duration(self):
        frame = AudioFrame(np.zeros(OUTPUT_SAMPLE_RATE // 2, dtype=np.float32))
        assert frame.duration_seconds == pytest.approx(0.5)

    def test_samples_are_cast_to_float32(self):
        assert AudioFrame(np.zeros(4, dtype=np.float64)).samples.dtype == np.float32

    def test_rejects_multichannel_audio(self):
        with pytest.raises(ValueError, match="mono 1-D"):
            AudioFrame(np.zeros((2, 4), dtype=np.float32))

    def test_rejects_a_nonsense_sample_rate(self):
        with pytest.raises(ValueError, match="sample_rate"):
            AudioFrame(np.zeros(4, dtype=np.float32), sample_rate=0)

    def test_is_final_defaults_to_false(self):
        assert AudioFrame(np.zeros(1, dtype=np.float32)).is_final is False


class TestImportsStayLight:
    def test_types_do_not_pull_in_transformers(self):
        # torch is not checked here only because kova_codec's package __init__ imports the codec
        # eagerly; nothing in this module needs either framework.
        import subprocess
        import sys

        code = "import sys, kova_tts.engine.types; assert 'transformers' not in sys.modules"
        assert subprocess.run([sys.executable, "-c", code]).returncode == 0
