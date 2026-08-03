"""The facade: text splitting, prompt assembly, the reference trim, and one real generation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from kova_codec.constants import SAMPLE_RATE, TOKEN_RATE
from kova_tts import audio as audio_io
from kova_tts import prompt as prompt_module
from kova_tts.engine import tts as tts_module
from kova_tts.engine.tts import KovaTTS, split_sentences
from kova_tts.engine.types import CLONE_SAMPLING, TTS_SAMPLING, Voice


class StubGenerator:
    """Everything :class:`KovaTTS` needs from a generator to build a prompt."""

    device = torch.device("cpu")


@pytest.fixture
def facade() -> KovaTTS:
    return KovaTTS(StubGenerator())


@pytest.fixture
def clone_voice() -> Voice:
    return Voice(name="ref", ref_codes=tuple(range(160)), ref_text="This is the reference.")


# ------------------------------------------------------------------------------ text splitting


class TestSplitSentences:
    def test_empty_text_yields_nothing(self):
        assert split_sentences("") == []
        assert split_sentences("   \n ") == []

    def test_one_sentence_stays_whole(self):
        assert split_sentences("Hello world, this is a single sentence.") == [
            "Hello world, this is a single sentence."
        ]

    def test_splits_on_sentence_punctuation(self):
        text = "The first sentence is here. The second sentence follows! And a third one?"
        assert split_sentences(text) == [
            "The first sentence is here.",
            "The second sentence follows!",
            "And a third one?",
        ]

    def test_keeps_closing_quotes_with_their_sentence(self):
        text = '"Absolutely not," she said. The room went very quiet indeed.'
        assert split_sentences(text) == [
            '"Absolutely not," she said.',
            "The room went very quiet indeed.",
        ]

    def test_short_fragments_are_glued_to_the_next_segment(self):
        segments = split_sentences("Yes. That is exactly what I have been saying all along.")
        assert segments == ["Yes. That is exactly what I have been saying all along."]

    def test_a_long_sentence_breaks_at_a_clause(self):
        text = "a" * 120 + ", " + "b" * 120 + ", " + "c" * 120
        segments = split_sentences(text, max_chars=200)
        assert segments == ["a" * 120 + ",", "b" * 120 + ",", "c" * 120]

    def test_a_long_sentence_with_no_punctuation_breaks_between_words(self):
        text = " ".join(["word"] * 100)
        segments = split_sentences(text, max_chars=60)
        assert all(len(s) <= 60 for s in segments)
        assert " ".join(segments) == text

    def test_nothing_is_lost_or_duplicated(self):
        text = "One. Two things happened here today. Three, four, and five as well."
        assert "".join(split_sentences(text)).replace(" ", "") == text.replace(" ", "")


# ------------------------------------------------------------------------------ prompt shapes


class TestPrompt:
    def test_plain_synthesis_gets_bos_and_nothing_else(self, facade):
        got = facade._prompt("Hello world.", None, None)
        assert got == "<|begin_of_text|>" + prompt_module.tts_prompt("Hello world.")

    def test_bos_appears_exactly_once(self, facade, clone_voice):
        for voice in (None, clone_voice):
            assert facade._prompt("Hello.", voice, None).count("<|begin_of_text|>") == 1

    def test_a_cloned_voice_uses_the_clone_layout(self, facade, clone_voice):
        got = facade._prompt("Say this.", clone_voice, None)
        assert got == prompt_module.clone_prompt(
            clone_voice.ref_text, "Say this.", clone_voice.ref_codes
        )

    def test_a_carry_puts_the_previous_text_in_front_and_its_codes_behind(self, facade):
        carry = tts_module._Carry("The first sentence.", (1, 2, 3))
        got = facade._prompt("The second one.", None, carry)
        assert got == prompt_module.clone_prompt(
            "The first sentence.", "The second one.", (1, 2, 3)
        )
        assert "The first sentence. The second one." in got

    def test_a_carry_follows_the_reference_for_a_cloned_voice(self, facade, clone_voice):
        carry = tts_module._Carry("Already said.", (7, 8))
        got = facade._prompt("Now this.", clone_voice, carry)
        # Text order: reference transcript, then what was already spoken, then the new text.
        assert "This is the reference. Already said. Now this." in got
        # Code order: the reference clip, then the carried codes.
        codes = prompt_module.parse_audio_tokens(got)
        assert codes == list(clone_voice.ref_codes) + [7, 8]

    def test_a_carry_of_codes_alone_is_never_built(self, facade):
        """Codes with no text make the model stop on the first step; the pair travels together."""
        carry = tts_module._Carry("", (1, 2, 3))
        got = facade._prompt("Next.", None, carry)
        assert prompt_module.parse_audio_tokens(got) == [1, 2, 3]
        assert got.startswith("<|begin_of_text|><|text_prompt_start|>Next.")


# ---------------------------------------------------------------------------- sampling + trim


class TestSamplingChoice:
    def test_plain_synthesis_uses_the_tts_preset(self):
        assert KovaTTS._sampling(None, None, None) is TTS_SAMPLING

    def test_a_lora_voice_uses_the_tts_preset(self, tmp_path):
        voice = Voice(name="alpha", lora_path=tmp_path)
        assert KovaTTS._sampling(voice, None, None) is TTS_SAMPLING

    def test_a_cloned_voice_uses_the_clone_preset(self, clone_voice):
        assert KovaTTS._sampling(clone_voice, None, None) is CLONE_SAMPLING

    def test_an_explicit_params_wins(self, clone_voice):
        mine = TTS_SAMPLING.replace(temperature=0.5)
        assert KovaTTS._sampling(clone_voice, mine, None) is mine

    def test_a_seed_is_applied_to_whichever_preset_was_chosen(self, clone_voice):
        assert KovaTTS._sampling(clone_voice, None, 42).seed == 42
        assert KovaTTS._sampling(clone_voice, None, 42).temperature == CLONE_SAMPLING.temperature


class TestReferenceTrim:
    """Cloned output opens with a re-rendering of the reference; the trim has to be exact."""

    def test_reference_seconds_is_the_code_count_over_the_token_rate(self, clone_voice):
        assert clone_voice.ref_seconds == pytest.approx(160 / TOKEN_RATE)

    def test_trimming_drops_exactly_the_reference(self, clone_voice):
        total = 5 * SAMPLE_RATE
        wav = np.arange(total, dtype=np.float32)
        trimmed = audio_io.trim_leading(wav, clone_voice.ref_seconds)
        assert trimmed.size == total - int(clone_voice.ref_seconds * SAMPLE_RATE)
        assert trimmed[0] == int(clone_voice.ref_seconds * SAMPLE_RATE)

    def test_the_whole_reference_is_the_default_preroll(self, facade, clone_voice):
        assert facade._preroll(clone_voice) == clone_voice.ref_codes

    def test_a_shorter_preroll_takes_the_tail_of_the_reference(self, clone_voice):
        facade = KovaTTS(StubGenerator(), clone_preroll=80)
        preroll = facade._preroll(clone_voice)
        assert preroll == clone_voice.ref_codes[-80:]

    def test_trimming_a_short_clip_leaves_nothing_rather_than_going_negative(self):
        assert audio_io.trim_leading(np.zeros(100, dtype=np.float32), 10.0).size == 0


class TestCloneErrors:
    def test_a_missing_transcript_points_at_the_seam(self, facade):
        with pytest.raises(ValueError, match="transcriber="):
            facade.clone("reference.wav", transcript=None)

    def test_a_configured_transcriber_is_used(self, tmp_path, monkeypatch):
        calls = []

        class FakeCodec:
            device = torch.device("cpu")

            def encode(self, wav):
                calls.append(np.asarray(wav).size)
                return torch.arange(200, dtype=torch.long)

        facade = KovaTTS(StubGenerator(), transcriber=lambda path: "Transcribed text.")
        facade._encoding_codec = FakeCodec()
        wav = (0.3 * np.sin(np.arange(3 * SAMPLE_RATE) / 50)).astype(np.float32)
        path = audio_io.save_wav(tmp_path / "ref.wav", wav)
        voice = facade.clone(path)
        assert voice.ref_text == "Transcribed text." and calls


# ------------------------------------------------------------------------------- end to end


def _free_cuda_device() -> torch.device:
    """The CUDA device with the most free memory. This box has two and another job may own one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    return torch.device("cuda", best)


def _local(env_var: str) -> str:
    from kova_tts import paths

    paths.load_dotenv()
    value = os.environ.get(env_var, "").strip()
    if not value or not Path(value).exists():
        pytest.skip(f"set {env_var} to a local path to run this test")
    return value


@pytest.fixture(scope="module")
def tts() -> KovaTTS:
    from kova_tts import paths

    model = _local(paths.ENV_MODEL)
    codec = _local(paths.ENV_CODEC)
    return KovaTTS.from_pretrained(
        model, codec=codec, device=_free_cuda_device(), max_cache_len=1536
    )


@pytest.mark.gpu
@pytest.mark.weights
def test_a_sentence_becomes_real_audio(tts):
    wav = tts.generate("The quick brown fox jumps over the lazy dog.", seed=7)
    assert wav.dtype == np.float32 and wav.ndim == 1
    assert wav.size / SAMPLE_RATE > 0.5, "less than half a second of speech for a whole sentence"
    peak = float(np.abs(wav).max())
    assert 0.02 < peak <= 1.0, f"peak {peak} is silence or clipping, not speech"
    # Speech has a syllable-rate envelope; a constant hum or noise floor does not.
    usable = wav[: wav.size - wav.size % 1600]
    envelope = np.abs(usable).reshape(-1, 1600).mean(axis=1)
    assert envelope.std() / envelope.mean() > 0.3


@pytest.mark.gpu
@pytest.mark.weights
def test_streaming_yields_the_same_audio_as_generating(tts):
    text = "The quick brown fox jumps over the lazy dog."
    whole = tts.generate(text, seed=99)
    frames = list(tts.stream(text, seed=99))
    assert frames[-1].is_final
    assert all(f.sample_rate == SAMPLE_RATE for f in frames)
    streamed = np.concatenate([f.samples for f in frames])
    assert streamed.size == whole.size
    np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.weights
def test_several_sentences_produce_more_audio_than_one(tts):
    """Cross-sentence continuation: carrying codes with no text makes segments after the first
    stop immediately, which shows up here as a paragraph no longer than its first sentence."""
    first = "The quick brown fox jumps over the lazy dog."
    paragraph = f"{first} Pack my box with five dozen liquor jugs. How quickly daft zebras jump."
    one = tts.generate(first, seed=3)
    three = tts.generate(paragraph, seed=3)
    assert three.size > 2 * one.size


@pytest.mark.gpu
@pytest.mark.weights
def test_a_clone_does_not_begin_with_its_reference(tts):
    """The single easiest thing to get subtly wrong: the reference is re-rendered first.

    The assertion is about the trim, so it holds whatever the reference says -- set
    ``KOVA_TEST_TRANSCRIPT`` to the clip's real words for an intelligible sample as well.
    """
    reference = _local("KOVA_TEST_AUDIO")
    _local(  # cloning needs the encoder, and therefore WavLM
        __import__("kova_tts", fromlist=["paths"]).paths.ENV_WAVLM
    )
    transcript = os.environ.get("KOVA_TEST_TRANSCRIPT") or "This is a short spoken recording."
    voice = tts.clone(reference, transcript=transcript)
    assert voice.is_clone and voice.ref_seconds > 0.5

    wav = tts.generate("The quick brown fox jumps over the lazy dog.", voice=voice, seed=5)
    assert wav.size / SAMPLE_RATE > 0.5

    original = audio_io.load_audio(reference)
    n = min(int(voice.ref_seconds * SAMPLE_RATE), wav.size, original.size)
    head, start = wav[:n], original[:n]
    correlation = float(
        np.dot(head - head.mean(), start - start.mean())
        / (np.linalg.norm(head - head.mean()) * np.linalg.norm(start - start.mean()) + 1e-9)
    )
    assert abs(correlation) < 0.2, (
        f"the first {n / SAMPLE_RATE:.1f}s of output correlates {correlation:.2f} with the "
        f"reference clip -- Voice.ref_seconds was not trimmed"
    )
