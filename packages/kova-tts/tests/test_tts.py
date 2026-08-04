"""The facade: text splitting, prompt assembly, the reference trim, and one real generation."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch

from kova_codec.constants import HOP_LENGTH, SAMPLE_RATE, TOKEN_RATE
from kova_tts import audio as audio_io
from kova_tts import prompt as prompt_module
from kova_tts import tokens as tokens_module
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
        text = "The first sentence is here. The second follows along! And a third one comes too?"
        assert split_sentences(text, max_chars=30) == [
            "The first sentence is here.",
            "The second follows along!",
            "And a third one comes too?",
        ]

    def test_keeps_closing_quotes_with_their_sentence(self):
        text = '"Absolutely not," she said. The room went very quiet indeed.'
        assert split_sentences(text, max_chars=35) == [
            '"Absolutely not," she said.',
            "The room went very quiet indeed.",
        ]

    def test_short_fragments_are_glued_to_the_next_segment(self):
        segments = split_sentences("Yes. That is exactly what I have been saying all along.")
        assert segments == ["Yes. That is exactly what I have been saying all along."]

    def test_whole_sentences_are_packed_up_to_the_limit(self):
        """Short sentences ride in one generation rather than each being started cold."""
        text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
        assert split_sentences(text, max_chars=36) == [
            "One two three. Four five six.",
            "Seven eight nine. Ten eleven twelve.",
        ]

    def test_only_a_merged_tail_may_exceed_the_limit(self):
        text = " ".join(f"Sentence number {i} says something worth hearing." for i in range(40))
        segments = split_sentences(text)
        assert len(segments) > 4
        assert all(len(s) <= tts_module.MAX_SEGMENT_CHARS for s in segments[:-1])
        assert len(segments[-1]) < tts_module.MAX_SEGMENT_CHARS + tts_module.MIN_SEGMENT_CHARS

    def test_a_trailing_fragment_is_merged_backwards(self):
        """Only the last chunk can come out under min_chars, and it is never carried."""
        text = "This first sentence is a perfectly reasonable length on its own. Yes."
        assert split_sentences(text, max_chars=64) == [text]

    def test_a_long_sentence_breaks_at_a_clause(self):
        text = "a" * 120 + ", " + "b" * 120 + ", " + "c" * 120
        segments = split_sentences(text, max_chars=200)
        assert segments == ["a" * 120 + ",", "b" * 120 + ",", "c" * 120]

    def test_a_long_sentence_with_no_punctuation_breaks_between_words(self):
        text = " ".join(["word"] * 100)
        segments = split_sentences(text, max_chars=60, min_chars=0)
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


# ------------------------------------------------------------------------------ carry across

#: Ordinary prose with a realistic spread of sentence lengths, including sentences long enough
#: that the splitter has to break them. Text of this shape is what exercises the carry: every
#: segment of a paragraph of short sentences fits the carry limit whether or not it is sized
#: correctly, so short text cannot tell the two apart.
PARAGRAPHS = [
    "The committee spent the better part of the afternoon working through the revised "
    "schedule, and by the time the last item was settled it was clear that the original "
    "deadline had never been realistic. Nobody wanted to say so out loud, partly because the "
    "schedule had been agreed in public and partly because the alternative meant reopening a "
    "question everyone had assumed was closed.",
    "It rained all week. The gutters had been blocked since autumn, so the water came off the "
    "roof in a single sheet and pooled against the doors until somebody propped them open. "
    "That helped for about an hour. Nobody complained.",
    "She had learned to read the river the way other people read a timetable, noticing the "
    "small changes in colour that meant the current had shifted overnight. On a good morning "
    "the water ran clear enough to see the gravel, and the boats went out early. On a bad one "
    "it came down brown and fast from the hills, carrying branches and the occasional fence "
    "post, and the village found other things to do until it settled. Either way the routine "
    "was the same: walk to the bend, look for a long minute, and then decide.",
]


class RecordingGenerator:
    """A generator that renders text at a plausible rate and keeps every prompt it was given.

    Token counts are approximated the way the real tokenizer behaves on these prompts: one
    token per audio code, and roughly one per four characters of everything else. That only has
    to be close enough for the KV-cache guard to be exercised at a realistic scale.
    """

    device = torch.device("cpu")

    def __init__(self, counts: list[int], *, max_cache_len: int = 4096) -> None:
        self.counts = list(counts)
        self.max_cache_len = max_cache_len
        self.encoded: list[str] = []
        #: The prompt actually generated from, which is the second one whenever a carry was
        #: built, found not to fit and rebuilt without it.
        self.used: list[str] = []

    def encode(self, prompt: str) -> list[int]:
        self.encoded.append(prompt)
        codes = prompt_module.parse_audio_tokens(prompt)
        text = tokens_module.AUDIO_TOKEN_RE.sub("", prompt)
        return [0] * (len(codes) + len(text) // 4)

    def stream_ids(self, ids, params=None, *, greedy: bool = False):
        self.used.append(self.encoded[-1])
        return iter(range(1, self.counts.pop(0) + 1))

    def unload_lora(self) -> None:
        """No adapters here, but every generation asks for the base weights first."""


def _plausible_counts(chunks: list[str], rate: float = 4.5) -> list[int]:
    """Codes each chunk would really generate, at the rate measured on ordinary prose."""
    return [math.ceil(rate * len(chunk)) for chunk in chunks]


def _carried_codes(prompt: str) -> list[int]:
    return prompt_module.parse_audio_tokens(prompt)


class TestCarryAcrossChunks:
    """Continuity across a join is the carry, and the carry only survives if the chunk fits."""

    def test_the_carry_limit_is_derived_from_the_chunk_size(self):
        assert tts_module.MAX_CARRY_CODES == tts_module._codes_for_chars(
            tts_module.MAX_SEGMENT_CHARS
        )

    @pytest.mark.parametrize("text", PARAGRAPHS)
    def test_every_chunk_the_splitter_emits_is_small_enough_to_carry(self, text):
        """The invariant. As two independent constants these drifted apart and the carry died."""
        for chunk in split_sentences(text)[:-1]:  # the last chunk is never carried
            assert tts_module._codes_for_chars(len(chunk)) <= tts_module.MAX_CARRY_CODES

    @pytest.mark.parametrize("text", PARAGRAPHS)
    def test_every_chunk_after_the_first_is_given_the_previous_one(self, text):
        chunks = split_sentences(text)
        assert len(chunks) > 2, "this paragraph no longer exercises a join"
        generator = RecordingGenerator(_plausible_counts(chunks))
        list(KovaTTS(generator)._stream_codes(text, None, TTS_SAMPLING))

        assert len(generator.used) == len(chunks)
        for prompt, previous in zip(generator.used[1:], chunks[:-1], strict=True):
            assert previous in prompt, "the previous chunk's text is missing from the prompt"
            assert _carried_codes(prompt), "the previous chunk's codes are missing"

    def test_a_chunk_that_ran_long_is_not_carried(self):
        """A matched pair or nothing: codes with no text make the model stop on step one."""
        text = PARAGRAPHS[0]
        chunks = split_sentences(text)
        counts = _plausible_counts(chunks)
        counts[0] = tts_module.MAX_CARRY_CODES + 1
        generator = RecordingGenerator(counts)
        list(KovaTTS(generator)._stream_codes(text, None, TTS_SAMPLING))
        assert not _carried_codes(generator.used[1])
        assert _carried_codes(generator.used[2]), "only the one join should be lost"

    def test_a_carry_that_would_not_fit_the_kv_cache_is_dropped_not_raised(self):
        """Failure mode of a cramped cache is a lost join, never a crash or a cut-off chunk."""
        text = PARAGRAPHS[0]
        chunks = split_sentences(text)
        generator = RecordingGenerator(_plausible_counts(chunks), max_cache_len=256)
        list(KovaTTS(generator)._stream_codes(text, None, TTS_SAMPLING))
        assert not any(_carried_codes(p) for p in generator.used)

    def test_a_cloned_reference_still_leaves_room_for_the_carry(self, clone_voice):
        """The dangerous combination: a whole reference clip in front of a full-size carry."""
        text = PARAGRAPHS[0]
        chunks = split_sentences(text)
        # Twenty seconds of reference, which is longer than any clip the docs suggest cloning.
        voice = Voice(name="ref", ref_codes=tuple(range(1600)), ref_text=clone_voice.ref_text)
        generator = RecordingGenerator(_plausible_counts(chunks))
        list(KovaTTS(generator)._stream_codes(text, voice, CLONE_SAMPLING))
        for prompt in generator.used[1:]:
            assert len(_carried_codes(prompt)) > len(voice.ref_codes)


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


class SineCodec:
    """A codec stand-in that decodes code `c` to the 400 samples of a tone at position `c`.

    Position-aware on purpose. The generator below emits codes 1, 2, 3, ..., so a whole
    utterance decodes to one continuous sine -- which means any discontinuity found in the
    output was put there by the code under test and not by the stand-in.
    """

    device = torch.device("cpu")
    sample_rate = SAMPLE_RATE

    @staticmethod
    def _tone(codes: torch.Tensor) -> torch.Tensor:
        offsets = codes[..., None] * HOP_LENGTH + torch.arange(HOP_LENGTH)
        return 0.5 * torch.sin(2 * math.pi * 220.0 * offsets / SAMPLE_RATE).flatten(-2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._tone(codes)

    def decode_with_lstm(self, codes, state=None, return_lstm_state=None, conv_padding=None):
        trimmed = codes[:, conv_padding:-conv_padding] if conv_padding else codes
        return self._tone(trimmed), (None if return_lstm_state is None else torch.zeros(1))


class TestOutputSampleRate:
    """`sample_rate=` on the public calls, and the property that makes it usable: streaming at
    a converted rate has to give the same audio as generating at it, joins included."""

    TEXT = "One sentence here. And a second one, a little longer than the first."

    def facade(self, **kwargs) -> KovaTTS:
        chunks = split_sentences(self.TEXT)
        generator = RecordingGenerator(_plausible_counts(chunks))
        return KovaTTS(generator, codec=SineCodec(), **kwargs)

    def streamed(self, tts: KovaTTS, **kwargs) -> tuple[np.ndarray, list]:
        frames = list(tts.stream(self.TEXT, **kwargs))
        return np.concatenate([f.samples for f in frames]), frames

    def test_the_default_is_the_model_rate(self):
        assert (
            self.facade().generate(self.TEXT).size
            == self.facade().generate(self.TEXT, sample_rate=SAMPLE_RATE).size
        )

    @pytest.mark.parametrize("rate", [16_000, 24_000, 8_000, 44_100])
    def test_generate_returns_the_requested_rate(self, rate):
        native = self.facade().generate(self.TEXT)
        out = self.facade().generate(self.TEXT, sample_rate=rate)
        assert out.dtype == np.float32 and out.ndim == 1
        assert out.size == math.ceil(native.size * rate / SAMPLE_RATE)

    @pytest.mark.parametrize("rate", [16_000, 8_000])
    def test_frames_report_the_rate_actually_delivered(self, rate):
        _, frames = self.streamed(self.facade(), sample_rate=rate)
        assert all(f.sample_rate == rate for f in frames)
        assert frames[-1].is_final

    @pytest.mark.parametrize("rate", [16_000, 24_000, 8_000, 44_100])
    def test_streaming_matches_generating_at_a_converted_rate(self, rate):
        """The reason the streaming resampler is stateful. A stateless one applied per frame
        gives the right length and the wrong samples at every join."""
        whole = self.facade().generate(self.TEXT, sample_rate=rate)
        streamed, _ = self.streamed(self.facade(), sample_rate=rate)
        assert streamed.shape == whole.shape
        np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-6)

    def test_the_tail_is_flushed_into_the_final_frame(self):
        """The resampler holds output back until its right-hand context arrives; those samples
        have to come out somewhere, and the last frame is the only place left."""
        whole = self.facade().generate(self.TEXT, sample_rate=16_000)
        streamed, frames = self.streamed(self.facade(), sample_rate=16_000)
        assert frames[-1].samples.size > 0
        assert streamed.size == whole.size

    @pytest.mark.parametrize("rate", [16_000, 8_000])
    def test_the_frame_joins_carry_no_step(self, rate):
        """A per-frame resample leaves a filter discontinuity at every join. Measured against
        the signal's own step distribution, a seam is an outlier; there should be none."""
        streamed, frames = self.streamed(self.facade(), sample_rate=rate)
        joins = np.cumsum([f.samples.size for f in frames])[:-1]
        joins = joins[(joins > 0) & (joins < streamed.size)]
        assert joins.size > 2, "this text no longer produces enough frames to have joins"
        steps = np.abs(np.diff(streamed))
        assert steps[joins - 1].max() <= np.percentile(steps, 99.9)

    def test_the_native_rate_is_not_run_through_a_resampler_at_all(self):
        whole = self.facade().generate(self.TEXT)
        streamed, _ = self.streamed(self.facade())
        np.testing.assert_array_equal(streamed, whole)

    @pytest.mark.parametrize("rate", [0, -1])
    def test_a_non_positive_rate_is_refused(self, rate):
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            self.facade().generate(self.TEXT, sample_rate=rate)
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            next(self.facade().stream(self.TEXT, sample_rate=rate))

    @pytest.mark.parametrize("rate", [SAMPLE_RATE, 16_000, 8_000])
    def test_the_reference_trim_still_removes_the_reference(self, rate, clone_voice):
        """The trim is expressed in seconds, so it should survive a rate change -- but it is
        applied either side of the conversion, and getting that order wrong leaves part of the
        reference clip in the output at exactly the rates a caller is most likely to ask for."""
        tts = self.facade()
        codes = sum(_plausible_counts(split_sentences(self.TEXT)))
        wav = tts.generate(self.TEXT, clone_voice, sample_rate=rate)
        assert wav.size == math.ceil(codes * HOP_LENGTH * rate / SAMPLE_RATE)

    @pytest.mark.parametrize("rate", [SAMPLE_RATE, 16_000])
    def test_a_cloned_stream_drops_the_reference_too(self, rate, clone_voice):
        whole = self.facade().generate(self.TEXT, clone_voice, sample_rate=rate)
        streamed, _ = self.streamed(self.facade(), voice=clone_voice, sample_rate=rate)
        assert streamed.shape == whole.shape
        np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-6)


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


@pytest.fixture(scope="module")
def tts(cuda_device, local_artifact) -> KovaTTS:
    from kova_tts import paths

    model = local_artifact(paths.ENV_MODEL)
    codec = local_artifact(paths.ENV_CODEC)
    return KovaTTS.from_pretrained(model, codec=codec, device=cuda_device, max_cache_len=1536)


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
def test_a_clone_does_not_begin_with_its_reference(tts, local_artifact):
    """The single easiest thing to get subtly wrong: the reference is re-rendered first.

    The assertion is about the trim, so it holds whatever the reference says -- set
    ``KOVA_TEST_TRANSCRIPT`` to the clip's real words for an intelligible sample as well.
    """
    from kova_tts import paths

    reference = local_artifact("KOVA_TEST_AUDIO")
    local_artifact(paths.ENV_WAVLM)  # cloning needs the encoder, and therefore WavLM
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
