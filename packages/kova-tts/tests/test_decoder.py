"""Streaming decode: the window arithmetic on its own, then against the real codec."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pytest
import torch

from kova_codec.constants import HOP_LENGTH, OUTPUT_HOP_LENGTH
from kova_tts.engine.decoder import (
    CONV_PADDING,
    LOOKAHEAD,
    WINDOW,
    DecodeWindow,
    StreamingDecoder,
    cudnn,
    decode_all,
    plan_window,
)

#: Real codes a steady-state window needs before it can be decoded: what it emits, plus the
#: right-hand lookahead, plus the convolution's context. Spelled out rather than imported, so a
#: change to the window constants has to be reckoned with here too.
FIRST_WINDOW_CODES = WINDOW + 2 * LOOKAHEAD + CONV_PADDING


def plan_windows(total: int) -> Iterator[DecodeWindow]:
    """Every window a finished stream of `total` codes decodes through, in order.

    The sequence :class:`StreamingDecoder` walks, without a codec in the way, so the tiling can
    be checked on its own.
    """
    position = 0
    while (plan := plan_window(position, total, finished=True)) is not None:
        yield plan
        if plan.is_last:
            return
        position += plan.advance


# --------------------------------------------------------------------------- window arithmetic


class TestWindowPlan:
    @pytest.mark.parametrize("total", [1, 30, 39, 40, 41, 52, 53, 100, 240, 398, 1000])
    def test_every_code_is_emitted_exactly_once(self, total):
        """The whole point: the windows tile the utterance with no gap and no overlap."""
        emitted = 0
        for plan in plan_windows(total):
            frames = (plan.stop - plan.start) - 2 * CONV_PADDING
            stop = frames if plan.emit_stop is None else plan.emit_stop // OUTPUT_HOP_LENGTH
            emitted += stop - plan.emit_start // OUTPUT_HOP_LENGTH
        assert emitted == total

    def test_the_first_window_emits_its_lookahead_too(self):
        first = next(plan_windows(1000))
        assert first.emit_start == 0
        assert first.emit_stop == (LOOKAHEAD + WINDOW) * OUTPUT_HOP_LENGTH

    def test_steady_state_windows_emit_one_window_each(self):
        plans = list(plan_windows(1000))
        for plan in plans[1:-1]:
            assert plan.emit_start == LOOKAHEAD * OUTPUT_HOP_LENGTH
            assert plan.emit_stop == (LOOKAHEAD + WINDOW) * OUTPUT_HOP_LENGTH
            assert plan.advance == WINDOW

    def test_the_last_window_keeps_no_state_and_runs_to_the_end(self):
        last = list(plan_windows(1000))[-1]
        assert last.is_last
        assert last.return_state_at is None
        assert last.emit_stop is None
        assert last.advance == 0

    def test_the_last_window_stops_at_the_last_real_code(self):
        """Trap 1: audio the LSTM produced from padding must never be emitted."""
        total = 200
        last = list(plan_windows(total))[-1]
        # The slice ends at the padded array's end: exactly `conv_padding` pad codes, which the
        # codec trims off after the convolution, so the LSTM stops at the last real code.
        assert last.stop == total + 2 * CONV_PADDING

    @pytest.mark.parametrize("total", [41, 52, 100, 240, 1000])
    def test_no_window_asks_for_state_at_its_own_end(self, total):
        """Trap 2: ``return_lstm_state=k`` on a window of exactly k frames raises in torch."""
        for plan in plan_windows(total):
            if plan.return_state_at is None:
                continue
            frames = (plan.stop - plan.start) - 2 * CONV_PADDING
            assert frames > plan.return_state_at

    def test_a_short_utterance_is_one_window(self):
        plans = list(plan_windows(20))
        assert len(plans) == 1
        assert plans[0].is_last and plans[0].emit_start == 0

    def test_no_lookahead_is_rejected_by_name(self):
        with pytest.raises(ValueError, match="larger than 0 in RNN"):
            plan_window(0, 100, finished=False, lookahead=0)


class TestStreamingReadiness:
    def test_nothing_decodes_before_a_window_is_complete(self):
        for available in range(FIRST_WINDOW_CODES):
            assert plan_window(0, available, finished=False) is None

    def test_the_first_window_fires_at_exactly_52_codes(self):
        plan = plan_window(0, FIRST_WINDOW_CODES, finished=False)
        assert plan is not None and not plan.is_last
        assert plan.stop - plan.start == WINDOW + 2 * LOOKAHEAD + 2 * CONV_PADDING

    def test_an_unfinished_stream_never_produces_the_last_window(self):
        assert plan_window(0, 30, finished=False) is None
        assert plan_window(0, 30, finished=True) is not None

    def test_an_unfinished_window_is_never_short(self):
        """A window must not count a trailing pad that only exists once the stream ends:
        firing three codes early drifts the window's last frames by up to 0.05."""
        width = WINDOW + 2 * LOOKAHEAD + 2 * CONV_PADDING
        for available in range(FIRST_WINDOW_CODES, 400):
            plan = plan_window(31, available, finished=False)
            if plan is None:
                continue
            # Only the leading pad exists while streaming, so this is the array's real length.
            assert plan.stop <= available + CONV_PADDING
            assert plan.stop - plan.start == width
            break


class TestStreamingDecoderBookkeeping:
    class _FakeCodec:
        """Returns one identifiable sample per frame, so slicing can be checked exactly."""

        device = torch.device("cpu")
        sample_rate = 32_000

        def decode_with_lstm(self, codes, state=None, return_lstm_state=None, conv_padding=None):
            trimmed = codes[:, conv_padding:-conv_padding] if conv_padding else codes
            audio = trimmed.repeat_interleave(HOP_LENGTH, dim=1).float()
            return audio, (None if return_lstm_state is None else torch.zeros(1))

    @pytest.fixture
    def decoder(self) -> StreamingDecoder:
        return StreamingDecoder(self._FakeCodec())

    def frames(self, audio: np.ndarray) -> list[int]:
        return audio.reshape(-1, HOP_LENGTH)[:, 0].astype(int).tolist()

    def test_pushed_codes_come_back_in_order_exactly_once(self, decoder):
        codes = list(range(1, 201))
        out = [decoder.push([c]) for c in codes]
        out.append(decoder.finish())
        assert self.frames(np.concatenate([o for o in out if o.size])) == codes

    def test_priming_discards_exactly_the_primed_audio(self, decoder):
        decoder.prime([9999] * 40)
        pieces = [decoder.push([c]) for c in range(1, 121)]
        pieces.append(decoder.finish())
        got = self.frames(np.concatenate([p for p in pieces if p.size]))
        assert got == list(range(1, 121))

    def test_priming_after_real_codes_is_refused(self, decoder):
        decoder.push([1, 2, 3])
        with pytest.raises(RuntimeError, match="before the first push"):
            decoder.prime([4])

    def test_finish_is_idempotent_and_closes_the_decoder(self, decoder):
        decoder.push(range(10))
        decoder.finish()
        assert decoder.finish().size == 0
        with pytest.raises(RuntimeError, match="reset"):
            decoder.push([1])

    def test_reset_starts_a_fresh_utterance(self, decoder):
        decoder.push(range(1, 100))
        decoder.finish()
        decoder.reset()
        out = [decoder.push(range(1, 60)), decoder.finish()]
        assert self.frames(np.concatenate([o for o in out if o.size])) == list(range(1, 60))

    def test_empty_input_decodes_to_nothing(self, decoder):
        assert decoder.push([]).size == 0
        assert decoder.finish().size == 0


# ----------------------------------------------------------------------------- rate conversion


class ToneCodec:
    """Decodes code `c` to the 400 samples of a tone at position `c`.

    Position-aware, so a run of consecutive codes decodes to one continuous waveform however it
    is cut into windows. That is what lets a discontinuity in the output be attributed to the
    code under test rather than to the stand-in.
    """

    device = torch.device("cpu")
    sample_rate = 32_000

    @staticmethod
    def _tone(codes: torch.Tensor) -> torch.Tensor:
        offsets = codes[..., None] * HOP_LENGTH + torch.arange(HOP_LENGTH)
        return 0.5 * torch.sin(2 * np.pi * 220.0 * offsets / 32_000).flatten(-2)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._tone(codes)

    def decode_with_lstm(self, codes, state=None, return_lstm_state=None, conv_padding=None):
        trimmed = codes[:, conv_padding:-conv_padding] if conv_padding else codes
        return self._tone(trimmed), (None if return_lstm_state is None else torch.zeros(1))


class TestRateConversionOverWindows:
    """Window cadence meets rate conversion. The decoder emits a chunk every ~390 ms, and each
    chunk boundary is a place a stateless resampler would leave a step."""

    CODES = list(range(600))  # 7.5 seconds, ~19 windows

    def streamed(self, rate: int) -> tuple[np.ndarray, np.ndarray]:
        """Chunked decode + conversion, and where the chunk joins landed in the output."""
        from kova_tts.audio import StreamingResampler

        decoder = StreamingDecoder(ToneCodec())
        resampler = StreamingResampler(32_000, rate)
        pieces = [resampler.process(decoder.push([code])) for code in self.CODES]
        pieces.append(resampler.process(decoder.finish()))
        pieces.append(resampler.flush())
        pieces = [p for p in pieces if p.size]
        return np.concatenate(pieces), np.cumsum([p.size for p in pieces])[:-1]

    @pytest.mark.parametrize("rate", [16_000, 24_000, 8_000, 44_100])
    def test_chunked_decode_and_conversion_equals_doing_both_at_once(self, rate):
        from kova_tts.audio import resample

        whole = resample(decode_all(ToneCodec(), self.CODES), 32_000, rate)
        got, _ = self.streamed(rate)
        assert got.shape == whole.shape
        np.testing.assert_allclose(got, whole, rtol=0, atol=1e-6)

    @pytest.mark.parametrize("rate", [16_000, 8_000])
    def test_the_chunk_joins_are_not_visible_in_the_waveform(self, rate):
        got, joins = self.streamed(rate)
        steps = np.abs(np.diff(got))
        joins = joins[(joins > 0) & (joins < got.size)]
        assert joins.size > 5, "this many codes should produce more chunks than that"
        assert steps[joins - 1].max() <= np.percentile(steps, 99.9)


# ------------------------------------------------------------------------------- with the codec


@pytest.fixture(scope="module")
def codec(cuda_device, local_artifact):
    """An encode-capable codec, so the test can build *realistic* codes to decode.

    The seam error is entirely signal-dependent -- codes drawn at random decode to something
    far wider-band than speech and leave 0.02 at the seams, where encoded speech leaves 0.002 --
    so a meaningful tolerance needs codes that came out of the encoder.
    """
    from kova_tts import paths
    from kova_tts.engine.decoder import load_codec

    checkpoint = local_artifact(paths.ENV_CODEC)
    wavlm = local_artifact(paths.ENV_WAVLM)
    return load_codec(checkpoint, device=cuda_device, encode=True, wavlm=wavlm)


@pytest.fixture(scope="module")
def speech_codes(codec) -> list[int]:
    """Codes for five seconds of synthetic speech-like audio.

    A gliding harmonic stack under three fixed formant resonances, gated by a syllable-rate
    envelope -- the same signal the codec's own tests use, because no audio is committed here.
    """
    import math

    rate, seconds = 32_000, 5.0
    n = int(seconds * rate)
    t = np.arange(n) / rate
    f0 = 110.0 + 40.0 * np.sin(2 * math.pi * 0.7 * t)
    phase = 2 * math.pi * np.cumsum(f0) / rate
    wav = np.zeros(n)
    for k in range(1, 41):
        hz = k * 130.0
        gain = sum(math.exp(-(((hz - f) / 250.0) ** 2)) for f in (700.0, 1200.0, 2600.0))
        wav += (gain / k) * np.sin(k * phase)
    wav *= (0.5 + 0.5 * np.sin(2 * math.pi * 3.5 * t)) ** 2 * (t % 1.0 > 0.15)
    wav = (0.7 * wav / np.abs(wav).max()).astype(np.float32)
    return codec.encode(torch.from_numpy(wav)).cpu().tolist()


@pytest.mark.gpu
@pytest.mark.weights
def test_windowed_streaming_matches_a_single_decode(codec, speech_codes):
    """The property the streaming server rests on, on the real checkpoint."""
    whole = decode_all(codec, speech_codes)

    decoder = StreamingDecoder(codec)
    pieces = [decoder.push([code]) for code in speech_codes]
    pieces.append(decoder.finish())
    streamed = np.concatenate([p for p in pieces if p.size])

    assert streamed.shape == whole.shape
    # Nine frames of lookahead is a shade under the post-LSTM receptive field, so the seams
    # carry a little error: ~60 dB SNR, a max |diff| of around 6e-3 on these codes.
    np.testing.assert_allclose(streamed, whole, rtol=0, atol=1e-2)


@pytest.mark.gpu
@pytest.mark.weights
def test_pushing_in_batches_gives_the_same_audio_as_one_at_a_time(codec, speech_codes):
    single = StreamingDecoder(codec)
    pieces = [single.push([c]) for c in speech_codes] + [single.finish()]
    one_at_a_time = np.concatenate([p for p in pieces if p.size])

    batched = StreamingDecoder(codec)
    chunks = [batched.push(speech_codes[i : i + 37]) for i in range(0, len(speech_codes), 37)]
    chunks.append(batched.finish())
    in_batches = np.concatenate([c for c in chunks if c.size])

    np.testing.assert_array_equal(one_at_a_time, in_batches)


@pytest.mark.gpu
@pytest.mark.weights
def test_the_captured_window_decodes_exactly_what_the_eager_one_does(
    codec, speech_codes, monkeypatch
):
    """The steady window replays as a CUDA graph; the audio must not change by a sample.

    Primed, so the replay starts from a carried LSTM state, and interleaved with a second
    decoder on the same shared graph, which is what two sessions on one engine thread do.
    """

    def decode(codes, *, other=None) -> np.ndarray:
        mine = StreamingDecoder(codec)
        theirs = StreamingDecoder(codec)
        mine.prime(codes[:80])
        pieces = []
        for index, code in enumerate(codes[80:]):
            pieces.append(mine.push([code]))
            if other is not None:
                theirs.push([other[index % len(other)]])
        pieces.append(mine.finish())
        return np.concatenate([piece for piece in pieces if piece.size])

    monkeypatch.setenv("KOVA_DISABLE_CUDA_GRAPH", "1")
    eager = decode(speech_codes)
    monkeypatch.setenv("KOVA_DISABLE_CUDA_GRAPH", "0")
    captured = decode(speech_codes, other=list(reversed(speech_codes)))

    assert captured.size == eager.size
    np.testing.assert_array_equal(captured, eager)


@pytest.mark.gpu
@pytest.mark.weights
def test_the_captured_window_works_on_a_gpu_that_is_not_the_current_one(
    local_artifact, monkeypatch
):
    """Capture defaults to a stream on whichever device was current first; with two GPUs that
    can be the wrong one, and the graph would record nothing and replay as silence."""
    from kova_codec import KovaCodec

    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    other = torch.device("cuda", (torch.cuda.current_device() + 1) % torch.cuda.device_count())
    codec = KovaCodec.from_checkpoint(
        local_artifact("KOVA_CODEC_PATH"), device=other, decode_only=True
    )
    codes = torch.randint(0, 8192, (300,), generator=torch.Generator().manual_seed(3)).tolist()

    def decode() -> np.ndarray:
        decoder = StreamingDecoder(codec)
        pieces = [decoder.push([code]) for code in codes] + [decoder.finish()]
        return np.concatenate([piece for piece in pieces if piece.size])

    monkeypatch.setenv("KOVA_DISABLE_CUDA_GRAPH", "1")
    eager = decode()
    monkeypatch.setenv("KOVA_DISABLE_CUDA_GRAPH", "0")
    np.testing.assert_array_equal(decode(), eager)


def test_cudnn_is_restored_after_a_decode_changes_it():
    before = torch.backends.cudnn.enabled
    with cudnn(enabled=not before):
        assert torch.backends.cudnn.enabled is (not before)
    assert torch.backends.cudnn.enabled is before


@pytest.mark.gpu
@pytest.mark.weights
def test_priming_drops_the_reference_and_keeps_the_rest(codec, speech_codes):
    reference, target = speech_codes[:120], speech_codes[120:]
    decoder = StreamingDecoder(codec)
    decoder.prime(reference)
    pieces = [decoder.push([c]) for c in target] + [decoder.finish()]
    primed = np.concatenate([p for p in pieces if p.size])

    assert primed.size == len(target) * codec.hop_length
    # The primed decoder is warm, so what it emits is the tail of a whole-utterance decode.
    whole = decode_all(codec, speech_codes)
    np.testing.assert_allclose(
        primed, whole[len(reference) * codec.hop_length :], rtol=0, atol=1e-2
    )
