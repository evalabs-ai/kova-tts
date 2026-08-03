"""Streaming decode: the window arithmetic on its own, then against the real codec."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from kova_codec.constants import HOP_LENGTH
from kova_tts.engine.decoder import (
    CONV_PADDING,
    LOOKAHEAD,
    WINDOW,
    StreamingDecoder,
    decode_all,
    plan_window,
    plan_windows,
)

#: Real codes a steady-state window needs before it can be decoded: what it emits, plus the
#: right-hand lookahead, plus the convolution's context. Spelled out rather than imported, so a
#: change to the window constants has to be reckoned with here too.
FIRST_WINDOW_CODES = WINDOW + 2 * LOOKAHEAD + CONV_PADDING


# --------------------------------------------------------------------------- window arithmetic


class TestWindowPlan:
    @pytest.mark.parametrize("total", [1, 30, 39, 40, 41, 52, 53, 100, 240, 398, 1000])
    def test_every_code_is_emitted_exactly_once(self, total):
        """The whole point: the windows tile the utterance with no gap and no overlap."""
        emitted = 0
        for plan in plan_windows(total):
            frames = (plan.stop - plan.start) - 2 * CONV_PADDING
            stop = frames if plan.emit_stop is None else plan.emit_stop // HOP_LENGTH
            emitted += stop - plan.emit_start // HOP_LENGTH
        assert emitted == total

    def test_the_first_window_emits_its_lookahead_too(self):
        first = next(plan_windows(1000))
        assert first.emit_start == 0
        assert first.emit_stop == (LOOKAHEAD + WINDOW) * HOP_LENGTH

    def test_steady_state_windows_emit_one_window_each(self):
        plans = list(plan_windows(1000))
        for plan in plans[1:-1]:
            assert plan.emit_start == LOOKAHEAD * HOP_LENGTH
            assert plan.emit_stop == (LOOKAHEAD + WINDOW) * HOP_LENGTH
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
        """The bug this guards: counting a trailing pad that only exists once the stream ends
        fires the window three codes early, and its last frames drift by up to 0.05."""
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

    def frames(self, audio: np.ndarray) -> list[int]:
        return audio.reshape(-1, HOP_LENGTH)[:, 0].astype(int).tolist()

    def test_pushed_codes_come_back_in_order_exactly_once(self):
        decoder = StreamingDecoder(self._FakeCodec())
        codes = list(range(1, 201))
        out = [decoder.push([c]) for c in codes]
        out.append(decoder.finish())
        assert self.frames(np.concatenate([o for o in out if o.size])) == codes

    def test_priming_discards_exactly_the_primed_audio(self):
        decoder = StreamingDecoder(self._FakeCodec())
        decoder.prime([9999] * 40)
        pieces = [decoder.push([c]) for c in range(1, 121)]
        pieces.append(decoder.finish())
        got = self.frames(np.concatenate([p for p in pieces if p.size]))
        assert got == list(range(1, 121))

    def test_priming_after_real_codes_is_refused(self):
        decoder = StreamingDecoder(self._FakeCodec())
        decoder.push([1, 2, 3])
        with pytest.raises(RuntimeError, match="before the first push"):
            decoder.prime([4])

    def test_finish_is_idempotent_and_closes_the_decoder(self):
        decoder = StreamingDecoder(self._FakeCodec())
        decoder.push(range(10))
        decoder.finish()
        assert decoder.finish().size == 0
        with pytest.raises(RuntimeError, match="reset"):
            decoder.push([1])

    def test_reset_starts_a_fresh_utterance(self):
        decoder = StreamingDecoder(self._FakeCodec())
        decoder.push(range(1, 100))
        decoder.finish()
        decoder.reset()
        out = [decoder.push(range(1, 60)), decoder.finish()]
        assert self.frames(np.concatenate([o for o in out if o.size])) == list(range(1, 60))

    def test_empty_input_decodes_to_nothing(self):
        decoder = StreamingDecoder(self._FakeCodec())
        assert decoder.push([]).size == 0
        assert decoder.finish().size == 0


# ------------------------------------------------------------------------------- with the codec


def _free_cuda_device() -> torch.device:
    """The CUDA device with the most free memory. This box has two and another job may own one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    return torch.device("cuda", best)


def _local_checkpoint(env_var: str) -> str:
    """A locally configured artifact path, or skip. Never downloads: a test that needs
    gigabytes of weights should say so rather than fetching them behind the runner's back."""
    from kova_tts import paths

    paths.load_dotenv()
    configured = os.environ.get(env_var, "").strip()
    if not configured or not Path(configured).exists():
        pytest.skip(f"set {env_var} to a local checkpoint to run this test")
    return configured


@pytest.fixture(scope="module")
def codec():
    """An encode-capable codec, so the test can build *realistic* codes to decode.

    The seam error is entirely signal-dependent -- codes drawn at random decode to something
    far wider-band than speech and leave 0.02 at the seams, where encoded speech leaves 0.002 --
    so a meaningful tolerance needs codes that came out of the encoder.
    """
    from kova_tts import paths
    from kova_tts.engine.decoder import load_codec

    checkpoint = _local_checkpoint(paths.ENV_CODEC)
    wavlm = _local_checkpoint(paths.ENV_WAVLM)
    return load_codec(checkpoint, device=_free_cuda_device(), encode=True, wavlm=wavlm)


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
def test_priming_drops_the_reference_and_keeps_the_rest(codec, speech_codes):
    reference, target = speech_codes[:120], speech_codes[120:]
    decoder = StreamingDecoder(codec)
    decoder.prime(reference)
    pieces = [decoder.push([c]) for c in target] + [decoder.finish()]
    primed = np.concatenate([p for p in pieces if p.size])

    assert primed.size == len(target) * HOP_LENGTH
    # The primed decoder is warm, so what it emits is the tail of a whole-utterance decode.
    whole = decode_all(codec, speech_codes)
    np.testing.assert_allclose(primed, whole[len(reference) * HOP_LENGTH :], rtol=0, atol=1e-2)
