"""Decoding codes to audio, one window at a time, while the LM is still generating.

The codec decoder has exactly two sources of cross-frame context, and streaming means
reproducing both by hand (see :meth:`~kova_codec.KovaCodec.decode_with_lstm`):

* an **LSTM**, unbounded in the past -- carried from window to window as an explicit state;
* a **first convolution** reaching 3 frames either side -- fed :data:`CONV_PADDING` extra codes
  of real context on each end, which the codec trims off again after the convolution.

Everything after the LSTM has a finite receptive field to the right, so each window is also
decoded with :data:`LOOKAHEAD` frames of future codes that are computed and thrown away. What
is left, :data:`WINDOW` frames, is bit-comparable with a whole-utterance decode.

Two failure modes are baked into the window arithmetic here:

1. **Never emit audio the LSTM produced from padding.** If the final window runs past the last
   real code and that audio is emitted, its last ~20 frames drift from a whole-utterance decode
   by up to 0.065 -- an order of magnitude worse than the ~0.006 the seams themselves cost.
   :func:`plan_window` sizes the final window to end exactly at the last real code and asks for
   no state back from it.
2. **Never ask for the LSTM state at the very end of a window.** ``return_lstm_state=k`` splits
   the window at ``x[:k]`` / ``x[k:]``, so a window whose conv-trimmed length is exactly ``k``
   makes the second half empty and torch raises ``Expected sequence length to be larger than 0
   in RNN``. The invariant that avoids it -- a non-final window always has more than
   :data:`WINDOW` frames of output -- holds only while ``lookahead >= 1``, so
   :func:`plan_window` asserts it.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from kova_codec.constants import HOP_LENGTH, SAMPLE_RATE

#: Codes of real context fed to each end of a window to fill the first convolution's receptive
#: field. The codec trims exactly these frames back off after the convolution.
CONV_PADDING = 3

#: Frames of future context decoded but not emitted, covering the post-LSTM receptive field.
LOOKAHEAD = 9

#: Frames emitted per window in steady state. 31 frames is 387.5 ms at 80 codes/second.
WINDOW = 31

#: Code value the codec reads as padding: its embedding is zeroed, which is what a
#: whole-utterance decode sees beyond the ends of the sequence.
PAD_CODE = -1


@dataclass(frozen=True, slots=True)
class DecodeWindow:
    """One call to the codec: which codes to feed it, and which samples to keep.

    Offsets are into the *padded* code array (``CONV_PADDING`` pad codes on each side), which is
    what :class:`StreamingDecoder` holds, so a window is a plain slice.
    """

    #: Slice of the padded code array to decode, inclusive of conv padding on both ends.
    start: int
    stop: int

    #: Samples to keep from this window's output. ``emit_stop`` of ``None`` means "to the end".
    emit_start: int
    emit_stop: int | None

    #: Frame index whose LSTM state the next window resumes from, or ``None`` on the last one.
    return_state_at: int | None

    #: True when this window ends at the last real code of the utterance.
    is_last: bool

    @property
    def advance(self) -> int:
        """How far the write position moves after this window."""
        return 0 if self.is_last else (self.return_state_at or 0)


def plan_window(
    position: int,
    available: int,
    *,
    finished: bool,
    window: int = WINDOW,
    lookahead: int = LOOKAHEAD,
    conv_padding: int = CONV_PADDING,
) -> DecodeWindow | None:
    """The next window to decode, or ``None`` if more codes are needed first.

    Args:
        position: Index of the next code to emit audio for, in unpadded code coordinates.
        available: How many real codes have been produced so far.
        finished: Whether `available` is the final count. Only a finished stream can produce
            the last window, which is the one that emits its lookahead region.

    Pure arithmetic -- no codec, no torch -- so the window layout can be tested on its own.
    """
    if lookahead < 1:
        raise ValueError(
            "lookahead must be >= 1: with no lookahead a window's output is exactly `window` "
            "frames long, and asking the codec for the LSTM state at its end raises "
            "'Expected sequence length to be larger than 0 in RNN'."
        )
    if position >= available:
        return None

    width = window + 2 * lookahead + 2 * conv_padding
    padded_len = available + 2 * conv_padding
    is_last = position + window + lookahead >= available

    if not is_last:
        # A steady-state window reads real codes out to `position + width - conv_padding - 1`:
        # what it emits, plus the right-hand lookahead, plus the convolution's own context.
        # Firing before all of those exist silently shortens the window and puts a measurable
        # error (up to 0.05) on its last few frames -- the padding on the right of a finished
        # stream is real padding, but the end of an unfinished one is not.
        if not finished and position + width - conv_padding > available:
            return None
        stop = min(position + width, padded_len)
        emit_stop: int | None = (lookahead + window) * HOP_LENGTH
        trimmed = stop - position - 2 * conv_padding
        if trimmed <= window:
            raise AssertionError(
                f"Non-final window at {position} has only {trimmed} output frames but the LSTM "
                f"state is wanted at frame {window}; the codec cannot split a window there."
            )
        return DecodeWindow(
            start=position,
            stop=stop,
            emit_start=0 if position == 0 else lookahead * HOP_LENGTH,
            emit_stop=emit_stop,
            return_state_at=window,
            is_last=False,
        )

    if not finished:
        return None
    # The last window runs to the end of the padded array. Its trailing conv padding is trimmed
    # off after the convolution, so the LSTM never sees past the last real code.
    return DecodeWindow(
        start=position,
        stop=padded_len,
        emit_start=0 if position == 0 else lookahead * HOP_LENGTH,
        emit_stop=None,
        return_state_at=None,
        is_last=True,
    )


class StreamingDecoder:
    """Decodes a growing sequence of codes into audio, window by window.

    Feed codes in with :meth:`push` as the LM produces them and it returns whatever audio is
    complete; call :meth:`finish` once to flush the tail. The output is sample-for-sample
    comparable with :func:`decode_all` on the same codes: ~60 dB SNR on the shipped
    checkpoint, all of it at the seams.

    :meth:`prime` exists for voice cloning: the reference clip's codes are pushed through the
    decoder so the LSTM and the convolution start from real context rather than from silence,
    and the audio they produce is dropped. The alternative -- decoding the generated codes cold
    -- puts an audible discontinuity in the first frames.
    """

    def __init__(
        self,
        codec,
        *,
        window: int = WINDOW,
        lookahead: int = LOOKAHEAD,
        conv_padding: int = CONV_PADDING,
    ) -> None:
        self.codec = codec
        self.window = window
        self.lookahead = lookahead
        self.conv_padding = conv_padding
        self.sample_rate = int(getattr(codec, "sample_rate", SAMPLE_RATE))
        self.reset()

    # ------------------------------------------------------------------ state

    def reset(self) -> None:
        """Forget every code and all decoder state, ready for a new utterance."""
        # The leading pad is the left conv context of the very first window. The matching
        # trailing pad is only appended by finish(), since until then more codes may arrive.
        self._codes: list[int] = [PAD_CODE] * self.conv_padding
        self._n_codes = 0
        self._position = 0
        self._lstm_state: torch.Tensor | None = None
        self._finished = False
        self._discard_samples = 0

    @property
    def n_codes(self) -> int:
        """Real codes pushed so far."""
        return self._n_codes

    # ------------------------------------------------------------------ input

    def prime(self, codes: Sequence[int] | Iterable[int]) -> None:
        """Push `codes` as decoder context and drop the audio they produce.

        Must be called before any :meth:`push`, since the discard is counted in samples from
        the start of the stream.
        """
        codes = [int(c) for c in codes]
        if self.n_codes:
            raise RuntimeError(
                "prime() must come before the first push(): it discards audio from the start "
                "of the stream, so real codes must not already be queued behind it."
            )
        self._discard_samples += len(codes) * HOP_LENGTH
        self.push_codes(codes)

    def push_codes(self, codes: Sequence[int] | Iterable[int]) -> None:
        """Append codes without decoding anything."""
        if self._finished:
            raise RuntimeError("This decoder is finished; call reset() before reusing it.")
        before = len(self._codes)
        self._codes.extend(int(c) for c in codes)
        self._n_codes += len(self._codes) - before

    def push(self, codes: Sequence[int] | Iterable[int] = ()) -> np.ndarray:
        """Append codes and decode every window they completed. May return no samples."""
        self.push_codes(codes)
        return self._drain(finished=False)

    def finish(self) -> np.ndarray:
        """Decode everything left, including the final partial window."""
        if self._finished:
            return _empty()
        # Now that no more codes are coming, close the padded array off on the right. Without
        # this the last window's slice runs short and the codec's conv trim eats the final
        # `conv_padding` frames of real audio.
        self._codes.extend([PAD_CODE] * self.conv_padding)
        audio = self._drain(finished=True)
        self._finished = True
        return audio

    # ------------------------------------------------------------------ decode

    def _drain(self, *, finished: bool) -> np.ndarray:
        pieces: list[np.ndarray] = []
        while (
            plan := plan_window(
                self._position,
                self.n_codes,
                finished=finished,
                window=self.window,
                lookahead=self.lookahead,
                conv_padding=self.conv_padding,
            )
        ) is not None:
            pieces.append(self._decode_window(plan))
            self._position += plan.advance
            if plan.is_last:
                break
        return np.concatenate(pieces) if pieces else _empty()

    def _decode_window(self, plan: DecodeWindow) -> np.ndarray:
        chunk = torch.tensor(
            self._codes[plan.start : plan.stop], dtype=torch.long, device=self.codec.device
        ).unsqueeze(0)
        with torch.inference_mode():
            audio, state = self.codec.decode_with_lstm(
                chunk,
                self._lstm_state,
                return_lstm_state=plan.return_state_at,
                conv_padding=self.conv_padding,
            )
        if not plan.is_last:
            self._lstm_state = state
        samples = audio[0, plan.emit_start : plan.emit_stop].float().cpu().numpy()
        return self._discard(samples)

    def _discard(self, samples: np.ndarray) -> np.ndarray:
        """Drop the leading samples :meth:`prime` asked for, across as many windows as it takes."""
        if self._discard_samples <= 0:
            return samples
        n = min(self._discard_samples, samples.size)
        self._discard_samples -= n
        return samples[n:]


def decode_all(codec, codes: Sequence[int] | np.ndarray) -> np.ndarray:
    """Whole-utterance decode: every code at once, no windows, no LSTM bookkeeping.

    The right choice when the codes are already complete -- it is one kernel launch per layer
    instead of one per window, and there are no seams at all.
    """
    array = np.asarray(codes, dtype=np.int64)
    if array.size == 0:
        return _empty()
    with torch.inference_mode():
        audio = codec.decode(torch.from_numpy(array))
    return np.ascontiguousarray(audio.float().cpu().numpy(), dtype=np.float32)


def load_codec(
    checkpoint: str | os.PathLike[str] | None = None,
    *,
    device: torch.device | str | None = None,
    encode: bool = False,
    wavlm: str | os.PathLike[str] | None = None,
):
    """Load the codec, WavLM-free unless `encode` is asked for.

    A decode-only codec skips WavLM entirely: ~1.2 GB lighter, several seconds faster to start,
    and it never imports ``transformers`` inside the codec. Plain TTS -- even with a LoRA voice
    -- only ever decodes, so encoding is opt-in and paid for only by voice cloning.
    """
    from kova_codec import KovaCodec
    from kova_tts import paths

    path = paths.codec_path(checkpoint)
    if not encode:
        return KovaCodec.from_checkpoint(path, device=device, decode_only=True)
    return KovaCodec.from_checkpoint(path, device=device, wavlm=paths.wavlm_path(wavlm))


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=np.float32)
