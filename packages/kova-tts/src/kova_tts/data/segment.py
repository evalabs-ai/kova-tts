"""Cutting recordings down to trainable clips: split on silence, then trim the edges.

Two separate jobs, both driven by the same frame-energy measurement:

**Splitting** exists because of the token budget. Audio costs 80 tokens per second, so a
4096-token training row holds about 51 seconds *including* the text prompt -- and rows over the
limit are dropped by :class:`~kova_tts.finetune.dataset.MaskedCausalDataset`, not truncated.
A half-hour recording must therefore become clips before it becomes training data. Cuts are
placed in the middle of a silence so no word is ever sliced in half; when a stretch of audio
holds no silence at all, nothing is cut and the caller reports the clip as too long. Inventing
a cut mid-word would be worse than dropping it.

**Trimming** removes the leading and trailing silence every recording has. Left in, that
silence is tokens the model is taught to emit before speaking and after finishing -- which is
exactly how an adapter learns to trail off instead of stopping.

The threshold is relative to the loudest frame in the recording (``top_db``, as in the usual
convention) rather than an absolute level, because trimming happens *before* loudness
normalisation and the input level is whatever the user's recording chain produced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kova_codec.constants import SAMPLE_RATE
from kova_tts.audio import as_waveform, trim_leading

#: Analysis window. 25 ms is short enough to find the gap between two words and long enough
#: that a single glottal pulse does not read as speech.
FRAME_SECONDS = 0.025

#: Window step. Every boundary this module reports is a multiple of the hop.
HOP_SECONDS = 0.010

#: Frames quieter than this many dB below the recording's loudest frame count as silence.
DEFAULT_TOP_DB = 40.0

#: Peak amplitude below which a waveform is digital silence and no threshold means anything.
_SILENCE_PEAK = 1e-4

_EPS = 1e-10


@dataclass(frozen=True, slots=True)
class SegmentSettings:
    """How a recording is cut up. Every duration is in seconds.

    The defaults suit narration recorded in one take: split at 30 s (2400 audio tokens, well
    under the 4096-token row limit), keep anything over half a second, and treat 400 ms of
    quiet as a sentence boundary.
    """

    #: Clips longer than this are split. Also the point at which an un-splittable clip is
    #: reported as too long.
    max_seconds: float = 30.0

    #: Clips shorter than this are dropped. Below ~0.4 s loudness cannot be measured at all
    #: (:func:`~kova_tts.audio.normalize_loudness` returns such clips unchanged), and a clip
    #: that short rarely holds a usable utterance.
    min_seconds: float = 0.5

    #: Silence threshold, in dB below the recording's loudest frame.
    top_db: float = DEFAULT_TOP_DB

    #: A quiet stretch must last this long to be treated as a boundary worth cutting at.
    min_silence: float = 0.4

    #: Silence deliberately left at each end after trimming. A hard cut on the first sample of
    #: speech clips the attack of a plosive; 50 ms is inaudible and safe.
    pad_seconds: float = 0.05

    def validate(self) -> None:
        if self.max_seconds <= self.min_seconds:
            raise ValueError(
                f"max_seconds ({self.max_seconds}) must exceed min_seconds ({self.min_seconds}); "
                f"every clip would be dropped for being both too long and too short."
            )
        if self.top_db <= 0:
            raise ValueError(f"top_db must be > 0 dB below the peak, got {self.top_db}.")
        if self.min_silence <= 0:
            raise ValueError(f"min_silence must be > 0 seconds, got {self.min_silence}.")
        if self.pad_seconds < 0:
            raise ValueError(f"pad_seconds must be >= 0, got {self.pad_seconds}.")


# ------------------------------------------------------------------------------- measurement


def frame_db(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    frame_seconds: float = FRAME_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> np.ndarray:
    """Per-frame RMS in dB relative to the loudest frame. Frame ``i`` starts at ``i * hop``.

    Relative rather than absolute so a quiet recording and a hot one are trimmed the same way.
    """
    audio = as_waveform(wav)
    frame = max(1, int(round(frame_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    if audio.size < frame:
        # Too short to window: the whole clip is one frame, so it is all speech or all silence.
        energy = float(np.mean(audio.astype(np.float64) ** 2)) if audio.size else 0.0
        rms = np.array([np.sqrt(energy)])
    else:
        windows = np.lib.stride_tricks.sliding_window_view(audio, frame)[::hop]
        rms = np.sqrt(np.mean(windows.astype(np.float64) ** 2, axis=-1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= _EPS:
        return np.full(rms.shape, -np.inf)
    return 20.0 * np.log10(np.maximum(rms, _EPS) / peak)


def _speech_mask(wav: np.ndarray, sample_rate: int, top_db: float) -> tuple[np.ndarray, int, int]:
    """Boolean per-frame "this is not silence" mask, plus the frame and hop in samples."""
    db = frame_db(wav, sample_rate)
    frame = max(1, int(round(FRAME_SECONDS * sample_rate)))
    hop = max(1, int(round(HOP_SECONDS * sample_rate)))
    return db > -top_db, frame, hop


def is_silent(wav: np.ndarray) -> bool:
    """True when a waveform holds no signal at all -- empty, or below the digital noise floor."""
    audio = as_waveform(wav)
    return audio.size == 0 or float(np.max(np.abs(audio))) < _SILENCE_PEAK


def silent_spans(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    top_db: float = DEFAULT_TOP_DB,
    min_silence: float = 0.4,
) -> list[tuple[int, int]]:
    """Sample ranges of every silence at least `min_silence` long, in order.

    Interior silences only: a span touching either end of the recording is leading or trailing
    silence, which :func:`trim_silence` handles and which is never a useful place to cut.
    """
    audio = as_waveform(wav)
    speech, frame, hop = _speech_mask(audio, sample_rate, top_db)
    if not speech.any():
        return []

    quiet = ~speech
    # Frame-index run boundaries, via the transitions in the padded mask.
    padded = np.concatenate(([0], quiet.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    minimum = int(round(min_silence * sample_rate))

    spans: list[tuple[int, int]] = []
    for start_frame, stop_frame in zip(edges[::2], edges[1::2], strict=True):
        # Leading and trailing silence is trim_silence's job, and is never a place to cut.
        # Tested on frame indices rather than sample positions: the last window starts at most
        # `frame` samples before the end, so a trailing run never reaches audio.size.
        if start_frame == 0 or stop_frame == speech.size:
            continue
        start = int(start_frame) * hop
        # The last quiet frame is stop_frame - 1 and covers `frame` samples.
        stop = min((int(stop_frame) - 1) * hop + frame, audio.size)
        if stop - start >= minimum:
            spans.append((start, stop))
    return spans


# ---------------------------------------------------------------------------------- cutting


def split_on_silence(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    max_seconds: float = 30.0,
    top_db: float = DEFAULT_TOP_DB,
    min_silence: float = 0.4,
) -> list[np.ndarray]:
    """Split a recording into pieces of at most `max_seconds`, cutting only inside silences.

    Greedy and rightmost-first: each piece runs to the last silence that still fits under the
    limit, which yields the fewest cuts and keeps whole sentences together. A stretch with no
    silence in it is left whole and comes back over-length -- the caller reports it rather than
    this function cutting mid-word.

    Returns the input as a single-element list when it already fits.
    """
    audio = as_waveform(wav)
    limit = int(round(max_seconds * sample_rate))
    if audio.size <= limit:
        return [audio]

    spans = silent_spans(audio, sample_rate, top_db=top_db, min_silence=min_silence)
    cuts = [(start + stop) // 2 for start, stop in spans]
    if not cuts:
        return [audio]

    pieces: list[np.ndarray] = []
    start = 0
    while audio.size - start > limit:
        fitting = [c for c in cuts if start < c <= start + limit]
        if fitting:
            cut = fitting[-1]
        else:
            # No boundary inside the window. Overshoot to the next one rather than cut blind;
            # the piece stays over-length and the caller decides what to do about it.
            later = [c for c in cuts if c > start + limit]
            if not later:
                break
            cut = later[0]
        pieces.append(audio[start:cut])
        start = cut
    pieces.append(audio[start:])
    return [p for p in pieces if p.size]


def trim_silence(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    top_db: float = DEFAULT_TOP_DB,
    pad_seconds: float = 0.05,
) -> np.ndarray:
    """Drop leading and trailing silence, keeping `pad_seconds` of it at each end.

    Returns an empty array when the whole clip is below the threshold, which is the signal the
    caller turns into a "silent" skip.
    """
    audio = as_waveform(wav)
    if audio.size == 0:
        return audio
    speech, frame, hop = _speech_mask(audio, sample_rate, top_db)
    voiced = np.flatnonzero(speech)
    if voiced.size == 0:
        return audio[:0]

    pad = int(round(pad_seconds * sample_rate))
    start = max(0, int(voiced[0]) * hop - pad)
    # A clip that is still speaking in its final window has no trailing silence to remove:
    # the last window begins up to `frame` samples before the end and would otherwise take a
    # few milliseconds of speech with it.
    ends_speaking = int(voiced[-1]) == speech.size - 1
    stop = audio.size if ends_speaking else min(audio.size, int(voiced[-1]) * hop + frame + pad)

    trimmed = trim_leading(audio, start / sample_rate, sample_rate)
    tail = audio.size - stop
    if tail > 0:
        trimmed = trimmed[: max(0, trimmed.size - tail)]
    return np.ascontiguousarray(trimmed, dtype=np.float32)
