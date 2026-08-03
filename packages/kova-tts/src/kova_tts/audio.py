"""Waveform I/O and conditioning.

One convention throughout: a waveform is a **1-D float32 numpy array, mono, nominally in
[-1, 1]**, at :data:`~kova_codec.constants.SAMPLE_RATE` unless a function says otherwise.
Anything crossing a module boundary is in that form, so nothing downstream has to guess about
channel order or dtype.

:func:`normalize_loudness` is not a nicety: the training clips were LUFS-normalised to -23 with
exactly this procedure before being tokenized, so reference audio fed to the codec must be
normalised the same way or cloning quality drops.
"""

from __future__ import annotations

import io
import math
import os
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from kova_codec.constants import SAMPLE_RATE, TARGET_LUFS

#: Peak the loudness normaliser backs off to, to avoid clipping after the gain is applied.
_PEAK_CEILING = 0.99

#: Below this peak amplitude a clip is treated as digital silence: measuring its loudness is
#: meaningless (and returns -inf), and applying gain would only amplify noise.
_SILENCE_PEAK = 1e-3

#: pyloudnorm's integration block is 400 ms; shorter clips cannot be measured at all.
_LUFS_BLOCK_SECONDS = 0.4


def as_waveform(wav: np.ndarray) -> np.ndarray:
    """Coerce an array to the module's convention: 1-D float32 mono, downmixing if needed."""
    arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim == 2:
        # soundfile hands back (samples, channels); torch and the codec use (channels, samples).
        # Averaging the shorter axis is right in both layouts.
        axis = 1 if arr.shape[0] >= arr.shape[1] else 0
        arr = arr.mean(axis=axis, dtype=np.float32)
    elif arr.ndim != 1:
        raise ValueError(f"Expected a mono or 2-D waveform, got shape {arr.shape}.")
    return np.ascontiguousarray(arr, dtype=np.float32)


def resample(wav: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling to `target_rate`. Returns the input unchanged when rates match."""
    if orig_rate == target_rate:
        return as_waveform(wav)
    if orig_rate <= 0 or target_rate <= 0:
        raise ValueError(f"Sample rates must be positive, got {orig_rate} -> {target_rate}.")

    from scipy.signal import resample_poly

    g = math.gcd(orig_rate, target_rate)
    resampled = resample_poly(as_waveform(wav), target_rate // g, orig_rate // g)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def load_audio(path: str | os.PathLike[str], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Read an audio file as a mono float32 waveform at `sample_rate`.

    Loudness is left alone; call :func:`normalize_loudness` before encoding to the codec.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"Audio file not found: {file}")
    data, file_rate = sf.read(str(file), dtype="float32", always_2d=False)
    return resample(as_waveform(data), int(file_rate), sample_rate)


def normalize_loudness(
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    target_lufs: float = TARGET_LUFS,
) -> np.ndarray:
    """Loudness-normalise to `target_lufs` (ITU-R BS.1770), then limit the peak.

    The unmeasurable cases matter as much as the normal one: a clip that is silent, shorter
    than one 400 ms integration block, or that pyloudnorm refuses is returned unchanged rather
    than being scaled by an infinite gain.
    """
    audio = as_waveform(wav)
    if audio.size == 0 or float(np.max(np.abs(audio))) < _SILENCE_PEAK:
        return audio
    if audio.size < int(_LUFS_BLOCK_SECONDS * sample_rate):
        return audio

    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)
    try:
        loudness = float(meter.integrated_loudness(audio))
    except ValueError:
        # Raised for clips pyloudnorm considers unmeasurable. Returning them unchanged keeps a
        # marginal clip out of the pipeline's way; scaling by an unmeasured gain would not.
        return audio
    if not math.isfinite(loudness):
        return audio

    with warnings.catch_warnings():
        # pyloudnorm warns when the gain would clip; the peak limiter below is the answer to
        # that, so the warning is noise for the caller.
        warnings.filterwarnings("ignore", message="Possible clipped samples")
        gained = pyln.normalize.loudness(audio, loudness, target_lufs)
    normalized = np.asarray(gained, np.float32)
    peak = float(np.max(np.abs(normalized)))
    if peak > _PEAK_CEILING:
        normalized = normalized / peak * _PEAK_CEILING
    return np.ascontiguousarray(normalized, dtype=np.float32)


def trim_leading(wav: np.ndarray, seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Drop the first `seconds` of audio.

    Used on cloned output, where the model re-renders the reference clip before the target
    text. Rounding to the nearest sample -- rather than truncating -- is deliberate: a few
    leftover milliseconds of the reference are audible, a few missing ones are not.
    """
    audio = as_waveform(wav)
    if seconds <= 0:
        return audio
    return audio[min(round(seconds * sample_rate), audio.size) :]


def to_pcm_bytes(wav: np.ndarray) -> bytes:
    """Waveform -> 16-bit little-endian PCM, the raw payload the streaming server sends."""
    audio = np.clip(as_waveform(wav), -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


def to_wav_bytes(wav: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Waveform -> a complete 16-bit WAV file in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, as_waveform(wav), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def save_wav(
    path: str | os.PathLike[str],
    wav: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Write a 16-bit WAV, creating the parent directory. Returns the path written."""
    file = Path(path).expanduser()
    file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file), as_waveform(wav), sample_rate, format="WAV", subtype="PCM_16")
    return file
