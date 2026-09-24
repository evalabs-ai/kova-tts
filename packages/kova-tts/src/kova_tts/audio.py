"""Waveform I/O, conditioning, rate conversion and container encoding.

One convention throughout: a waveform is a **1-D float32 numpy array, mono, nominally in
[-1, 1]**. Anything crossing a module boundary is in that form, so nothing downstream has to
guess about channel order or dtype.

Two rates meet here. Audio going *into* the codec -- reference clips, training data -- is at
:data:`~kova_codec.constants.SAMPLE_RATE`, 32 kHz, the only rate the encoder takes, and the
loading and conditioning helpers default to it. Audio coming *out* is at the decoder's rate,
:data:`~kova_codec.constants.OUTPUT_SAMPLE_RATE`, 48 kHz, and the writers default to that.

Reference audio fed to the codec goes through :func:`normalize_loudness` first, so that clips
recorded at different levels all reach it at -23 LUFS.

The encoder also takes 16 kHz natively, and :func:`encoder_input_rate` decides which of the two
a source goes in at.

**Rate conversion.** The model speaks at 48 kHz; voice-agent pipelines run at 16 kHz and
telephony at 8 kHz. :func:`resample` converts a finished waveform; :class:`StreamingResampler`
converts a stream, and its concatenated output equals resampling the whole signal at once. A
stateless resample applied per chunk does not: it restarts the anti-alias filter at every chunk
boundary, which puts a discontinuity into the audio several times a second.

**Container encoding.** :func:`encode_audio` writes ``pcm``, ``wav``, ``mp3``, ``flac`` and
``opus`` through the libsndfile :mod:`soundfile` already bundles -- no ffmpeg, no extra
dependency. Which of those a given install can actually write is *probed*, not assumed, into
:data:`SUPPORTED_FORMATS`, because a soundfile wheel may carry a libsndfile older than the
1.1 release that added MPEG.

**Full-scale 16-bit is asymmetric.** ``int16`` runs from -32768 to +32767, so the two signs do
not share a scale factor; :func:`to_pcm_bytes` uses both.
"""

from __future__ import annotations

import io
import math
import os
import struct
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import soundfile as sf

from kova_codec.constants import LOW_SAMPLE_RATE, OUTPUT_SAMPLE_RATE, SAMPLE_RATE, TARGET_LUFS

if TYPE_CHECKING:  # torch is imported lazily; importing this module must stay cheap.
    import torch

#: Peak the loudness normaliser backs off to, to avoid clipping after the gain is applied.
_PEAK_CEILING = 0.99

#: Below this peak amplitude a clip is treated as digital silence: measuring its loudness is
#: meaningless (and returns -inf), and applying gain would only amplify noise.
_SILENCE_PEAK = 1e-3

#: pyloudnorm's integration block is 400 ms; shorter clips cannot be measured at all.
_LUFS_BLOCK_SECONDS = 0.4

#: Zero crossings kept either side of the resampling filter's centre. Six is torchaudio's
#: default and the value the streaming and one-shot paths must agree on, since they are the
#: same filter and their outputs are compared sample for sample.
_FILTER_WIDTH = 6

#: Fraction of Nyquist the anti-alias filter rolls off at. Below 1.0 so the transition band
#: fits below Nyquist instead of folding back as aliasing.
_ROLLOFF = 0.99

#: 16-bit PCM scale factors. ``int16`` spans [-32768, +32767], so the negative side has one
#: more code than the positive one and the two cannot share a factor: scaling both by 32767
#: leaves -1.0 a step short of the rail, and scaling both by 32768 makes +1.0 overflow.
_PCM_SCALE_NEGATIVE = 32768.0
_PCM_SCALE_POSITIVE = 32767.0


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


# ------------------------------------------------------------------------------ rate conversion


@lru_cache(maxsize=16)
def _sinc_resample_kernel(orig: int, new: int) -> tuple[torch.Tensor, int]:
    """Windowed-sinc polyphase kernel for `orig` -> `new`, already reduced by their GCD.

    Returns the kernel, shape ``[new, 1, 2 * width + orig]``, and `width`: the filter's
    one-sided context in input samples.

    Built by torchaudio so that this module's output matches ``torchaudio.functional.resample``
    sample for sample. That entry point is private, so the import is guarded and
    :func:`_build_sinc_kernel` -- the same construction, written out -- takes over if a release
    moves it. Cached: the kernel depends only on the reduced rate pair, and every frame of a
    stream wants the same one.
    """
    import torch

    try:
        from torchaudio.functional.functional import _get_sinc_resample_kernel

        kernel, width = _get_sinc_resample_kernel(
            orig,
            new,
            gcd=1,
            lowpass_filter_width=_FILTER_WIDTH,
            rolloff=_ROLLOFF,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
    except (ImportError, AttributeError, TypeError):
        kernel, width = _build_sinc_kernel(orig, new)
    return kernel, int(width)


def _build_sinc_kernel(orig: int, new: int) -> tuple[torch.Tensor, int]:
    """Build the polyphase kernel directly: a Hann-squared windowed sinc, one row per phase.

    Output sample ``m`` is ``x`` interpolated at ``m / new``; row ``m % new`` holds the taps for
    that fractional offset, so one strided convolution produces every phase at once.
    """
    import torch

    base_freq = min(orig, new) * _ROLLOFF
    width = math.ceil(_FILTER_WIDTH * orig / base_freq)

    idx = torch.arange(-width, width + orig, dtype=torch.float32)[None, None] / orig
    t = torch.arange(0, -new, -1, dtype=torch.float32)[:, None, None] / new + idx
    t *= base_freq
    t = t.clamp_(-_FILTER_WIDTH, _FILTER_WIDTH)
    window = torch.cos(t * math.pi / _FILTER_WIDTH / 2) ** 2
    t *= math.pi
    kernel = torch.where(t == 0, torch.ones_like(t), t.sin() / t)
    kernel *= window * (base_freq / orig)
    return kernel, width


class StreamingResampler:
    """Rate conversion for a stream, whose output equals resampling the whole signal at once.

    That equality is what the class is for. A stateless resampler called once per chunk restarts
    the anti-alias filter at every chunk boundary, and the step it leaves behind is an audible
    click -- the streaming decoder emits a chunk roughly every 390 ms, so that is several clicks
    a second. Here the filter's context crosses the boundary, so there is no join to hear.

    Reduce the rates by their GCD to ``orig``/``new``. Output sample ``m`` lives in block
    ``b = m // new`` at phase ``m % new``, and block ``b`` is one stride-``orig`` convolution of
    the polyphase kernel over input samples ``[b * orig - width, b * orig + width + orig)``. A
    rolling buffer holds those input samples, left-padded with `width` zeros so block 0 sees
    exactly what a whole-signal resample's left zero-pad would show it. :meth:`process` emits
    every block whose right-hand context has arrived and keeps the rest; :meth:`flush` zero-pads
    the right and emits the remainder, truncated to the same length a whole-signal resample
    would produce.

    >>> r = StreamingResampler(48_000, 16_000)
    >>> out = [r.process(chunk) for chunk in chunks] + [r.flush()]

    Equal rates are a pass-through, so a caller can construct one unconditionally.
    """

    def __init__(self, orig_rate: int, target_rate: int) -> None:
        if orig_rate <= 0 or target_rate <= 0:
            raise ValueError(f"Sample rates must be positive, got {orig_rate} -> {target_rate}.")
        self.orig_rate = int(orig_rate)
        self.target_rate = int(target_rate)

        divisor = math.gcd(self.orig_rate, self.target_rate)
        self._orig = self.orig_rate // divisor
        self._new = self.target_rate // divisor
        if self.orig_rate == self.target_rate:
            self._kernel, self._width = None, 0
        else:
            self._kernel, self._width = _sinc_resample_kernel(self._orig, self._new)
        self.reset()

    def reset(self) -> None:
        """Forget the stream so far, ready for a new one."""
        # The leading zeros are the left context of block 0, which is what makes the first
        # chunk's output equal to the first samples of a whole-signal resample.
        self._buffer = np.zeros(self._width, dtype=np.float32)
        #: Global input index the buffer starts at; negative while the left pad is still there.
        self._buffer_start = -self._width
        self._seen = 0
        self._next_block = 0
        self._flushed = False

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Convert `chunk` and return whatever output is now complete. May return nothing.

        Output is held back only for as long as the filter needs future input to produce it:
        `width` input samples, 19 of them at 48 kHz -> 16 kHz.
        """
        audio = as_waveform(chunk)
        if self._kernel is None:
            return audio
        if self._flushed:
            raise RuntimeError("This resampler has been flushed; call reset() before reusing it.")
        if audio.size:
            self._buffer = np.concatenate((self._buffer, audio))
            self._seen += audio.size

        # A block is ready once its right-hand context exists, which is `width` samples past the
        # last input sample it convolves. Integer division floors, and the count is clamped
        # because a stream shorter than the filter's context has produced no blocks at all.
        available = max((self._seen - self._width) // self._orig, 0)
        blocks = available - self._next_block
        if blocks <= 0:
            return _empty()
        out = self._run(blocks)
        self._trim()
        return out

    def flush(self) -> np.ndarray:
        """Emit the tail: the blocks whose right context is the end of the signal.

        Idempotent, and the output length is pinned to ``ceil(new * total_input / orig)`` --
        the same count a whole-signal resample returns, no rounding drift over a long stream.
        """
        if self._kernel is None or self._flushed:
            return _empty()
        self._flushed = True

        total = (self._new * self._seen + self._orig - 1) // self._orig
        emitted = self._next_block * self._new
        if total <= emitted:
            return _empty()

        blocks = -(-(total - emitted) // self._new)
        needed = (self._next_block + blocks) * self._orig + self._width - self._buffer_start
        if needed > self._buffer.size:
            self._buffer = np.concatenate(
                (self._buffer, np.zeros(needed - self._buffer.size, dtype=np.float32))
            )
        return self._run(blocks)[: total - emitted]

    # ------------------------------------------------------------------ internals

    def _run(self, blocks: int) -> np.ndarray:
        """Convolve `blocks` blocks out of the buffer and interleave their phases."""
        import torch

        start = self._next_block * self._orig - self._width - self._buffer_start
        window = self._buffer[start : start + blocks * self._orig + 2 * self._width]
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))[None, None]
        out = torch.nn.functional.conv1d(x, self._kernel, stride=self._orig)
        self._next_block += blocks
        # [1, new, blocks] -> block-major, which is the order the output samples come in.
        return np.ascontiguousarray(out.transpose(1, 2).reshape(-1).numpy(), dtype=np.float32)

    def _trim(self) -> None:
        """Drop the input the next block no longer needs as left context."""
        start = self._next_block * self._orig - self._width
        offset = start - self._buffer_start
        if offset > 0:
            self._buffer = self._buffer[offset:]
            self._buffer_start = start


def resample(wav: np.ndarray, orig_rate: int, target_rate: int) -> np.ndarray:
    """Polyphase resampling to `target_rate`. Returns the input unchanged when rates match.

    The same filter :class:`StreamingResampler` applies, so a waveform converted here is equal
    to the same waveform converted chunk by chunk and concatenated. That is what keeps
    ``KovaTTS.generate(sample_rate=...)`` and ``KovaTTS.stream(sample_rate=...)`` returning the
    same audio.
    """
    if orig_rate == target_rate:
        return as_waveform(wav)
    resampler = StreamingResampler(orig_rate, target_rate)
    head = resampler.process(wav)
    tail = resampler.flush()
    if not tail.size:
        return head
    return np.ascontiguousarray(np.concatenate((head, tail)), dtype=np.float32)


def _empty() -> np.ndarray:
    return np.zeros(0, dtype=np.float32)


def file_sample_rate(path: str | os.PathLike[str]) -> int:
    """The rate an audio file was recorded at, read from its header."""
    file = Path(path).expanduser()
    if not file.is_file():
        raise FileNotFoundError(f"Audio file not found: {file}")
    return int(sf.info(str(file)).samplerate)


def encoder_input_rate(source_rate: int) -> int:
    """The rate to hand the encoder audio that was recorded at `source_rate`.

    A source at 16 kHz or below -- phone audio, most ASR corpora -- goes in at 16 kHz, through
    the encoder's own 16 kHz path. It has nothing above 8 kHz either way, and upsampling it to
    32 kHz first leaves the decoder with a dull top octave, where encoding it natively lets the
    decoder fill that octave in. Everything else goes in at 32 kHz.
    """
    return LOW_SAMPLE_RATE if source_rate <= LOW_SAMPLE_RATE else SAMPLE_RATE


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


def trim_leading(
    wav: np.ndarray, seconds: float, sample_rate: int = OUTPUT_SAMPLE_RATE
) -> np.ndarray:
    """Drop the first `seconds` of audio.

    Used on cloned output, where the model re-renders the reference clip before the target text.
    The count rounds to the nearest sample rather than truncating: a few leftover milliseconds
    of the reference are audible, a few missing ones are not.
    """
    audio = as_waveform(wav)
    if seconds <= 0:
        return audio
    return audio[min(round(seconds * sample_rate), audio.size) :]


def to_pcm_bytes(wav: np.ndarray) -> bytes:
    """Waveform -> 16-bit little-endian PCM, the raw payload the streaming server sends.

    Negative and non-negative samples are scaled by different factors, which is what puts -1.0
    and +1.0 on the format's rails. See :data:`_PCM_SCALE_NEGATIVE`.
    """
    audio = np.clip(as_waveform(wav), -1.0, 1.0)
    scaled = np.where(audio < 0, audio * _PCM_SCALE_NEGATIVE, audio * _PCM_SCALE_POSITIVE)
    return np.rint(scaled).astype("<i2").tobytes()


def to_wav_bytes(wav: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> bytes:
    """Waveform -> a complete 16-bit WAV file in memory."""
    buffer = io.BytesIO()
    sf.write(buffer, as_waveform(wav), sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


# --------------------------------------------------------------------------- container formats


@dataclass(frozen=True, slots=True)
class _Container:
    """One value of `fmt`, and everything needed to write it.

    Args:
        sf_format: libsndfile format name, or ``None`` when this module writes the bytes itself.
        sf_subtype: libsndfile subtype within that format.
        media_type: What an HTTP layer should put in ``Content-Type``.
        rates: Sample rates the codec accepts, or ``None`` for "any". MP3 and Opus each accept
            a fixed list and refuse everything else, so a rate outside it is snapped to the
            nearest one at or above it rather than turning into an error the caller cannot act
            on -- the container states its own rate, so the audio stays correct either way.
        streams: Whether an incremental encoder for it delivers bytes within
            :data:`STREAMING_LATENCY_BUDGET_MS`. See :class:`StreamingEncoder`.
    """

    sf_format: str | None
    sf_subtype: str | None
    media_type: str
    rates: tuple[int, ...] | None
    streams: bool


#: Rates MPEG-1/2/2.5 layer III is defined for. libsndfile refuses anything else outright.
_MP3_RATES = (8000, 11025, 12000, 16000, 22050, 24000, 32000, 44100, 48000)

#: Rates Opus is defined for. The model's native 48 kHz is among them; a rate that is not, such
#: as the 32 kHz of the older decoder, is carried at the nearest rate above it -- upsampling
#: discards nothing.
_OPUS_RATES = (8000, 12000, 16000, 24000, 48000)

_CONTAINERS: dict[str, _Container] = {
    "pcm": _Container(None, None, "application/octet-stream", None, True),
    "wav": _Container("WAV", "PCM_16", "audio/wav", None, True),
    "mp3": _Container("MP3", "MPEG_LAYER_III", "audio/mpeg", _MP3_RATES, True),
    "flac": _Container("FLAC", "PCM_16", "audio/flac", None, True),
    "opus": _Container("OGG", "OPUS", "audio/ogg", _OPUS_RATES, False),
}

#: Added latency an incremental encoder is allowed to cost over raw PCM, in milliseconds.
STREAMING_LATENCY_BUDGET_MS = 100


def _probe_formats() -> tuple[str, ...]:
    """Which of :data:`_CONTAINERS` this install can actually write.

    Probed rather than assumed: :mod:`soundfile` wheels bundle their own libsndfile, MPEG
    support only arrived in libsndfile 1.1, and Opus needs a build with the Ogg/Opus libraries
    linked in. Probing turns an older wheel into a message naming what it can write instead of a
    traceback from inside the C library.
    """
    available = set(sf.available_formats())
    usable = []
    for name, container in _CONTAINERS.items():
        if container.sf_format is None:
            usable.append(name)
        elif container.sf_format in available:
            if container.sf_subtype in sf.available_subtypes(container.sf_format):
                usable.append(name)
    return tuple(usable)


#: Formats :func:`encode_audio` can produce **in this install**. Read it rather than hard-coding
#: the list: it is what an HTTP layer should validate against and list to clients.
SUPPORTED_FORMATS: tuple[str, ...] = _probe_formats()

#: Formats whose :class:`StreamingEncoder` delivers bytes within
#: :data:`STREAMING_LATENCY_BUDGET_MS` of the audio being handed to it. ``opus`` is missing on
#: purpose -- see :class:`StreamingEncoder`.
STREAMING_FORMATS: tuple[str, ...] = tuple(f for f in SUPPORTED_FORMATS if _CONTAINERS[f].streams)


def _container(fmt: str) -> _Container:
    """Look up a format name, with an error that says what this install can do instead."""
    name = str(fmt).strip().lower()
    if name in SUPPORTED_FORMATS:
        return _CONTAINERS[name]
    usable = ", ".join(SUPPORTED_FORMATS)
    if name in _CONTAINERS:
        raise ValueError(
            f"Audio format {name!r} is not available: the libsndfile bundled with this "
            f"soundfile install (version {sf.__libsndfile_version__}) cannot write it. "
            f"Upgrade soundfile, or ask for one of: {usable}."
        )
    raise ValueError(f"Unknown audio format {name!r}. This install can produce: {usable}.")


def content_type(fmt: str) -> str:
    """Media type for `fmt`: what an HTTP layer must send in ``Content-Type``.

    Lives next to :func:`encode_audio` so the header cannot drift from the bytes -- a response
    labelled ``audio/wav`` carrying MP3 is a bug no client can work around.
    """
    return _container(fmt).media_type


def container_rate(fmt: str, sample_rate: int) -> int:
    """The rate `fmt` will actually be written at, which is not always the one asked for.

    MP3 and Opus are each defined for a fixed set of rates. A request outside the set is snapped
    up to the nearest one the codec accepts -- up, so no bandwidth is thrown away -- and the
    audio is resampled to it. The container records that rate, so the result plays at the right
    speed and pitch; only the sample grid changed. Call this when you need to know the rate
    before you have the bytes, as an HTTP layer does.
    """
    container = _container(fmt)
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    if container.rates is None or sample_rate in container.rates:
        return int(sample_rate)
    return next((r for r in container.rates if r >= sample_rate), container.rates[-1])


def encode_audio(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """Encode a whole waveform into `fmt` and return the bytes.

    Args:
        wav: Mono float32 waveform, nominally in [-1, 1].
        sample_rate: Rate `wav` is at. Also the rate written into the container, unless the
            codec does not accept it -- see :func:`container_rate`.
        fmt: One of :data:`SUPPORTED_FORMATS`.

    ``pcm`` is raw 16-bit little-endian samples with no container and therefore no rate in the
    bytes: whatever plays it has to be told the rate out of band. Everything else is a real
    file, self-describing.

    Use this whenever the audio is already complete; :class:`StreamingEncoder` is for the case
    where it is not, and it costs a little latency that this path does not.
    """
    container = _container(fmt)
    audio = as_waveform(wav)
    if container.sf_format is None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
        return to_pcm_bytes(audio)

    rate = container_rate(fmt, sample_rate)
    if rate != sample_rate:
        audio = resample(audio, sample_rate, rate)
    buffer = io.BytesIO()
    sf.write(buffer, audio, rate, format=container.sf_format, subtype=container.sf_subtype)
    return buffer.getvalue()


#: Size fields of a WAV whose length is not yet known. 0xFFFFFFFF rather than 0 because it is
#: what the common streaming servers send and what players are tolerant of: a decoder that
#: trusts it simply reads until the connection ends.
_UNKNOWN_SIZE = 0xFFFFFFFF


def streaming_wav_header(sample_rate: int) -> bytes:
    """A 44-byte RIFF header with placeholder sizes, to put in front of a PCM stream.

    Describes the mono 16-bit samples :func:`to_pcm_bytes` produces, which is the only shape
    anything here writes.

    A WAV file states its own length in two places, and neither is known until the last sample
    has been generated. Writing ``0xFFFFFFFF`` into both lets the header go out first and the
    samples follow as they are produced.

    The result is a valid *stream*, not a well-formed file: a player fed it live reads until the
    stream ends, but a file on disk carrying these sizes will have its duration misreported and
    may not seek. When the length is known -- which it is whenever the audio is already in hand
    -- use :func:`encode_audio` or :func:`to_wav_bytes` instead and get real sizes.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}.")
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        _UNKNOWN_SIZE,
        b"WAVE",
        b"fmt ",
        16,  # size of the PCM fmt chunk
        1,  # WAVE_FORMAT_PCM
        1,  # channels
        sample_rate,
        sample_rate * 2,  # bytes per second, at 2 bytes per mono frame
        2,  # block align
        16,  # bits per sample
        b"data",
        _UNKNOWN_SIZE,
    )


#: Layer III bitrates in kbps, indexed by the header's 4-bit field. Keyed by MPEG generation:
#: 1 for MPEG-1, 2 for MPEG-2 and MPEG-2.5, which share a table.
_MPEG_BITRATES = {
    1: (0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 0),
    2: (0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160, 0),
}

#: Sample rates by the header's 2-bit version field, then its 2-bit rate index.
_MPEG_RATES = {3: (44100, 48000, 32000), 2: (22050, 24000, 16000), 0: (11025, 12000, 8000)}


def _mpeg_frame_length(data: bytes) -> int | None:
    """Byte length of the MPEG layer III frame `data` starts with, or ``None`` if it is not one.

    Only enough of the header is decoded to measure the frame: sync word, generation, layer,
    bitrate, sample rate and the padding bit. Anything that does not parse as layer III is left
    alone, which is the safe answer -- the one caller uses this to decide whether to *drop*
    bytes, and dropping real audio would be worse than keeping a redundant frame.
    """
    if len(data) < 4 or data[0] != 0xFF or (data[1] & 0xE0) != 0xE0:
        return None
    version = (data[1] >> 3) & 0b11  # 3 = MPEG-1, 2 = MPEG-2, 0 = MPEG-2.5, 1 = reserved
    layer = (data[1] >> 1) & 0b11  # 1 = layer III
    bitrate_index = data[2] >> 4
    rate_index = (data[2] >> 2) & 0b11
    if layer != 1 or version == 1 or rate_index == 3 or bitrate_index in (0, 15):
        return None
    bitrate = _MPEG_BITRATES[1 if version == 3 else 2][bitrate_index] * 1000
    samples = 1152 if version == 3 else 576
    return samples // 8 * bitrate // _MPEG_RATES[version][rate_index] + ((data[2] >> 1) & 1)


class StreamingEncoder:
    """Encode a growing waveform into `fmt`, returning bytes as soon as the codec produces them.

    Feed audio in with :meth:`encode` as it is decoded and send whatever comes back; call
    :meth:`finish` once to get the trailer. Concatenating everything yields a file a decoder
    reads back as the same audio :func:`encode_audio` would have produced from the same input.

    An empty return from the first call or two is normal: a frame-based codec cannot emit
    anything until it has a whole frame. What this never does is hold the utterance.

    Two of the formats never touch a codec at all. ``pcm`` is the samples themselves, and
    ``wav`` is :func:`streaming_wav_header` once followed by those same samples -- the header
    has to go out before the length is known, which is what the placeholder sizes are for. Both
    add zero latency. Their samples come from :func:`to_pcm_bytes`, which rounds where
    libsndfile truncates, so a streamed ``wav`` can differ from :func:`encode_audio`'s by one
    least significant bit, at -90 dBFS.

    ``opus`` is the one format whose bytes do not keep up with the audio: libsndfile emits an
    Ogg page only once the page has filled, which at speech bitrates is about a second, so bytes
    arrive in second-apart bursts however finely they are fed in. That is well outside
    :data:`STREAMING_LATENCY_BUDGET_MS`, which is why ``opus`` is absent from
    :data:`STREAMING_FORMATS`. Encoding it incrementally still works and still never holds the
    whole utterance, but a latency-sensitive caller should render it with :func:`encode_audio`
    or choose a format that streams.

    **What the caller receives is a stream, not a saved file.** Both MP3 and FLAC keep a field
    at the *front* that is only correct once the length is known -- the MP3 Xing/LAME frame, the
    FLAC ``STREAMINFO`` sample count and MD5 -- and those bytes have long since been sent. FLAC
    states "length unknown", which the format allows and stream decoders handle. MP3 is worse:
    its placeholder frame decodes as 36 ms of leading silence, so it is dropped here
    (see :func:`_mpeg_frame_length`) and the stream begins at the first real frame. What remains
    is the codec's own priming delay, 1105 samples -- 23 ms at 48 kHz -- which is inherent to
    MP3 without gapless metadata. Every encoded frame after that is byte-identical to what
    :func:`encode_audio` produces.

    Args:
        fmt: One of :data:`SUPPORTED_FORMATS`.
        sample_rate: Rate of the audio that will be handed to :meth:`encode`. Audio is
            resampled on the way in when the codec does not accept that rate; :attr:`rate` is
            what actually goes into the container.
    """

    def __init__(self, fmt: str, sample_rate: int) -> None:
        self._container = _container(fmt)
        self.fmt = str(fmt).strip().lower()
        self.sample_rate = int(sample_rate)
        #: Rate written into the container, after any snapping. See :func:`container_rate`.
        self.rate = container_rate(self.fmt, self.sample_rate)
        self.media_type = self._container.media_type

        self._resampler = StreamingResampler(self.sample_rate, self.rate)
        self._buffer: io.BytesIO | None = None
        self._file: sf.SoundFile | None = None
        self._sent = 0
        self._started = False
        self._finished = False
        #: MP3 only: hold the first frame back until it can be identified and dropped.
        self._held = b""
        self._dropping = self.fmt == "mp3"

    def encode(self, wav: np.ndarray) -> bytes:
        """Encode a chunk and return the bytes that became available. May be empty."""
        if self._finished:
            raise RuntimeError("This encoder is finished; build a new one for the next stream.")
        return self._emit(self._resampler.process(wav))

    def finish(self) -> bytes:
        """Close the container and return the last bytes, including any trailer. May be empty."""
        if self._finished:
            return b""
        out = self._emit(self._resampler.flush())
        self._finished = True
        if self._file is not None:
            # close() patches the fields at the front of the file, which have already been sent.
            # It is still what writes the container's own trailer, so it has to happen.
            self._file.close()
            self._file = None
            out += self._filter(self._drain())
        # Whatever is still held back was never long enough to identify; send it as it is
        # rather than swallowing the tail of a very short utterance.
        held, self._held, self._dropping = self._held, b"", False
        return out + held

    # ------------------------------------------------------------------ internals

    def _emit(self, audio: np.ndarray) -> bytes:
        """Push resampled audio into the container and return whatever new bytes appeared."""
        prefix = b""
        if not self._started:
            self._started = True
            prefix = self._open()
        if self._file is None:
            return prefix + to_pcm_bytes(audio)
        if audio.size:
            self._file.write(audio)
            # Without the flush, libsndfile keeps the encoded bytes in its own buffer and
            # nothing reaches the caller until close(): the whole-utterance hold this class
            # exists to avoid.
            self._file.flush()
        return prefix + self._filter(self._drain())

    def _filter(self, data: bytes) -> bytes:
        """Drop MP3's leading placeholder frame, once enough bytes exist to recognise it.

        libsndfile reserves a frame at the head of an MP3 and fills in the Xing/LAME tag at
        close, by which time the placeholder -- an all-zero frame, which decodes as 36 ms of
        silence -- has already been sent. It is dropped here on two conditions, both required:
        the bytes parse as a layer III frame header, and its payload is entirely zero. A real
        audio frame fails the second, so the worst case of a wrong guess is a moment of leading
        digital silence surviving, not audio going missing.
        """
        if not self._dropping:
            return data
        self._held += data
        length = _mpeg_frame_length(self._held)
        if length is None and len(self._held) >= 4:
            self._dropping = False
            return self._take_held()
        if length is None or len(self._held) < length:
            return b""
        self._dropping = False
        held = self._take_held()
        return held[length:] if not any(held[4:length]) else held

    def _take_held(self) -> bytes:
        held, self._held = self._held, b""
        return held

    def _open(self) -> bytes:
        """Start the container, returning any bytes that precede the first sample."""
        if self.fmt == "pcm":
            return b""
        if self.fmt == "wav":
            # Written by hand rather than by libsndfile: libsndfile patches its RIFF sizes at
            # close(), which is far too late for bytes that went out on the first chunk.
            return streaming_wav_header(self.rate)
        self._buffer = io.BytesIO()
        self._file = sf.SoundFile(
            self._buffer,
            mode="w",
            samplerate=self.rate,
            channels=1,
            format=self._container.sf_format,
            subtype=self._container.sf_subtype,
        )
        return b""

    def _drain(self) -> bytes:
        """Bytes libsndfile has written into the buffer since the last call."""
        if self._buffer is None:
            return b""
        view = self._buffer.getbuffer()
        try:
            data = bytes(view[self._sent :])
        finally:
            # A live export pins the BytesIO against resizing, so the next write would fail.
            view.release()
        self._sent += len(data)
        return data


def save_wav(
    path: str | os.PathLike[str],
    wav: np.ndarray,
    sample_rate: int = OUTPUT_SAMPLE_RATE,
) -> Path:
    """Write a 16-bit WAV, creating the parent directory. Returns the path written."""
    file = Path(path).expanduser()
    file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file), as_waveform(wav), sample_rate, format="WAV", subtype="PCM_16")
    return file
