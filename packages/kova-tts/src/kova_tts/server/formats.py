"""Which container ``POST /v1/audio/speech`` answers with, and how it is produced.

Everything about *making* audio bytes lives in :mod:`kova_tts.audio` -- the container table, the
media types, the rates each codec accepts, the one-shot encoder and the incremental one. This
module is the policy on top of it: which of OpenAI's six format names this endpoint honours,
what a request that names none should get, and whether a given response is streamed or encoded
in one pass. It writes no bytes itself.

Two ways to produce the same container, and the difference is worth stating:

* **Streamed**, through :class:`kova_tts.audio.StreamingEncoder`: bytes leave as the codec
  decodes, so audio starts while the utterance is still being generated. What arrives is a
  valid *stream* -- containers that state their own length state "unknown", because at the time
  the header goes out nobody knows it.
* **Whole**, through :func:`kova_tts.audio.encode_audio`: the utterance is generated first and
  encoded in one pass, so sizes, duration metadata and seek tables are exact. What arrives is a
  well-formed *file*, and its first byte arrives no sooner than the last frame was decoded.

Streaming is the default, because a head start is the reason this server exists; ``stream:
false`` asks for the file instead. Either way the samples are the same audio.

``aac`` is the one name in OpenAI's set with no route here: every encoder for it is ffmpeg or a
licensed library, and a local TTS server should not need either to say a sentence.
"""

from __future__ import annotations

import numpy as np

from kova_codec.constants import OUTPUT_SAMPLE_RATE
from kova_tts import audio
from kova_tts.server.errors import InvalidRequest

#: OpenAI's default when a request omits ``response_format``, and this endpoint's too.
OPENAI_DEFAULT_FORMAT = "mp3"

#: Used when the default cannot be encoded here. Never unavailable: wav needs no codec.
FALLBACK_FORMAT = "wav"

_AAC_MESSAGE = (
    "aac needs an external encoder (ffmpeg or a licensed AAC library) that this server "
    "deliberately does not depend on"
)


def available() -> tuple[str, ...]:
    """Every ``response_format`` this installation can answer with."""
    return audio.SUPPORTED_FORMATS


def default_format() -> str:
    """The container for a request that named none.

    OpenAI's default is mp3 and its clients expect mp3 back, so that is what they get wherever
    it can be produced. Where it cannot -- a soundfile wheel older than libsndfile 1.1 -- the
    fallback is wav rather than an error: a request that named no format has nothing wrong with
    it, and correctly labelled ``audio/wav`` beats a 4xx for a field the client never sent. An
    *explicit* ``"response_format": "mp3"`` on such an installation is still refused, because
    there the client did say what it wanted.
    """
    return OPENAI_DEFAULT_FORMAT if OPENAI_DEFAULT_FORMAT in available() else FALLBACK_FORMAT


def resolve(name: str | None) -> str:
    """The container for a request's ``response_format``. ``None`` means the field was absent.

    Raises:
        InvalidRequest: for a name this server does not know, or one this installation cannot
            produce. Either way the message names what *is* available right now, because the
            client's next move is to pick one of them.
    """
    if name is None:
        return default_format()
    key = name.strip().lower()
    if key == "aac":
        raise InvalidRequest(f"{_AAC_MESSAGE}; this server can return: {', '.join(available())}")
    try:
        audio.content_type(key)
    except ValueError as exc:
        raise InvalidRequest(str(exc)) from exc
    return key


def media_type(fmt: str) -> str:
    """``Content-Type`` for `fmt`, from the module that writes the bytes."""
    return audio.content_type(fmt)


def output_rate(fmt: str, requested: int | None, model_rate: int = OUTPUT_SAMPLE_RATE) -> int:
    """The rate the response will really carry.

    Not always the rate that was asked for: MP3 and Opus are each defined for a fixed set of
    rates, and :func:`kova_tts.audio.container_rate` snaps anything else up to the nearest one
    they accept. The container records the rate it was written at, so the audio is correct
    either way -- but ``X-Sample-Rate`` has to say what actually happened, and a client
    reading raw pcm has nothing else to go on.

    Nothing requested means the model's own rate, `model_rate`.
    """
    return audio.container_rate(fmt, requested or model_rate)


def streamable(fmt: str, wanted: bool) -> bool:
    """Whether this response is sent as it is decoded, rather than encoded once and sent.

    Every container this server produces can be encoded incrementally, so the only reason not
    to is a client that asked for a file with ``stream: false``.

    `fmt` is taken and not consulted, and that is the policy: no container is exempt here. The
    marginal one is ``opus``, whose bytes arrive in bursts of roughly a second because
    libsndfile releases an Ogg page only once the page is full -- outside
    :data:`kova_tts.audio.STREAMING_LATENCY_BUDGET_MS`, so it is absent from
    :data:`kova_tts.audio.STREAMING_FORMATS` and a WebSocket session refuses it. Here it is
    streamed anyway: bursts still beat holding a long utterance to the end, and for a short one
    the two are the same.
    """
    return wanted


class ResampledEncoder:
    """A :class:`kova_tts.audio.StreamingEncoder` fed at a rate the model does not produce.

    Both halves come from the audio layer -- :class:`~kova_tts.audio.StreamingResampler` for the
    filter that has to carry state across frames, the encoder for the container. This only
    holds them together, because the alternative is an endpoint that quietly ignores
    ``sample_rate`` on the path where audio actually streams.
    """

    def __init__(self, encoder: audio.StreamingEncoder, orig_rate: int, target_rate: int) -> None:
        self.encoder = encoder
        self.rate = encoder.rate
        self._resampler = audio.StreamingResampler(orig_rate, target_rate)

    def encode(self, wav: np.ndarray) -> bytes:
        return self.encoder.encode(self._resampler.process(wav))

    def finish(self) -> bytes:
        tail = self.encoder.encode(self._resampler.flush())
        return tail + self.encoder.finish()


def stream_encoder(fmt: str, sample_rate: int, model_rate: int = OUTPUT_SAMPLE_RATE) -> object:
    """An incremental encoder for `fmt`, taking the model's frames and emitting `sample_rate`.

    `sample_rate` has already been through :func:`output_rate`, so the container will not snap
    it again and the audio is resampled exactly once, here, on the way in.

    Built before the response starts rather than inside it, so anything it objects to is a
    status code instead of a body that stops halfway.

    Raises:
        InvalidRequest: when this installation cannot stream that container at that rate.
    """
    try:
        encoder = audio.StreamingEncoder(fmt, sample_rate)
    except (ValueError, RuntimeError) as exc:
        raise InvalidRequest(f"cannot stream {fmt} at {sample_rate} Hz: {exc}") from exc
    if sample_rate == model_rate:
        return encoder
    return ResampledEncoder(encoder, model_rate, sample_rate)


def encode_whole(wav: np.ndarray, sample_rate: int, fmt: str) -> bytes:
    """A finished utterance as a complete `fmt` file.

    Raises:
        InvalidRequest: when this installation cannot encode that container at that rate.
    """
    try:
        return audio.encode_audio(wav, sample_rate, fmt)
    except (ValueError, RuntimeError) as exc:
        raise InvalidRequest(f"cannot encode {fmt} at {sample_rate} Hz: {exc}") from exc
