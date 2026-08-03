"""Every shape that crosses the wire, in one file.

Requests are strict -- ``extra="forbid"`` -- so a misspelled field is a 422 naming the field
rather than a silently ignored setting. That matters more here than compatibility: this is a
local server, the client is usually a script someone just wrote, and a typo that changes
nothing about the audio is a bad half-hour.

WebSocket frames are a union discriminated by key: each frame is a JSON object whose *key* says
what it is (``{"send_text": "..."}``), which reads well in a log and needs no version field. One
connection drives one stream, so no frame carries a session or request id.

Audio on the streaming paths is always **raw 16-bit little-endian PCM, mono, at 32 kHz**, the
codec's own output rate, base64-encoded. The server does not resample and does not re-encode;
ask ``POST /v1/tts`` for a container if you want one.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from kova_codec.constants import SAMPLE_RATE
from kova_tts.engine.types import SamplingParams

#: Longest text accepted in one request. Not a model limit -- a guard rail: the generator beats
#: realtime by only a small factor (~2.6x on one consumer GPU when this was measured), so a
#: novel pasted into ``text`` is a multi-hour request that looks like a hung server. Split it
#: client-side and send it as several calls, or stream it over the WebSocket, which is what the
#: incremental session is for.
MAX_TEXT_CHARS = 5000


class _WireModel(BaseModel):
    """Strict base: reject unknown fields, so a typo is reported instead of ignored."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------------- shared bits


class SamplingOverrides(_WireModel):
    """Per-request sampling knobs. Anything left unset keeps the tuned preset.

    Only send what you actually want changed: the presets in :mod:`kova_tts.engine.types` are
    the values this checkpoint was tuned with, and the model is sensitive to them.
    """

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    repetition_penalty: float | None = None
    max_tokens: int | None = None

    def apply(self, base: SamplingParams) -> SamplingParams:
        """`base` with the fields that were sent replaced.

        Validation lives in :class:`~kova_tts.engine.types.SamplingParams`, which raises
        ``ValueError`` with a message naming the offending value; the caller turns that into a
        422 rather than duplicating the bounds here and letting the two drift apart.
        """
        overrides = {name: value for name, value in self.model_dump().items() if value is not None}
        return base.replace(**overrides) if overrides else base


class ResponseFormat(_WireModel):
    """Audio format for a streaming session. Each field has exactly one legal value.

    Stated rather than assumed so a client can declare the format it is prepared to decode and
    be told plainly when that is not what this server sends, instead of misreading the bytes.
    """

    encoding: Literal["pcm"] = "pcm"
    sample_rate: int = SAMPLE_RATE

    @field_validator("sample_rate")
    @classmethod
    def _only_the_codec_rate(cls, value: int) -> int:
        if value != SAMPLE_RATE:
            raise ValueError(
                f"this server streams the codec's own {SAMPLE_RATE} Hz output and does not "
                f"resample, so sample_rate must be {SAMPLE_RATE}, not {value}; resample the "
                f"PCM yourself, or POST /v1/tts for a wav file"
            )
        return value


# ------------------------------------------------------------------------------- HTTP requests


class SynthesisRequest(_WireModel):
    """What to say, and how. Shared by ``/v1/tts`` and ``/v1/tts/stream``."""

    text: str = Field(max_length=MAX_TEXT_CHARS)
    voice: str | None = Field(
        default=None,
        description="A LoRA voice name from GET /v1/voices. Omit for the base model's voice.",
    )
    sampling: SamplingOverrides | None = None
    seed: int | None = Field(
        default=None,
        description="Fixes the sampler, so the same request returns the same audio.",
    )

    @field_validator("text")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text is empty; send the words you want spoken")
        return value


class TtsRequest(SynthesisRequest):
    """``POST /v1/tts``: the whole utterance, in one response body."""

    response_format: Literal["wav", "pcm"] = Field(
        default="wav",
        description="wav is a complete 16-bit file; pcm is headerless 16-bit little-endian.",
    )


# ------------------------------------------------------------------------------ HTTP responses


class HealthResponse(BaseModel):
    """``GET /health``."""

    status: Literal["ok"] = "ok"
    model_loaded: bool
    device: str
    sample_rate: int
    voices: int
    busy: bool
    version: str


class VoiceInfo(BaseModel):
    """One voice the server can speak as."""

    name: str
    kind: Literal["lora"] = "lora"


class VoicesResponse(BaseModel):
    """``GET /v1/voices``.

    ``lora_dir`` is echoed because the commonest voice problem is a directory pointed at the
    wrong place, and an empty list with no explanation is a poor way to find that out.
    """

    voices: list[VoiceInfo]
    lora_dir: str | None = None


class ErrorResponse(BaseModel):
    """The body of every failure, whatever the status code."""

    error: str
    message: str


# ------------------------------------------------------------------- server-sent event payloads


class ChunkEvent(BaseModel):
    """``event: chunk`` -- one decoded frame, roughly 390 ms of speech."""

    index: int
    audio: str  # base64 of 16-bit little-endian PCM
    sample_rate: int = SAMPLE_RATE


class DoneEvent(BaseModel):
    """``event: done`` -- the terminal event of a successful stream."""

    chunks: int
    samples: int
    duration_seconds: float
    sample_rate: int = SAMPLE_RATE


class ErrorEvent(BaseModel):
    """``event: error`` -- the terminal event of a failed stream.

    Generation begins after the response headers are on the wire, so a failure partway through
    cannot change the status code; it arrives as this event instead. A stream always ends with
    exactly one terminal event, ``done`` or ``error``.
    """

    error: str
    message: str


# ------------------------------------------------------------- WebSocket frames: client -> server


class StartConfig(_WireModel):
    """Settings for the session, fixed for its lifetime."""

    voice: str | None = None
    sampling: SamplingOverrides | None = None
    seed: int | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)


class StartContext(_WireModel):
    start_context: StartConfig


class SendText(_WireModel):
    """Text to add to the buffer. Buffered text is not spoken until a flush."""

    send_text: str


class Flush(_WireModel):
    """Synthesize everything buffered so far and keep the session open."""

    flush: bool = True
    flush_id: str | None = None


class CloseContext(_WireModel):
    """Synthesize anything still buffered, then end the session."""

    close_context: bool = True
    flush_id: str | None = None


IncomingFrame = StartContext | SendText | Flush | CloseContext

#: The four keys a client frame may be discriminated by, in the order they are usually sent.
INCOMING_KEYS = ("start_context", "send_text", "flush", "close_context")


def parse_incoming(frame: Any) -> IncomingFrame:
    """Parse an inbound frame by its discriminator key.

    Raises ``ValueError`` for anything unrecognised, including a JSON value that is not an
    object at all -- the endpoint turns that into an ``error`` frame and keeps the socket open,
    since one bad frame is not a reason to drop a session mid-sentence.
    """
    if not isinstance(frame, dict):
        raise ValueError(
            f"a frame must be a JSON object keyed by its type, one of "
            f"{', '.join(INCOMING_KEYS)}; got {type(frame).__name__}"
        )
    if "start_context" in frame:
        return StartContext.model_validate(frame)
    if "send_text" in frame:
        return SendText.model_validate(frame)
    if "flush" in frame:
        return Flush.model_validate(frame)
    if "close_context" in frame:
        return CloseContext.model_validate(frame)
    keys = ", ".join(sorted(frame)) or "no keys at all"
    raise ValueError(f"unknown frame: expected one of {', '.join(INCOMING_KEYS)}; got {keys}")


# ------------------------------------------------------------- WebSocket frames: server -> client


class ContextStarted(_WireModel):
    """Echo of the accepted configuration, with defaults filled in."""

    context_started: StartConfig


class AudioChunk(_WireModel):
    audio_chunk: str  # base64 of 16-bit little-endian PCM at 32 kHz


class FlushCompleted(_WireModel):
    """All audio for one flush has been sent. Arrives even when the flush produced none."""

    flush_completed: bool = True
    flush_id: str


class ContextClosed(_WireModel):
    """The last frame of the session; the server closes the socket immediately after."""

    context_closed: bool = True


class Error(_WireModel):
    """Something went wrong. Carries ``flush_id`` when it can be blamed on one flush."""

    error: str
    flush_id: str | None = None


OutgoingFrame = ContextStarted | AudioChunk | FlushCompleted | ContextClosed | Error


def to_wire(frame: BaseModel) -> dict[str, Any]:
    """Render an outgoing frame as JSON-ready data.

    ``exclude_none`` keeps unset optionals off the wire entirely rather than sending nulls a
    client then has to filter.
    """
    return frame.model_dump(mode="json", exclude_none=True)
