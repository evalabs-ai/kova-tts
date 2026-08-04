"""Every shape that crosses the wire, in one file.

Requests are strict -- ``extra="forbid"`` -- so a misspelled field is a 422 naming the field
rather than a silently ignored setting. That matters more here than compatibility: this is a
local server, the client is usually a script someone just wrote, and a typo that changes
nothing about the audio is a bad half-hour. The one exception is :class:`SpeechRequest`, whose
shape belongs to somebody else's API and grows without asking; it says why in its own docstring.

WebSocket frames are a union discriminated by key: each frame is a JSON object whose *key* says
what it is (``{"send_text": "..."}``), which reads well in a log and needs no version field. One
connection drives one stream, so no frame carries a session or request id.

``POST /v1/tts/stream`` sends **raw 16-bit little-endian PCM, mono, at 32 kHz**, the codec's own
output rate, base64-encoded in each event. The WebSocket session chooses: its
:class:`ResponseFormat` names a container and a rate, and the session says in
``context_started`` which ones it settled on.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from kova_codec.constants import SAMPLE_RATE
from kova_tts.audio import STREAMING_FORMATS, content_type
from kova_tts.engine.types import SamplingParams

#: Longest text accepted in one request. Not a model limit -- a guard rail: generation beats
#: realtime by only a small factor, so a novel pasted into ``text`` is a multi-hour request that
#: looks like a hung server. Split it client-side and send it as several calls, or stream it
#: over the WebSocket, which is what the incremental session is for.
MAX_TEXT_CHARS = 5000

#: Bounds on every ``sample_rate`` a client can ask for: the extension on
#: ``POST /v1/audio/speech`` and the WebSocket session's ``response_format``. The floor is
#: telephony and the ceiling is the highest rate any container this server writes will take; the
#: point of the range is to reject a typo'd rate here rather than three layers down in the
#: encoder.
MIN_OUTPUT_RATE = 8000
MAX_OUTPUT_RATE = 48000


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


def streamable_encodings() -> str:
    """The containers a session can deliver, for an error message to name."""
    return ", ".join(STREAMING_FORMATS)


class ResponseFormat(_WireModel):
    """What a session's ``audio_chunk`` payloads are: a container, and the rate inside it.

    Both fields are honoured, and ``context_started`` echoes this model back holding the values
    actually in effect -- including a rate a container had to snap -- so a client reads what its
    bytes are rather than inferring them.

    Fixed for the life of the session, because the chunks are one continuous stream of bytes and
    a container cannot change its mind halfway through a file.
    """

    encoding: str = Field(
        default="pcm",
        description="Container for the audio_chunk payloads. pcm is headerless 16-bit "
        "little-endian samples; the rest are files, streamed from their first byte.",
    )
    sample_rate: int = Field(
        default=SAMPLE_RATE,
        ge=MIN_OUTPUT_RATE,
        le=MAX_OUTPUT_RATE,
        description="Rate to deliver at; the codec's own 32000 by default. Another rate is "
        "converted with one filter whose state runs through the whole session, so no chunk "
        "join and no flush boundary carries a step.",
    )

    @field_validator("encoding")
    @classmethod
    def _one_a_session_can_stream(cls, value: str) -> str:
        """Accept the containers whose bytes keep up with a live session.

        The set is :data:`kova_tts.audio.STREAMING_FORMATS` rather than a list written out
        here, so it follows what this install can actually produce: a soundfile wheel carrying a
        libsndfile older than the 1.1 release that added MPEG has no mp3 to offer, and saying so
        beats a traceback from inside the C library.
        """
        name = str(value).strip().lower()
        try:
            # A name that is not a container at all, or one this install cannot write, already
            # has a precise message in the audio layer. That message names every format a
            # *whole file* can be encoded in, which is a wider set than a session can stream.
            content_type(name)
        except ValueError as exc:
            raise ValueError(
                f"{exc} Of those, a session streams: {streamable_encodings()}."
            ) from exc
        if name not in STREAMING_FORMATS:
            raise ValueError(
                f"encoding {name!r} is not streamed over a session: libsndfile releases an Ogg "
                f"page only once the page has filled, which at speech bitrates is about a "
                f"second of audio, so a flush's last words would still be inside the encoder "
                f"when its flush_completed went out. A session streams: "
                f"{streamable_encodings()}. For {name}, POST /v1/audio/speech."
            )
        return name


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


# ------------------------------------------------------------------ OpenAI-compatible requests


class SpeechRequest(BaseModel):
    """``POST /v1/audio/speech``: OpenAI's audio-speech body, as its clients send it.

    The one request in this file that is **not** strict. Unknown fields are ignored rather than
    refused, because this shape is not ours: OpenAI adds fields to it -- ``instructions`` and
    ``stream_format`` are two recent ones -- and every client that follows sends them to every
    server it is pointed at. Refusing a field this server has never heard
    of would break working clients on somebody else's release schedule. The trade is real and
    worth naming: a misspelled ``respones_format`` here is silently the default, where on
    ``/v1/tts`` it would be a 422. Use ``/v1/tts`` when you want the strict door.

    Fields this model *does* declare are checked, including the two the model cannot honour:
    a request that asks for something this server would have to fake is refused, not quietly
    served with audio it did not ask for.
    """

    model_config = ConfigDict(extra="ignore")

    model: str | None = Field(
        default=None,
        description="Accepted and ignored: this server serves exactly one model. See "
        "GET /v1/models for the name it reports.",
    )
    input: str = Field(max_length=MAX_TEXT_CHARS, description="The text to speak.")
    voice: str | None = Field(
        default=None,
        description='A LoRA voice from GET /v1/voices. Omit it, or send "default", for the '
        "base voice. OpenAI's own voice names (alloy, echo, ...) do not exist here.",
    )
    response_format: str | None = Field(
        default=None,
        description="mp3, opus, flac, wav or pcm. Absent means mp3, as with OpenAI.",
    )
    speed: float = Field(default=1.0, description="Only 1.0; this model has no speed control.")
    instructions: str | None = Field(
        default=None,
        description="Only empty; this model takes no style instructions.",
    )
    stream_format: Literal["audio", "sse"] = Field(
        default="audio",
        description="audio streams the container's bytes; sse wraps them in OpenAI's "
        "speech.audio.delta events.",
    )
    sample_rate: int | None = Field(
        default=None,
        ge=MIN_OUTPUT_RATE,
        le=MAX_OUTPUT_RATE,
        description="Not part of OpenAI's schema: an extension for pipelines fixed at another "
        "rate (16 kHz agents, 8 kHz telephony). Absent means the model's own 32 kHz. Streams "
        "like any other response; the resampler carries its filter state across frames.",
    )
    stream: bool = Field(
        default=True,
        description="Not part of OpenAI's schema: false encodes the finished utterance in one "
        "pass, for a file with exact sizes and duration metadata. True, the default, starts "
        "sending as the codec decodes -- a valid stream, and audio that starts early.",
    )

    @field_validator("voice", mode="before")
    @classmethod
    def _unwrap_voice_object(cls, value: Any) -> Any:
        """Accept ``{"id": "..."}`` as well as a bare name.

        Recent OpenAI clients allow a custom-voice object there, so a client that sends one is
        following its own documentation; taking the id out of it costs a line and saves a
        confusing 422 about a string that was never going to be a string.
        """
        if isinstance(value, dict) and isinstance(value.get("id"), str):
            return value["id"]
        return value

    @field_validator("input")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input is empty; send the words you want spoken")
        return value

    @field_validator("speed")
    @classmethod
    def _only_natural_speed(cls, value: float) -> float:
        # Not silently ignored: a client that asks for 1.5x and is handed 1.0x audio has no way
        # to tell, and the audio is wrong for as long as nobody listens closely.
        if abs(value - 1.0) > 1e-6:
            raise ValueError(
                f"this model has no speed control, so only speed=1.0 is accepted, not {value}; "
                f"change the tempo after the fact instead (ffmpeg's atempo filter, or your "
                f"player's playback rate)"
            )
        return value

    @field_validator("instructions")
    @classmethod
    def _no_instructions(cls, value: str | None) -> str | None:
        # Same reasoning as `speed`: gpt-4o-mini-tts steers delivery from this field, this
        # model has no such conditioning, and pretending otherwise hides that from the caller.
        if value is not None and value.strip():
            raise ValueError(
                "this model takes no style instructions; delivery follows the text and the "
                "voice. Pick a LoRA voice with `voice`, or punctuate the text to shape it"
            )
        return value


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


#: What ``GET /v1/models`` reports as a model's creation time: this project's first release,
#: fixed. Nothing here is versioned by date, but the field is required by every client that
#: deserializes OpenAI's model object, and some of them sort or display it -- where a zero shows
#: up as 1 January 1970.
MODEL_CREATED = 1751328000  # 2025-07-01, UTC


class ModelInfo(BaseModel):
    """One entry of ``GET /v1/models``, in OpenAI's shape."""

    id: str
    object: Literal["model"] = "model"
    created: int = MODEL_CREATED
    owned_by: str = "kova-tts"


class ModelListResponse(BaseModel):
    """``GET /v1/models``: the one model this server has, listed the way clients expect."""

    object: Literal["list"] = "list"
    data: list[ModelInfo]


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


class VoiceReference(_WireModel):
    """A recording for the session to speak as, and the words that are in it.

    Two ways to hand the clip over, for two kinds of client:

    * ``audio`` -- one base64 audio file. Self-contained: a single frame and the session can
      speak. The server decodes and encodes it, which is the only thing here that needs WavLM.
    * ``codes`` -- the same clip already encoded to codec codes. An agent that speaks as one
      voice across many sessions encodes it once and starts every session from these, which
      needs no WavLM in the server at all and no encode on the way in.

    Either way ``transcript`` is required and must be what the clip actually says, word for
    word. Cloning is a *continuation*: the transcript goes in front of the text you send and the
    clip's codes lead the continuation, so the model reads the transcript as words it has
    already spoken and carries on from them. A transcript that does not match leaves the model
    speaking words the codes do not contain, and the output garbles.
    """

    transcript: str = Field(
        default="",
        description="Exactly what the reference says, word for word. Required.",
    )
    audio: str | None = Field(
        default=None,
        description="Base64 of an audio file -- anything soundfile reads. Resampled to mono "
        "32 kHz and loudness-normalised here, so send the recording as you have it.",
    )
    codes: list[int] | None = Field(
        default=None,
        description="The clip already encoded, 80 codes per second, as Voice.ref_codes holds "
        "them. Sent instead of audio by a client that encoded the reference once.",
    )

    @model_validator(mode="after")
    def _one_clip_and_its_transcript(self) -> VoiceReference:
        """Exactly one form of the clip, and the transcript that goes with it.

        Both checks produce their own message rather than a schema error, because both are
        things a client gets wrong for a reason: the two forms look interchangeable until you
        know one needs WavLM, and a transcript looks optional until you know there is no speech
        recognition here to fall back on.
        """
        if (self.audio is None) == (self.codes is None):
            raise ValueError(
                "a reference is either `audio` (base64 of a recording) or `codes` (that "
                "recording already encoded), and this one is "
                + ("both" if self.audio is not None else "neither")
            )
        if not self.transcript.strip():
            raise ValueError(
                "a reference needs its `transcript`: cloning continues the recording, so the "
                "model reads the transcript as words it has already spoken and carries on from "
                "them. There is no speech recognition in this server -- send what the clip "
                "says, word for word"
            )
        return self


class StartConfig(_WireModel):
    """Settings for the session, fixed for its lifetime.

    A session speaks in one voice, with one sampler, into one container: everything here is
    chosen once because the audio it produces is one continuous utterance.
    """

    voice: str | None = Field(
        default=None,
        description="A LoRA voice name from GET /v1/voices. Omit for the base model's voice.",
    )
    reference: VoiceReference | None = Field(
        default=None,
        description="Speak as the voice in a recording, with no adapter and no training. "
        "Validated when it arrives, so an unusable clip is refused before any text is sent.",
    )
    sampling: SamplingOverrides | None = None
    seed: int | None = None
    response_format: ResponseFormat = Field(default_factory=ResponseFormat)

    @model_validator(mode="after")
    def _one_voice_at_a_time(self) -> StartConfig:
        # Both fields answer "who speaks", and a session speaks as one voice for its whole life.
        # Honouring one and ignoring the other would be a session that sounds like nothing the
        # client asked for, with nothing in the echo to explain it.
        if self.voice is not None and self.reference is not None:
            raise ValueError(
                "`voice` and `reference` are two ways of saying who speaks, so send one of "
                "them: a name from GET /v1/voices, or a recording to clone"
            )
        return self


class ReferenceInEffect(_WireModel):
    """What ``context_started`` reports about a cloned reference: that there is one, and its size.

    The clip does not come back. A client that has just sent thirty seconds of base64 has no use
    for it echoed, and these two numbers are what it cannot work out for itself -- how much of
    the clip survived, in the units the prompt is actually built from.
    """

    seconds: float = Field(description="Duration of the reference the session will speak from.")
    codes: int = Field(description="Codes it encoded to; 80 per second.")


class StartContext(_WireModel):
    start_context: StartConfig


class SendText(_WireModel):
    """Text to append to the session's buffer, as often as the client has more of it.

    The session speaks into the buffer as it fills, so a flush ends a turn rather than starting
    one. Nothing here says when the audio for a given piece of text arrives.
    """

    send_text: str


class Flush(_WireModel):
    """Finish the turn -- speak whatever text is left -- and keep the session open."""

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


class StartedConfig(StartConfig):
    """The accepted configuration, as the session will really run it.

    Same settings the client sent, with one field narrowed: a reference comes back as the
    :class:`ReferenceInEffect` description of the clip rather than the clip.
    """

    reference: ReferenceInEffect | None = None


class ContextStarted(_WireModel):
    """Echo of the accepted configuration, with defaults filled in.

    ``response_format`` holds what the session will really deliver, so a client can set up its
    decoder from this frame alone, and ``reference`` says a cloned voice is in effect.
    """

    context_started: StartedConfig


class AudioChunk(_WireModel):
    """The next bytes of the session's audio stream, base64-encoded.

    Concatenating every chunk of a session, in the order they arrive, gives one file in
    ``response_format.encoding`` at ``response_format.sample_rate``. A chunk is a slice of that
    stream and nothing more: it is not separately decodable, and the boundaries between chunks
    are wherever the container happened to hand bytes over.
    """

    audio_chunk: str


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
