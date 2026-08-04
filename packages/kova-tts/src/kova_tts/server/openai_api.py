"""OpenAI's ``/v1/audio/speech``, so a tool that already speaks that API needs no new code.

Open WebUI, SillyTavern, LibreChat, AnythingLLM and most local voice stacks can already talk to
OpenAI's audio-speech endpoint. Answering it here turns "integrate this TTS" into "change the
base URL", which is the entire reason this module exists. It adds nothing to the model and
takes nothing away from :mod:`kova_tts.server.routes`: ``/v1/tts`` and ``/v1/tts/stream`` are
this server's own API, they keep every knob this one lacks (sampling, seed, strict validation),
and neither surface deprecates the other.

**Compatibility beats elegance here, and the differences are deliberate:**

* ``model`` is accepted whatever it says. There is one model on this machine; refusing a name
  would only break clients that hardcode ``tts-1``. ``GET /v1/models`` reports the name this
  server calls it, for clients that populate a dropdown from there.
* ``voice`` names a LoRA voice from ``GET /v1/voices``. OpenAI's ``alloy``/``nova``/... do not
  exist here, but they are *accepted* and speak in the base voice: its schema allows nothing
  else, so a client's voice field arrives holding ``alloy`` whether anyone chose it or not, and
  a 404 there would fail the first request made after changing a base URL. An operator can point
  them at real voices with ``--voice-alias``. A name that is neither installed nor one of
  OpenAI's is a typo, and still gets the 404 that lists what this machine has.
* ``speed`` and ``instructions`` are refused unless they are the no-op value. This model has
  neither control, and serving audio that ignores them looks like success.
* ``response_format`` defaults to mp3, as OpenAI's does, and every container is sent as the
  codec decodes it. :mod:`kova_tts.server.formats` covers what streaming costs a container and
  what ``stream: false`` buys back.
* ``sample_rate`` and ``stream`` are extensions, not part of OpenAI's schema. A client that
  sends neither is unaffected by either.

The response is audio bytes with the matching ``Content-Type``, never a JSON envelope -- that
is what the clients read. Failures are still the server's usual envelope, since a client that
gets a 4xx is already off the audio path.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import AsyncIterator, Iterable
from contextlib import aclosing
from typing import Any

import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from kova_tts.audio import resample
from kova_tts.paths import MissingArtifact
from kova_tts.server import formats
from kova_tts.server.engine import Engine
from kova_tts.server.protocol import (
    ErrorResponse,
    ModelInfo,
    ModelListResponse,
    SpeechRequest,
)

log = logging.getLogger(__name__)

router = APIRouter()

#: What ``GET /v1/models`` reports and what a client may as well send back as ``model``. Any
#: other value is accepted too; this is the name, not a gate.
MODEL_ID = "kova-tts"

#: Voice names that mean "the base model's own voice", for clients whose UI insists on a
#: non-empty voice field. A LoRA voice that really is called ``default`` still wins: the
#: adapter directory is consulted first.
BASE_VOICE_ALIASES = frozenset({"default", "base"})

#: OpenAI's own voice names. Its schema accepts nothing else, so every SDK example, every
#: tutorial and the stock configuration of most clients sends one of these -- ``alloy`` above
#: all, which several of them hardcode. They are accepted here and resolved to the base voice,
#: because a 404 on a client's very first request is precisely the thing this endpoint exists to
#: prevent. An operator with LoRA voices can point them somewhere real; see :func:`parse_aliases`.
#:
#: Kept as one list because OpenAI keeps adding to it: this is the set as of the 2.x SDKs.
STOCK_VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "cedar",
        "coral",
        "echo",
        "fable",
        "marin",
        "nova",
        "onyx",
        "sage",
        "shimmer",
        "verse",
    }
)

#: Prefix for the per-voice entries of ``GET /v1/models``, so a client whose only dropdown is a
#: model dropdown can still choose a voice. ``kova-tts:my_voice`` names the same one model.
VOICE_MODEL_SEPARATOR = ":"

#: Environment variable read when no ``--voice-alias`` is given, for a deployment configured by
#: environment rather than by flags: ``KOVA_VOICE_ALIASES="alloy=my_voice,nova=another"``.
VOICE_ALIAS_ENV = "KOVA_VOICE_ALIASES"

#: What ``X-Voice`` says when the base model spoke, rather than any adapter.
BASE_VOICE_LABEL = "base"

#: Documented failure bodies, so the generated OpenAPI shows the error envelope.
_ERRORS = {
    404: {"model": ErrorResponse, "description": "No such voice."},
    409: {"model": ErrorResponse, "description": "A generation is already in flight."},
    422: {"model": ErrorResponse, "description": "The request could not be honoured."},
}


def _engine(request: Request) -> Engine:
    return request.app.state.engine


@router.get("/v1/models", response_model=ModelListResponse, tags=["openai"])
async def models(request: Request) -> ModelListResponse:
    """The one model this server has, plus one entry per installed voice.

    Clients that offer a model dropdown fill it from here, and some refuse to send a request at
    all until the call succeeds. Nothing in a request is *checked* against this list -- ``model``
    is accepted whatever it says -- but a ``kova-tts:<voice>`` id picks that voice, which is the
    only handle a client that exposes no voice field has.
    """
    engine = _engine(request)
    entries = [ModelInfo(id=MODEL_ID)]
    entries += [
        ModelInfo(id=f"{MODEL_ID}{VOICE_MODEL_SEPARATOR}{name}") for name in engine.voices()
    ]
    return ModelListResponse(data=entries)


# ------------------------------------------------------------------------------- voice aliases


def parse_aliases(entries: Iterable[str] | None) -> dict[str, str]:
    """``["alloy=my_voice"]`` -> ``{"alloy": "my_voice"}``.

    Accepts the repeatable ``--voice-alias`` flag and the comma-separated
    :data:`VOICE_ALIAS_ENV` variable in one function, because they mean the same thing and
    should not be able to disagree about what a valid entry looks like.

    Raises:
        ValueError: for an entry that is not ``name=voice``, naming the offending entry. A
            misconfigured alias is worth stopping for; it is silent otherwise.
    """
    aliases: dict[str, str] = {}
    for entry in entries or ():
        for item in str(entry).split(","):
            text = item.strip()
            if not text:
                continue
            name, separator, target = text.partition("=")
            if not separator or not name.strip() or not target.strip():
                raise ValueError(
                    f"voice alias {text!r} is not of the form name=voice, e.g. alloy=my_voice"
                )
            aliases[name.strip().lower()] = target.strip()
    return aliases


def aliases_from_environment() -> dict[str, str]:
    """The alias map from :data:`VOICE_ALIAS_ENV`, for a deployment configured by environment."""
    return parse_aliases([os.environ.get(VOICE_ALIAS_ENV, "")])


def validate_aliases(engine: Engine, aliases: dict[str, str]) -> None:
    """Fail at startup for an alias pointing at a voice that is not installed.

    At startup and not per request: an operator who mistypes a voice name should find out when
    the server refuses to boot, not when a user reports that one client is broken.

    Raises:
        ValueError: naming every bad mapping and what is installed.
    """
    installed = set(engine.voices())
    missing = {name: target for name, target in aliases.items() if target not in installed}
    if not missing:
        return
    listed = ", ".join(f"{name}={target}" for name, target in sorted(missing.items()))
    available = ", ".join(sorted(installed)) or "none installed"
    raise ValueError(
        f"voice alias points at a voice this machine does not have: {listed}. "
        f"Installed voices: {available}."
    )


@router.post(
    "/v1/audio/speech",
    tags=["openai"],
    responses={
        200: {
            "content": {
                "audio/mpeg": {},
                "audio/ogg": {},
                "audio/flac": {},
                "audio/wav": {},
                "application/octet-stream": {},
                "text/event-stream": {},
            },
            "description": (
                "Audio bytes in the requested container, sent as the codec decodes them "
                'unless `stream` is false. `stream_format: "sse"` wraps the same bytes in '
                "OpenAI's speech.audio.delta events."
            ),
        },
        **_ERRORS,
    },
)
async def create_speech(body: SpeechRequest, request: Request) -> Response:
    """Synthesize ``input`` and return it as audio bytes, OpenAI-style."""
    engine = _engine(request)
    fmt = formats.resolve(body.response_format)
    rate = formats.output_rate(fmt, body.sample_rate)
    # The encoder is built here rather than inside the response body, and the voice is resolved
    # before the model is reserved: anything either objects to should be a status code, and
    # once a streaming response has started 200 has already been sent.
    streaming = formats.streamable(fmt, body.stream)
    encoder = formats.stream_encoder(fmt, rate, engine.sample_rate) if streaming else None
    voice = _voice(engine, body.voice, body.model, _aliases(request))

    if body.stream_format == "sse":
        await engine.acquire()
        return StreamingResponse(
            _sse_body(engine, fmt, rate, encoder, body.input, voice),
            media_type="text/event-stream",
            headers=_headers(fmt, rate, voice, streaming=True) | {"Cache-Control": "no-cache"},
        )

    if encoder is not None:
        # No Content-Length: the length is not known when the headers go out, and claiming one
        # would be worse than omitting it. Starlette sends this chunked.
        await engine.acquire()
        return StreamingResponse(
            _audio_body(engine, encoder, body.input, voice),
            media_type=formats.media_type(fmt),
            headers=_headers(fmt, rate, voice, streaming=True),
        )

    # The whole-file path: generate, then encode in one pass. It is also the friendlier
    # failure -- a generation that dies here is still a status code, where a stream that dies
    # mid-body is a truncated file.
    async with engine.reserve():
        wav = await engine.generate(body.input, voice, params=None, seed=None)
    samples = np.asarray(wav, dtype=np.float32)
    payload = await asyncio.to_thread(_encode_whole, samples, fmt, rate, engine.sample_rate)
    headers = _headers(fmt, rate, voice) | {
        "X-Duration-Seconds": f"{samples.size / engine.sample_rate:.3f}",
    }
    return Response(payload, media_type=formats.media_type(fmt), headers=headers)


# ------------------------------------------------------------------------------ request pieces


def _aliases(request: Request) -> dict[str, str]:
    """The operator's stock-name mapping, empty unless one was configured."""
    return getattr(request.app.state, "voice_aliases", None) or {}


def _voice(
    engine: Engine,
    requested: str | None,
    model: str | None = None,
    aliases: dict[str, str] | None = None,
) -> str | None:
    """The voice to generate in: a LoRA name, or ``None`` for the base voice.

    Resolved in this order, most deliberate first:

    1. a voice installed on this machine, so a real adapter always wins -- including one that
       happens to be named ``alloy`` or ``default``;
    2. the voice named by a ``kova-tts:<voice>`` model id, for clients whose only dropdown is a
       model dropdown. Ahead of the aliases below because somebody picked it: the ``voice``
       field of such a client is whatever its config shipped with;
    3. an operator alias, which is how ``alloy`` is pointed at a real voice
       (:func:`parse_aliases`);
    4. one of OpenAI's stock names, or an empty/``default`` field: the base model's own voice.
       This is the case that makes a stock client work at all. Its voice field arrives holding
       ``alloy`` whether the user chose it or not, and refusing that would fail the very first
       request somebody makes after changing a base URL.

    Raises:
        MissingArtifact: 404 for a name that is none of those -- a typo -- with the voices this
            machine actually has. That error is right for a typo and wrong for ``alloy``, which
            is why case 4 above catches ``alloy`` before it reaches here.
    """
    name = (requested or "").strip()
    known = _installed(engine, name)
    if known is not None:
        return known
    if name and not _is_generic(name, aliases):
        installed = ", ".join(engine.voices()) or "none installed"
        raise MissingArtifact(
            f"no voice named {name!r} on this server. Here `voice` names one of this "
            f"machine's LoRA voices (GET /v1/voices), not an OpenAI voice like 'alloy' "
            f"-- those are accepted and speak in the base voice. "
            f'Available: {installed}. Omit voice, or send "default", for the base voice.'
        ) from None
    selected = _model_voice(engine, model)
    if selected is not None:
        return selected
    mapped = (aliases or {}).get(name.lower()) if name else None
    if mapped is not None:
        # Validated at startup; re-resolved here so an adapter directory that changed under a
        # running server is a clear 404 rather than a failure three layers down.
        engine.check_voice(mapped)
        return mapped
    return None


def _is_generic(name: str, aliases: dict[str, str] | None) -> bool:
    """Is `name` something this server knows how to interpret, rather than a typo?

    True for OpenAI's stock names, for the base-voice words, and for anything an operator has
    mapped. Everything else that is not installed is a mistake worth reporting.
    """
    key = name.lower()
    return key in STOCK_VOICES or key in BASE_VOICE_ALIASES or key in (aliases or {})


def _installed(engine: Engine, name: str) -> str | None:
    """`name` if this machine really has that voice, else ``None``."""
    if not name:
        return None
    try:
        engine.check_voice(name)
    except MissingArtifact:
        return None
    return name


def _model_voice(engine: Engine, model: str | None) -> str | None:
    """The voice named by a ``kova-tts:<voice>`` model id, if it names one that exists.

    Anything else is ignored rather than refused: ``model`` is accepted whatever it says, and a
    client hardcoding ``tts-1`` must keep working.
    """
    if not model or VOICE_MODEL_SEPARATOR not in model:
        return None
    head, _, tail = model.partition(VOICE_MODEL_SEPARATOR)
    if head.strip() != MODEL_ID:
        return None
    return _installed(engine, tail.strip())


def voice_label(voice: str | None) -> str:
    """What ``X-Voice`` reports: the adapter that spoke, or that the base model did."""
    return voice or BASE_VOICE_LABEL


def _headers(fmt: str, rate: int, voice: str | None, *, streaming: bool = False) -> dict[str, str]:
    """Response headers.

    ``X-Sample-Rate`` matters most for pcm, which states it nowhere else. ``X-Voice`` matters
    because ``alloy`` now resolves to something rather than failing, and an operator should be
    able to see *which* voice spoke without guessing.
    """
    headers = {
        "X-Sample-Rate": str(rate),
        "X-Voice": voice_label(voice),
        "Content-Disposition": f'inline; filename="speech.{fmt}"',
    }
    if streaming:
        # Or an intermediate proxy holds every chunk back until the generation ends, which
        # costs exactly the head start this endpoint is streaming for.
        headers["X-Accel-Buffering"] = "no"
    return headers


def _encode_whole(samples: np.ndarray, fmt: str, rate: int, model_rate: int) -> bytes:
    """A finished utterance as a complete file, at the rate the response promised."""
    if rate != model_rate:
        samples = resample(samples, model_rate, rate)
    return formats.encode_whole(samples, rate, fmt)


# ------------------------------------------------------------------------------- the two bodies


async def _audio_body(
    engine: Engine,
    encoder: Any,
    text: str,
    voice: str | None,
) -> AsyncIterator[bytes]:
    """Container bytes, yielded as the codec decodes the utterance.

    Every frame goes straight into the audio layer's incremental encoder, whose empty returns
    are ordinary -- a frame-based codec emits nothing until it has a frame's worth -- and never
    mean the end. Encoding runs off the event loop for the same reason generation does: it is C
    code holding the GIL, and the loop has a WebSocket session to keep answering.

    The model is reserved by the caller, before the response headers go out, and released here
    however this ends.
    """
    try:
        # aclosing, because a client that hangs up mid-stream closes this generator, and only
        # an explicit aclose passes that on to the model's iterator; without it the abandoned
        # generation stays in flight and the next request is told the server is busy.
        stream = engine.stream(text, voice, params=None, seed=None)
        async with aclosing(stream) as frames:
            async for frame in frames:
                if not frame.samples.size:
                    continue
                chunk = await asyncio.to_thread(encoder.encode, frame.samples)
                if chunk:
                    yield chunk
        tail = await asyncio.to_thread(encoder.finish)
        if tail:
            yield tail
    except Exception:
        # The status line is long gone, so this cannot become a 500. Raising rather than
        # returning quietly is deliberate: it aborts the response mid-body, and a truncated
        # transfer is something the client can detect, where a short clean file is not.
        log.exception("openai-compatible synthesis failed")
        raise
    finally:
        engine.release()


async def _sse_body(
    engine: Engine,
    fmt: str,
    rate: int,
    encoder: Any | None,
    text: str,
    voice: str | None,
) -> AsyncIterator[bytes]:
    """The same bytes wrapped in OpenAI's audio events, for ``stream_format: "sse"``.

    ``speech.audio.delta`` carries base64 of the very same container bytes the plain response
    would have sent, so a client concatenating every delta gets the identical thing. The
    terminal ``speech.audio.done`` reports zero usage: this server meters nothing, and a made-up
    token count is worse than an honest zero.

    An ``error`` event is this server's own addition. OpenAI has no failure event here because
    its errors arrive before the stream starts; on a local GPU a generation can fail halfway,
    and saying so beats a stream that simply stops.
    """
    try:
        async with aclosing(_chunks(engine, fmt, rate, encoder, text, voice)) as chunks:
            async for chunk in chunks:
                yield _event(
                    {
                        "type": "speech.audio.delta",
                        "audio": base64.b64encode(chunk).decode("ascii"),
                    }
                )
        yield _event(
            {
                "type": "speech.audio.done",
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            }
        )
    except Exception as exc:  # noqa: BLE001 - already logged below; report it and stop
        yield _event({"type": "error", "error": {"type": type(exc).__name__, "message": str(exc)}})


async def _chunks(
    engine: Engine,
    fmt: str,
    rate: int,
    encoder: Any | None,
    text: str,
    voice: str | None,
) -> AsyncIterator[bytes]:
    """Whatever the plain response would have sent, as one or more pieces.

    A streamed container yields its frames; one encoded in a single pass yields exactly one
    piece, once the utterance is finished. Either way the concatenation is the same bytes.
    """
    if encoder is not None:
        async for chunk in _audio_body(engine, encoder, text, voice):
            yield chunk
        return
    try:
        wav = await engine.generate(text, voice, params=None, seed=None)
        samples = np.asarray(wav, dtype=np.float32)
        yield await asyncio.to_thread(_encode_whole, samples, fmt, rate, engine.sample_rate)
    except Exception:
        log.exception("openai-compatible synthesis failed")
        raise
    finally:
        engine.release()


def _event(payload: dict[str, object]) -> bytes:
    """One event of OpenAI's audio stream: a bare ``data:`` line, keyed by its ``type`` field.

    No ``event:`` name, unlike ``/v1/tts/stream``. OpenAI's clients switch on the JSON's
    ``type``, and an SSE event name they do not expect is at best ignored.
    """
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()
