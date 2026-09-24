"""The HTTP endpoints: health, voices, synthesis, and the SSE stream.

Two response shapes for the same audio. ``POST /v1/tts`` blocks until the utterance is finished
and hands back a file; ``POST /v1/tts/stream`` hands back the first ~390 ms as soon as the codec
has decoded it, which on a warm GPU is a fraction of the time the whole utterance takes. Use the
streaming one for anything a person is waiting on.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

import numpy as np
from fastapi import APIRouter, Request, Response
from fastapi.responses import StreamingResponse

from kova_tts import __version__
from kova_tts.audio import to_pcm_bytes, to_wav_bytes
from kova_tts.server.engine import Engine
from kova_tts.server.protocol import (
    ChunkEvent,
    DoneEvent,
    ErrorEvent,
    ErrorResponse,
    HealthResponse,
    SynthesisRequest,
    TtsRequest,
    VoiceInfo,
    VoicesResponse,
)

log = logging.getLogger(__name__)

router = APIRouter()

#: ``audio/pcm`` rather than ``audio/L16``: the registered L16 type is big-endian, and what
#: comes out of here is little-endian, which is what every tool that reads raw PCM assumes.
PCM_MEDIA_TYPE = "audio/pcm"

#: Documented failure bodies, so the generated OpenAPI shows the error envelope.
_ERRORS = {
    404: {"model": ErrorResponse, "description": "No such voice."},
    409: {"model": ErrorResponse, "description": "A generation is already in flight."},
    422: {"model": ErrorResponse, "description": "The request could not be honoured."},
}


def _engine(request: Request) -> Engine:
    return request.app.state.engine


# ----------------------------------------------------------------------------------- inspection


@router.get("/health", response_model=HealthResponse, tags=["status"])
async def health(request: Request) -> HealthResponse:
    """Is the model up, where does it live, and how many voices does it have?

    ``model_loaded`` is always true in a served response -- loading happens in the application
    lifespan, so the server does not accept connections until it is finished. The field is here
    so a client can tell this server apart from a proxy or a placeholder that answers /health.
    """
    engine = _engine(request)
    return HealthResponse(
        model_loaded=True,
        device=engine.device,
        backend=engine.backend,
        sample_rate=engine.sample_rate,
        voices=len(engine.voices()),
        busy=engine.busy,
        version=__version__,
    )


@router.get("/v1/voices", response_model=VoicesResponse, tags=["status"])
async def voices(request: Request) -> VoicesResponse:
    """The LoRA voices installed on this machine.

    An empty list with ``lora_dir: null`` means no adapter directory is configured, which is the
    usual reason a voice name does not resolve.
    """
    engine = _engine(request)
    return VoicesResponse(
        voices=[VoiceInfo(name=name) for name in engine.voices()],
        lora_dir=engine.lora_dir,
    )


# ----------------------------------------------------------------------------- full synthesis


@router.post(
    "/v1/tts",
    tags=["synthesis"],
    responses={
        200: {
            "content": {"audio/wav": {}, PCM_MEDIA_TYPE: {}},
            "description": "A 16-bit wav file, or headerless 16-bit little-endian PCM.",
        },
        **_ERRORS,
    },
)
async def synthesize(body: TtsRequest, request: Request) -> Response:
    """Synthesize the whole text and return it as audio bytes."""
    engine = _engine(request)
    params = engine.sampling(body.sampling)

    async with engine.reserve():
        wav = await engine.generate(body.text, body.voice, params=params)

    audio = np.asarray(wav, dtype=np.float32)
    rate = engine.sample_rate
    headers = {
        "X-Sample-Rate": str(rate),
        "X-Duration-Seconds": f"{audio.size / rate:.3f}",
    }
    if body.response_format == "pcm":
        return Response(to_pcm_bytes(audio), media_type=PCM_MEDIA_TYPE, headers=headers)
    headers["Content-Disposition"] = 'inline; filename="speech.wav"'
    return Response(to_wav_bytes(audio, rate), media_type="audio/wav", headers=headers)


# ------------------------------------------------------------------------------ streaming (SSE)


def _sse(event: str, payload: object) -> bytes:
    """One Server-Sent Event: a named event and a single-line JSON data field."""
    body = json.dumps(payload, separators=(",", ":"))
    return f"event: {event}\ndata: {body}\n\n".encode()


@router.post(
    "/v1/tts/stream",
    tags=["synthesis"],
    responses={
        200: {
            "content": {"text/event-stream": {}},
            "description": (
                "A `chunk` event per decoded frame, then exactly one terminal event: "
                "`done` on success, `error` on failure."
            ),
        },
        **_ERRORS,
    },
)
async def synthesize_stream(body: SynthesisRequest, request: Request) -> StreamingResponse:
    """Synthesize `text`, sending audio as it is decoded.

    Each ``chunk`` event carries base64 16-bit little-endian PCM -- roughly 390 ms of it, which
    is one codec window. Concatenating every chunk's decoded bytes gives exactly the PCM that
    ``POST /v1/tts`` would have returned with ``response_format: "pcm"``.
    """
    engine = _engine(request)
    params = engine.sampling(body.sampling)
    # Resolved up front for the same reason the reservation is: a voice that does not exist
    # should be a 404, and once the stream is open the only way left to say so is an event.
    engine.check_voice(body.voice)

    # Reserved here, not inside the generator below: a refusal has to happen while the status
    # code is still ours to choose. Once StreamingResponse starts, 200 has already been sent.
    await engine.acquire()

    async def events() -> AsyncIterator[bytes]:
        index = 0
        samples = 0
        try:
            # aclosing, because a client that hangs up mid-stream closes *this* generator, and
            # only an explicit aclose passes that on to the model's iterator. Without it the
            # abandoned generation stays marked in-flight and the next request is told the
            # server is busy.
            stream = engine.stream(body.text, body.voice, params=params)
            async with aclosing(stream) as frames:
                async for frame in frames:
                    if not frame.samples.size:
                        continue
                    chunk = ChunkEvent(
                        index=index,
                        audio=base64.b64encode(to_pcm_bytes(frame.samples)).decode("ascii"),
                        sample_rate=frame.sample_rate,
                    )
                    index += 1
                    samples += int(frame.samples.size)
                    yield _sse("chunk", chunk.model_dump())
            done = DoneEvent(
                chunks=index,
                samples=samples,
                duration_seconds=round(samples / engine.sample_rate, 3),
                sample_rate=engine.sample_rate,
            )
            yield _sse("done", done.model_dump())
        except Exception as exc:  # noqa: BLE001 - the status line is long gone; report and stop
            log.exception("streaming synthesis failed")
            failure = ErrorEvent(error=type(exc).__name__, message=str(exc))
            yield _sse("error", failure.model_dump())
        finally:
            engine.release()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tell an intermediate proxy not to buffer, or it will hold every chunk back until
            # the generation finishes and the whole point of the endpoint is lost.
            "X-Accel-Buffering": "no",
        },
    )
