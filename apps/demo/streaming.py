"""The page's audio source: :mod:`kova_tts.server`'s protocol, served from this session.

The wire format is that package's, event for event and field for field -- this reuses its
models rather than describing the protocol a second time, so the demo is a client of the
documented API. What it does not reuse is that package's ``Engine``, for two reasons: a voice
cloned on the other tab lives in this session and would not resolve against the adapters on
disk, and a second lock over the same non-reentrant generator is not a lock at all.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from kova_codec.constants import OUTPUT_SAMPLE_RATE
from kova_tts import audio as audio_io
from kova_tts.server import errors as server_errors
from kova_tts.server.engine import aiter_frames
from kova_tts.server.protocol import (
    ChunkEvent,
    DoneEvent,
    ErrorEvent,
    ErrorResponse,
    SynthesisRequest,
)

from content import BUSY
from session import DemoSession

log = logging.getLogger("kova_tts.demo")

#: Where the page streams from. The same path and the same protocol as ``kova_tts.server``.
STREAM_PATH = "/v1/tts/stream"

#: Longest text the demo will accept in one press. Not a model limit -- long text is split into
#: sentences and generated segment by segment -- but a public demo should not let one visitor
#: hold the only generator for ten minutes.
MAX_CHARS = 1200


def _sse(event: str, payload: object) -> bytes:
    """One server-sent event: a named event and a single-line JSON data field."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def add_stream_route(app: FastAPI, session: DemoSession, *, path: str = STREAM_PATH) -> FastAPI:
    """Attach the page's audio source to `app`."""

    @app.post(
        path,
        tags=["synthesis"],
        responses={
            200: {
                "content": {"text/event-stream": {}},
                "description": (
                    "A `chunk` event per decoded frame, carrying base64 16-bit little-endian "
                    "PCM, then exactly one terminal event: `done`, or `error`."
                ),
            },
            409: {"model": ErrorResponse, "description": "A generation is already in flight."},
            422: {"model": ErrorResponse, "description": "The request could not be honoured."},
            503: {"model": ErrorResponse, "description": "The model could not be loaded."},
        },
    )
    async def synthesize_stream(body: SynthesisRequest) -> StreamingResponse:
        """Synthesize `text`, sending audio as it is decoded."""
        if len(body.text) > MAX_CHARS:
            raise server_errors.InvalidRequest(
                f"text: that is {len(body.text):,} characters; this demo generates up to "
                f"{MAX_CHARS:,} at a time. Trim it, or use the Python API for a long passage."
            )
        params = session.sampling(body.sampling, body.voice)

        # Reserved before anything is generated and before the response starts, so a refusal is
        # still a status code. The model is loaded inside the reservation because loading is
        # not thread-safe and the first two visitors would otherwise both pay for it.
        await session.acquire()
        try:
            frames = await asyncio.to_thread(session.frames, body.text, body.voice, params=params)
        except BaseException:
            session.release()
            raise

        async def events() -> AsyncIterator[bytes]:
            index = 0
            samples = 0
            rate = OUTPUT_SAMPLE_RATE
            try:
                # aclosing, because a listener who presses Stop or reloads the page closes
                # *this* generator, and only an explicit aclose passes that on to the model's
                # iterator. Without it the abandoned generation stays marked in-flight and the
                # next press is told the model is busy.
                async with aclosing(aiter_frames(frames)) as decoded:
                    async for frame in decoded:
                        if not frame.samples.size:
                            continue
                        rate = frame.sample_rate
                        pcm = audio_io.to_pcm_bytes(frame.samples)
                        chunk = ChunkEvent(
                            index=index,
                            audio=base64.b64encode(pcm).decode("ascii"),
                            sample_rate=rate,
                        )
                        index += 1
                        samples += int(frame.samples.size)
                        yield _sse("chunk", chunk.model_dump())
                yield _sse(
                    "done",
                    DoneEvent(
                        chunks=index,
                        samples=samples,
                        duration_seconds=round(samples / rate, 3),
                        sample_rate=rate,
                    ).model_dump(),
                )
            except Exception as exc:  # noqa: BLE001 - the status line is long gone; report it
                log.exception("Streaming synthesis failed")
                message = BUSY if "already running" in str(exc) else str(exc)
                failure = ErrorEvent(error=type(exc).__name__, message=message)
                yield _sse("error", failure.model_dump())
            finally:
                session.release()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                # Tell an intermediate proxy not to buffer, or it holds every frame back until
                # the generation finishes and the whole point of the endpoint is lost.
                "X-Accel-Buffering": "no",
            },
        )

    return app
