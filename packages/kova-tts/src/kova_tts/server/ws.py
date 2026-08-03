"""``WS /v1/ws`` -- an incremental synthesis session.

One connection drives one stream. The client opens a context, pushes text as it becomes
available, and asks for audio when it wants some::

    -> {"start_context": {"voice": "some-voice"}}
    <- {"context_started": {...}}
    -> {"send_text": "The first half of a sentence "}
    -> {"send_text": "and the second half."}
    -> {"flush": true, "flush_id": "a"}
    <- {"audio_chunk": "<base64 pcm>"}      (repeated)
    <- {"flush_completed": true, "flush_id": "a"}
    -> {"close_context": true, "flush_id": "b"}
    <- {"audio_chunk": "<base64 pcm>"}      (anything still buffered)
    <- {"flush_completed": true, "flush_id": "b"}
    <- {"context_closed": true}

Text is buffered, never spoken, until a flush: that is the point of the session. It lets a
caller that is producing text a token at a time -- an LLM, a chat UI -- hand it over as it
arrives and choose the sentence boundaries where latency is worth spending.

**Ordering is guaranteed and the mechanism is worth knowing.** Inbound frames are read by one
task; flushes are executed one at a time, in arrival order, by a second; every outgoing frame
goes through a single queue drained by a third. So ``flush_completed`` can never overtake the
audio it terminates, and two flushes can never interleave their chunks, however fast the client
sends.

Two things this session deliberately does not do. It does not carry prosody across a flush
boundary -- each flush is its own generation, so a flush per word will sound like a flush per
word; flush on sentences. And it holds the model only for the duration of a flush, not for the
life of the connection, so an idle session costs nothing and a flush that collides with an HTTP
request gets an ``error`` frame rather than deadlocking.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
from dataclasses import dataclass

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from kova_tts.audio import to_pcm_bytes
from kova_tts.server import protocol as wire
from kova_tts.server.engine import Engine

log = logging.getLogger(__name__)

router = APIRouter()

#: Sentinel put on the outbound queue to retire the writer task.
_STOP = None


@dataclass(frozen=True, slots=True)
class _Work:
    """One flush: the text buffered when it was asked for, and whether it ends the session."""

    text: str
    flush_id: str
    close: bool


class Session:
    """The state behind one connection: buffered text, and the flushes it turns into.

    Args:
        engine: The shared model. Reserved per flush, never for the whole session.
        start: The accepted configuration, echoed back to the client.
        out: The connection's outbound queue. The endpoint owns the task draining it, so this
            class never touches the socket and cannot emit frames out of order.
    """

    def __init__(self, *, engine: Engine, start: wire.StartConfig, out: asyncio.Queue) -> None:
        self.engine = engine
        self.start = start
        self.out = out
        self.params = engine.sampling(start.sampling)

        self._buffer: list[str] = []
        self._work: asyncio.Queue[_Work | None] = asyncio.Queue()
        self._producer: asyncio.Task | None = None

    # ------------------------------------------------------------------ inbound frame handlers

    async def open(self) -> None:
        """Start the flush producer and acknowledge the context."""
        self._producer = asyncio.create_task(self._run())
        await self._emit(wire.ContextStarted(context_started=self.start))

    def send_text(self, text: str) -> None:
        self._buffer.append(text)

    def flush(self, flush_id: str | None) -> None:
        self._enqueue(flush_id or "flush", close=False)

    def close_context(self, flush_id: str | None) -> None:
        self._enqueue(flush_id or "close", close=True)

    def _enqueue(self, flush_id: str, *, close: bool) -> None:
        # Snapshot the buffer at the moment the flush was *asked for*, not when it runs: text
        # that arrives while an earlier flush is still generating belongs to the next one.
        text = "".join(self._buffer)
        self._buffer.clear()
        self._work.put_nowait(_Work(text=text, flush_id=flush_id, close=close))

    async def aclose(self) -> None:
        """Tear down after a disconnect, without waiting for work still queued."""
        if self._producer is not None:
            self._producer.cancel()
            try:
                await self._producer
            except asyncio.CancelledError:
                pass
            self._producer = None
        await self.out.put(_STOP)

    # ------------------------------------------------------------------ the flush producer

    async def _run(self) -> None:
        """Execute queued flushes, one at a time, in order."""
        try:
            while True:
                work = await self._work.get()
                if work is None:
                    break
                await self._process(work)
                if work.close:
                    await self._emit(wire.ContextClosed())
                    break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a session failure is a frame, not a crash
            log.exception("websocket session failed")
            await self._emit(wire.Error(error=str(exc)))
        finally:
            await self.out.put(_STOP)

    async def _process(self, work: _Work) -> None:
        """Synthesize one flush, then acknowledge it however it went.

        ``flush_completed`` is sent even when the flush failed or held no text, so a client that
        waits for it after every flush never hangs.
        """
        if work.text.strip():
            try:
                await self._synthesize(work.text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reported to the client, session survives
                log.exception("synthesis failed for flush %s", work.flush_id)
                await self._emit(wire.Error(error=str(exc), flush_id=work.flush_id))
        await self._emit(wire.FlushCompleted(flush_id=work.flush_id))

    async def _synthesize(self, text: str) -> None:
        async with self.engine.reserve():
            stream = self.engine.stream(
                text, self.start.voice, params=self.params, seed=self.start.seed
            )
            # aclosing, so a cancelled flush -- the client disconnected mid-sentence -- passes
            # the close on to the model's iterator instead of leaving it marked in-flight.
            async with contextlib.aclosing(stream) as frames:
                async for frame in frames:
                    if not frame.samples.size:
                        continue
                    payload = base64.b64encode(to_pcm_bytes(frame.samples)).decode("ascii")
                    await self._emit(wire.AudioChunk(audio_chunk=payload))

    async def _emit(self, frame: wire.OutgoingFrame) -> None:
        await self.out.put(frame)


# ---------------------------------------------------------------------------------- the endpoint


async def _writer(ws: WebSocket, out: asyncio.Queue) -> None:
    """Drain the outbound queue to the socket. The only task that sends."""
    while True:
        frame = await out.get()
        if frame is _STOP:
            return
        await ws.send_json(wire.to_wire(frame))


@router.websocket("/v1/ws")
async def stream(ws: WebSocket) -> None:
    """Read frames, dispatch them, and close cleanly once the session is done."""
    engine: Engine = ws.app.state.engine
    await ws.accept()

    out: asyncio.Queue = asyncio.Queue()
    writer = asyncio.create_task(_writer(ws, out))
    session: Session | None = None
    closing = False

    try:
        while not closing:
            try:
                raw = await ws.receive_json()
            except WebSocketDisconnect:
                break
            except (ValueError, KeyError, TypeError) as exc:  # not JSON, or not text
                await out.put(wire.Error(error=f"bad frame: {exc}"))
                continue

            try:
                frame = wire.parse_incoming(raw)
            except ValueError as exc:  # unknown key, or a field the model rejects
                await out.put(wire.Error(error=f"bad frame: {exc}"))
                continue

            if isinstance(frame, wire.StartContext):
                if session is not None:
                    await out.put(wire.Error(error="context already started"))
                    continue
                try:
                    engine.check_voice(frame.start_context.voice)
                    session = Session(engine=engine, start=frame.start_context, out=out)
                except Exception as exc:  # noqa: BLE001 - a rejected start is recoverable
                    # No session is created, so the client can correct the configuration and
                    # send start_context again on the same connection.
                    await out.put(wire.Error(error=str(exc)))
                    continue
                await session.open()
            elif session is None:
                await out.put(wire.Error(error="no active context; send start_context first"))
            elif isinstance(frame, wire.SendText):
                session.send_text(frame.send_text)
            elif isinstance(frame, wire.Flush):
                session.flush(frame.flush_id)
            elif isinstance(frame, wire.CloseContext):
                session.close_context(frame.flush_id)
                # Stop reading, but do not close the socket: the producer is still generating
                # the final flush, and the writer has to get those frames out first.
                closing = True
    except WebSocketDisconnect:
        log.debug("websocket client disconnected")
    finally:
        if not closing:
            # The reader stopped for some other reason -- a disconnect, an error -- so nothing
            # is going to produce the frames the writer is waiting for. Retire both.
            if session is not None:
                await session.aclose()
            else:
                await out.put(_STOP)
        try:
            # When closing, this ends by itself: the producer emits context_closed and stops.
            await writer
        except Exception:  # noqa: BLE001 - the socket died under the writer, nothing to send
            log.debug("websocket writer stopped early", exc_info=True)
            if session is not None:
                await session.aclose()
        with contextlib.suppress(RuntimeError):
            await ws.close()
