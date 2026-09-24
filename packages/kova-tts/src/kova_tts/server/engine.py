"""The loaded model, plus the one lock that keeps it to a single generation.

:class:`~kova_tts.engine.generator.Generator` is deliberately not reentrant: one static KV
cache, one set of CUDA graph input buffers, batch size one. Overlapping two generations raises
from three layers down, and that error is a 500 as far as a client can tell -- which is a lie,
because nothing is broken.

So everything that touches the model goes through :meth:`Engine.reserve`, and the refusal is
part of the API. The refusal is not instant: ``busy_timeout`` seconds of waiting come first.

**Why wait at all.** The overwhelmingly common "concurrent" request on a local server is not
concurrent -- it is a page reload, a client retry, or a second script started a moment too
early, landing while the previous generation's last frame is still being written. A couple of
seconds of waiting turns that race into a request that simply works. A *genuinely* parallel
second caller still gets a clean 409 within the timeout instead of being silently queued behind
a thirty-second generation, which is the failure mode a queue would give it. Set
``busy_timeout=0`` to refuse the instant the model is busy.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from kova_codec.constants import OUTPUT_SAMPLE_RATE
from kova_tts import paths
from kova_tts.engine.types import TTS_SAMPLING, AudioFrame, SamplingParams
from kova_tts.server.errors import Busy, InvalidRequest
from kova_tts.server.protocol import SamplingOverrides

log = logging.getLogger(__name__)

#: Seconds a second caller waits for the model before being refused. Long enough to absorb the
#: hand-off between two back-to-back requests, short enough that a real collision is reported
#: while the caller is still watching.
DEFAULT_BUSY_TIMEOUT = 5.0

_BUSY_MESSAGE = (
    "This server generates one utterance at a time -- the model holds a single KV cache, so "
    "there is nothing to parallelise onto. Wait for the request in flight to finish and send "
    "this one again."
)


class Engine:
    """A loaded :class:`~kova_tts.engine.tts.KovaTTS` and the lock that serialises it.

    Args:
        tts: The loaded model. Typed loosely so a test can pass a stub without importing torch.
        busy_timeout: Seconds a caller waits for the model before :class:`Busy` is raised.
            Zero refuses immediately.
    """

    def __init__(self, tts: Any, *, busy_timeout: float = DEFAULT_BUSY_TIMEOUT) -> None:
        self.tts = tts
        self.busy_timeout = float(busy_timeout)
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------ description

    @property
    def device(self) -> str:
        """Where the LM lives, for ``/health``.

        ``KovaTTS`` keeps its device private; its generator does not, and that is the object
        that actually decides where the decode loop runs.
        """
        generator = getattr(self.tts, "generator", None)
        return str(getattr(generator, "device", "unknown"))

    @property
    def backend(self) -> str:
        """Which decode loop is running -- ``torch`` or ``mlx``.

        Worth reporting next to the device because on an Apple machine both backends say
        ``mps``: the codec is torch either way, and the difference between four times slower
        than real time and faster than it is which loop drives the LM.
        """
        generator = getattr(self.tts, "generator", None)
        return str(getattr(generator, "backend", "unknown"))

    @property
    def sample_rate(self) -> int:
        return int(getattr(self.tts, "sample_rate", OUTPUT_SAMPLE_RATE))

    @property
    def busy(self) -> bool:
        """True while a generation is in flight."""
        return self._lock.locked()

    @property
    def lora_dir(self) -> str | None:
        """The adapter directory in use, or ``None`` when none is configured.

        Reported by ``/v1/voices`` because an empty voice list is almost always a directory
        pointed somewhere wrong, and the answer to that is visible here and nowhere else. A
        misconfigured directory is reported as *no* directory rather than raised: listing voices
        must not be the request that fails.
        """
        try:
            root = paths.lora_dir(getattr(self.tts, "lora_root", None))
        except paths.MissingArtifact:
            return None
        return str(root) if root is not None else None

    def voices(self) -> list[str]:
        return list(self.tts.voices())

    def check_voice(self, name: str | None) -> None:
        """Raise if `name` will not resolve, before anything expensive has started.

        The WebSocket session uses this at ``start_context``: a voice that does not exist should
        be reported while the client is still setting up, not two frames into the first flush.
        """
        if name is None:
            return
        self.tts.voice(name)

    # ------------------------------------------------------------------ the single-flight lock

    @asynccontextmanager
    async def reserve(self) -> AsyncIterator[None]:
        """Hold the model for the duration of the block, or raise :class:`Busy`."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    async def acquire(self) -> None:
        """Take the model, waiting up to ``busy_timeout`` seconds for it.

        Split out of :meth:`reserve` for the streaming endpoints, which have to know they hold
        the model *before* the response headers go out -- once a stream has started, a refusal
        can no longer be an HTTP status code.
        """
        if self.busy_timeout <= 0:
            if self._lock.locked():
                raise Busy(_BUSY_MESSAGE)
            await self._lock.acquire()
            return
        try:
            await asyncio.wait_for(self._lock.acquire(), self.busy_timeout)
        except TimeoutError:
            raise Busy(_BUSY_MESSAGE) from None

    def release(self) -> None:
        self._lock.release()

    # ------------------------------------------------------------------ synthesis

    def sampling(self, overrides: SamplingOverrides | None) -> SamplingParams | None:
        """Turn request overrides into sampling parameters, or ``None`` to keep the preset.

        ``None`` is passed straight through so :class:`~kova_tts.engine.tts.KovaTTS` picks the
        preset the voice calls for. When there *are* overrides they are applied to the plain-TTS
        preset: the only voices this server exposes are LoRA adapters, which are never clones.
        """
        if overrides is None:
            return None
        try:
            return overrides.apply(TTS_SAMPLING)
        except ValueError as exc:
            raise InvalidRequest(f"sampling: {exc}") from exc

    async def generate(
        self,
        text: str,
        voice: str | None,
        *,
        params: SamplingParams | None,
    ) -> Any:
        """The whole waveform, generated off the event loop."""
        return await asyncio.to_thread(self.tts.generate, text, voice, params=params)

    def stream(
        self,
        text: str,
        voice: str | None,
        *,
        params: SamplingParams | None,
    ) -> AsyncIterator[AudioFrame]:
        """Frames as the codec produces them, pumped off the event loop.

        The iterator ``KovaTTS.stream`` returns is lazy, so nothing runs until the first frame
        is awaited -- which is what lets a caller reserve the model, hand the response back to
        the ASGI server, and only then start generating.
        """
        return aiter_frames(self.tts.stream(text, voice, params=params))


@contextlib.contextmanager
def pinned_worker(name: str) -> Iterator[ThreadPoolExecutor]:
    """One dedicated thread, for the whole life of one MLX-backed iterator.

    Anything that steps an iterator belonging to the MLX generator has to step it from the *same*
    thread every time, which :func:`asyncio.to_thread` cannot promise: that helper submits to the
    default pool, which is free to pick a different worker per call. MLX binds a GPU stream to
    the thread that created it, so an utterance resumed elsewhere raises ``There is no
    Stream(gpu, N) in current thread`` the moment it touches an array the first thread made.

    Even where nothing breaks, a step on a new thread pays for a fresh GPU stream. One worker for
    the whole iterator fixes both, and costs a thread per in-flight request -- which is one, since
    the engine is batch-1.

    ``shutdown(wait=False)`` because a cancelled consumer must not block the event loop until the
    in-flight step finishes on the GPU. The worker is never reused, so letting it retire on its
    own is safe; closing the iterator is what actually releases the generator.
    """
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
    try:
        yield pool
    finally:
        pool.shutdown(wait=False)


async def aiter_frames(frames: Iterator[AudioFrame]) -> AsyncIterator[AudioFrame]:
    """Drive a blocking frame iterator from async code, one thread hop per frame.

    A hop costs microseconds and frames are ~390 ms apart, so the overhead is invisible; what it
    buys is an event loop that stays responsive while the GPU works, which is the difference
    between a WebSocket that answers a ``close_context`` promptly and one that answers it after
    the current utterance. Every hop lands on the same thread -- see :func:`pinned_worker`.
    """
    done = object()
    loop = asyncio.get_running_loop()
    try:
        with pinned_worker("kova-frames") as pump:
            while True:
                frame = await loop.run_in_executor(pump, next, frames, done)
                if frame is done:
                    return
                yield frame  # type: ignore[misc]
    finally:
        # A consumer that gives up -- a disconnected SSE client, a cancelled flush -- leaves the
        # underlying generator suspended mid-decode with its "one request in flight" flag still
        # set. Closing it runs that flag's `finally` now, rather than whenever the garbage
        # collector gets to it, which is the difference between the next request working and
        # the next request being told the model is busy.
        #
        # Suppressed because cancellation does not stop the worker thread, only the wait for
        # it: for the tens of milliseconds it takes the in-flight frame to finish decoding, the
        # generator really is still executing and refuses to be closed. It closes itself when
        # that frame lands, so the flag is cleared either way.
        with contextlib.suppress(ValueError, RuntimeError):
            frames.close()
