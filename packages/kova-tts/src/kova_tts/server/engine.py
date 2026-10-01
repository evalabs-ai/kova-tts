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
import functools
import logging
import threading
from collections.abc import AsyncIterator, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any, TypeVar

from kova_codec.constants import OUTPUT_SAMPLE_RATE
from kova_tts import paths
from kova_tts.engine.types import TTS_SAMPLING, AudioFrame, SamplingParams
from kova_tts.server.errors import Busy, InvalidRequest
from kova_tts.server.protocol import SamplingOverrides

log = logging.getLogger(__name__)

T = TypeVar("T")

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

    def check_timestamps(self, timestamps: bool) -> None:
        """Refuse word timestamps up front when no aligner is loaded, while a status code can
        still say so."""
        if timestamps and getattr(self.tts, "aligner", None) is None:
            raise InvalidRequest(
                "timestamps: this server has no word aligner loaded. Start it with "
                "KOVA_ALIGNMENT_PATH pointing at alignment.pt, or let it download from the Hub."
            )

    async def generate(
        self,
        text: str,
        voice: str | None,
        *,
        params: SamplingParams | None,
        normalize: bool = True,
        timestamps: bool = False,
    ) -> Any:
        """The whole waveform -- and its word timestamps, if asked -- on the engine thread."""
        return await on_engine_thread(
            self.tts.generate,
            text,
            voice,
            params=params,
            normalize=normalize,
            timestamps=timestamps,
        )

    def stream(
        self,
        text: str,
        voice: str | None,
        *,
        params: SamplingParams | None,
        normalize: bool = True,
        timestamps: bool = False,
    ) -> AsyncIterator[AudioFrame]:
        """Frames as the codec produces them, pumped off the event loop.

        The iterator ``KovaTTS.stream`` returns is lazy, so nothing runs until the first frame
        is awaited -- which is what lets a caller reserve the model, hand the response back to
        the ASGI server, and only then start generating.
        """
        frames = self.tts.stream(
            text, voice, params=params, normalize=normalize, timestamps=timestamps
        )
        return aiter_frames(frames)


#: What the engine thread is called in a stack dump or a profiler.
ENGINE_THREAD_NAME = "kova-engine"

_engine_thread: ThreadPoolExecutor | None = None
_engine_thread_lock = threading.Lock()


def engine_thread() -> ThreadPoolExecutor:
    """The one thread every call into the model and the codec runs on, for the life of the process.

    One thread, and always the same one, for two reasons:

    * **PyTorch keeps cuDNN's execution plans per thread.** The codec's upsampling stack has
      dozens of convolution shapes, and the first decode on a thread that has never run it
      spends about 1.2 s building plans for all of them, against ~25 ms once they exist. A new
      thread per request paid that before every first frame -- it was most of the time to first
      audio. One thread, warmed at startup (:func:`on_engine_thread` from the server's and the
      demo's warm-ups), pays it once.
    * **MLX binds a GPU stream to the thread that created it**, so an iterator belonging to the
      MLX generator has to be stepped from the same thread every time; resumed anywhere else it
      raises ``There is no Stream(gpu, N) in current thread``. One thread for everything is the
      simplest way to never break that.

    One is also all there is work for: the engine is batch-1, and :class:`Engine` serialises
    requests before they get here. Never shut down; it lives as long as the model it serves.
    """
    global _engine_thread
    with _engine_thread_lock:
        if _engine_thread is None:
            _engine_thread = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix=ENGINE_THREAD_NAME
            )
        return _engine_thread


async def on_engine_thread(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """``fn(*args, **kwargs)`` on :func:`engine_thread`, awaited from the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(engine_thread(), functools.partial(fn, *args, **kwargs))


def close_on_engine_thread(iterator: Any) -> None:
    """Close a generator that the engine thread may be in the middle of stepping.

    A consumer that gives up -- a disconnected client, a cancelled burst -- leaves the model's
    generator suspended with its "one request in flight" flag still set, and the caller usually
    still holds it, so nothing else will ever close it: every later request would be told the
    model is busy. Closing runs that flag's ``finally``.

    Closed here when it can be. But cancellation stops the wait for the engine thread, not the
    thread, so a consumer that left mid-step finds the generator still executing there, and an
    executing generator refuses to be closed. Then the close is queued behind that step instead
    and runs the moment it lands, on the engine thread. Not awaited: a cancelled consumer must
    not wait for the GPU.
    """
    try:
        iterator.close()
    except ValueError:
        engine_thread().submit(iterator.close)


async def aiter_frames(frames: Iterator[AudioFrame]) -> AsyncIterator[AudioFrame]:
    """Drive a blocking frame iterator from async code, one thread hop per frame.

    A hop costs microseconds and frames are ~390 ms apart, so the overhead is invisible; what it
    buys is an event loop that stays responsive while the GPU works, which is the difference
    between a WebSocket that answers a ``close_context`` promptly and one that answers it after
    the current utterance. Every hop lands on the same thread -- see :func:`engine_thread`.
    """
    done = object()
    loop = asyncio.get_running_loop()
    try:
        while True:
            frame = await loop.run_in_executor(engine_thread(), next, frames, done)
            if frame is done:
                return
            yield frame  # type: ignore[misc]
    finally:
        close_on_engine_thread(frames)
