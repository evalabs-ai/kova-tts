"""The state behind the page: one engine, one lock, and the voices cloned this session.

The engine is not reentrant and neither is this page. One :class:`Generator`, one KV cache: a
second overlapping request raises. A single lock guards every path that touches the model --
the streaming endpoint and the clone tab alike -- and losing the race is a sentence the page
prints, not a traceback.

The engine loads lazily, on the first generation rather than at import, so the page comes up on
a machine with no weights configured and says what to fix.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from kova_codec.constants import SAMPLE_RATE
from kova_tts import CLONE_SAMPLING, TTS_SAMPLING, MissingArtifact, SamplingParams, Voice, paths
from kova_tts import audio as audio_io
from kova_tts.server import errors as server_errors
from kova_tts.server.protocol import SamplingOverrides
from kova_tts.voices import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS

from content import BUSY

log = logging.getLogger("kova_tts.demo")

#: Value of the voice picker for "the model's own voice". Empty rather than ``None`` because a
#: Gradio dropdown treats ``None`` as *nothing selected* and refuses to show a label for it.
BASE_VOICE = ""
BASE_LABEL = "Base voice (no LoRA)"

#: Seconds a second request waits for the model before it is refused. The overwhelmingly common
#: collision is one person pressing Speak again: the previous request is abandoned a moment
#: earlier and its engine is still being released. Waiting turns that into a request that works.
BUSY_TIMEOUT = 5.0

#: Below this peak amplitude a reference recording is silence, and saying so is kinder than
#: letting the codec return no codes and reporting that.
SILENT_PEAK = 1e-3


class DemoError(RuntimeError):
    """Something the person at the browser can fix, phrased for them rather than for a log."""


class DemoSession:
    """Everything the page needs: one engine, one lock, and the voices cloned this session.

    Args:
        tts: An already-built :class:`~kova_tts.engine.tts.KovaTTS`. Tests inject a fake here;
            leaving it ``None`` means the real one is built by `loader` on first use.
        loader: ``callable() -> KovaTTS``, called at most once.
        lora_root: Where to look for LoRA voices *before* the engine exists, so the voice picker
            is populated on the first paint.
        device: Only used to describe the machine in the banner; the loader owns the real one.
    """

    def __init__(
        self,
        tts: Any = None,
        *,
        loader: Callable[[], Any] | None = None,
        lora_root: str | os.PathLike[str] | None = None,
        device: str | None = None,
    ) -> None:
        self._tts = tts
        self._loader = loader
        self._lora_root = lora_root
        self._device = device
        self._cloned: dict[str, Voice] = {}
        # Non-reentrant engine, non-reentrant page: the streaming endpoint and the clone tab
        # both take this, so a concurrent request is answered instead of crashing the model.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------------------- engine

    @property
    def loaded(self) -> bool:
        """True once the weights are in memory. The first generation pays for them."""
        return self._tts is not None

    def engine(self) -> Any:
        """The engine, loading it on first call.

        Every failure is re-raised as a :class:`DemoError` whose message names the fix, because
        this is the one place a misconfigured machine shows up.
        """
        if self._tts is not None:
            return self._tts
        if self._loader is None:
            raise DemoError("This demo was built without an engine, so it cannot generate audio.")
        try:
            self._tts = self._loader()
        except MissingArtifact as exc:
            raise DemoError(
                f"{exc}\n\nRun `kova-tts paths` to see where each artifact resolved to, and set "
                f"the missing ones in your `.env`."
            ) from exc
        except Exception as exc:
            log.exception("Loading the model failed")
            raise DemoError(
                f"The model could not be loaded: {exc}\n\nRun `kova-tts paths` to check where "
                f"the weights are being looked for."
            ) from exc
        return self._tts

    # ------------------------------------------------------------------------------- voices

    def voices(self) -> list[str]:
        """LoRA voices installed on this machine, without loading the model.

        A broken ``KOVA_LORA_DIR`` returns nothing rather than raising: the base voice and
        cloning still work, and the banner already says the directory is wrong.
        """
        try:
            if self._tts is not None:
                return sorted(self._tts.voices())
            return sorted(paths.available_loras(self._lora_root))
        except (MissingArtifact, OSError):
            return []

    def choices(self) -> list[tuple[str, str]]:
        """``(label, value)`` pairs for the voice picker: base voice, LoRAs, then clones."""
        cloned = [(f"{name} (cloned)", name) for name in sorted(self._cloned)]
        return [(BASE_LABEL, BASE_VOICE), *((name, name) for name in self.voices()), *cloned]

    def resolve(self, name: str | None) -> str | Voice | None:
        """The voice picker's value as the engine wants it: a clone, a LoRA name, or ``None``."""
        if not name:
            return None
        return self._cloned.get(name, name)

    def preset(self, name: str | None = None) -> SamplingParams:
        """The sampling preset a voice calls for.

        The two presets currently differ only in token budget -- cloning is given a longer one,
        because a cloned generation carries the reference through the same budget -- but the
        picker drives the Advanced sliders through this either way, so what the sliders show is
        always what the engine would have chosen on its own.
        """
        return CLONE_SAMPLING if name and name in self._cloned else TTS_SAMPLING

    def sampling(self, overrides: SamplingOverrides | None, voice: str | None) -> SamplingParams:
        """The preset for `voice`, with whatever the request actually asked to change."""
        preset = self.preset(voice)
        if overrides is None:
            return preset
        try:
            return overrides.apply(preset)
        except ValueError as exc:
            raise server_errors.InvalidRequest(f"sampling: {exc}") from exc

    # ---------------------------------------------------------------------- the single-flight

    async def acquire(self) -> None:
        """Take the model, waiting up to :data:`BUSY_TIMEOUT` for it, or raise ``Busy``.

        Off the event loop, because this is a threading lock shared with the clone tab, which
        Gradio calls from a worker thread. Split from :meth:`release` rather than offered as a
        context manager because a streaming response has to know it holds the model *before*
        the status code goes out: once the stream is open, 200 has already been sent.
        """
        if not await asyncio.to_thread(self._lock.acquire, True, BUSY_TIMEOUT):
            raise server_errors.Busy(BUSY)

    def release(self) -> None:
        self._lock.release()

    def frames(
        self,
        text: str,
        voice: str | None,
        *,
        params: SamplingParams | None = None,
        seed: int | None = None,
    ) -> Iterator[Any]:
        """Audio frames for `text`, in this session's voice namespace.

        Blocking, and it loads the model on the first call, so callers on an event loop run it
        in a thread. The iterator itself is lazy: nothing is generated until it is advanced.
        """
        return self.engine().stream(text, self.resolve(voice), params=params, seed=seed)

    # ---------------------------------------------------------------------------- callbacks

    def clone_voice(
        self,
        reference: str | None,
        transcript: str = "",
        name: str = "",
    ) -> tuple[str, str | None]:
        """Turn a reference recording into a voice, returning ``(status, new voice name)``.

        The clip is checked here rather than left to the codec: "that recording is 0.4 seconds
        long" is a useful thing to read, and "the reference clip encoded to no codes" is not.
        """
        if not reference:
            return "Upload a recording, or record one with the microphone, first.", None
        transcript = (transcript or "").strip()

        try:
            wav = audio_io.load_audio(reference, SAMPLE_RATE)
        except Exception as exc:
            log.warning("Could not read reference audio %s: %s", reference, exc)
            return f"That file could not be read as audio: {exc}", None

        seconds = wav.size / SAMPLE_RATE
        if seconds < MIN_REFERENCE_SECONDS:
            return (
                f"That recording is {seconds:.1f} s long. Cloning needs at least "
                f"{MIN_REFERENCE_SECONDS:.0f} s of clean speech -- five to ten is better.",
                None,
            )
        if float(np.max(np.abs(wav))) < SILENT_PEAK:
            return (
                "That recording is silent. If you used the microphone, check that the browser "
                "was allowed to record, and that the meter moved while you spoke.",
                None,
            )

        if not self._lock.acquire(blocking=False):
            return BUSY, None
        try:
            tts = self.engine()
            if not transcript and getattr(tts, "transcriber", None) is None:
                return (
                    "This demo has no transcriber, so it needs the transcript: type exactly what "
                    "the recording says into the box above.",
                    None,
                )
            voice = tts.clone(reference, transcript or None, name=self._unique(name, reference))
        except DemoError as exc:
            return str(exc), None
        except (ImportError, ValueError) as exc:
            # A missing `data` extra, a transcript that does not match, an unusable clip: all
            # of these already explain themselves.
            return f"That did not work: {exc}", None
        except Exception as exc:
            log.exception("Cloning failed")
            return f"Cloning failed: {exc}", None
        finally:
            self._lock.release()

        self._cloned[voice.name] = voice
        used = min(seconds, MAX_REFERENCE_SECONDS)
        heard = "transcribed automatically" if not transcript else "as typed"
        return (
            f"Cloned **{voice.name}** from {used:.1f} s of audio. It is selected on the Speak "
            f"tab now.\n\nReference transcript ({heard}): “{voice.ref_text}”",
            voice.name,
        )

    def _unique(self, name: str, reference: str) -> str:
        """A voice name that collides with nothing already in the picker."""
        base = (name or Path(reference).stem or "cloned").strip()
        taken = set(self._cloned) | set(self.voices())
        if base not in taken:
            return base
        for index in range(2, 100):
            candidate = f"{base}-{index}"
            if candidate not in taken:
                return candidate
        return f"{base}-{random.randrange(10_000)}"

    # ------------------------------------------------------------------------------ banner

    def notices(self) -> list[str]:
        """Lines for the banner: what this machine has, and what it is missing.

        Everything here is cheap and offline. Resolving the codec would download it from the
        Hub, which is not something a page load should do.
        """
        lines: list[str] = []
        device = self._device or _default_device()
        if device.startswith("cpu"):
            lines.append(
                "**Running on the CPU.** Generation will work, but a sentence takes minutes "
                "rather than seconds; a CUDA device is strongly recommended."
            )
        else:
            lines.append(f"Running on `{device}`.")

        if self._tts is None:
            try:
                paths.model_path()
            except MissingArtifact as exc:
                lines.append(f"**No usable model weights.** {exc}\n\nRun `kova-tts paths`.")

        found = self.voices()
        if found:
            lines.append(
                f"{len(found)} installed voice{'s' if len(found) > 1 else ''}: "
                f"{', '.join(f'`{name}`' for name in found)}."
            )
        else:
            lines.append(
                "No LoRA voices installed -- set `KOVA_LORA_DIR` to a directory of adapters to "
                "add some. The base voice works without them, and so does cloning."
            )
        return lines


def _default_device() -> str:
    """What torch would pick, named, so the banner can be honest about it."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is a hard dependency; this is belt and braces
        return "cpu"


def build_transcriber(**options: Any) -> Callable[[str], str]:
    """The ASR callable :meth:`KovaTTS.clone` uses when no transcript is typed.

    Built lazily and once, so a visitor who never clones does not wait for a Whisper checkpoint
    to load. A missing ``data`` extra raises something the clone tab can print, rather than an
    ImportError from three frames down.
    """
    loaded: list[Any] = []

    def transcribe(path: str) -> str:
        if not loaded:
            from kova_tts.data.asr import MissingDependency, load_transcriber

            try:
                loaded.append(load_transcriber(**options))
            except MissingDependency as exc:
                raise DemoError(
                    f"{exc}\n\nOr type the transcript into the box and clone again."
                ) from exc
        return loaded[0].transcribe(audio_io.load_audio(path, SAMPLE_RATE), SAMPLE_RATE)

    return transcribe


def load_engine(
    *,
    model: str | None = None,
    codec: str | None = None,
    wavlm: str | None = None,
    lora_dir: str | None = None,
    device: str | None = None,
) -> Any:
    """Build the real engine. Called at most once per process, on the first generation."""
    from kova_tts import KovaTTS

    log.info("Loading the model; this takes a few seconds the first time.")
    return KovaTTS.from_pretrained(
        model,
        codec=codec,
        wavlm=wavlm,
        device=device,
        lora_root=lora_dir,
        transcriber=build_transcriber(),
    )


def warm_up(tts: Any) -> Any:
    """Generate a throwaway sentence, so the first visitor does not pay for the warm-up.

    Loading the weights is only half of a cold start: the first generation also builds the
    codec and captures the CUDA graphs. Measured on one consumer GPU, that is the difference
    between about five seconds to first audio and about a quarter of one.
    """
    log.info("Warming up: one short generation to build the codec and capture CUDA graphs.")
    try:
        tts.generate("Warming up.", params=SamplingParams(max_tokens=256), seed=0)
    except Exception as exc:  # noqa: BLE001 - a warm-up that fails is not worth refusing to serve
        log.warning("Warm-up generation failed, continuing anyway: %s", exc)
    return tts
