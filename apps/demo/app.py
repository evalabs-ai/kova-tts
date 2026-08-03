"""The Gradio demo: for most people, the first time they hear this model.

Run it with ``kova-tts demo``, or directly::

    uv run python apps/demo/app.py --port 7860

Two things shape everything here.

**Audio has to start before generation finishes.** :meth:`KovaTTS.stream` yields ~390 ms frames
as the codec produces them, and a Gradio generator function feeding a ``streaming=True`` audio
output turns those frames into an HLS stream the browser plays while the rest is still being
written. So the callbacks in this file are generators from top to bottom; nothing waits for a
whole clip. A second, plain player receives the finished waveform at the end, because a
streaming player is for hearing and a normal one is for scrubbing and downloading.

**The engine is not reentrant and neither is this page.** One :class:`Generator`, one KV cache:
a second overlapping request raises. Gradio will happily fire concurrent events -- two browser
tabs, or a clone started while a generation runs -- so a single lock guards every path that
touches the model, and losing the race produces a sentence in the status line instead of a
traceback.

The engine loads lazily, on the first generation rather than at import, so the page comes up on
a machine with no weights configured and says what to fix.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np

from kova_codec.constants import SAMPLE_RATE
from kova_tts import CLONE_SAMPLING, TTS_SAMPLING, MissingArtifact, SamplingParams, Voice, paths
from kova_tts import audio as audio_io
from kova_tts.voices import MAX_REFERENCE_SECONDS, MIN_REFERENCE_SECONDS

log = logging.getLogger("kova_tts.demo")

TITLE = "Kova TTS"
TAGLINE = "Expressive speech that starts playing before it has finished generating."

#: Value of the voice picker for "the model's own voice". Empty rather than ``None`` because a
#: Gradio dropdown treats ``None`` as *nothing selected* and refuses to show a label for it.
BASE_VOICE = ""
BASE_LABEL = "Base voice (no LoRA)"

#: Longest text the demo will accept in one press. Not a model limit -- long text is split into
#: sentences and generated segment by segment -- but a public demo should not let one visitor
#: hold the only generator for ten minutes.
MAX_CHARS = 1200

#: Below this peak amplitude a reference recording is silence, and saying so is kinder than
#: letting the codec return no codes and reporting that.
SILENT_PEAK = 1e-3

#: Prompts written for this demo. Each one is here to show something: a held pause, a change of
#: register, digits and units read aloud, and a long passage whose later sentences are still
#: being generated while the first are playing.
EXAMPLES = [
    "The kettle clicked off, and for a moment the whole kitchen was completely quiet.",
    "Wait. You're telling me the entire thing runs on one graphics card? That cannot be right.",
    "Take the second left, carry on for about four hundred metres, and if you reach the bridge, "
    "you have gone too far.",
    "Speech comes back in small pieces, eighty of them a second, so the first words are already "
    "playing while the last ones are still being written.",
    "Here is the part I find strange. It does not plan the sentence before it starts talking. "
    "It writes the sound one fragment at a time, left to right, and somehow the pauses still "
    "land where a person would put them, and the question at the end still rises.",
]

READY = "Ready when you are."

BUSY = (
    "The model is already speaking. It generates one clip at a time -- give it a moment and "
    "press Speak again."
)

CLONE_HELP = """\
Upload or record **five to twenty seconds** of clean speech, and it becomes a voice you can use
on the Speak tab straight away. Nothing is saved to disk: the clone lives in this session only.

A transcript is optional. Left empty, the recording is transcribed automatically, which needs
the `data` extra (`uv sync --extra data`). Typed in, it has to match the recording word for
word -- cloning continues the reference, so a wrong transcript garbles the output.
"""

CSS = """
.kova-status { min-height: 1.6em; font-variant-numeric: tabular-nums; }
footer { display: none !important; }
"""


class DemoError(RuntimeError):
    """Something the person at the browser can fix, phrased for them rather than for a log."""


class _Clear:
    """Sentinel for an output that should be emptied.

    A callback yields three values per step and usually wants to touch only one of them, so
    ``None`` has to mean *leave this alone*; that leaves nothing to say *empty this player*.
    :func:`_updates` turns this sentinel into Gradio's ``None`` and ``None`` into a no-op
    update, which keeps every Gradio idiom out of :class:`DemoSession`.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "CLEAR"


CLEAR = _Clear()


# --------------------------------------------------------------------------------- the session


class DemoSession:
    """Everything the page needs: one engine, one lock, and the voices cloned this session.

    Args:
        tts: An already-built :class:`~kova_tts.engine.tts.KovaTTS`. Tests inject a fake here;
            leaving it ``None`` means the real one is built by `loader` on first use.
        loader: ``callable() -> KovaTTS``, called at most once. Deferred so that a missing
            checkpoint becomes a message under the Speak button rather than a traceback before
            the browser has ever connected.
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
        # Non-reentrant engine, non-reentrant page: every callback that reaches the model takes
        # this without blocking, so a concurrent request is answered instead of queued forever.
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

        Cloning is hotter and barely repetition-penalised; plain synthesis is the opposite. The
        picker drives the Advanced sliders through this, so what the sliders show is always what
        the engine would have chosen on its own.
        """
        return CLONE_SAMPLING if name and name in self._cloned else TTS_SAMPLING

    # ---------------------------------------------------------------------------- callbacks

    def speak(
        self,
        text: str,
        voice: str = BASE_VOICE,
        temperature: float = TTS_SAMPLING.temperature,
        top_p: float = TTS_SAMPLING.top_p,
        top_k: int = TTS_SAMPLING.top_k,
        repetition_penalty: float = TTS_SAMPLING.repetition_penalty,
        max_tokens: int = TTS_SAMPLING.max_tokens,
        seed: float | int | None = -1,
    ) -> Iterator[tuple[Any, Any, str]]:
        """Synthesize `text`, yielding ``(stream chunk, finished clip, status)`` as it goes.

        The argument order is the order of the controls in :func:`build_ui`; Gradio calls this
        positionally. Nothing here raises: every failure becomes a final status line, because a
        traceback in the terminal is invisible to whoever is looking at the page.
        """
        text = (text or "").strip()
        if not text:
            yield CLEAR, CLEAR, "Type something for the model to say, then press Speak."
            return
        if len(text) > MAX_CHARS:
            yield (
                CLEAR,
                CLEAR,
                f"That is {len(text):,} characters; this demo generates up to {MAX_CHARS:,} at a "
                f"time. Trim it, or use the Python API for a long passage.",
            )
            return

        try:
            params = SamplingParams(
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=int(top_k),
                repetition_penalty=float(repetition_penalty),
                max_tokens=int(max_tokens),
            )
        except (ValueError, TypeError) as exc:
            yield CLEAR, CLEAR, f"Those sampling settings will not work: {exc}"
            return

        if not self._lock.acquire(blocking=False):
            yield None, None, BUSY
            return
        try:
            yield CLEAR, CLEAR, "Loading the model..." if not self.loaded else "Generating..."
            chosen = _seed(seed)
            tts = self.engine()

            pieces: list[np.ndarray] = []
            started = time.perf_counter()
            first: float | None = None
            rate = SAMPLE_RATE
            for frame in tts.stream(text, self.resolve(voice), params=params, seed=chosen):
                if not frame.samples.size:
                    continue
                rate = frame.sample_rate
                if first is None:
                    first = time.perf_counter() - started
                pieces.append(frame.samples)
                spoken = sum(piece.size for piece in pieces) / rate
                yield (rate, _pcm16(frame.samples)), None, _progress(first, spoken)

            if not pieces:
                yield (
                    CLEAR,
                    CLEAR,
                    (
                        "The model produced no audio for that text. Try rephrasing it, or add some "
                        "punctuation so it has a sentence to work with."
                    ),
                )
                return
            wav = np.concatenate(pieces)
            yield (
                None,
                (rate, _pcm16(wav)),
                _summary(first or 0.0, time.perf_counter() - started, wav.size / rate, chosen),
            )
        except DemoError as exc:
            yield CLEAR, CLEAR, str(exc)
        except RuntimeError as exc:
            # The generator refuses to interleave two requests. The lock above should make this
            # unreachable from the page, but a server sharing the same engine can still win.
            log.warning("Generation failed: %s", exc)
            yield CLEAR, CLEAR, BUSY if "already running" in str(exc) else f"That failed: {exc}"
        except Exception as exc:
            log.exception("Generation failed")
            yield CLEAR, CLEAR, f"Generation failed: {exc}"
        finally:
            self._lock.release()

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


# --------------------------------------------------------------------------------- formatting


def _pcm16(wav: np.ndarray) -> np.ndarray:
    """Waveform to 16-bit PCM, which is what the browser is going to play anyway.

    Gradio converts float32 itself, but warns while doing it, once per chunk -- and a streamed
    generation is one chunk every 390 ms.
    """
    return (np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)


def _seed(value: float | int | None) -> int:
    """A concrete seed. Negative or missing means draw one, so every result is reproducible."""
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = -1
    return random.randrange(2**31 - 1) if number < 0 else number


def _progress(first: float | None, spoken: float) -> str:
    ttfa = f"first audio in **{first:.2f} s** · " if first is not None else ""
    return f"{ttfa}{spoken:.1f} s generated..."


def _summary(first: float, elapsed: float, spoken: float, seed: int) -> str:
    speed = spoken / elapsed if elapsed > 0 else 0.0
    return (
        f"First audio in **{first:.2f} s** · {spoken:.1f} s of speech in {elapsed:.1f} s "
        f"({speed:.1f}× real time) · seed `{seed}`"
    )


def _updates(values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Translate a callback's sentinels into what Gradio expects.

    ``CLEAR`` empties a component, ``None`` leaves it exactly as it is -- which is what keeps a
    finished stream audible while the full clip appears beside it.
    """
    return tuple(
        None if value is CLEAR else gr.update() if value is None else value for value in values
    )


# ------------------------------------------------------------------------------------ the page


def build_ui(session: DemoSession | None = None, *, title: str = TITLE) -> gr.Blocks:
    """Construct the interface, without launching it.

    Args:
        session: The state the callbacks run against. ``None`` builds one that loads the real
            engine from the environment on first use.
        title: Browser tab title and page heading.

    Returns:
        A :class:`gradio.Blocks` ready for ``.launch()``, or for a test to inspect.

    The theme and stylesheet are not set here: Gradio 6 moved both to ``launch()``, so
    :func:`main` applies them and a caller mounting these Blocks elsewhere picks its own.
    """
    session = session or DemoSession(loader=load_engine)

    # Analytics off: this runs on someone else's machine, against their weights, and Gradio's
    # default is to phone home on launch and on error.
    with gr.Blocks(title=title, fill_width=False, analytics_enabled=False) as ui:
        gr.Markdown(f"# {title}\n{TAGLINE}")
        gr.Markdown("\n\n".join(f"- {line}" for line in session.notices()))

        with gr.Tabs() as tabs:
            with gr.Tab("Speak", id="speak"):
                with gr.Row():
                    with gr.Column(scale=3):
                        text = gr.Textbox(
                            label="Text",
                            lines=4,
                            max_lines=14,
                            autofocus=True,
                            placeholder="Say something...",
                        )
                    with gr.Column(scale=2):
                        voice = gr.Dropdown(
                            choices=session.choices(),
                            value=BASE_VOICE,
                            label="Voice",
                            info="LoRA voices installed here, plus anything you clone.",
                        )
                        seed = gr.Number(
                            value=-1,
                            precision=0,
                            label="Seed",
                            info="-1 draws a new one; the seed used is printed below.",
                        )
                        with gr.Row():
                            speak_button = gr.Button("Speak", variant="primary", scale=3)
                            stop_button = gr.Button("Stop", variant="stop", scale=1)

                gr.Examples(
                    examples=[[prompt] for prompt in EXAMPLES],
                    inputs=[text],
                    label="Or try one of these",
                )

                with gr.Accordion("Advanced", open=False):
                    gr.Markdown(
                        "These start at the preset for the selected voice, which is what the "
                        "model was tuned with. Cloned voices use a hotter, less penalised "
                        "preset than plain synthesis -- switching voice resets them."
                    )
                    with gr.Row():
                        temperature = gr.Slider(
                            0.1, 1.5, TTS_SAMPLING.temperature, step=0.05, label="Temperature"
                        )
                        top_p = gr.Slider(0.05, 1.0, TTS_SAMPLING.top_p, step=0.01, label="Top-p")
                    with gr.Row():
                        top_k = gr.Slider(
                            0, 200, TTS_SAMPLING.top_k, step=1, label="Top-k", info="0 turns it off"
                        )
                        repetition_penalty = gr.Slider(
                            1.0,
                            2.0,
                            TTS_SAMPLING.repetition_penalty,
                            step=0.05,
                            label="Repetition penalty",
                        )
                    max_tokens = gr.Slider(
                        256,
                        4096,
                        TTS_SAMPLING.max_tokens,
                        step=64,
                        label="Token budget",
                        info="80 tokens is one second of audio; this caps a single sentence.",
                    )

                stream_audio = gr.Audio(
                    label="Streaming",
                    streaming=True,
                    autoplay=True,
                    interactive=False,
                )
                full_audio = gr.Audio(label="Finished clip", interactive=False)
                status = gr.Markdown(READY, elem_classes=["kova-status"])

            with gr.Tab("Clone a voice", id="clone"):
                gr.Markdown(CLONE_HELP)
                with gr.Row():
                    with gr.Column():
                        reference = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Reference recording",
                        )
                        clone_name = gr.Textbox(
                            label="Name this voice",
                            placeholder="taken from the filename if you leave it empty",
                        )
                    with gr.Column():
                        clone_transcript = gr.Textbox(
                            label="Transcript (optional)",
                            lines=4,
                            placeholder="Exactly what the recording says.",
                        )
                        clone_button = gr.Button("Clone this voice", variant="primary")
                clone_status = gr.Markdown("", elem_classes=["kova-status"])

        # ----------------------------------------------------------------------- behaviour

        def on_speak(*values: Any) -> Iterator[tuple[Any, ...]]:
            for step in session.speak(*values):
                yield _updates(step)

        def on_voice_change(name: str) -> tuple[float, float, int, float, int]:
            preset = session.preset(name)
            return (
                preset.temperature,
                preset.top_p,
                preset.top_k,
                preset.repetition_penalty,
                preset.max_tokens,
            )

        def on_clone(*values: Any) -> tuple[Any, ...]:
            message, cloned = session.clone_voice(*values)
            if cloned is None:
                return message, gr.update(), gr.update()
            # Land the user where the new voice is usable, already selected.
            return (
                message,
                gr.update(choices=session.choices(), value=cloned),
                gr.Tabs(selected="speak"),
            )

        controls = [text, voice, temperature, top_p, top_k, repetition_penalty, max_tokens, seed]
        outputs = [stream_audio, full_audio, status]
        events = [
            speak_button.click(on_speak, controls, outputs, concurrency_limit=1),
            text.submit(on_speak, controls, outputs, concurrency_limit=1),
        ]
        stop_button.click(lambda: READY, None, status, cancels=events)
        voice.change(
            on_voice_change,
            voice,
            [temperature, top_p, top_k, repetition_penalty, max_tokens],
        )
        clone_button.click(
            on_clone,
            [reference, clone_transcript, clone_name],
            [clone_status, voice, tabs],
            concurrency_limit=1,
        )

    # One generation at a time, and one queue position per visitor: the engine has a single KV
    # cache, so a higher limit would only convert waiting into failing.
    ui.queue(default_concurrency_limit=1)
    return ui


# ---------------------------------------------------------------------------------- the engine


def _default_device() -> str:
    """What torch would pick, named, so the banner can be honest about it."""
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover - torch is a hard dependency; this is belt and braces
        return "cpu"


def build_transcriber(**options: Any) -> Callable[[str], str]:
    """The ASR callable :meth:`KovaTTS.clone` uses when no transcript is typed.

    Built lazily and once: a visitor who never clones, or who always types the transcript, does
    not wait for a Whisper checkpoint to load. When the ``data`` extra is missing this raises
    something the clone tab can print, rather than an ImportError from three frames down.
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
    between about five seconds to first audio and about a quarter of one. One short sentence
    buys all of it.
    """
    log.info("Warming up: one short generation to build the codec and capture CUDA graphs.")
    try:
        tts.generate("Warming up.", params=SamplingParams(max_tokens=256), seed=0)
    except Exception as exc:  # noqa: BLE001 - a warm-up that fails is not worth refusing to serve
        log.warning("Warm-up generation failed, continuing anyway: %s", exc)
    return tts


# ------------------------------------------------------------------------------------- the CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts demo",
        description="Launch the Kova TTS demo in a browser.",
    )
    parser.add_argument(
        "--host", default="127.0.0.1", help="interface to bind (default: %(default)s)"
    )
    parser.add_argument("--port", type=int, default=7860, help="port (default: %(default)s)")
    parser.add_argument("--share", action="store_true", help="expose a public gradio.live link")
    parser.add_argument("--open", action="store_true", help="open a browser window on startup")
    parser.add_argument(
        "--preload",
        action="store_true",
        help="load and warm up the model at startup, so the first visitor waits for none of it",
    )
    parser.add_argument("--model", default=None, help="model directory or Hub repo id")
    parser.add_argument("--codec", default=None, help="codec checkpoint")
    parser.add_argument("--wavlm", default=None, help="WavLM directory or Hub repo id")
    parser.add_argument("--lora-dir", default=None, help="directory of LoRA voices")
    parser.add_argument("--device", default=None, help="torch device, e.g. cuda:1")
    parser.add_argument("-v", "--verbose", action="store_true", help="log what the engine is doing")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch the demo. Blocks until the server is stopped."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    session = DemoSession(
        loader=lambda: load_engine(
            model=args.model,
            codec=args.codec,
            wavlm=args.wavlm,
            lora_dir=args.lora_dir,
            device=args.device,
        ),
        lora_root=args.lora_dir,
        device=args.device,
    )
    if args.preload:
        try:
            warm_up(session.engine())
        except DemoError as exc:
            # Not fatal: the page is still worth serving, and it will say the same thing.
            print(f"warning: {exc}", file=sys.stderr)

    ui = build_ui(session)
    ui.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.open,
        show_error=True,
        theme=gr.themes.Soft(),
        css=CSS,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
