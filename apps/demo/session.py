"""The state behind the page: one engine, one lock, and the voices cloned this session.

The engine is not reentrant and neither is this page. One :class:`Generator`, one KV cache: a
second overlapping request raises. A single lock guards every path that touches the model --
the streaming endpoint and the clone panel alike -- and losing the race is a sentence the page
prints, not a traceback.

The engine loads lazily, on the first generation rather than at import, so the page comes up on
a machine with no weights configured and says what to fix.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import os
import random
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, NamedTuple

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

#: Picker values for zero-shot presets carry this prefix, so a preset can never collide with a
#: LoRA or a clone of the same name.
PRESET_PREFIX = "preset:"

#: The bundled zero-shot presets: audio files plus one manifest naming each file's transcript.
DEFAULT_ZERO_SHOT_DIR = Path(__file__).with_name("zero_shot_voices")
PRESET_MANIFEST = "metadata.csv"

_AUDIO_SUFFIXES = (".flac", ".wav", ".ogg", ".opus", ".mp3")

#: The published LoRA voices, offered under professional cloning when nothing else is configured.
DEFAULT_VOICES_REPO = "kova-ai/kova-tts-1-voices"
#: All the demo needs from a voices repo: each adapter, not the licences and README beside them.
_ADAPTER_FILES = ["*/adapter_config.json", "*/adapter_model.safetensors"]

#: Where a voice comes from. The prompt box picks one of these first, then a voice within it.
#: In order of the effort behind the voice; professional cloning is a trained LoRA adapter.
SOURCE_BASE = "Base model"
SOURCE_PRESET = "Zero-shot preset"
SOURCE_CLONE = "Your recording"
SOURCE_LORA = "Professional cloning"
SOURCES = (SOURCE_BASE, SOURCE_PRESET, SOURCE_CLONE, SOURCE_LORA)


class Preset(NamedTuple):
    """One zero-shot preset: its reference clip, the clip's exact transcript, and how to list it."""

    audio: Path
    transcript: str
    gender: str = ""
    #: A few words that set the voice apart, e.g. ``("British", "gravelly", "storyteller")``.
    tags: tuple[str, ...] = ()
    #: 0-10, best first in the picker. ``None`` sorts after every rated preset.
    rating: int | None = None


def resolve_lora_dir(value: str | None) -> str | None:
    """The LoRA directory to serve: `value`, else ``KOVA_LORA_DIR``, else the published voices.

    A Hub repo id is downloaded (adapters only) and its local snapshot returned; ``''`` means no
    LoRA voices at all. A download that fails leaves professional cloning empty rather than
    stopping the page: the other three sources still work.
    """
    if value is None:
        paths.load_dotenv()
        value = os.environ.get(paths.ENV_LORA_DIR, "").strip() or DEFAULT_VOICES_REPO
    if not value:
        return None
    if value.startswith(("/", "~", ".")) or Path(value).expanduser().exists():
        return value
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(value, allow_patterns=_ADAPTER_FILES)
    except Exception as exc:  # noqa: BLE001 - no LoRA voices is a smaller demo, not a broken one
        log.warning("Could not download LoRA voices from %s: %s", value, exc)
        return None


def load_presets(directory: str | os.PathLike[str] | None) -> dict[str, Preset]:
    """``{preset id: Preset}`` from ``<directory>/metadata.csv``, best-rated first.

    The CSV needs a ``file_name`` and a ``text`` column. ``gender``, ``tags`` (``;``-separated)
    and ``rating`` (0-10) are optional and only change how the picker lists the preset; any other
    column rides along unread. The id is the file's stem. A row with no text, or whose audio file
    is missing, is skipped rather than guessed at: a zero-shot reference whose text does not
    match the audio garbles everything generated from it.
    """
    if directory is None:
        return {}
    manifest = Path(directory) / PRESET_MANIFEST
    if not manifest.is_file():
        return {}
    presets = {}
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            audio = Path(directory) / (row.get("file_name") or "").strip()
            transcript = (row.get("text") or "").strip()
            if transcript and audio.suffix.lower() in _AUDIO_SUFFIXES and audio.is_file():
                rating = (row.get("rating") or "").strip()
                presets[audio.stem] = Preset(
                    audio,
                    transcript,
                    gender=(row.get("gender") or "").strip(),
                    tags=tuple(t.strip() for t in (row.get("tags") or "").split(";") if t.strip()),
                    rating=int(float(rating)) if rating else None,
                )

    def best_first(item: tuple[str, Preset]) -> tuple[int, str]:
        rating = item[1].rating
        return (-(rating if rating is not None else -1), item[0])

    return dict(sorted(presets.items(), key=best_first))


#: Seconds a second request waits for the model before it is refused. The overwhelmingly common
#: collision is one person pressing Generate again: the previous request is abandoned a moment
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
        backend: Likewise -- which decode loop the banner should name.
        zero_shot_dir: Directory of zero-shot presets (see :func:`load_presets`). ``None``
            offers none.
        transcriber: ``path -> text`` for the clone panel's automatic transcript. ``None`` falls
            back to the engine's own transcriber, which means loading the engine first.
    """

    def __init__(
        self,
        tts: Any = None,
        *,
        loader: Callable[[], Any] | None = None,
        lora_root: str | os.PathLike[str] | None = None,
        device: str | None = None,
        backend: str | None = None,
        zero_shot_dir: str | os.PathLike[str] | None = None,
        transcriber: Callable[[str], str] | None = None,
    ) -> None:
        self._tts = tts
        self._loader = loader
        self._lora_root = lora_root
        self._device = device
        self._backend = backend
        self._transcriber = transcriber
        self._cloned: dict[str, Voice] = {}
        self._presets = load_presets(zero_shot_dir)
        self.zero_shot_dir = Path(zero_shot_dir) if self._presets else None
        # A preset is encoded into a Voice the first time it is used, then kept.
        self._preset_voices: dict[str, Voice] = {}
        # Non-reentrant engine, non-reentrant page: the streaming endpoint and the clone panel
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

    def choices(self, source: str | None = None) -> list[tuple[str, str]]:
        """``(label, value)`` pairs for the voice picker.

        With no `source`, every voice: base, presets, clones, then LoRAs. With one of
        :data:`SOURCES`, only that kind -- which is how the prompt box fills its picker.
        """
        groups = {
            SOURCE_BASE: [(BASE_LABEL, BASE_VOICE)],
            SOURCE_PRESET: [(self.preset_label(pid), PRESET_PREFIX + pid) for pid in self._presets],
            SOURCE_CLONE: [(f"{name} (cloned)", name) for name in sorted(self._cloned)],
            SOURCE_LORA: [(name, name) for name in self.voices()],
        }
        if source is not None:
            return groups[source]
        return [pair for group in groups.values() for pair in group]

    def sources(self) -> list[str]:
        """The voice sources worth offering here: all of them, bar presets when none are set up.

        Cloning is always possible, and professional cloning stays on offer with no adapters
        installed -- picking it says how to add one.
        """
        return [s for s in SOURCES if s != SOURCE_PRESET or self._presets]

    def source_of(self, name: str | None) -> str:
        """Which of :data:`SOURCES` a picker value belongs to."""
        if not name:
            return SOURCE_BASE
        if name.startswith(PRESET_PREFIX):
            return SOURCE_PRESET
        if name in self._cloned:
            return SOURCE_CLONE
        return SOURCE_LORA

    def preset_label(self, preset_id: str) -> str:
        """``07 · Female · British, crisp, storyteller`` when the manifest tags the preset.

        Untagged, the transcript is what tells presets apart: ``voice_07 · "The train
        station smelled…"``. The rating only orders the list; it is never shown.
        """
        preset = self._presets[preset_id]
        if not (preset.gender or preset.tags):
            text = preset.transcript
            short = text if len(text) <= 48 else text[:47].rsplit(" ", 1)[0] + "…"
            return f"{preset_id} · “{short}”"
        number = preset_id.rsplit("_", 1)[-1]
        parts = [number, preset.gender.capitalize(), ", ".join(preset.tags)]
        return " · ".join(p for p in parts if p)

    def preset_reference(self, name: str | None) -> tuple[str, str] | None:
        """``(audio path, transcript)`` behind a preset picker value, for the preview player."""
        if not name or not name.startswith(PRESET_PREFIX):
            return None
        found = self._presets.get(name.removeprefix(PRESET_PREFIX))
        return (str(found.audio), found.transcript) if found else None

    def resolve(self, name: str | None) -> str | Voice | None:
        """The voice picker's value as the engine wants it: a clone, a LoRA name, or ``None``.

        A preset is encoded on its first use, which needs the engine -- so this is called only
        from inside the model lock, as :meth:`frames` is.
        """
        if not name:
            return None
        if name.startswith(PRESET_PREFIX):
            return self._preset_voice(name.removeprefix(PRESET_PREFIX))
        return self._cloned.get(name, name)

    def _preset_voice(self, preset_id: str) -> Voice:
        if preset_id not in self._preset_voices:
            if preset_id not in self._presets:
                raise server_errors.InvalidRequest(
                    f"voice: no zero-shot preset named {preset_id!r}"
                )
            preset = self._presets[preset_id]
            self._preset_voices[preset_id] = self.engine().clone(
                str(preset.audio), preset.transcript, name=PRESET_PREFIX + preset_id
            )
        return self._preset_voices[preset_id]

    def preset(self, name: str | None = None) -> SamplingParams:
        """The sampling preset a voice calls for.

        The two presets currently differ only in token budget -- cloning is given a longer one,
        because a cloned generation carries the reference through the same budget -- but the
        picker drives the Advanced sliders through this either way, so what the sliders show is
        always what the engine would have chosen on its own. A zero-shot preset is a clone.
        """
        cloning = bool(name) and (name in self._cloned or name.startswith(PRESET_PREFIX))
        return CLONE_SAMPLING if cloning else TTS_SAMPLING

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

        Off the event loop, because this is a threading lock shared with the clone panel, which
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
    ) -> Iterator[Any]:
        """Audio frames for `text`, in this session's voice namespace, with word timestamps
        whenever the engine has its aligner.

        Blocking, and it loads the model on the first call, so callers on an event loop run it
        in a thread. The iterator itself is lazy: nothing is generated until it is advanced.
        """
        engine = self.engine()
        timestamps = getattr(engine, "aligner", None) is not None
        return engine.stream(text, self.resolve(voice), params=params, timestamps=timestamps)

    # ---------------------------------------------------------------------------- callbacks

    def transcribe_reference(self, reference: str | None) -> tuple[str, str | None]:
        """``(status, transcript)`` for a recording that just arrived on the clone panel.

        The transcript is ``None`` when there is nothing to put in the box -- no clip, or ASR
        could not run -- and the status says why, so a typed transcript is never wiped.
        """
        if not reference:
            return "", None
        if not Path(reference).is_file():
            upload_dir = os.environ.get("GRADIO_TEMP_DIR", "/tmp/gradio")
            return (
                "The recording did not reach the server. Record or upload it again; if this "
                "keeps happening, check that the demo can write to its upload directory "
                f"(`GRADIO_TEMP_DIR`, currently `{upload_dir}`).",
                None,
            )
        transcriber = self._transcriber or getattr(self._tts, "transcriber", None)
        if transcriber is None:
            return "No transcriber here -- type exactly what the recording says.", None
        try:
            text = transcriber(reference)
        except DemoError as exc:
            return str(exc), None
        except Exception as exc:
            log.exception("Transcribing the reference failed")
            return f"Automatic transcription failed ({exc}); type the transcript instead.", None
        if not text:
            return "No speech was heard in that recording. Try again, a little closer.", None
        return (
            "Transcribed automatically. Check it matches the recording word for word, then "
            "press **Clone**.",
            text,
        )

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

        if not transcript and self._transcriber is not None:
            # Outside the model lock: ASR is its own model and need not wait for a generation.
            status, heard = self.transcribe_reference(reference)
            if heard is None:
                return status, None
            transcript, auto = heard, True
        else:
            auto = not transcript

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
        heard = "transcribed automatically" if auto else "as typed"
        return (
            f"Cloned **{voice.name}** from {used:.1f} s of audio, and it is selected -- press "
            f"Generate to hear it.\n\nReference transcript ({heard}): “{voice.ref_text}”",
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

    def summary(self) -> str:
        """The header's one-line status, e.g. ``cuda:0 · torch · 52 voices``.

        The count is every ready-made voice: installed LoRAs plus zero-shot presets.
        """
        found = len(self.voices()) + len(self._presets)
        voices = f"{found} voice{'s' if found != 1 else ''}"
        parts = [self._device or _default_device(), self._backend, voices]
        return " · ".join(part for part in parts if part)

    def warnings(self) -> list[str]:
        """The banner lines that need acting on: a slow device, or weights that are not there.

        Everything here is cheap and offline. Resolving the codec would download it from the
        Hub, which is not something a page load should do.
        """
        lines: list[str] = []
        device = self._device or _default_device()
        if device.startswith("cpu"):
            lines.append(
                "**Running on the CPU.** Generation will work, but a sentence takes minutes "
                "rather than seconds; a CUDA device or Apple Silicon is strongly recommended."
            )
        elif device.startswith("mps") and self._backend != "mlx":
            lines.append(
                "**Running the torch backend on Metal.** It works, but an MLX checkpoint runs "
                "a lot faster; an official one is coming soon. See `docs/apple-silicon.md`."
            )

        if self._tts is None:
            try:
                paths.model_path()
            except MissingArtifact as exc:
                lines.append(f"**No usable model weights.** {exc}\n\nRun `kova-tts paths`.")
        return lines

    def notices(self) -> list[str]:
        """Everything worth knowing about this machine: :meth:`warnings`, then the routine lines."""
        lines = self.warnings()
        device = self._device or _default_device()
        slow = device.startswith("cpu") or (device.startswith("mps") and self._backend != "mlx")
        if not slow:
            via = f" via the {self._backend} backend" if self._backend else ""
            lines.insert(0, f"Running on `{device}`{via}.")

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
        if self._presets:
            lines.append(
                f"{len(self._presets)} zero-shot preset voices, each a reference clip with its "
                f"transcript -- pick **{SOURCE_PRESET}** to use one without recording anything."
            )
        return lines


def _default_device() -> str:
    """What torch would pick, named, so the banner can be honest about it."""
    try:
        from kova_codec.devices import default_device

        return default_device().type
    except Exception:  # pragma: no cover - torch is a hard dependency; this is belt and braces
        return "cpu"


#: The recogniser behind automatic transcripts: NVIDIA Parakeet TDT, run through transformers
#: (a base dependency). Punctuated and cased, which matters here -- the transcript becomes the
#: text the cloned voice is conditioned on.
PARAKEET_MODEL = os.environ.get("KOVA_DEMO_ASR_MODEL", "nvidia/parakeet-tdt-0.6b-v3")

#: Parakeet's input rate.
ASR_SAMPLE_RATE = 16_000


def load_parakeet(model: str = PARAKEET_MODEL, device: str | None = None) -> Any:
    """A transformers ASR pipeline for `model`, on `device` (default: CUDA when there is one)."""
    import torch
    from transformers import pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Loading %s on %s for automatic transcripts.", model, device)
    return pipeline("automatic-speech-recognition", model=model, device=device)


def build_transcriber(
    *, model: str = PARAKEET_MODEL, device: str | None = None
) -> Callable[[str], str]:
    """``path -> transcript``: what the clone panel fills in, and what cloning uses when no
    transcript is typed.

    Built lazily and once, so a visitor who never clones does not wait for the checkpoint. A
    missing dependency raises something the clone panel can print, rather than an ImportError from
    three frames down. Thread-safe: Gradio may call it from two workers at once.
    """
    loaded: list[Any] = []
    lock = threading.Lock()

    def transcribe(path: str) -> str:
        with lock:
            if not loaded:
                try:
                    loaded.append(load_parakeet(model, device))
                except ImportError as exc:
                    raise DemoError(
                        f"Automatic transcription could not start: {exc}\n\nInstall the demo "
                        f"extra (`uv sync --extra demo`), or type the transcript into the box."
                    ) from exc
            wav = audio_io.load_audio(path, ASR_SAMPLE_RATE)
            result = loaded[0]({"raw": wav, "sampling_rate": ASR_SAMPLE_RATE})
        return str(result["text"]).strip()

    return transcribe


def load_engine(
    *,
    model: str | None = None,
    codec: str | None = None,
    wavlm: str | None = None,
    lora_dir: str | None = None,
    device: str | None = None,
    backend: str | None = None,
    decode_window: int | None = None,
    transcriber: Callable[[str], str] | None = None,
) -> Any:
    """Build the real engine. Called at most once per process, on the first generation.

    Pass the session's `transcriber` so the clone panel and the engine share one ASR model.
    """
    from kova_tts import KovaTTS

    log.info("Loading the model; this takes a few seconds the first time.")
    return KovaTTS.from_pretrained(
        model,
        codec=codec,
        wavlm=wavlm,
        backend=backend,
        device=device,
        lora_root=lora_dir,
        transcriber=transcriber or build_transcriber(device=device),
        # Omitted rather than passed as None, so the engine keeps owning the default.
        **({} if decode_window is None else {"decode_window": decode_window}),
    )


def warm_up(tts: Any) -> Any:
    """Generate a throwaway sentence, so the first visitor does not pay for the warm-up.

    Loading the weights is only half of a cold start: the first generation also builds the
    codec and captures the CUDA graphs. Measured on one consumer GPU, that is the difference
    between about five seconds to first audio and about a quarter of one.
    """
    log.info("Warming up: one short generation to build the codec and capture CUDA graphs.")
    try:
        tts.generate("Warming up.", params=SamplingParams(max_tokens=256))
    except Exception as exc:  # noqa: BLE001 - a warm-up that fails is not worth refusing to serve
        log.warning("Warm-up generation failed, continuing anyway: %s", exc)
    return tts
