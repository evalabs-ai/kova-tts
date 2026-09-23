"""The nodes themselves: load once, generate, clone.

ComfyUI's conventions and nothing else: ``INPUT_TYPES`` declares the widgets, ``RETURN_TYPES``
the sockets, ``FUNCTION`` names the method that runs, ``CATEGORY`` places the node in the menu.
Two custom socket types carry things ComfyUI has no idea about -- ``KOVA_TTS`` is a loaded
engine and ``KOVA_VOICE`` is a :class:`~kova_tts.engine.types.Voice`. Custom types are opaque to
ComfyUI, which is exactly what is wanted: it wires them from node to node without touching them.

Nothing here imports ComfyUI. Nothing here imports torch or transformers at module import time
either -- a node pack that costs four seconds to import is a node pack that makes ComfyUI feel
broken on startup, so the engine is pulled in inside the functions that need it.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE
from kova_tts import TTS_SAMPLING, SamplingParams

from .audio import from_comfy_audio, to_comfy_audio

log = logging.getLogger("kova_tts.comfyui")

#: Where these nodes appear in ComfyUI's node menu.
CATEGORY = "audio/Kova TTS"

#: The loaded engine, keyed by the loader's settings. ComfyUI re-runs a node whenever anything
#: upstream changes, and a 1B model plus a codec is far too expensive to rebuild per execution.
#: One entry only: two engines is two copies of the weights in VRAM, and the second one is
#: usually just the first with a different device typed into it.
_ENGINES: dict[tuple[Any, ...], Any] = {}


def _clean(value: str | None) -> str | None:
    """An empty widget means *unset*, which is not the same as the empty string."""
    text = (value or "").strip()
    return text or None


def _transcriber(**options: Any) -> Callable[[str], str]:
    """The ASR callable :meth:`KovaTTS.clone` falls back on when no transcript is given.

    Loaded on first use, so a graph that never clones -- or always supplies the transcript --
    never pays for a Whisper checkpoint. Without the ``data`` extra this raises a ValueError,
    which ComfyUI shows on the node, rather than an ImportError from inside the engine.
    """
    loaded: list[Any] = []

    def transcribe(path: str) -> str:
        if not loaded:
            from kova_tts.data.asr import MissingDependency, load_transcriber

            try:
                loaded.append(load_transcriber(**options))
            except MissingDependency as exc:
                raise ValueError(f"{exc}") from exc
        from kova_tts.audio import load_audio

        return loaded[0].transcribe(load_audio(path, SAMPLE_RATE), SAMPLE_RATE)

    return transcribe


class KovaTTSLoader:
    """Load the model once and hand it to every node downstream."""

    DESCRIPTION = (
        "Loads the Kova TTS model and keeps it in memory across executions. Leave the paths "
        "empty to use KOVA_* environment variables, or the Hugging Face Hub."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "device": (
                    ["auto", "cuda", "cuda:0", "cuda:1", "mps", "cpu"],
                    {
                        "default": "auto",
                        "tooltip": "auto uses CUDA, then Apple's Metal backend, then the CPU.",
                    },
                ),
                "precision": (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
                "backend": (
                    ["auto", "torch", "mlx"],
                    {
                        "default": "auto",
                        "tooltip": "Decode loop. auto reads it off the checkpoint; mlx needs an "
                        "Apple-converted model and ignores device and precision.",
                    },
                ),
            },
            "optional": {
                "model": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Model directory or Hub repo id. Empty: KOVA_MODEL_PATH.",
                    },
                ),
                "codec": (
                    "STRING",
                    {"default": "", "tooltip": "Codec checkpoint. Empty: KOVA_CODEC_PATH."},
                ),
                "lora_dir": (
                    "STRING",
                    {"default": "", "tooltip": "Directory of LoRA voices. Empty: KOVA_LORA_DIR."},
                ),
            },
        }

    RETURN_TYPES = ("KOVA_TTS",)
    RETURN_NAMES = ("tts",)
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(
        self,
        device: str = "auto",
        precision: str = "bfloat16",
        backend: str = "auto",
        model: str = "",
        codec: str = "",
        lora_dir: str = "",
    ) -> tuple[Any]:
        key = (device, precision, backend, model.strip(), codec.strip(), lora_dir.strip())
        engine = _ENGINES.get(key)
        if engine is None:
            import torch

            from kova_tts import KovaTTS
            from kova_tts.engine import backends

            _ENGINES.clear()
            resolved = backends.resolve(_clean(model), backend)
            log.info("Loading Kova TTS (%s, %s, %s)", resolved, device, precision)
            # device and precision describe the torch loop and have no counterpart in MLX;
            # forwarding them there would only produce a warning per load.
            hardware = (
                {}
                if resolved == "mlx"
                else {
                    "device": None if device == "auto" else device,
                    "dtype": getattr(torch, precision),
                }
            )
            engine = KovaTTS.from_pretrained(
                _clean(model),
                codec=_clean(codec),
                backend=backend,
                lora_root=_clean(lora_dir),
                transcriber=_transcriber(),
                **hardware,
            )
            _ENGINES[key] = engine
        return (engine,)


class KovaTTSGenerate:
    """Text in, an ``AUDIO`` output ComfyUI can preview or save."""

    DESCRIPTION = (
        "Synthesizes text. Connect a cloned voice, or type the name of an installed LoRA "
        "voice; leave both empty for the model's own voice."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "tts": ("KOVA_TTS",),
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": True},
                ),
                "temperature": (
                    "FLOAT",
                    {"default": TTS_SAMPLING.temperature, "min": 0.05, "max": 2.0, "step": 0.05},
                ),
                "top_p": (
                    "FLOAT",
                    {"default": TTS_SAMPLING.top_p, "min": 0.05, "max": 1.0, "step": 0.01},
                ),
                "top_k": (
                    "INT",
                    {
                        "default": TTS_SAMPLING.top_k,
                        "min": 0,
                        "max": 500,
                        "tooltip": "0 disables it.",
                    },
                ),
                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": TTS_SAMPLING.repetition_penalty,
                        "min": 1.0,
                        "max": 2.0,
                        "step": 0.05,
                    },
                ),
                "max_tokens": (
                    "INT",
                    {
                        "default": TTS_SAMPLING.max_tokens,
                        "min": 256,
                        "max": 8192,
                        "step": 64,
                        "tooltip": "80 tokens is one second of audio. Caps a single sentence.",
                    },
                ),
            },
            "optional": {
                "voice": ("KOVA_VOICE",),
                "voice_name": (
                    "STRING",
                    {"default": "", "tooltip": "An installed LoRA voice, when none is connected."},
                ),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = CATEGORY

    def generate(
        self,
        tts: Any,
        text: str,
        seed: int = 0,
        temperature: float = TTS_SAMPLING.temperature,
        top_p: float = TTS_SAMPLING.top_p,
        top_k: int = TTS_SAMPLING.top_k,
        repetition_penalty: float = TTS_SAMPLING.repetition_penalty,
        max_tokens: int = TTS_SAMPLING.max_tokens,
        voice: Any = None,
        voice_name: str = "",
    ) -> tuple[dict[str, Any]]:
        if not (text or "").strip():
            raise ValueError("Nothing to say: the text input is empty.")

        params = SamplingParams(
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            repetition_penalty=float(repetition_penalty),
            max_tokens=int(max_tokens),
        )
        wav = tts.generate(
            text,
            voice if voice is not None else _clean(voice_name),
            params=params,
            seed=int(seed),
        )
        if wav.size == 0:
            raise RuntimeError(
                "The model produced no audio for that text. Rephrase it, or add punctuation so "
                "it has a sentence to work with."
            )
        return (to_comfy_audio(wav, getattr(tts, "sample_rate", OUTPUT_SAMPLE_RATE)),)


class KovaTTSCloneVoice:
    """An ``AUDIO`` input and its transcript, encoded into a voice the generate node can use."""

    DESCRIPTION = (
        "Turns a reference recording into a voice. Five to twenty seconds of clean speech "
        "works best. Leave the transcript empty to transcribe it automatically (needs the "
        "kova-tts 'data' extra)."
    )

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "tts": ("KOVA_TTS",),
                "audio": ("AUDIO",),
                "name": ("STRING", {"default": "cloned"}),
            },
            "optional": {
                "transcript": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "dynamicPrompts": False,
                        "tooltip": "Exactly what the recording says. Empty transcribes it.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("KOVA_VOICE",)
    RETURN_NAMES = ("voice",)
    FUNCTION = "clone"
    CATEGORY = CATEGORY

    def clone(
        self,
        tts: Any,
        audio: dict[str, Any],
        name: str = "cloned",
        transcript: str = "",
    ) -> tuple[Any]:
        # The encoder takes 32 kHz, whatever rate the model speaks at.
        wav = from_comfy_audio(audio, SAMPLE_RATE)
        text = _clean(transcript)
        voice_name = _clean(name) or "cloned"

        if text is not None:
            return (tts.clone(wav, text, name=voice_name),)

        if getattr(tts, "transcriber", None) is None:
            raise ValueError(
                "This engine has no transcriber, so cloning needs the transcript: type what the "
                "recording says into the transcript box."
            )
        # A transcriber reads a file, and the audio arrived as a tensor. Writing it out is the
        # whole cost of not making the user type the transcript, and the file goes away again.
        with tempfile.TemporaryDirectory(prefix="kova-clone-") as directory:
            reference = Path(directory) / "reference.wav"
            tts.save(wav, reference)
            return (tts.clone(reference, name=voice_name),)


#: What ComfyUI imports. The keys are the node ids saved inside a workflow file, so they carry
#: the project name and must never change; the display names are free to.
NODE_CLASS_MAPPINGS = {
    "KovaTTSLoader": KovaTTSLoader,
    "KovaTTSGenerate": KovaTTSGenerate,
    "KovaTTSCloneVoice": KovaTTSCloneVoice,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KovaTTSLoader": "Kova TTS Loader",
    "KovaTTSGenerate": "Kova TTS Generate",
    "KovaTTSCloneVoice": "Kova TTS Clone Voice",
}

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "KovaTTSCloneVoice",
    "KovaTTSGenerate",
    "KovaTTSLoader",
]
