"""Kova TTS: expressive text-to-speech with voice cloning and LoRA finetuning.

The engine is imported lazily. ``from kova_tts import tts_prompt`` costs a plain module import;
naming :class:`~kova_tts.engine.tts.KovaTTS` or :class:`~kova_tts.engine.generator.Generator` is
what drags in ``transformers`` and, for LoRA voices, ``peft``. The prompt, token, voice and path
helpers are used by tooling with no business loading a model stack, so they stay free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kova_tts.engine.types import (
    CLONE_SAMPLING,
    TTS_SAMPLING,
    AudioFrame,
    SamplingParams,
    Voice,
    WordTimestamp,
)
from kova_tts.paths import (
    MissingArtifact,
    alignment_path,
    available_loras,
    codec_path,
    load_dotenv,
    lora_dir,
    lora_path,
    model_path,
    wavlm_path,
)
from kova_tts.prompt import (
    clone_prompt,
    format_audio_tokens,
    parse_audio_tokens,
    training_example,
    tts_prompt,
)
from kova_tts.tokens import VocabMap, vocab_map
from kova_tts.voices import VoiceRegistry
from kova_tts.voices import available as available_voices
from kova_tts.voices import resolve as resolve_voice

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps torch out of import time
    from kova_tts.engine.generator import Generator
    from kova_tts.engine.tts import KovaTTS, generate, split_sentences

__version__ = "0.1.0"

#: Names re-exported from :mod:`kova_tts.engine.tts`, loaded on first attribute access.
_LAZY = {
    "Generator": "kova_tts.engine.generator",
    "KovaTTS": "kova_tts.engine.tts",
    "generate": "kova_tts.engine.tts",
    "split_sentences": "kova_tts.engine.tts",
}


def __getattr__(name: str) -> Any:
    """Import the torch-backed engine only when something actually asks for it."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module), name)


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "CLONE_SAMPLING",
    "TTS_SAMPLING",
    "AudioFrame",
    "Generator",
    "KovaTTS",
    "MissingArtifact",
    "SamplingParams",
    "VocabMap",
    "Voice",
    "VoiceRegistry",
    "WordTimestamp",
    "__version__",
    "alignment_path",
    "available_loras",
    "available_voices",
    "clone_prompt",
    "codec_path",
    "format_audio_tokens",
    "generate",
    "load_dotenv",
    "lora_dir",
    "lora_path",
    "model_path",
    "parse_audio_tokens",
    "resolve_voice",
    "split_sentences",
    "training_example",
    "tts_prompt",
    "vocab_map",
    "wavlm_path",
]
