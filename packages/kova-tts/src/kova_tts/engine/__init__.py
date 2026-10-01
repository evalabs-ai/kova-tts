"""Inference engine: the LM that turns text into codec codes, and the pieces around it.

Only the plain data types are re-exported here; importing them never pulls in torch. The
torch-backed modules (:mod:`~kova_tts.engine.generator`, :mod:`~kova_tts.engine.decoder`,
:mod:`~kova_tts.engine.tts`) are imported by name.
"""

from __future__ import annotations

from kova_tts.engine.types import (
    CLONE_SAMPLING,
    TTS_SAMPLING,
    AudioFrame,
    SamplingParams,
    Voice,
    WordTimestamp,
)

__all__ = [
    "CLONE_SAMPLING",
    "TTS_SAMPLING",
    "AudioFrame",
    "SamplingParams",
    "Voice",
    "WordTimestamp",
]
