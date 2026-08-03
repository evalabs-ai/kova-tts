"""Kova neural audio codec: 32 kHz waveforms <-> discrete codes at 80 tokens/second."""

from __future__ import annotations

from typing import TYPE_CHECKING

from kova_codec.constants import (
    CODE_MAX,
    CODE_MIN,
    CODEBOOK_SIZE,
    HOP_LENGTH,
    SAMPLE_RATE,
    TARGET_LUFS,
    TOKEN_RATE,
    WAVLM_LAYER,
    WAVLM_MODEL,
    WAVLM_SAMPLE_RATE,
    codes_to_seconds,
    seconds_to_codes,
)

if TYPE_CHECKING:  # so type checkers and IDEs still see the export
    from kova_codec.codec import KovaCodec

__version__ = "0.1.0"


def __getattr__(name: str) -> object:
    """Load :class:`KovaCodec` on first use.

    Defining the model imports torch, which costs ~1.8 s. Callers that only want the constants
    -- a CLI reporting where its checkpoints resolved, a test asserting the token rate -- should
    not pay that, and ``kova_tts`` reaches these constants from its own package root, so an eager
    import here would tax every entry point in the project.
    """
    if name == "KovaCodec":
        from kova_codec.codec import KovaCodec

        return KovaCodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CODEBOOK_SIZE",
    "CODE_MAX",
    "CODE_MIN",
    "HOP_LENGTH",
    "SAMPLE_RATE",
    "TARGET_LUFS",
    "TOKEN_RATE",
    "WAVLM_LAYER",
    "WAVLM_MODEL",
    "WAVLM_SAMPLE_RATE",
    "KovaCodec",
    "__version__",
    "codes_to_seconds",
    "seconds_to_codes",
]
