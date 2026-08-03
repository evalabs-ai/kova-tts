"""Turning prepared waveforms into codec codes.

Encoding runs at roughly 34x realtime on one consumer GPU, so a corpus of any reasonable size
is a matter of minutes.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from kova_codec.constants import HOP_LENGTH
from kova_tts import paths

logger = logging.getLogger(__name__)


class Encoder(Protocol):
    """The one method this package needs from a codec.

    Declared structurally so tests can substitute a deterministic fake, which gives a much
    sharper assertion about the pipeline than 1.2 GB of WavLM would.
    """

    def encode(self, wav: object) -> object: ...


def load_codec(
    *,
    checkpoint: str | os.PathLike[str] | None = None,
    wavlm: str | os.PathLike[str] | None = None,
    device: str | None = None,
    dtype: object | None = None,
) -> Encoder:
    """Load a codec that can encode, resolving the checkpoint and WavLM through ``paths``.

    Encoding needs WavLM (~1.2 GB) where decoding does not, so this is deliberately not the
    same construction the TTS engine uses. Float32 throughout: fp16 encode flips a small
    fraction of codes, and a corpus is written once and trained on many times.
    """
    import torch

    from kova_codec import KovaCodec

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return KovaCodec.from_checkpoint(
        paths.codec_path(checkpoint),
        device=resolved_device,
        dtype=dtype or torch.float32,
        wavlm=paths.wavlm_path(wavlm),
    )


def code_count(samples: int) -> int:
    """Codes the codec produces for a clip of `samples` samples."""
    return math.ceil(samples / HOP_LENGTH)


def encode_clips(
    encoder: Encoder,
    waveforms: Sequence[np.ndarray],
    *,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[list[int]]:
    """Encode every waveform, returning one list of codes per input, in input order.

    `on_progress` is called with ``(clips_done, clips_total)`` after each clip, which is how the
    CLI shows progress on a corpus that takes minutes.
    """
    codes: list[list[int]] = []
    for done, waveform in enumerate(waveforms, start=1):
        # Two-dimensional so the encoder always sees the same [B, T] contract.
        encoded = np.asarray(encoder.encode(waveform[np.newaxis, :].astype(np.float32)))
        if encoded.ndim != 2 or encoded.shape[0] != 1:
            raise ValueError(
                f"Encoder returned shape {encoded.shape} for a single clip; expected [1, T] codes."
            )
        codes.append(encoded[0, : code_count(waveform.size)].astype(int).tolist())
        if on_progress is not None:
            on_progress(done, len(waveforms))
    return codes
