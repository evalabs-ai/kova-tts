"""The boundary between ComfyUI's ``AUDIO`` type and this project's waveforms.

ComfyUI passes audio as ``{"waveform": tensor [B, C, T], "sample_rate": int}``, batched and
often at whatever rate the file had. ``kova_tts`` works in 1-D float32 numpy: it takes 32 kHz
in, for the encoder, and gives 48 kHz out. Both conversions live here, in one small module with
no ComfyUI imports, so that no ComfyUI shape ever reaches the engine and none of this needs
ComfyUI installed to be tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE
from kova_tts import audio as audio_io


def to_comfy_audio(wav: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> dict[str, Any]:
    """Waveform -> an ``AUDIO`` dict: one batch item, one channel.

    The tensor is built on the CPU. ComfyUI's own audio nodes expect that, and a waveform that
    is about to be written to a file has no business holding on to VRAM.
    """
    import torch

    mono = audio_io.as_waveform(wav)
    return {
        "waveform": torch.from_numpy(np.ascontiguousarray(mono)).reshape(1, 1, -1),
        "sample_rate": int(sample_rate),
    }


def from_comfy_audio(audio: dict[str, Any], sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """``AUDIO`` dict -> 1-D float32 mono at `sample_rate`, downmixed and resampled as needed."""
    mono, rate = native_comfy_audio(audio)
    return audio_io.resample(mono, rate, int(sample_rate))


def native_comfy_audio(audio: dict[str, Any]) -> tuple[np.ndarray, int]:
    """``AUDIO`` dict -> ``(1-D float32 mono, the rate it is at)``, downmixed but not resampled.

    Only the first item of a batch is used: a voice is cloned from one recording, and silently
    averaging a batch of different speakers together would be worse than ignoring the rest.
    """
    if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError(
            "Expected a ComfyUI AUDIO input -- a dict with 'waveform' and 'sample_rate'. "
            "Connect the output of a Load Audio node here."
        )

    waveform = audio["waveform"]
    array = np.asarray(
        waveform.detach().to("cpu").float().numpy() if hasattr(waveform, "detach") else waveform,
        dtype=np.float32,
    )
    if array.ndim == 3:
        if array.shape[0] == 0:
            raise ValueError("That AUDIO input holds no samples.")
        array = array[0]
    elif array.ndim > 3:
        raise ValueError(f"AUDIO waveform must be [B, C, T], [C, T] or [T]; got {array.shape}.")

    mono = audio_io.as_waveform(array)
    if mono.size == 0:
        raise ValueError("That AUDIO input holds no samples.")
    return mono, int(audio["sample_rate"])
