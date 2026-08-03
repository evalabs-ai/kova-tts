"""WavLM semantic features for the encode path.

Only :meth:`KovaCodec.encode` needs this. It is imported lazily from ``codec.py`` so a
decode-only process never imports ``transformers`` or allocates WavLM's ~1.2 GB of fp32
weights.

Layer 23 of WavLM-large is the layer the codec was trained against; earlier layers are more
acoustic and later ones drift towards the pre-training objective, so this is not a knob.
"""

from __future__ import annotations

import torch
from torch import nn
from torchaudio.transforms import Resample

from kova_codec.constants import SAMPLE_RATE, WAVLM_LAYER, WAVLM_MODEL, WAVLM_SAMPLE_RATE


class WavLMFeatures(nn.Module):
    """WavLM-large plus the 32 kHz -> 16 kHz resampler in front of it.

    Weights and the resample kernel are held in float32 whatever dtype the codec runs at:
    both are applied to raw waveforms, and casting them to fp16/bf16 would either mismatch
    the float32 input or cost accuracy for no speed gain (WavLM is not the bottleneck).
    """

    def __init__(
        self,
        model_name: str = WAVLM_MODEL,
        *,
        device: torch.device | None = None,
        layer: int = WAVLM_LAYER,
        sample_rate: int = WAVLM_SAMPLE_RATE,
    ) -> None:
        super().__init__()
        from transformers import WavLMModel

        self.layer = layer
        self.resample = Resample(orig_freq=SAMPLE_RATE, new_freq=sample_rate)
        self.wavlm = WavLMModel.from_pretrained(model_name)
        self.wavlm.eval()
        self.float()
        if device is not None:
            self.to(device)

    @torch.inference_mode()
    def forward(self, wav_32k: torch.Tensor) -> torch.Tensor:
        """Mono 32 kHz audio ``[T]`` or ``[B, T]`` -> hidden states ``[B, T_ssl, 1024]``."""
        if wav_32k.dim() == 1:
            wav_32k = wav_32k.unsqueeze(0)
        device = next(self.wavlm.parameters()).device
        wav_32k = wav_32k.to(device=device, dtype=torch.float32)
        outputs = self.wavlm(self.resample(wav_32k), output_hidden_states=True)
        return outputs.hidden_states[self.layer]
