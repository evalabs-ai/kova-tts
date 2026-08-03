"""Residual vector quantizer. Kova uses a single stage, but the loop is kept general."""

from __future__ import annotations

import torch
from torch import nn

from kova_codec.vq.factorized_vector_quantize import FactorizedVectorQuantize


class ResidualVQ(nn.Module):
    """Stack of quantizers, each one coding the residual left by the previous."""

    def __init__(self, *, num_quantizers: int, codebook_size: int | list[int], **kwargs) -> None:
        super().__init__()
        if isinstance(codebook_size, int):
            codebook_size = [codebook_size] * num_quantizers
        self.layers = nn.ModuleList(
            [FactorizedVectorQuantize(codebook_size=size, **kwargs) for size in codebook_size]
        )
        self.num_quantizers = num_quantizers

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, D, T]`` -> ``(quantized [B, D, T], codes [Q, B, T])``."""
        residual = x
        quantized_out = torch.zeros_like(x)
        all_indices: list[torch.Tensor] = []
        for layer in self.layers:
            quantized, indices = layer(residual)
            residual = residual - quantized
            quantized_out = quantized_out + quantized
            all_indices.append(indices)
        return quantized_out, torch.stack(all_indices)

    def vq2emb(self, vq: torch.Tensor, proj: bool = True) -> torch.Tensor:
        """Codes ``[B, T, num_quantizers]`` -> summed embeddings ``[B, T, D]``."""
        emb = self.layers[0].vq2emb(vq[:, :, 0], proj=proj)
        for idx, layer in enumerate(self.layers[1:], start=1):
            emb = emb + layer.vq2emb(vq[:, :, idx], proj=proj)
        return emb
