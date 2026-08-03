"""Factorized vector quantizer: project to a low-dimensional space, then look up a codebook.

Quantizing in 8 dimensions instead of 1024 keeps the 8192-entry codebook dense enough that
almost every entry stays alive, which is what makes a single stream at 80 tok/s viable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import weight_norm


class FactorizedVectorQuantize(nn.Module):
    """One codebook of ``codebook_size`` entries in ``codebook_dim`` dimensions."""

    def __init__(self, dim: int, codebook_size: int, codebook_dim: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

        if dim != codebook_dim:
            self.in_proj = weight_norm(nn.Linear(dim, codebook_dim))
            self.out_proj = weight_norm(nn.Linear(codebook_dim, dim))
        else:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
        self._codebook = nn.Embedding(codebook_size, codebook_dim)

    @property
    def codebook(self) -> nn.Embedding:
        return self._codebook

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``[B, D, T]`` latents -> ``(quantized [B, D, T], codes [B, T])``."""
        z_e = self.in_proj(z.transpose(1, 2)).transpose(1, 2)
        z_q, indices = self.decode_latents(z_e)
        z_q = self.out_proj(z_q.transpose(1, 2)).transpose(1, 2)
        return z_q, indices

    def vq2emb(self, vq: torch.Tensor, proj: bool = True) -> torch.Tensor:
        """Codes ``[B, T]`` -> embeddings ``[B, T, dim]`` (or ``[B, T, codebook_dim]``)."""
        emb = self.embed_code(vq)
        return self.out_proj(emb) if proj else emb

    def embed_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return F.embedding(embed_id, self.codebook.weight)

    def decode_code(self, embed_id: torch.Tensor) -> torch.Tensor:
        return self.embed_code(embed_id).transpose(1, 2)

    def decode_latents(self, latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Nearest codebook entry per frame, matched by cosine distance (both sides L2-normed)."""
        batch = latents.shape[0]
        encodings = F.normalize(latents.transpose(1, 2).reshape(-1, latents.shape[1]))
        codebook = F.normalize(self.codebook.weight)
        dist = (
            encodings.pow(2).sum(1, keepdim=True)
            - 2 * encodings @ codebook.t()
            + codebook.pow(2).sum(1, keepdim=True).t()
        )
        indices = (-dist).max(1)[1].view(batch, -1)
        return self.decode_code(indices), indices
