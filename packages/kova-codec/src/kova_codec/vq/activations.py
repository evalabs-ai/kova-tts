"""The SnakeBeta activation.

Adapted from https://github.com/EdwardDixon/snake (MIT), via BigVGAN. The periodic term
gives the generator an inductive bias towards the periodic structure of speech, which a
ReLU/LeakyReLU stack has to learn from scratch.
"""

from __future__ import annotations

import torch
from torch import nn


class SnakeBeta(nn.Module):
    """``x + sin^2(a * x) / b``: separate per-channel frequency ``a`` and magnitude ``b``."""

    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        # Log-scale alphas start at 0 (so exp(alpha) == 1); linear-scale ones start at `alpha`.
        init = torch.zeros(in_features) if alpha_logscale else torch.ones(in_features) * alpha
        self.alpha = nn.Parameter(init.clone())
        self.beta = nn.Parameter(init.clone())
        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable
        self.no_div_by_zero = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # [C] -> [1, C, 1], lines up with [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + (1.0 / (beta + self.no_div_by_zero)) * torch.sin(x * alpha).pow(2)
