"""Anti-aliased activation wrapper.

Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0), via BigVGAN.
"""

from __future__ import annotations

import torch
from torch import nn

from kova_codec.vq.alias_free_torch.resample import DownSample1d, UpSample1d


class Activation1d(nn.Module):
    """Upsample, apply a pointwise nonlinearity, downsample.

    The nonlinearity generates harmonics above Nyquist; oversampling by 2x first and
    band-limiting on the way back keeps them from folding into the audible band.
    """

    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.up_ratio = up_ratio
        self.down_ratio = down_ratio
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))
