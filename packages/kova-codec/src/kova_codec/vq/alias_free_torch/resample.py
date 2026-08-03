"""Band-limited up/downsampling around a nonlinearity.

Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0), via BigVGAN.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kova_codec.vq.alias_free_torch.filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    """Insert zeros and interpolate with a sinc kernel: ``[B, C, T] -> [B, C, ratio * T]``."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.stride = ratio
        self.pad = self.kernel_size // ratio - 1
        self.pad_left = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.register_buffer(
            "filter",
            kaiser_sinc_filter1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    """Low-pass then decimate: ``[B, C, ratio * T] -> [B, C, T]``."""

    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)
