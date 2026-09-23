"""Kaiser-windowed sinc low-pass filter.

Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0), via BigVGAN.
Modified by Kova AI for integration with Kova TTS.
The low-pass filter also derives from julius (MIT), via BigVGAN.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def kaiser_sinc_filter1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Low-pass kernel of shape ``[1, 1, kernel_size]``, normalised to sum to 1.

    The Kaiser beta follows the standard stopband-attenuation heuristic; normalising the
    taps keeps the DC component from leaking through the up/downsample pair.
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)

    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros_like(time).view(1, 1, kernel_size)
    taps = 2 * cutoff * window * torch.sinc(2 * cutoff * time)
    taps /= taps.sum()
    return taps.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    """Depthwise low-pass convolution over ``[B, C, T]``, optionally strided."""

    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        if cutoff < 0.0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer("filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x, self.filter.expand(channels, -1, -1), stride=self.stride, groups=channels
        )
