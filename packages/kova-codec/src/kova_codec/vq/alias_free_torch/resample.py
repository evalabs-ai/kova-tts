"""Band-limited up/downsampling around a nonlinearity.

Adapted from https://github.com/junjun3518/alias-free-torch (Apache 2.0), via BigVGAN.
Modified by Kova AI for integration with Kova TTS.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from kova_codec.vq.alias_free_torch.filter import LowPassFilter1d, kaiser_sinc_filter1d


class UpSample1d(nn.Module):
    """Insert zeros and interpolate with a sinc kernel: ``[B, C, T] -> [B, C, ratio * T]``.

    Written as a **polyphase** filter bank rather than the strided transposed convolution the
    reference implementation uses. The transposed form spends most of its multiplications on
    the zeros it has just inserted, and depthwise ``conv_transpose1d`` is far less well
    optimised than depthwise ``conv1d`` on at least one backend.

    Measured on Metal at the decoder's own shapes, one second of audio through the whole
    ladder: **19.0 ms transposed against 3.9 ms polyphase, 4.8x**. That module was 41% of the
    codec's decode time before this change.

    The two forms agree **exactly** on Metal at float16 and float32 and to within a couple of
    ULPs (< 1e-6 absolute) on the CPU at float32, where the summation order differs. That is
    the same class of difference as changing a convolution algorithm, and well under the
    ~39 dB the shipped float16 decode already sits at.

    The identity: writing ``n = ratio*m + p``,

        out[ratio*m + p] = sum_i in[i] * w[ratio*m + p - ratio*i]
                         = sum_k in[m - k] * w[ratio*k + p]

    so output phase ``p`` is a plain convolution of the input with the filter's ``p``-th
    polyphase component -- ``ratio`` convolutions of ``kernel_size / ratio`` taps each, over
    the *unexpanded* input, interleaved at the end.
    """

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
        # Derived from `filter`, and therefore rebuilt whenever it is loaded or moved rather
        # than stored: `persistent=False` keeps these out of the state dict, so a checkpoint
        # written before this change still loads strictly.
        self.register_buffer("phases", torch.empty(0), persistent=False)

    def _polyphase(self, like: torch.Tensor) -> torch.Tensor:
        """The filter's polyphase components, ``[ratio, 1, kernel_size // ratio]``.

        Taps go in reversed because ``conv1d`` cross-correlates where the identity above is a
        convolution; the matching ``padding`` in :meth:`forward` turns that into the *full*
        convolution the transposed form produces.
        """
        if self.phases.numel() == 0 or self.phases.dtype != like.dtype:
            phases = torch.cat(
                [self.filter[..., p :: self.ratio].flip(-1) for p in range(self.ratio)]
            )
            self.phases = phases.to(device=like.device, dtype=like.dtype).contiguous()
        return self.phases

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        phases = self._polyphase(x)
        taps = phases.shape[-1]
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        outputs = [
            F.conv1d(
                x,
                phases[p : p + 1].expand(channels, -1, -1),
                groups=channels,
                padding=taps - 1,
            )
            for p in range(self.ratio)
        ]
        # Interleave the phases: stacking on a new last axis and flattening it into the time
        # axis is exactly out[ratio*m + p] = outputs[p][m].
        x = self.ratio * torch.stack(outputs, dim=-1).flatten(-2)
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
