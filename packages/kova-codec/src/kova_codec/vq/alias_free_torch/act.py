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

    This is the single most expensive module in the decoder -- 43 instances, half of decode
    time -- because the three stages are individually cheap but each one reads and writes a
    tensor twice the width of the layer it sits in. On Metal the whole sandwich is instead run
    as one fused kernel (:func:`~kova_codec.mps_kernels.fused_activation1d`), which touches the
    tensor once; see :meth:`_fused_constants` for what it needs precomputed. Every other
    backend takes the three-stage path below unchanged.
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
        # Resolved on the first forward and reused: the constants the Metal kernel reads, or
        # None where it does not apply. `_fused_device` is what that answer was resolved for.
        self._fused: tuple[torch.Tensor, ...] | None = None
        self._fused_device: torch.device | None = None

    def _fused_constants(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        """The per-channel and per-tap constants the fused kernel reads, all float32 on `device`.

        The kernel wants ``exp(alpha)`` and ``1 / (exp(beta) + eps)`` rather than the raw
        log-scale parameters, and the two filters flat: hoisting all of that out of the inner
        loop is part of why the fused pass is cheap. All four are float32 whatever the module's
        dtype is, so only a change of device invalidates them.
        """
        if self._fused is None or self._fused_device != device:
            act = self.act
            beta = torch.exp(act.beta.detach().float()) + act.no_div_by_zero
            self._fused = (
                torch.exp(act.alpha.detach().float()).to(device).contiguous(),
                (1.0 / beta).to(device).contiguous(),
                self.upsample.filter.detach().float().flatten().to(device).contiguous(),
                self.downsample.lowpass.filter.detach().float().flatten().to(device).contiguous(),
            )
            self._fused_device = device
        return self._fused

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Asked per call rather than cached: measured at ~2.5 us against ~158 us of GPU time per
        # activation, so it is not worth pinning -- and pinning it would make the fused path
        # impossible to turn off from a test or a benchmark, which is how it gets checked.
        if x.device.type == "mps" and x.dim() == 3:
            from kova_codec import mps_kernels

            if mps_kernels.available() and mps_kernels.activation1d_supported(self):
                return mps_kernels.fused_activation1d(x, *self._fused_constants(x.device))
        return self.downsample(self.act(self.upsample(x)))
