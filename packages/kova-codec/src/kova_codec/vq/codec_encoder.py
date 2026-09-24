"""Acoustic encoder: 32 kHz waveform (or 16 kHz, with a dual-rate stem) -> latent frames."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from kova_codec.vq import activations
from kova_codec.vq.alias_free_torch import Activation1d
from kova_codec.vq.module import EncoderBlock, ResidualUnit, ResLSTMBasic, WNConv1d


class CodecEncoder(nn.Module):
    """Strided conv stack that downsamples by ``prod(up_ratios)`` samples per frame.

    With ``low_rate_stem=True`` it also carries the dual-rate checkpoints' 16 kHz stem: a small
    trained front end that stands in for the initial convolution and the first stride-2 block,
    so 16 kHz audio joins the shared tail at the frame rate 32 kHz audio reaches it at. Half the
    input rate through one fewer factor of two gives the same codes per second.
    """

    def __init__(
        self,
        ngf: int = 48,
        use_rnn: bool = True,
        rnn_bidirectional: bool = False,
        rnn_num_layers: int = 2,
        up_ratios: tuple[int, ...] = (2, 2, 2, 5, 5),
        dilations: tuple[int, ...] = (1, 3, 9),
        out_channels: int = 1024,
        low_rate_stem: bool = False,
    ) -> None:
        super().__init__()
        self.hop_length = int(np.prod(up_ratios))

        d_model = ngf
        blocks: list[nn.Module] = [WNConv1d(1, d_model, kernel_size=7, padding=3)]
        for i, stride in enumerate(up_ratios):
            # The first block downsamples without widening; the rest double the width.
            if i > 0:
                d_model *= 2
            blocks.append(
                EncoderBlock(d_model, stride=stride, dilations=dilations, double_channels=i > 0)
            )
        if use_rnn:
            blocks.append(
                ResLSTMBasic(d_model, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional)
            )
        blocks += [
            Activation1d(activation=activations.SnakeBeta(d_model, alpha_logscale=True)),
            WNConv1d(d_model, out_channels, kernel_size=3, padding=1),
        ]

        self.block = nn.Sequential(*blocks)

        #: The 16 kHz front end, or ``None`` on a 32 kHz-only checkpoint.
        self.low_rate_stem: nn.Sequential | None = None
        #: Input samples per frame through :attr:`low_rate_stem`, or ``None`` without one.
        self.low_rate_hop_length: int | None = None
        if low_rate_stem:
            if not up_ratios or up_ratios[0] != 2:
                raise ValueError("The 16 kHz stem replaces a stride-2 first block; this has none.")
            self.low_rate_stem = nn.Sequential(
                WNConv1d(1, ngf, kernel_size=7, padding=3),
                *[ResidualUnit(ngf, dilation=d) for d in dilations],
                Activation1d(activation=activations.SnakeBeta(ngf, alpha_logscale=True)),
                WNConv1d(ngf, ngf, kernel_size=3, stride=1, padding=1),
            )
            self.low_rate_hop_length = int(np.prod(up_ratios[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, T]`` waveform -> ``[B, out_channels, T / hop_length]``."""
        return self.block(x)

    def forward_low_rate(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, T]`` 16 kHz waveform -> ``[B, out_channels, T / low_rate_hop_length]``.

        The stem replaces ``block[0]`` (the initial convolution) and ``block[1]`` (the first
        stride-2 block); everything after that is the shared 32 kHz tail.
        """
        if self.low_rate_stem is None:
            raise RuntimeError("This encoder has no trained 16 kHz stem.")
        return self.block[2:](self.low_rate_stem(x))

    def remove_weight_norm(self) -> None:
        """Fold weight norm into the conv weights; inference never needs the g/v split."""

        def _remove(m: nn.Module) -> None:
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # module had no weight norm
                return

        self.apply(_remove)
