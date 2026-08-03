"""Acoustic encoder: 32 kHz waveform -> latent frames at the token rate."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from kova_codec.vq import activations
from kova_codec.vq.alias_free_torch import Activation1d
from kova_codec.vq.module import EncoderBlock, ResLSTMBasic, WNConv1d


class CodecEncoder(nn.Module):
    """Strided conv stack that downsamples by ``prod(up_ratios)`` samples per frame."""

    def __init__(
        self,
        ngf: int = 48,
        use_rnn: bool = True,
        rnn_bidirectional: bool = False,
        rnn_num_layers: int = 2,
        up_ratios: tuple[int, ...] = (2, 2, 2, 5, 5),
        dilations: tuple[int, ...] = (1, 3, 9),
        out_channels: int = 1024,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, 1, T]`` waveform -> ``[B, out_channels, T / hop_length]``."""
        return self.block(x)

    def remove_weight_norm(self) -> None:
        """Fold weight norm into the conv weights; inference never needs the g/v split."""

        def _remove(m: nn.Module) -> None:
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # module had no weight norm
                return

        self.apply(_remove)
