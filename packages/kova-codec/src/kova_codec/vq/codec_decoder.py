"""Quantizer plus the transposed-conv generator: latent frames -> waveform."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from kova_codec.vq import activations
from kova_codec.vq.alias_free_torch import Activation1d
from kova_codec.vq.module import DecoderBlock, LSTMState, ResLSTM, WNConv1d
from kova_codec.vq.residual_vq import ResidualVQ


class CodecDecoder(nn.Module):
    """Generator that upsamples by ``prod(up_ratios)`` samples per frame.

    ``self.model`` is a ``Sequential`` only so the checkpoint's ``model.<i>.*`` keys line up;
    it is never called end to end. Inference walks it in two stages (:meth:`run_conv1d` and
    :meth:`run_lstm_onwards`) so the caller can hold on to the LSTM state between windows.
    """

    def __init__(
        self,
        in_channels: int = 1024,
        upsample_initial_channel: int = 1536,
        ngf: int = 48,
        use_rnn: bool = True,
        rnn_bidirectional: bool = False,
        rnn_num_layers: int = 2,
        up_ratios: tuple[int, ...] = (5, 5, 2, 2, 2),
        dilations: tuple[int, ...] = (1, 3, 9),
        vq_num_quantizers: int = 1,
        vq_dim: int = 1024,
        codebook_size: int = 8192,
        codebook_dim: int = 8,
    ) -> None:
        super().__init__()
        self.hop_length = int(np.prod(up_ratios))

        self.quantizer = ResidualVQ(
            num_quantizers=vq_num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )

        channels = upsample_initial_channel
        layers: list[nn.Module] = [WNConv1d(in_channels, channels, kernel_size=7, padding=3)]
        if use_rnn:
            layers.append(
                ResLSTM(channels, num_layers=rnn_num_layers, bidirectional=rnn_bidirectional)
            )
        output_dim = channels
        for i, stride in enumerate(up_ratios):
            input_dim = channels // 2**i
            output_dim = channels // 2 ** (i + 1)
            layers.append(DecoderBlock(input_dim, output_dim, stride, dilations))
        layers += [
            Activation1d(activation=activations.SnakeBeta(output_dim, alpha_logscale=True)),
            WNConv1d(output_dim, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

        self._lstm_index: int | None = None
        for i, layer in enumerate(self.model):
            if isinstance(layer, ResLSTM):
                self._lstm_index = i
                break

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize ``[B, D, T]`` latents. Returns ``(quantized, codes)``."""
        return self.quantizer(x)

    def vq2emb(self, vq: torch.Tensor) -> torch.Tensor:
        """Codes ``[B, T, num_quantizers]`` -> latent embeddings ``[B, T, D]``."""
        return self.quantizer.vq2emb(vq)

    def run_conv1d(self, x: torch.Tensor) -> torch.Tensor:
        """First conv only: ``[B, D, T]`` -> ``[B, C, T]``, time length unchanged."""
        if self._lstm_index is None:
            raise RuntimeError("No LSTM layer found in the decoder.")
        return self.model[0](x)

    def run_lstm_onwards(
        self,
        x: torch.Tensor,
        lstm_state: LSTMState | None = None,
        return_at_idx: int | None = None,
    ) -> tuple[torch.Tensor, LSTMState, LSTMState | None]:
        """LSTM and everything after it: ``[B, C, T]`` -> ``[B, 1, T * hop_length]``.

        Returns ``(audio, state_after_window, state_at_return_at_idx)``.
        """
        if self._lstm_index is None:
            raise RuntimeError("No LSTM layer found in the decoder.")
        lstm_layer: ResLSTM = self.model[self._lstm_index]
        x, state, state_at_idx = lstm_layer(x, hidden=lstm_state, return_at_idx=return_at_idx)
        for i in range(self._lstm_index + 1, len(self.model)):
            x = self.model[i](x)
        return x, state, state_at_idx

    def remove_weight_norm(self) -> None:
        """Fold weight norm into the conv weights; inference never needs the g/v split."""

        def _remove(m: nn.Module) -> None:
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # module had no weight norm
                return

        self.apply(_remove)
