"""Building blocks shared by the codec encoder and decoder.

The convolutional stack derives from BigCodec (https://github.com/Aria-K-Alethia/BigCodec).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils import weight_norm

from kova_codec.vq import activations
from kova_codec.vq.alias_free_torch import Activation1d

LSTMState = tuple[torch.Tensor, torch.Tensor]


def WNConv1d(*args, **kwargs) -> nn.Module:  # noqa: N802 - mirrors the upstream BigCodec name
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs) -> nn.Module:  # noqa: N802
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))


class ResidualUnit(nn.Module):
    """Dilated residual block: two anti-aliased SnakeBeta activations around a 7-tap conv."""

    def __init__(self, dim: int = 16, dilation: int = 1) -> None:
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad),
            Activation1d(activation=activations.SnakeBeta(dim, alpha_logscale=True)),
            WNConv1d(dim, dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class EncoderBlock(nn.Module):
    """Residual units at three dilations, then a strided conv that downsamples by ``stride``."""

    def __init__(
        self,
        dim: int = 16,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 9),
        double_channels: bool = True,
    ) -> None:
        super().__init__()
        # The first encoder block keeps its channel count; every later one doubles it.
        input_dim = dim // 2 if double_channels else dim
        self.block = nn.Sequential(
            *[ResidualUnit(input_dim, dilation=d) for d in dilations],
            Activation1d(activation=activations.SnakeBeta(input_dim, alpha_logscale=True)),
            WNConv1d(
                input_dim,
                dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """Transposed conv that upsamples by ``stride``, then residual units at three dilations."""

    def __init__(
        self,
        input_dim: int = 16,
        output_dim: int = 8,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 9),
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            Activation1d(activation=activations.SnakeBeta(input_dim, alpha_logscale=True)),
            WNConvTranspose1d(
                input_dim,
                output_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=stride // 2 + stride % 2,
                output_padding=stride % 2,
            ),
        )
        self.block.extend([ResidualUnit(output_dim, dilation=d) for d in dilations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResLSTM(nn.Module):
    """Residual LSTM over ``[B, C, T]``, with the hidden state exposed for streaming.

    This is the only stateful layer in the decoder, so carrying ``(h, c)`` across code
    windows is what makes chunked decoding equal whole-utterance decoding.
    """

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension,
            dimension if not bidirectional else dimension // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(
        self,
        x: torch.Tensor,
        hidden: LSTMState | None = None,
        return_at_idx: int | None = None,
    ) -> tuple[torch.Tensor, LSTMState, LSTMState | None]:
        """Run the LSTM over ``x`` ``[B, C, T]``.

        Args:
            hidden: ``(h, c)`` carried in from the previous window; ``None`` zero-inits.
            return_at_idx: Also return the state as of timestep ``return_at_idx`` (third
                element). Streaming needs this: a window is decoded with lookahead context
                on both sides, but the next window must resume from where the *emitted*
                audio ended, not from the end of the lookahead.

        Returns:
            ``(y, hidden_after_window, hidden_at_idx)``.
        """
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C], nn.LSTM is batch_first
        if return_at_idx is None:
            y, hidden_out = self.lstm(x, hidden)
            hidden_at_idx = None
        else:
            # Two calls so the intermediate state falls out of the split point.
            y_1, hidden_at_idx = self.lstm(x[:, :return_at_idx, :].contiguous(), hidden)
            y_2, hidden_out = self.lstm(x[:, return_at_idx:, :].contiguous(), hidden_at_idx)
            y = torch.cat([y_1, y_2], dim=1)
        if self.skip:
            y = y + x
        return y.transpose(1, 2), hidden_out, hidden_at_idx


class ResLSTMBasic(nn.Module):
    """Residual LSTM over ``[B, C, T]`` without the streaming state plumbing (encoder side)."""

    def __init__(
        self,
        dimension: int,
        num_layers: int = 2,
        bidirectional: bool = False,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.skip = skip
        self.lstm = nn.LSTM(
            dimension,
            dimension if not bidirectional else dimension // 2,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2).contiguous()
        y, _ = self.lstm(x)
        if self.skip:
            y = y + x
        return y.transpose(1, 2)


class SemanticEncoder(nn.Module):
    """Projects WavLM features to the codec's latent width before they are fused with audio."""

    def __init__(
        self,
        input_channels: int,
        code_dim: int,
        encode_channels: int,
        kernel_size: int = 3,
        bias: bool = True,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.initial_conv = nn.Conv1d(
            input_channels, encode_channels, kernel_size, stride=1, padding=pad, bias=False
        )
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels, encode_channels, kernel_size, stride=1, padding=pad, bias=bias
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                encode_channels, encode_channels, kernel_size, stride=1, padding=pad, bias=bias
            ),
        )
        self.final_conv = nn.Conv1d(
            encode_channels, code_dim, kernel_size, stride=1, padding=pad, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(x)
        x = self.residual_blocks(x) + x
        return self.final_conv(x)
