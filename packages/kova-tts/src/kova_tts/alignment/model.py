"""The alignment model: codec codes in, per-frame character log-probabilities out.

It reads the codes the LM generated, never the audio, so a word's timing costs no decode and is
exact to the frame the codec will play it at. The layers are the codec's own quantizer embedding,
a projection, a small convolutional decoder and a CTC head over :data:`VOCAB`::

    codes --quantizer.vq2emb--> fc_post_s --semantic_decoder--> ctc_head --> log_softmax

Trained separately from the codec; the checkpoint is ``alignment.pt`` in the model repository.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from kova_codec.vq.residual_vq import ResidualVQ

#: CTC blank, the letters, apostrophe, ``@`` (what a ``[tag]`` is spelled as) and ``<star>``, the
#: wildcard that absorbs silence and anything not in the transcript.
VOCAB: tuple[str, ...] = ("<blank>", *"abcdefghijklmnopqrstuvwxyz", "'", "@", "<star>")
CHAR_TO_ID: dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
BLANK_ID = CHAR_TO_ID["<blank>"]
STAR_ID = CHAR_TO_ID["<star>"]


class AlignmentModel(nn.Module):
    """Codes ``[T]`` -> log-probabilities ``[T, len(VOCAB)]``. Sizes are read off the checkpoint."""

    def __init__(
        self,
        *,
        num_quantizers: int,
        vq_dim: int,
        codebook_size: int,
        codebook_dim: int,
        hidden: int,
        vocab_size: int = len(VOCAB),
    ) -> None:
        super().__init__()
        self.quantizer = ResidualVQ(
            num_quantizers=num_quantizers,
            dim=vq_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
        )
        self.fc_post_s = nn.Linear(vq_dim, hidden)
        self.semantic_decoder = _SemanticDecoder(hidden)
        self.ctc_head = nn.Linear(hidden, vocab_size)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], device: str = "cpu") -> AlignmentModel:
        state = torch.load(os.fspath(path), map_location="cpu", weights_only=False)
        weights = state.get("model", state)
        weights = {k.removeprefix("_orig_mod."): v for k, v in weights.items()}
        num_quantizers = 0
        while f"quantizer.layers.{num_quantizers}._codebook.weight" in weights:
            num_quantizers += 1
        codebook_size, codebook_dim = weights["quantizer.layers.0._codebook.weight"].shape
        model = cls(
            num_quantizers=num_quantizers,
            vq_dim=weights["quantizer.layers.0.out_proj.weight_v"].shape[0],
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            hidden=weights["fc_post_s.weight"].shape[0],
            vocab_size=weights["ctc_head.weight"].shape[0],
        )
        model.load_state_dict(weights)
        return model.to(device).eval()

    @torch.inference_mode()
    def log_probs(self, codes: torch.Tensor) -> torch.Tensor:
        """Codes ``[T]`` (one quantizer) -> float32 log-probabilities ``[T, V]`` on the CPU."""
        device = next(self.parameters()).device
        x = codes.to(device=device, dtype=torch.long).view(1, -1, 1)
        h = self.fc_post_s(self.quantizer.vq2emb(x)).transpose(1, 2)
        logits = self.ctc_head(self.semantic_decoder(h).transpose(1, 2))
        return F.log_softmax(logits, dim=-1)[0].float().cpu()


class _SemanticDecoder(nn.Module):
    """Three same-padded convolutions with a residual around the middle two.

    The ReLUs are in-place, and the first one runs on the residual's input before the residual
    is added back, so the skip connection carries ``relu(x)``. That is how the checkpoint was
    trained, so it is kept exactly.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.initial_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.residual_blocks = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
        )
        self.final_conv = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(z)
        x = self.residual_blocks(x) + x
        return self.final_conv(x)
