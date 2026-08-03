"""Network shapes and the streaming contract, on small randomly-initialised modules.

Everything here runs on the CPU in a second or two with no checkpoint, so CI covers the
wiring even though it can never cover the weights.
"""

from __future__ import annotations

import pytest
import torch

from kova_codec import CODE_MAX, CODE_MIN, CODEBOOK_SIZE, HOP_LENGTH, TOKEN_RATE
from kova_codec.vq.codec_decoder import CodecDecoder
from kova_codec.vq.codec_encoder import CodecEncoder
from kova_codec.vq.module import SemanticEncoder

# Kova's real ratios: both stacks move 400 samples per code.
FULL_ENCODER_RATIOS = (2, 2, 2, 2, 5, 5)
FULL_DECODER_RATIOS = (5, 5, 2, 2, 2, 2)

# A ~200x smaller stack with the same topology, for tests that only care about shapes.
TINY_HOP = 20
TINY_ENCODER = dict(ngf=16, up_ratios=(2, 2, 5), dilations=(1, 3), out_channels=64)
TINY_DECODER = dict(
    in_channels=64,
    upsample_initial_channel=128,
    ngf=16,
    up_ratios=(5, 2, 2),
    dilations=(1, 3),
    vq_dim=64,
    codebook_size=64,
    codebook_dim=8,
)


@pytest.fixture(scope="module")
def tiny_decoder() -> CodecDecoder:
    decoder = CodecDecoder(**TINY_DECODER)
    decoder.eval()
    decoder.remove_weight_norm()
    return decoder


def test_hop_length_is_the_product_of_the_stride_ladder():
    assert CodecEncoder(up_ratios=FULL_ENCODER_RATIOS).hop_length == HOP_LENGTH
    assert CodecDecoder(up_ratios=FULL_DECODER_RATIOS).hop_length == HOP_LENGTH
    assert HOP_LENGTH * TOKEN_RATE == 32_000


def test_encoder_emits_one_frame_per_hop():
    encoder = CodecEncoder(**TINY_ENCODER).eval()
    with torch.inference_mode():
        out = encoder(torch.randn(2, 1, TINY_HOP * 10))
    assert out.shape == (2, TINY_ENCODER["out_channels"], 10)


def test_removing_weight_norm_does_not_change_the_encoder_output():
    encoder = CodecEncoder(**TINY_ENCODER).eval()
    x = torch.randn(1, 1, TINY_HOP * 8)
    with torch.inference_mode():
        before = encoder(x)
        encoder.remove_weight_norm()
        after = encoder(x)
    torch.testing.assert_close(before, after, rtol=0, atol=1e-5)


def test_quantizer_emits_codes_inside_the_codebook():
    decoder = CodecDecoder(**TINY_DECODER).eval()
    with torch.inference_mode():
        quantized, codes = decoder(torch.randn(2, 64, 10))
    assert quantized.shape == (2, 64, 10)
    assert codes.shape == (1, 2, 10)  # [num_quantizers, B, T]
    assert codes.min() >= 0
    assert codes.max() < TINY_DECODER["codebook_size"]


def test_vq2emb_shapes(tiny_decoder: CodecDecoder):
    codes = torch.randint(0, TINY_DECODER["codebook_size"], (2, 10, 1))
    with torch.inference_mode():
        emb = tiny_decoder.vq2emb(codes)
    assert emb.shape == (2, 10, TINY_DECODER["vq_dim"])


def test_decoder_upsamples_by_the_hop_length(tiny_decoder: CodecDecoder):
    frames = tiny_decoder.run_conv1d(torch.randn(1, 64, 12))
    with torch.inference_mode():
        audio, _, _ = tiny_decoder.run_lstm_onwards(frames)
    assert audio.shape == (1, 1, 12 * TINY_HOP)
    assert audio.abs().max() <= 1.0  # the generator ends in a tanh


def test_semantic_encoder_preserves_length():
    encoder = SemanticEncoder(input_channels=32, code_dim=32, encode_channels=32).eval()
    with torch.inference_mode():
        out = encoder(torch.randn(1, 32, 25))
    assert out.shape == (1, 32, 25)


def test_codebook_range_constants_agree():
    assert (CODE_MIN, CODE_MAX) == (0, CODEBOOK_SIZE - 1)


def _decode(decoder: CodecDecoder, codes: torch.Tensor, state=None, at=None, conv_padding=None):
    """One ``decode_with_lstm``-equivalent call against a bare decoder."""
    pad_mask = codes < 0
    emb = decoder.vq2emb(codes.clamp(min=0).unsqueeze(-1)).masked_fill(pad_mask.unsqueeze(-1), 0.0)
    frames = decoder.run_conv1d(emb.transpose(1, 2))
    if conv_padding is not None:
        frames = frames[:, :, conv_padding:-conv_padding]
    audio, _, at_idx = decoder.run_lstm_onwards(frames, state, return_at_idx=at)
    return audio.squeeze(1), at_idx


def test_windowed_decoding_reproduces_a_single_decode(tiny_decoder: CodecDecoder):
    """The property the streaming server is built on.

    Decoding fixed windows with the LSTM state carried across calls, three frames of
    conv context trimmed off each end, and a lookahead margin that is decoded but not
    emitted, must reproduce a whole-sequence decode of the same codes.
    """
    conv_padding, lookahead, window = 3, 12, 16
    torch.manual_seed(0)
    codes = torch.randint(0, TINY_DECODER["codebook_size"], (1, 96))
    total = codes.shape[1]

    with torch.inference_mode():
        reference, _ = _decode(tiny_decoder, codes)

        # Pad codes stand in for the conv context either side of the real sequence.
        pad = torch.full((1, conv_padding), -1, dtype=torch.long)
        padded = torch.cat([pad, codes, pad], dim=1)
        width = window + 2 * lookahead + 2 * conv_padding

        pieces, state, pos = [], None, 0
        while pos < total:
            # The final window stops at the last real code: running the LSTM on into pad
            # frames and then emitting that audio is *not* equivalent to a single decode.
            last = pos + window + lookahead >= total
            chunk = padded[:, pos : padded.shape[1] if last else pos + width]
            audio, state = _decode(
                tiny_decoder,
                chunk,
                state,
                at=None if last else window,
                conv_padding=conv_padding,
            )
            start = 0 if pos == 0 else lookahead * TINY_HOP
            end = audio.shape[1] if last else (lookahead + window) * TINY_HOP
            pieces.append(audio[:, start:end])
            if last:
                break
            pos += window
        streamed = torch.cat(pieces, dim=1)

    assert streamed.shape == reference.shape
    # 12 frames of lookahead covers this stack's receptive field, so the match is exact to
    # within float32 noise (measured max |diff| ~2e-7 against a signal peaking at 0.17).
    torch.testing.assert_close(streamed, reference, rtol=0, atol=1e-5)
