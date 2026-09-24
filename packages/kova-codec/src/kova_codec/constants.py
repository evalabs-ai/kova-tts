"""Fixed properties of the codec. Changing any of these means a different codebook."""

from __future__ import annotations

#: The encoder's input: 32 kHz mono, float32 in [-1, 1]. Reference clips and training audio
#: are resampled to this rate before :meth:`~kova_codec.KovaCodec.encode`.
SAMPLE_RATE = 32_000

#: Input samples consumed per code.
HOP_LENGTH = 400

#: Codes per second of audio: 32000 / 400 = 80. Every decoder shares this grid.
TOKEN_RATE = SAMPLE_RATE // HOP_LENGTH

#: The second input rate a dual-rate encoder takes, through its own trained stem: 16 kHz, 200
#: samples per code, onto the same codes. Older checkpoints take :data:`SAMPLE_RATE` only;
#: ``KovaCodec.supported_input_sample_rates`` says which a loaded codec accepts.
LOW_SAMPLE_RATE = 16_000
LOW_HOP_LENGTH = LOW_SAMPLE_RATE // TOKEN_RATE

#: The decoder's output: 48 kHz, 600 samples per code. This is the rate of the shipped
#: decoder; a checkpoint states its own, and :attr:`KovaCodec.sample_rate
#: <kova_codec.KovaCodec.sample_rate>` on a loaded codec is the one to trust.
OUTPUT_SAMPLE_RATE = 48_000
OUTPUT_HOP_LENGTH = OUTPUT_SAMPLE_RATE // TOKEN_RATE

#: Single codebook, 8192 entries. Codes are ints in [CODE_MIN, CODE_MAX].
CODEBOOK_SIZE = 8192
CODE_MIN = 0
CODE_MAX = CODEBOOK_SIZE - 1

#: WavLM supplies the semantic half of the encoder input, at its own sample rate.
WAVLM_MODEL = "microsoft/wavlm-large"
WAVLM_LAYER = 23
WAVLM_SAMPLE_RATE = 16_000

#: Reference loudness applied before encoding, so every clip arrives at one level.
TARGET_LUFS = -23.0


def codes_to_seconds(n_codes: int) -> float:
    """Duration of `n_codes` codes, in seconds."""
    return n_codes / TOKEN_RATE


def seconds_to_codes(seconds: float) -> int:
    """Number of codes covering `seconds` of audio."""
    return int(seconds * TOKEN_RATE)
