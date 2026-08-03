"""Fixed properties of the codec. Changing any of these means a different codebook."""

from __future__ import annotations

#: Waveforms are 32 kHz mono, float32 in [-1, 1].
SAMPLE_RATE = 32_000

#: Samples consumed per code.
HOP_LENGTH = 400

#: Codes per second of audio: 32000 / 400 = 80.
TOKEN_RATE = SAMPLE_RATE // HOP_LENGTH

#: Single codebook, 8192 entries. Codes are ints in [CODE_MIN, CODE_MAX].
CODEBOOK_SIZE = 8192
CODE_MIN = 0
CODE_MAX = CODEBOOK_SIZE - 1

#: WavLM supplies the semantic half of the encoder input, at its own sample rate.
WAVLM_MODEL = "microsoft/wavlm-large"
WAVLM_LAYER = 23
WAVLM_SAMPLE_RATE = 16_000

#: Reference loudness applied before encoding, matching the training data.
TARGET_LUFS = -23.0


def codes_to_seconds(n_codes: int) -> float:
    """Duration of `n_codes` codes, in seconds."""
    return n_codes / TOKEN_RATE


def seconds_to_codes(seconds: float) -> int:
    """Number of codes covering `seconds` of audio."""
    return int(seconds * TOKEN_RATE)
