"""The codec's fixed rates. These are baked into the checkpoint and the LM vocabulary."""

from __future__ import annotations

import kova_codec


def test_token_rate_follows_from_sample_rate_and_hop():
    assert kova_codec.TOKEN_RATE == kova_codec.SAMPLE_RATE // kova_codec.HOP_LENGTH
    assert kova_codec.TOKEN_RATE == 80


def test_codebook_matches_the_audio_token_block_in_the_lm_vocab():
    assert kova_codec.CODEBOOK_SIZE == 8192
    assert (kova_codec.CODE_MIN, kova_codec.CODE_MAX) == (0, 8191)


def test_duration_conversions_round_trip():
    assert kova_codec.codes_to_seconds(160) == 2.0
    assert kova_codec.seconds_to_codes(2.0) == 160
    assert kova_codec.seconds_to_codes(kova_codec.codes_to_seconds(333)) == 333
