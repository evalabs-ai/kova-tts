"""The vocabulary map: audio codes <-> token ids, and the lexicographic-ordering trap.

The CPU tests build a stand-in vocabulary that reproduces the shipped tokenizer's layout
exactly (audio tokens numbered in sorted *string* order from 128256); the ``weights`` tests
check that claim against the real checkpoint.
"""

from __future__ import annotations

import numpy as np
import pytest

from kova_codec.constants import CODEBOOK_SIZE
from kova_tts import prompt, tokens

#: First id of the audio block in the shipped tokenizer.
AUDIO_ID_MIN = 128256


class FakeTokenizer:
    """Anything exposing ``get_vocab()`` is enough to build a VocabMap."""

    def __init__(self, vocab: dict[str, int]) -> None:
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)


def build_vocab(n_codes: int = CODEBOOK_SIZE, base: int = AUDIO_ID_MIN) -> dict[str, int]:
    """A vocabulary laid out like the real one: audio tokens in sorted string order."""
    audio = sorted(tokens.AUDIO_TOKEN.format(i) for i in range(n_codes))
    vocab = {tok: base + i for i, tok in enumerate(audio)}
    for offset, tok in enumerate(
        (tokens.SPEECH_END, tokens.SPEECH_START, tokens.TEXT_PROMPT_END, tokens.TEXT_PROMPT_START)
    ):
        vocab[tok] = base + n_codes + 2 + offset
    vocab[tokens.BEGIN_OF_TEXT] = 128000
    return vocab


@pytest.fixture(scope="module")
def vmap() -> tokens.VocabMap:
    return tokens.VocabMap.from_tokenizer(FakeTokenizer(build_vocab()))


@pytest.fixture(scope="module")
def resolved() -> str:
    from kova_tts import paths

    return paths.model_path()


@pytest.fixture(scope="module")
def tokenizer(resolved: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(resolved)


@pytest.fixture(scope="module")
def real(tokenizer) -> tokens.VocabMap:
    return tokens.VocabMap.from_tokenizer(tokenizer)


class TestConstruction:
    def test_covers_the_whole_codebook(self, vmap):
        assert vmap.code_to_id.shape == (CODEBOOK_SIZE,)
        assert vmap.id_to_code.shape == (CODEBOOK_SIZE,)

    def test_block_bounds(self, vmap):
        assert (vmap.audio_id_min, vmap.audio_id_max) == (
            AUDIO_ID_MIN,
            AUDIO_ID_MIN + CODEBOOK_SIZE - 1,
        )

    def test_ids_are_assigned_in_lexicographic_string_order_not_arithmetic(self, vmap):
        # The whole reason this module exists: <|s_1|> sorts after <|s_10|>..<|s_1999|>.
        assert vmap.code_to_id[0] == 128256
        assert vmap.code_to_id[1] == 129367
        assert vmap.code_to_id[2] == 130478
        assert vmap.code_to_id[1] != vmap.audio_id_min + 1

    def test_output_ids_are_the_audio_block_plus_eos(self, vmap):
        assert vmap.output_ids.size == CODEBOOK_SIZE + 1
        assert np.array_equal(vmap.output_ids, np.sort(vmap.output_ids))
        assert vmap.speech_end_id in vmap.output_ids.tolist()

    def test_missing_audio_tokens_are_reported(self):
        with pytest.raises(ValueError, match=f"Expected {CODEBOOK_SIZE} audio tokens"):
            tokens.VocabMap.from_tokenizer(FakeTokenizer(build_vocab(n_codes=16)))

    def test_a_gap_in_the_block_is_reported(self):
        vocab = build_vocab()
        vocab[tokens.AUDIO_TOKEN.format(0)] += 500_000
        with pytest.raises(ValueError, match="not a contiguous block"):
            tokens.VocabMap.from_tokenizer(FakeTokenizer(vocab))

    def test_missing_speech_end_is_reported(self):
        vocab = build_vocab()
        del vocab[tokens.SPEECH_END]
        with pytest.raises(ValueError, match="never stop"):
            tokens.VocabMap.from_tokenizer(FakeTokenizer(vocab))


class TestConversion:
    def test_round_trips_every_code(self, vmap):
        codes = np.arange(CODEBOOK_SIZE)
        assert np.array_equal(vmap.ids_to_codes(vmap.codes_to_ids(codes)), codes)

    def test_accepts_a_plain_list(self, vmap):
        assert vmap.codes_to_ids([0, 1]).tolist() == [128256, 129367]

    def test_empty_input(self, vmap):
        assert vmap.codes_to_ids([]).size == 0
        assert vmap.ids_to_codes([]).size == 0

    def test_ids_are_int64_for_torch(self, vmap):
        assert vmap.codes_to_ids([0]).dtype == np.int64

    def test_rejects_a_code_outside_the_codebook(self, vmap):
        with pytest.raises(ValueError, match="outside the codebook"):
            vmap.codes_to_ids([0, CODEBOOK_SIZE])

    def test_rejects_a_non_audio_id(self, vmap):
        with pytest.raises(ValueError, match="not an audio token"):
            vmap.ids_to_codes([vmap.audio_id_min, 5])

    def test_eos_is_called_out_by_name(self, vmap):
        with pytest.raises(ValueError, match="speech_end"):
            vmap.ids_to_codes([vmap.speech_end_id])

    def test_is_audio_id_masks_the_block(self, vmap):
        ids = [vmap.audio_id_min, vmap.audio_id_max, vmap.speech_end_id, 0]
        assert vmap.is_audio_id(ids).tolist() == [True, True, False, False]

    def test_decode_codes_drops_the_trailing_eos(self, vmap):
        ids = vmap.codes_to_ids([7, 8]).tolist() + [vmap.speech_end_id]
        assert vmap.decode_codes(ids).tolist() == [7, 8]


@pytest.mark.weights
class TestRealTokenizer:
    """Against the checkpoint at KOVA_MODEL_PATH (or the Hub)."""

    def test_audio_block_is_contiguous_and_starts_at_128256(self, real):
        assert real.audio_id_min == 128256
        assert real.audio_id_max == 136447
        assert real.audio_id_max - real.audio_id_min + 1 == CODEBOOK_SIZE

    def test_the_lexicographic_quirk_is_real(self, tokenizer):
        vocab = tokenizer.get_vocab()
        assert vocab["<|s_0|>"] == 128256
        assert vocab["<|s_1|>"] == 129367
        assert vocab["<|s_2|>"] == 130478
        assert vocab["<|s_8191|>"] != 128256 + 8191

    def test_structural_token_ids(self, tokenizer):
        vocab = tokenizer.get_vocab()
        assert vocab[tokens.SPEECH_END] == 136450
        assert vocab[tokens.SPEECH_START] == 136451
        assert vocab[tokens.TEXT_PROMPT_END] == 136452
        assert vocab[tokens.TEXT_PROMPT_START] == 136453
        assert vocab[tokens.BEGIN_OF_TEXT] == 128000

    def test_non_verbal_tags_are_single_tokens(self, tokenizer):
        vocab = tokenizer.get_vocab()
        assert all(tag in vocab for tag in tokens.NON_VERBAL_TAGS)

    def test_speech_end_is_the_models_eos(self, real, resolved):
        from transformers import AutoConfig

        assert real.speech_end_id == 136450
        assert AutoConfig.from_pretrained(resolved).eos_token_id == real.speech_end_id

    def test_round_trips_every_code(self, real):
        codes = np.arange(CODEBOOK_SIZE)
        assert np.array_equal(real.ids_to_codes(real.codes_to_ids(codes)), codes)

    def test_ids_match_what_the_tokenizer_encodes(self, real, tokenizer):
        codes = [0, 1, 2, 4095, 8191]
        encoded = tokenizer(prompt.format_audio_tokens(codes), add_special_tokens=False).input_ids
        assert encoded == real.codes_to_ids(codes).tolist()

    def test_decoded_generation_parses_back_to_codes(self, real, tokenizer):
        codes = [0, 1, 8191, 42]
        ids = real.codes_to_ids(codes).tolist() + [real.speech_end_id]
        assert prompt.parse_audio_tokens(tokenizer.decode(ids)) == codes

    def test_prompt_tokenizes_without_an_implicit_bos(self, tokenizer):
        ids = tokenizer(prompt.tts_prompt("Hello there."), add_special_tokens=False).input_ids
        vocab = tokenizer.get_vocab()
        assert ids[0] == vocab[tokens.TEXT_PROMPT_START]
        assert ids[-1] == vocab[tokens.SPEECH_START]
        assert vocab[tokens.BEGIN_OF_TEXT] not in ids

    def test_clone_prompt_carries_exactly_one_bos(self, tokenizer):
        built = prompt.clone_prompt("Reference.", "Target.", [0, 1])
        ids = tokenizer(built, add_special_tokens=False).input_ids
        bos = tokenizer.get_vocab()[tokens.BEGIN_OF_TEXT]
        assert ids[0] == bos
        assert ids.count(bos) == 1

    def test_vocab_map_is_cached(self, resolved):
        assert tokens.vocab_map(resolved) is tokens.vocab_map(resolved)
