"""The MLX decode loop: what it refuses to load, and what it produces when it does."""

from __future__ import annotations

import json

import numpy as np
import pytest

from kova_tts import paths
from kova_tts.engine.types import TTS_SAMPLING

pytest.importorskip("mlx.core", reason="the MLX backend needs Apple Silicon")

from kova_tts.engine import mlx_generator  # noqa: E402  (after the platform guard)


class TestHeadGeometry:
    def test_reads_the_offset_and_size(self, tmp_path):
        (tmp_path / "config.json").write_text(
            json.dumps({"head_vocab_offset": 128256, "head_vocab_size": 8195}), encoding="utf-8"
        )
        assert mlx_generator._head_geometry(tmp_path) == (128256, 8195)

    def test_a_directory_without_a_config_is_rejected(self, tmp_path):
        with pytest.raises(paths.MissingArtifact, match="no config.json"):
            mlx_generator._head_geometry(tmp_path)

    def test_an_unconverted_checkpoint_says_what_it_is_missing(self, tmp_path):
        # The failure this guards against is silent: mlx_lm would build a full-width head,
        # find no weights for it, and generate noise.
        (tmp_path / "config.json").write_text(
            json.dumps({"vocab_size": 136576, "tie_word_embeddings": True}), encoding="utf-8"
        )
        with pytest.raises(paths.MissingArtifact, match="head_vocab_size"):
            mlx_generator._head_geometry(tmp_path)


class TestRowTables:
    """The head is a slice of the id space; these are the sums that make a row a code."""

    def test_a_head_one_row_too_short_names_the_id_it_cannot_reach(self):
        vocab = _vocab_map()
        generator = object.__new__(mlx_generator.MLXGenerator)
        generator.vocab = vocab
        generator.head_offset = int(vocab.output_ids[0])
        generator.head_size = int(vocab.output_ids[-1] - vocab.output_ids[0])  # one short
        with pytest.raises(ValueError, match=str(int(vocab.output_ids[-1]))):
            generator._row_tables()

    def test_every_emittable_id_maps_back_to_itself(self):
        vocab = _vocab_map()
        generator = object.__new__(mlx_generator.MLXGenerator)
        generator.vocab = vocab
        generator.head_offset = int(vocab.output_ids[0])
        generator.head_size = int(vocab.output_ids[-1] - vocab.output_ids[0]) + 1
        codes, emittable = generator._row_tables()

        assert emittable.sum() == vocab.output_ids.size
        # Every audio id: id -> row -> code -> id is the identity.
        audio = vocab.output_ids[vocab.is_audio_id(vocab.output_ids)]
        rows = audio - generator.head_offset
        assert np.array_equal(vocab.codes_to_ids(codes[rows]), audio)
        # EOS is the one emittable row with no code, which is how the loop stops.
        assert codes[vocab.speech_end_id - generator.head_offset] == -1


@pytest.fixture(scope="module")
def generator(local_artifact):
    """One loaded model for the whole module: it is 0.6 GB and several seconds to build."""
    path = local_artifact(paths.ENV_MODEL)
    from kova_tts.engine import backends

    if backends.resolve(path) != "mlx":
        pytest.skip(f"{paths.ENV_MODEL} is not an MLX artifact")
    return mlx_generator.MLXGenerator.from_pretrained(path)


@pytest.mark.weights
class TestRealModel:
    """Against the converted checkpoint ``KOVA_MODEL_PATH`` points at."""

    def test_the_head_covers_exactly_the_emittable_vocabulary(self, generator):
        assert generator.head_size >= generator.vocab.output_ids.size
        assert generator.backend == "mlx"

    def test_greedy_generation_is_repeatable(self, generator):
        ids = generator.encode("<|begin_of_text|>")
        first = list(generator.stream_ids(ids, TTS_SAMPLING.replace(max_tokens=24), greedy=True))
        second = list(generator.stream_ids(ids, TTS_SAMPLING.replace(max_tokens=24), greedy=True))
        assert first == second

    def test_every_code_is_inside_the_codebook(self, generator):
        params = TTS_SAMPLING.replace(max_tokens=32)
        codes = list(generator.stream_ids(generator.encode("<|begin_of_text|>"), params))
        assert codes and all(0 <= code < 8192 for code in codes)

    def test_the_budget_is_honoured(self, generator):
        params = TTS_SAMPLING.replace(max_tokens=16)
        codes = list(generator.stream_ids(generator.encode("<|begin_of_text|>"), params))
        assert len(codes) <= 16

    def test_a_prompt_longer_than_the_cache_is_refused(self, generator):
        with pytest.raises(ValueError, match="KV cache holds"):
            list(generator.stream_ids([0] * (generator.max_cache_len + 1)))

    def test_two_overlapping_generations_are_refused(self, generator):
        params = TTS_SAMPLING.replace(max_tokens=8)
        running = generator.stream_ids(generator.encode("<|begin_of_text|>"), params)
        next(running)
        try:
            with pytest.raises(RuntimeError, match="already running a request"):
                next(generator.stream_ids(generator.encode("<|begin_of_text|>"), params))
        finally:
            running.close()

    def test_a_lora_voice_says_where_to_merge_it(self, generator, tmp_path):
        with pytest.raises(NotImplementedError, match="Merge the adapter"):
            generator.load_lora(tmp_path)
        generator.unload_lora()  # the no-op counterpart; KovaTTS calls it on every generation
        assert generator.lora is None


def _vocab_map():
    """The real tokenizer's vocabulary map, or skip: these are arithmetic on its exact ids."""
    from kova_tts.tokens import vocab_map

    try:
        return vocab_map()
    except Exception as exc:  # noqa: BLE001 - no weights configured is a skip, not a failure
        pytest.skip(f"needs a local tokenizer: {exc}")
