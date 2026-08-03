"""The decode loop: on a tiny model that needs no checkpoint, then on the real one.

The CPU half builds a two-layer Llama with the shipped vocabulary layout. That is a sharper
test than the real checkpoint for everything except numerics -- it runs in a second, so the
stopping rules, the narrowed head and the repetition-penalty bookkeeping can all be asserted
exactly rather than sampled.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from kova_codec.constants import CODEBOOK_SIZE
from kova_tts import tokens
from kova_tts.engine.generator import Generator, decode_attention, register_attention
from kova_tts.engine.types import TTS_SAMPLING

# The shipped checkpoint's layout, reproduced so the fake tokenizer exercises the same
# non-arithmetic id mapping the real one has.
AUDIO_ID_MIN = 128256
SPEECH_END_ID = 136450
VOCAB_SIZE = 136576


class FakeTokenizer:
    """Just enough tokenizer for the generator: a vocabulary and ``encode``.

    Audio tokens are numbered in *lexicographic* string order, as they are in the real
    tokenizer, so ``<|s_1|>`` is nowhere near ``<|s_0|> + 1``.
    """

    def __init__(self) -> None:
        ordered = sorted(range(CODEBOOK_SIZE), key=str)
        self.vocab = {f"<|s_{code}|>": AUDIO_ID_MIN + i for i, code in enumerate(ordered)}
        self.vocab.update(
            {
                tokens.SPEECH_END: SPEECH_END_ID,
                tokens.SPEECH_START: SPEECH_END_ID + 1,
                tokens.TEXT_PROMPT_END: SPEECH_END_ID + 2,
                tokens.TEXT_PROMPT_START: SPEECH_END_ID + 3,
                tokens.BEGIN_OF_TEXT: 128000,
            }
        )

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert not add_special_tokens, "prompts are tokenized with add_special_tokens=False"
        ids, i = [], 0
        while i < len(text):
            if text[i] == "<":
                end = text.index("|>", i) + 2
                ids.append(self.vocab[text[i:end]])
                i = end
            else:
                ids.append(ord(text[i]) % 1000)  # any text id; the model never emits these
                i += 1
        return ids


def build_tiny_generator(device: str = "cpu") -> Generator:
    """A two-layer Llama over the real vocabulary: ~9 M parameters, no checkpoint needed."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        tie_word_embeddings=True,
        attn_implementation=register_attention(),
    )
    model = LlamaForCausalLM(config).eval().to(device)
    return Generator(model, FakeTokenizer(), device=device, max_cache_len=256)


@pytest.fixture(scope="module")
def tiny_generator() -> Generator:
    return build_tiny_generator("cpu")


def audio_prompt(generator: Generator, codes: list[int]) -> str:
    from kova_tts.prompt import format_audio_tokens

    return f"{tokens.SPEECH_START}{format_audio_tokens(codes)}"


# --------------------------------------------------------------------------- fused attention


class TestDecodeAttention:
    """The hand-rolled single-token GQA attention, against torch's own SDPA."""

    def attn(self, heads=8, kv_heads=2, length=16, dim=8, masked=5):
        torch.manual_seed(3)
        query = torch.randn(1, heads, 1, dim)
        key = torch.randn(1, kv_heads, length, dim)
        value = torch.randn(1, kv_heads, length, dim)
        allowed = torch.zeros(1, 1, 1, length, dtype=torch.bool)
        allowed[..., :masked] = True
        return query, key, value, allowed

    def test_matches_sdpa_with_the_kv_heads_expanded(self):
        query, key, value, allowed = self.attn()
        groups = query.shape[1] // key.shape[1]
        want = F.scaled_dot_product_attention(
            query,
            key.repeat_interleave(groups, dim=1),
            value.repeat_interleave(groups, dim=1),
            attn_mask=allowed,
        ).transpose(1, 2)
        additive = torch.where(allowed, 0.0, float("-inf"))
        got = decode_attention(query, key, value, additive, query.shape[-1] ** -0.5)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-5)

    def test_ignores_positions_the_mask_closes(self):
        query, key, value, allowed = self.attn(masked=5)
        additive = torch.where(allowed, 0.0, float("-inf"))
        before = decode_attention(query, key, value, additive, 0.35)
        key[:, :, 5:] = 999.0  # the unfilled tail of a static cache holds anything at all
        after = decode_attention(query, key, value, additive, 0.35)
        torch.testing.assert_close(before, after)

    def test_returns_the_layout_the_model_expects(self):
        query, key, value, allowed = self.attn(heads=8, dim=8)
        out = decode_attention(query, key, value, None, 0.35)
        assert out.shape == (1, 1, 8, 8)  # [batch, query, heads, head_dim]

    def test_refuses_more_than_one_query(self):
        with pytest.raises(ValueError, match="one query position"):
            decode_attention(
                torch.randn(1, 8, 2, 8),
                torch.randn(1, 2, 4, 8),
                torch.randn(1, 2, 4, 8),
                None,
                0.35,
            )


# ------------------------------------------------------------------------------- the loop


class TestTinyModelLoop:
    def test_the_narrowed_head_agrees_with_the_full_head(self, tiny_generator):
        """The load-time optimisation that has to be exactly free: same rows, same reduction."""
        ids = tiny_generator.encode(audio_prompt(tiny_generator, [1, 2, 3]))
        with torch.inference_mode():
            narrowed = tiny_generator._prefill(ids)
            tiny_generator._cache.reset()
            out = tiny_generator._lm(
                input_ids=torch.tensor([ids]),
                attention_mask=None,
                position_ids=torch.arange(len(ids)).unsqueeze(0),
                past_key_values=tiny_generator._cache,
                use_cache=True,
            )
            full = F.linear(
                out.last_hidden_state[:, -1],
                tiny_generator.model.get_output_embeddings().weight,
            ).float()[0]
        emitted = torch.from_numpy(tiny_generator._row_to_id)
        torch.testing.assert_close(narrowed, full[emitted], rtol=0, atol=1e-6)
        assert int(tiny_generator._row_to_id[int(narrowed.argmax())]) == int(full.argmax())

    def test_every_generated_code_is_inside_the_codebook(self, tiny_generator):
        codes = tiny_generator.generate(
            audio_prompt(tiny_generator, [7]), TTS_SAMPLING.replace(max_tokens=64, seed=1)
        )
        assert codes and all(0 <= c < CODEBOOK_SIZE for c in codes)

    def test_max_tokens_caps_the_generation(self, tiny_generator):
        codes = tiny_generator.generate(
            audio_prompt(tiny_generator, [7]), TTS_SAMPLING.replace(max_tokens=12, seed=1)
        )
        assert len(codes) == 12

    def test_generation_stops_on_speech_end(self, tiny_generator, monkeypatch):
        """An untrained model rarely emits EOS on its own, so force it and watch the loop stop."""
        eos_row = tiny_generator._eos_row
        real = tiny_generator._logits

        def biased(hidden):
            out = real(hidden)
            out[eos_row] = 1e4
            return out

        monkeypatch.setattr(tiny_generator, "_logits", biased)
        assert tiny_generator.generate(audio_prompt(tiny_generator, [7])) == []

    def test_the_cache_budget_shortens_max_tokens(self, tiny_generator):
        prompt = audio_prompt(tiny_generator, list(range(200)))
        codes = tiny_generator.generate(prompt, TTS_SAMPLING.replace(max_tokens=2048, seed=1))
        assert len(codes) == tiny_generator.max_cache_len - len(tiny_generator.encode(prompt))

    def test_a_prompt_longer_than_the_cache_says_what_to_do(self, tiny_generator):
        prompt = audio_prompt(tiny_generator, list(range(300)))
        with pytest.raises(ValueError, match="max_cache_len"):
            tiny_generator.generate(prompt)

    def test_the_same_seed_reproduces_the_same_codes(self, tiny_generator):
        params = TTS_SAMPLING.replace(max_tokens=24, seed=99)
        prompt = audio_prompt(tiny_generator, [3, 4])
        assert tiny_generator.generate(prompt, params) == tiny_generator.generate(prompt, params)

    def test_a_different_seed_gives_different_codes(self, tiny_generator):
        prompt = audio_prompt(tiny_generator, [3, 4])
        first = tiny_generator.generate(prompt, TTS_SAMPLING.replace(max_tokens=24, seed=1))
        second = tiny_generator.generate(prompt, TTS_SAMPLING.replace(max_tokens=24, seed=2))
        assert first != second

    def test_streaming_and_blocking_agree(self, tiny_generator):
        params = TTS_SAMPLING.replace(max_tokens=20, seed=5)
        prompt = audio_prompt(tiny_generator, [1])
        assert list(tiny_generator.stream(prompt, params)) == tiny_generator.generate(
            prompt, params
        )

    def test_audio_tokens_in_the_prompt_count_as_already_seen(self, tiny_generator):
        """The reference clip and any carried context are prompt tokens, and the repetition
        penalty is defined over prompt and output alike."""
        with torch.inference_mode():
            tiny_generator._prefill(tiny_generator.encode(audio_prompt(tiny_generator, [5, 900])))
        rows = np.searchsorted(
            tiny_generator._row_to_id, tiny_generator.vocab.codes_to_ids([5, 900])
        )
        assert tiny_generator._seen[torch.from_numpy(rows)].all()
        assert int(tiny_generator._seen.sum()) == 2

    def test_two_streams_at_once_are_refused(self, tiny_generator):
        prompt = audio_prompt(tiny_generator, [1])
        first = tiny_generator.stream(prompt, TTS_SAMPLING.replace(max_tokens=8, seed=1))
        next(first)
        with pytest.raises(RuntimeError, match="batch size is 1|Batch size is 1"):
            tiny_generator.generate(prompt)
        first.close()

    def test_an_abandoned_stream_releases_the_generator(self, tiny_generator):
        prompt = audio_prompt(tiny_generator, [1])
        stream = tiny_generator.stream(prompt, TTS_SAMPLING.replace(max_tokens=8, seed=1))
        next(stream)
        stream.close()
        assert tiny_generator.generate(prompt, TTS_SAMPLING.replace(max_tokens=4, seed=1))


# ------------------------------------------------------------------------------ graph capture


@pytest.mark.gpu
def test_a_captured_graph_actually_records_work():
    """A graph that records nothing replays as a no-op, and torch only warns about it.

    Cheap enough to run on the tiny model, and worth running: the symptom is not a crash but
    logits frozen at their captured values, which reads downstream as a model that has
    forgotten how to stop talking.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    generator = build_tiny_generator("cuda:0")
    assert generator.cuda_graph
    prompt = audio_prompt(generator, [1, 2, 3])
    params = TTS_SAMPLING.replace(max_tokens=16)
    assert generator.generate(prompt, params, greedy=True) == _eager(generator, prompt, params)


@pytest.mark.gpu
def test_capturing_on_a_second_device_records_on_that_device():
    """``torch.cuda.graph`` keeps *one* process-wide capture stream, built on whichever device
    captured first. Without an explicit per-device stream the second generator captures an
    empty graph and silently generates nonsense -- which is exactly what happened, and only
    showed up as one decoder test failing in a full-suite run."""
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two CUDA devices")
    params = TTS_SAMPLING.replace(max_tokens=16)
    for device in ("cuda:1", "cuda:0", "cuda:1"):
        generator = build_tiny_generator(device)
        prompt = audio_prompt(generator, [1, 2, 3])
        captured = generator.generate(prompt, params, greedy=True)
        assert captured == _eager(generator, prompt, params), f"graph is stale on {device}"
        del generator


def _eager(generator: Generator, prompt: str, params) -> list[int]:
    generator.cuda_graph = False
    try:
        return generator.generate(prompt, params, greedy=True)
    finally:
        generator.cuda_graph = True


# ------------------------------------------------------------------------ the real checkpoint


def _free_cuda_device() -> torch.device:
    """The CUDA device with the most free memory. This box has two and another job may own one."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    return torch.device("cuda", best)


@pytest.fixture(scope="module")
def real_generator() -> Generator:
    from kova_tts import paths

    paths.load_dotenv()
    configured = os.environ.get(paths.ENV_MODEL, "").strip()
    if not configured or not Path(configured).is_dir():
        pytest.skip(f"set {paths.ENV_MODEL} to a local checkpoint to run this test")
    return Generator.from_pretrained(configured, device=_free_cuda_device(), max_cache_len=1024)


@pytest.fixture(scope="module")
def real_prompt(real_generator) -> str:
    from kova_tts.prompt import tts_prompt

    return f"{tokens.BEGIN_OF_TEXT}{tts_prompt('The quick brown fox jumps over the lazy dog.')}"


@pytest.mark.gpu
@pytest.mark.weights
def test_the_narrowed_head_produces_the_same_tokens_as_the_full_vocabulary(
    real_generator, real_prompt
):
    """Narrowing the head is a memory-traffic saving, not an approximation.

    Generation is greedy so a single differing logit would show up as a different code, and the
    comparison is against the full 136576-row head on the very same hidden states.
    """
    generator = real_generator
    ids = generator.encode(real_prompt)
    full_head = generator.model.get_output_embeddings().weight
    narrowed, unnarrowed = [], []

    with torch.inference_mode():
        logits = generator._prefill(ids)
        for _ in range(48):
            row = int(logits.argmax())
            code = int(generator._row_to_code[row])
            if code < 0:
                break
            narrowed.append(code)
            generator._in_ids.fill_(int(generator._row_to_id[row]))
            logits = generator._step()

        # The same walk, with the logits taken over the whole 136576-token vocabulary.
        logits_full = F.linear(generator._prefill_hidden(ids), full_head).float()[0]
        for _ in range(48):
            token = int(logits_full.argmax())
            if token == generator.vocab.speech_end_id:
                break
            unnarrowed.append(int(generator.vocab.ids_to_codes([token])[0]))
            generator._in_ids.fill_(token)
            logits_full = F.linear(generator._decode_hidden(), full_head).float()[0]

    assert len(narrowed) == 48
    assert narrowed == unnarrowed


@pytest.mark.gpu
@pytest.mark.weights
def test_the_cuda_graph_and_the_eager_loop_agree(real_generator, real_prompt):
    generator = real_generator
    assert generator.cuda_graph, "this box should be capturing graphs"
    params = TTS_SAMPLING.replace(max_tokens=64)

    captured = generator.generate(real_prompt, params, greedy=True)
    generator.cuda_graph = False
    try:
        eager = generator.generate(real_prompt, params, greedy=True)
    finally:
        generator.cuda_graph = True

    assert captured and captured == eager


@pytest.mark.gpu
@pytest.mark.weights
def test_a_real_generation_stops_on_its_own_inside_the_codebook(real_generator, real_prompt):
    codes = real_generator.generate(real_prompt, TTS_SAMPLING.replace(max_tokens=900, seed=7))
    assert len(codes) > 40, "less than half a second of speech for a whole sentence"
    assert len(codes) < 900, "generation ran to the token limit instead of stopping on EOS"
    assert all(0 <= c < CODEBOOK_SIZE for c in codes)
    assert len(set(codes)) > len(codes) // 4, "the codes barely move; this is not speech"


@pytest.mark.gpu
@pytest.mark.weights
def test_a_seed_reproduces_a_real_generation(real_generator, real_prompt):
    params = TTS_SAMPLING.replace(max_tokens=64, seed=1234)
    assert real_generator.generate(real_prompt, params) == real_generator.generate(
        real_prompt, params
    )
