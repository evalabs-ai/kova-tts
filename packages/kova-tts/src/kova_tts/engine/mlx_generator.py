"""The batch-1 decode loop on Apple Silicon, through MLX.

Same contract as :class:`~kova_tts.engine.generator.Generator` -- prompt in, codec codes out,
one request at a time -- and the same sampler, transcribed. What differs is everything
underneath: MLX instead of torch, a quantized checkpoint instead of bf16, and a model whose
output head has already been cut down on disk rather than at load time.

**Why a separate backend at all.** The torch path runs on Metal and produces correct audio, at
19 codes/second on a base M1 -- 0.24x real time, because the LM alone reads 2.5 GB per decode
step and the memory bandwidth is not there. Quantizing to 4 bits and slicing the output head
takes that to ~0.5 GB per step and 82 codes/second, which is what puts the LM above real time.
Neither change is expressible in the torch path: MLX owns the quantized kernels, and the sliced
head is baked into the artifact.

**The artifact.** This backend loads a *converted* model directory, not the bf16 checkpoint:

* ``lm_head`` is untied from the input embedding and holds only the rows generation can emit,
  ``head_vocab_size`` of them starting at token id ``head_vocab_offset``. Both numbers live in
  ``config.json`` -- nothing here assumes 128256, or that the emittable block begins where the
  text vocabulary ends.
* the input embedding keeps all its rows, because prompts contain text and reference audio.
  It is read as a gather, so carrying it in full costs footprint and nothing per step.
* everything is quantized; the shipped conversion is 4-bit at group 128.

A directory without ``head_vocab_size`` is rejected at load rather than run: mlx_lm would
build a full-width head, fail to find weights for it, and generate noise.

**Row space.** The head's rows are a slice of the token id space, so row ``r`` is token
``head_vocab_offset + r``. The slice is a little wider than the set of tokens the model may
legally emit -- it runs from the first audio token to ``<|speech_end|>`` inclusive, and the
vocabulary has a couple of unused ids in between -- so rows outside
:attr:`~kova_tts.tokens.VocabMap.output_ids` are biased to ``-inf`` before sampling. Feeding a
sampled row back in means adding the offset again: the input side still speaks full ids.

**The host stays one token behind.** Sampling happens inside the graph
(:mod:`kova_tts.engine.mlx_sampling`), so a step's whole chain -- forward, penalty, top-k,
top-p, draw, and the update to the seen mask -- is one unevaluated expression. The loop queues
step *n+1* before it reads step *n*'s token, which is what keeps the GPU busy across the read.
Sampling on the host instead costs 8%: 74.9 tok/s against 81.6 on a base M1, 0.94x real time
against 1.02x. The price is one wasted decode step per generation, discarded when the token
before it turns out to be ``<|speech_end|>``.

**Seeds are per backend.** ``SamplingParams.seed`` makes a run here reproducible, but MLX's
generator is not torch's, so the same seed gives different audio on the two backends. They
sample from the same distribution; they do not walk it in the same order.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kova_tts import paths
from kova_tts.engine import mlx_sampling
from kova_tts.engine.types import TTS_SAMPLING, SamplingParams
from kova_tts.tokens import VocabMap

try:
    import mlx.core as mx
except ImportError as exc:  # pragma: no cover - depends on the platform, not the code
    raise ImportError(
        "The MLX backend needs mlx, which ships for Apple Silicon only. Install it with "
        "`uv sync --extra mlx` on a Mac, or use the torch backend."
    ) from exc

log = logging.getLogger(__name__)

#: Prompt plus generation. MLX's KV cache grows as it is written, so unlike the torch backend
#: this costs nothing up front -- it is a budget, not an allocation, and it is here so that
#: :class:`~kova_tts.engine.tts.KovaTTS` sizes its prompts the same way on both backends.
DEFAULT_MAX_CACHE_LEN = 4096

#: Prompt tokens per forward pass during prefill. A 4096-token prompt in one call materialises
#: attention over the whole length at once; 512 keeps the peak down on a 16 GB machine and
#: costs nothing measurable, since prefill runs at ~900 tokens/second either way.
PREFILL_CHUNK = 512

#: Config keys the conversion writes. Their absence is what tells this backend it has been
#: pointed at an unconverted model.
HEAD_SIZE_KEY = "head_vocab_size"
HEAD_OFFSET_KEY = "head_vocab_offset"


class MLXGenerator:
    """Autoregressive generation of codec codes on MLX, one request at a time.

    Construct with :meth:`from_pretrained`; the constructor takes already-loaded objects so a
    test can drive the loop with a small model.

    Not reentrant: one KV cache. Starting a second generation while a :meth:`stream` is still
    running raises rather than silently interleaving.
    """

    #: Which decode loop this is, for ``/health`` and the logs.
    backend = "mlx"

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        head_offset: int,
        head_size: int,
        vocab: VocabMap | None = None,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else VocabMap.from_tokenizer(tokenizer)
        self.max_cache_len = int(max_cache_len)
        self.head_offset = int(head_offset)
        self.head_size = int(head_size)
        # The codec is a torch model and shares the chip with MLX; this is the device it runs
        # on, and the attribute KovaTTS reads to place it.
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self._busy = False
        self._cache: list[Any] | None = None
        self._row_to_code, emittable = self._row_tables()
        # -inf on every row that is neither an audio token nor EOS, added to the logits before
        # sampling. A bias rather than a mask, so it folds into one add inside the graph.
        self._bias = mx.where(mx.array(emittable), mx.zeros(self.head_size), float("-inf"))
        # Row indices, for turning a sampled row into a one-hot update of the seen mask
        # without leaving the graph.
        self._rows = mx.arange(self.head_size)
        mx.eval(self._bias, self._rows)

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model: str | os.PathLike[str] | None = None,
        *,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
    ) -> MLXGenerator:
        """Load a converted MLX model directory, resolving it through :mod:`kova_tts.paths`.

        The path must point at an artifact produced by the Apple conversion -- see the module
        docstring for what makes one. Pointing it at the bf16 checkpoint raises.
        """
        from mlx_lm.utils import hf_repo_to_path, load_model
        from transformers import AutoTokenizer

        resolved = paths.model_path(model)
        path = Path(resolved) if Path(resolved).is_dir() else Path(hf_repo_to_path(resolved))
        offset, size = _head_geometry(path)

        log.info("Loading the MLX model from %s (head of %d rows at %d)", path, size, offset)
        lm, _ = load_model(path, get_model_classes=_sliced_head_classes(size))
        tokenizer = AutoTokenizer.from_pretrained(str(path))

        generator = cls(
            lm,
            tokenizer,
            head_offset=offset,
            head_size=size,
            max_cache_len=max_cache_len,
        )
        generator.warmup()
        return generator

    def _row_tables(self) -> tuple[np.ndarray, np.ndarray]:
        """``row -> code`` and ``row -> may this row be emitted``.

        Both are derived from the tokenizer through :class:`~kova_tts.tokens.VocabMap`, so an
        artifact whose head does not cover every emittable token is caught here rather than by
        a sampler that silently cannot reach half the codebook.
        """
        ids = np.asarray(self.vocab.output_ids, dtype=np.int64)
        rows = ids - self.head_offset
        outside = ids[(rows < 0) | (rows >= self.head_size)]
        if outside.size:
            raise ValueError(
                f"This model's output head covers token ids "
                f"[{self.head_offset}, {self.head_offset + self.head_size}), which leaves out "
                f"{outside.size} token(s) the model has to be able to emit, starting with "
                f"{int(outside[0])}. The head was sliced against a different tokenizer than "
                f"the one in this directory."
            )

        code_of = np.full(self.head_size, -1, dtype=np.int64)
        audio = self.vocab.is_audio_id(ids)
        code_of[rows[audio]] = self.vocab.ids_to_codes(ids[audio])
        emittable = np.zeros(self.head_size, dtype=bool)
        emittable[rows] = True
        return code_of, emittable

    def warmup(self) -> None:
        """Compile the kernels now rather than inside somebody's first request.

        MLX compiles lazily on first use, which without this lands ~0.6 s in the first
        request's time-to-first-audio. Safe to call repeatedly.
        """
        first = int(self.vocab.output_ids[0])
        logits, seen = self._prefill([first] * 4)
        token, seen = self._sample(logits, seen, TTS_SAMPLING, None, TTS_SAMPLING.temperature)
        mx.eval(self._forward(token), token, seen)
        self._cache = None

    # ------------------------------------------------------------------ LoRA

    @property
    def lora(self) -> Path | None:
        """Always ``None``: this backend speaks in one voice, the one in its weights."""
        return None

    def load_lora(self, adapter: str | os.PathLike[str], *, merge: bool = True) -> None:
        """Not available on MLX. Raises, with what to do instead.

        A peft adapter is a pair of bf16 matrices per attention projection, and this model's
        projections are 4-bit MLX tensors -- there is nothing to add them to without
        dequantizing the layer on every step, which would cost more than the quantization
        saves. Merging is the answer, and it belongs before the conversion rather than at
        runtime: a merged voice quantizes as well as the base model does.
        """
        raise NotImplementedError(
            f"The MLX backend cannot apply the LoRA voice at {os.fspath(adapter)} at runtime: "
            f"its weights are quantized. Merge the adapter into the base checkpoint and "
            f"convert that -- one model directory per voice -- then point KOVA_MODEL_PATH at "
            f"the one you want. Cloning from a reference clip needs no adapter and works here."
        )

    def unload_lora(self) -> None:
        """No-op: there is never an adapter loaded to unload."""

    # ------------------------------------------------------------------ generation

    def generate(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        *,
        greedy: bool = False,
    ) -> list[int]:
        """Generate a whole utterance and return its codec codes."""
        return list(self.stream(prompt, params, greedy=greedy))

    def stream(
        self,
        prompt: str,
        params: SamplingParams | None = None,
        *,
        greedy: bool = False,
    ) -> Iterator[int]:
        """Yield codec codes as they are produced.

        Stops on ``<|speech_end|>`` or after ``params.max_tokens`` codes, whichever comes
        first. `greedy` forces argmax, which no preset can express and which the tests use to
        make a run exactly repeatable.
        """
        return self.stream_ids(self.encode(prompt), params, greedy=greedy)

    def encode(self, prompt: str) -> list[int]:
        """Tokenize a prompt string. Never adds special tokens: BOS belongs to the prompt."""
        return list(self.tokenizer.encode(prompt, add_special_tokens=False))

    def stream_ids(
        self,
        ids: Sequence[int],
        params: SamplingParams | None = None,
        *,
        greedy: bool = False,
    ) -> Iterator[int]:
        """:meth:`stream`, for a prompt that is already tokenized."""
        params = params or TTS_SAMPLING
        budget = self._budget(len(ids), params.max_tokens)
        if self._busy:
            raise RuntimeError(
                "This generator is already running a request. Batch size is 1 by design; "
                "finish or close the running stream before starting another."
            )
        self._busy = True
        try:
            yield from self._loop(ids, params, budget, greedy)
        finally:
            self._busy = False

    def _budget(self, prompt_len: int, max_tokens: int) -> int:
        if prompt_len >= self.max_cache_len:
            raise ValueError(
                f"Prompt is {prompt_len} tokens but the KV cache holds {self.max_cache_len}. "
                f"Shorten the text or the reference clip, or rebuild the generator with a "
                f"larger max_cache_len."
            )
        room = self.max_cache_len - prompt_len
        if max_tokens > room:
            log.debug("Clamping max_tokens %d -> %d to fit the KV cache", max_tokens, room)
        return min(max_tokens, room)

    def _loop(
        self,
        ids: Sequence[int],
        params: SamplingParams,
        budget: int,
        greedy: bool,
    ) -> Iterator[int]:
        """Decode, keeping exactly one step of work in flight ahead of the host."""
        state = mx.random.key(params.seed) if params.seed is not None else None
        temperature = 0.0 if greedy else params.temperature

        logits, seen = self._prefill(ids)
        state, draw = _split(state)
        token, seen = self._sample(logits, seen, params, draw, temperature)
        mx.async_eval(token, seen)

        for step in range(budget):
            pending: tuple[mx.array, mx.array] | None = None
            if step + 1 < budget:
                # Queued *before* the host looks at this step's token, so the GPU has the next
                # step to work on while the copy and the consumer's frame both happen.
                state, draw = _split(state)
                next_logits = self._forward(token)
                pending = self._sample(next_logits, seen, params, draw, temperature)
                mx.async_eval(*pending)

            code = int(self._row_to_code[int(token)])
            if code < 0:  # <|speech_end|>; the step queued above is discarded
                return
            yield code
            if pending is None:
                return  # the budget is spent
            token, seen = pending

    # ------------------------------------------------------------------ steps

    def _prefill(self, ids: Sequence[int]) -> tuple[mx.array, mx.array]:
        """Run the prompt through the model. Returns its logits and the initial seen mask."""
        from mlx_lm.models.cache import make_prompt_cache

        self._cache = make_prompt_cache(self.model)

        # Every audio token in the prompt counts as "already seen" for the repetition penalty:
        # the reference clip's codes and any carried context are prompt tokens, and the penalty
        # is defined over prompt and output alike (see :mod:`kova_tts.engine.sampling`).
        prompt = np.asarray(ids, dtype=np.int64)
        seen = np.zeros(self.head_size, dtype=bool)
        audio = prompt[self.vocab.is_audio_id(prompt)]
        if audio.size:
            seen[audio - self.head_offset] = True

        tokens = mx.array(prompt[None, :])
        logits = None
        for start in range(0, tokens.shape[1], PREFILL_CHUNK):
            logits = self.model(tokens[:, start : start + PREFILL_CHUNK], cache=self._cache)
            # Without this the whole prompt stays one unevaluated graph and the peak
            # allocation is the sum of every chunk rather than the largest of them.
            mx.eval([c.state for c in self._cache])
        return self._scores(logits), mx.array(seen)

    def _forward(self, token: mx.array) -> mx.array:
        """One decode step. `token` is a *row*, unevaluated; the model wants a token id."""
        ids = (token + self.head_offset).reshape(1, 1)
        return self._scores(self.model(ids, cache=self._cache))

    def _scores(self, logits: mx.array) -> mx.array:
        """The last position's logits, biased so only emittable rows can be drawn."""
        return logits[0, -1, :].astype(mx.float32) + self._bias

    def _sample(
        self,
        logits: mx.array,
        seen: mx.array,
        params: SamplingParams,
        key: Any | None,
        temperature: float,
    ) -> tuple[mx.array, mx.array]:
        """Draw one row and fold it into the seen mask, all inside the graph."""
        token = mlx_sampling.sample(
            logits,
            temperature=temperature,
            top_p=params.top_p,
            top_k=params.top_k,
            repetition_penalty=params.repetition_penalty,
            previous=seen,
            key=key,
        )
        return token, seen | (self._rows == token)


def _split(state: Any | None) -> tuple[Any | None, Any | None]:
    """Advance an MLX PRNG state, returning the new state and a key to draw with.

    ``None`` means "no seed was asked for" and passes straight through, which draws from MLX's
    global generator. Splitting rather than reusing keeps the key that produced a token from
    also being the key that produces the next one.
    """
    if state is None:
        return None, None
    keys = mx.random.split(state)
    return keys[0], keys[1]


def _head_geometry(path: Path) -> tuple[int, int]:
    """Read ``head_vocab_offset`` and ``head_vocab_size`` out of a converted model's config."""
    config_file = path / "config.json"
    if not config_file.is_file():
        raise paths.MissingArtifact(f"{path} has no config.json, so it is not a model directory.")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    size = config.get(HEAD_SIZE_KEY)
    offset = config.get(HEAD_OFFSET_KEY)
    if not size or offset is None:
        raise paths.MissingArtifact(
            f"{path} is not an MLX artifact for this engine: its config.json has no "
            f"{HEAD_SIZE_KEY!r}, so its output head is the full vocabulary. The MLX backend "
            f"needs the converted, head-sliced model -- point KOVA_MODEL_PATH at that, or run "
            f"the torch backend against this directory instead."
        )
    return int(offset), int(size)


def _sliced_head_classes(head_size: int) -> Any:
    """A ``get_model_classes`` for ``mlx_lm.utils.load_model`` that narrows ``lm_head``.

    mlx_lm sizes the input embedding and the output projection from one ``vocab_size``, so a
    model with a full input table and a sliced head cannot be described in config alone. This
    subclasses the stock Llama model and rebuilds ``lm_head`` at the width the artifact
    actually stores -- an injection point ``load_model`` already offers, which is a good deal
    safer than patching the class in place and hoping nothing else in the process is loading a
    Llama while it is patched.
    """
    import mlx.nn as nn
    import mlx_lm.models.llama as llama

    class SlicedHeadModel(llama.Model):  # type: ignore[misc, valid-type]
        def __init__(self, args: Any) -> None:
            super().__init__(args)
            if args.tie_word_embeddings:
                raise ValueError(
                    "A head-sliced model must have tie_word_embeddings=false: a tied head is "
                    "the input embedding, which this artifact deliberately keeps full width."
                )
            self.lm_head = nn.Linear(args.hidden_size, head_size, bias=False)

    def classes(config: dict) -> tuple[type, type]:
        if config.get("model_type") != "llama":
            raise ValueError(
                f"The MLX backend supports the Llama architecture; this artifact declares "
                f"model_type={config.get('model_type')!r}."
            )
        return SlicedHeadModel, llama.ModelArgs

    return classes
