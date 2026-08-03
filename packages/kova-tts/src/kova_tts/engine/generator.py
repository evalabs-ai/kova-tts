"""The batch-1 decode loop: prompt string in, codec codes out.

This is where the throughput is. One request at a time, no scheduler, no padding, no batching --
which is exactly why a plain torch loop can win here, provided the two things that dominate a
1B-parameter decode step at batch 1 are dealt with:

**Python.** An eager ``transformers`` forward costs ~15 ms of host time per step while the GPU
work is under 3 ms, so the loop is launch-bound by a factor of five. A preallocated static KV
cache plus a CUDA graph capture of the single-token step takes the host out of the inner loop
entirely: measured **62 -> 341 codes/second, 5.5x**, on an RTX 5090 with a 4096-token cache.
Prefill stays eager -- it happens once, its shape changes per request, and capturing it would
buy nothing.

**Memory traffic.** Two fixes, both A/B'd on the same box:

* *The LM head.* ``tie_word_embeddings`` is true, so ``lm_head.weight`` **is** the
  136576x2048 embedding matrix -- 559 MB of the 2.51 GB read per step, 22% of the traffic, to
  produce logits for 128k text tokens the model must never emit mid-utterance. The 8193 rows
  that matter (8192 audio tokens plus ``<|speech_end|>``) are gathered once at load into their
  own 34 MB buffer; the input side keeps the full embedding. Worth **1.12x** (3.04 -> 2.72 ms
  per step), against 1.27x if the step were purely bandwidth-bound.
* *Attention.* ``transformers`` materialises the GQA expansion (``repeat_kv``) before calling
  SDPA, because SDPA's native GQA path refuses any attention mask. At batch 1 with one query
  that means writing and re-reading 4x the KV cache every layer: 224 us per layer at a 4096
  cache, against 30 us for the obvious hand-rolled gemv. :func:`decode_attention` does the
  gemv, and only for the single-token step; prefill goes through SDPA unchanged. Worth
  **2.14x** on the decode step, and it is what keeps the step almost flat in cache length.

Both paths -- CUDA graph and eager -- run the same arithmetic and produce the same tokens at
``temperature=0``, which is what makes the fallback testable. Set ``KOVA_DISABLE_CUDA_GRAPH=1``
(or pass ``cuda_graph=False``) to take the eager path.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from kova_tts import paths
from kova_tts.engine import sampling
from kova_tts.engine.types import TTS_SAMPLING, SamplingParams
from kova_tts.tokens import VocabMap, vocab_map

log = logging.getLogger(__name__)

#: Set to 1 to force the eager decode loop. Exists so the two paths can be compared on one box.
ENV_DISABLE_CUDA_GRAPH = "KOVA_DISABLE_CUDA_GRAPH"

#: Prompt plus generation must fit here. 4096 codes is ~51 seconds of audio, which comfortably
#: covers a reference clip plus a long sentence; the cache is allocated up front, so raising it
#: costs memory whether or not a request uses it.
DEFAULT_MAX_CACHE_LEN = 4096

#: Name the fused single-token attention registers itself under with ``transformers``.
ATTENTION_IMPLEMENTATION = "kova_decode"

_registered = False


# --------------------------------------------------------------------------------- attention


def decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
) -> torch.Tensor:
    """Grouped-query attention for a single query position: ``[B, H, 1, D]`` -> ``[B, 1, H, D]``.

    With one query there is no tiling to do and no reason to expand the KV heads: the query
    heads of one group are just four rows of a matrix multiplied against that group's keys.
    Softmax is taken in float32, as SDPA does internally, so the result tracks it to bf16
    rounding.

    `attention_mask` is additive and broadcast over the head dimensions -- ``0`` where a
    position may be attended to, ``-inf`` in the unfilled tail of the static cache.
    """
    batch, heads, queries, dim = query.shape
    if queries != 1:
        raise ValueError(f"decode_attention handles one query position, got {queries}.")
    kv_heads = key.shape[1]
    grouped = query.reshape(batch, kv_heads, heads // kv_heads, dim)

    scores = torch.matmul(grouped, key.transpose(-1, -2)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[..., : key.shape[2]]
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(value.dtype)
    out = torch.matmul(weights, value)
    return out.reshape(batch, heads, queries, dim).transpose(1, 2).contiguous()


def register_attention() -> str:
    """Register the fused decode attention with ``transformers`` and return its name.

    Idempotent, and safe to call before any model is loaded. Anything that is not a
    single-token step falls through to stock SDPA, so prefill behaviour is unchanged.
    """
    global _registered
    if _registered:
        return ATTENTION_IMPLEMENTATION

    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def _attention(
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        dropout: float = 0.0,
        scaling: float | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        if query.shape[2] == 1 and (attention_mask is None or attention_mask.dtype != torch.bool):
            scale = scaling if scaling is not None else query.shape[-1] ** -0.5
            return decode_attention(query, key, value, attention_mask, scale), None
        return sdpa_attention_forward(
            module, query, key, value, attention_mask, dropout=dropout, scaling=scaling, **kwargs
        )

    AttentionInterface.register(ATTENTION_IMPLEMENTATION, _attention)
    _registered = True
    return ATTENTION_IMPLEMENTATION


# --------------------------------------------------------------------------------- generator


class Generator:
    """Autoregressive generation of codec codes, one request at a time.

    Construct with :meth:`from_pretrained`; the constructor takes already-loaded objects so a
    test can drive the whole loop with a small model.

    Not reentrant: one static KV cache, one set of graph input buffers. Starting a second
    generation while a :meth:`stream` is still running raises rather than silently interleaving.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        vocab: VocabMap | None = None,
        device: torch.device | str | None = None,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        cuda_graph: bool | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.vocab = vocab if vocab is not None else VocabMap.from_tokenizer(tokenizer)
        self.device = torch.device(device) if device is not None else _model_device(model)
        self.max_cache_len = int(max_cache_len)
        self.dtype = next(model.parameters()).dtype

        self._lm = model.model  # the bare LlamaModel: no lm_head, so the narrow head can be used
        self._busy = False
        self._lora: _LoraState | None = None

        # Row space: the generator works in indices into `output_ids` rather than token ids,
        # because that is what the narrowed head produces. searchsorted rather than arithmetic,
        # so nothing here assumes where <|speech_end|> sits relative to the audio block.
        out_ids = np.asarray(self.vocab.output_ids, dtype=np.int64)
        self._row_to_id = out_ids
        codes = np.full(out_ids.size, -1, dtype=np.int64)
        audio = self.vocab.is_audio_id(out_ids)
        codes[audio] = self.vocab.ids_to_codes(out_ids[audio])
        self._row_to_code = codes
        self._eos_row = int(np.searchsorted(out_ids, self.vocab.speech_end_id))

        self._head = self._narrow_head(out_ids)
        self._cache: Any = None
        self._graph: dict[str, torch.cuda.CUDAGraph] = {}
        self._graph_pool: Any = None
        self._graph_logits: dict[str, torch.Tensor] = {}
        self._capture_stream: torch.cuda.Stream | None = None
        self.cuda_graph = self._resolve_cuda_graph(cuda_graph)
        self._build_buffers()

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_pretrained(
        cls,
        model: str | os.PathLike[str] | None = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_cache_len: int = DEFAULT_MAX_CACHE_LEN,
        cuda_graph: bool | None = None,
    ) -> Generator:
        """Load the LM and tokenizer, resolving `model` through :mod:`kova_tts.paths`.

        Voices are applied afterwards with :meth:`load_lora`.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        resolved = paths.model_path(model)
        target = torch.device(device) if device is not None else _default_device()
        lm = AutoModelForCausalLM.from_pretrained(
            resolved, dtype=dtype, attn_implementation=register_attention()
        )
        lm.to(target).eval()
        tokenizer = AutoTokenizer.from_pretrained(resolved)
        generator = cls(
            lm,
            tokenizer,
            vocab=vocab_map(resolved),
            device=target,
            max_cache_len=max_cache_len,
            cuda_graph=cuda_graph,
        )
        generator.warmup()
        return generator

    def warmup(self) -> None:
        """Pay the one-time costs now rather than inside somebody's first request.

        Capturing the decode step takes ~1 s, and the first prefill another ~200 ms of cuBLAS
        and cuDNN autotuning. Left lazy, all of that lands in the first request's
        time-to-first-audio -- 1.2 s against the 17 ms a warm prefill costs -- which is the
        difference between a demo that feels instant and one that does not. Load time is the
        right place to spend it: nobody is waiting on a response yet.

        Safe to call repeatedly; work already done is skipped. An adapter loaded later with
        ``merge=False`` needs its own capture, and pays for it on its first use.
        """
        with torch.inference_mode():
            self._ensure_graph()
            self._prefill([int(self._row_to_id[0])] * 4)
            self._step()

    def _narrow_head(self, out_ids: np.ndarray) -> torch.Tensor:
        """Gather the rows of the (tied) embedding the model may actually emit.

        The same arithmetic as the full head: gathering rows of the matrix and multiplying
        reduces over the same 2048 values, so this is a traffic saving rather than an
        approximation. Only the GEMM's tiling differs, which on the shipped bf16 checkpoint
        leaves the logits bit-identical and can never move an argmax.
        """
        weight = self.model.get_output_embeddings().weight
        index = torch.from_numpy(out_ids).to(weight.device)
        with torch.inference_mode():
            return weight[index].detach().clone().contiguous()

    def _resolve_cuda_graph(self, requested: bool | None) -> bool:
        if os.environ.get(ENV_DISABLE_CUDA_GRAPH, "").strip() not in ("", "0"):
            return False
        if requested is False:
            return False
        if self.device.type != "cuda":
            if requested:
                log.warning("CUDA graphs need a CUDA device; falling back to the eager loop.")
            return False
        return True

    def _build_buffers(self) -> None:
        """Allocate everything the decode step reads or writes, once.

        Every one of these keeps its address for the life of the generator, which is what makes
        the captured graph replayable: the sampled token is written into `_in_ids`, the mask and
        the position advance themselves on the device, and the cache writes itself at
        `cumulative_length`.
        """
        from transformers import StaticCache

        self._cache = StaticCache(config=self.model.config, max_cache_len=self.max_cache_len)
        self._in_ids = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        self._position = torch.zeros(1, 1, dtype=torch.long, device=self.device)
        # Additive mask over the whole cache: 0 where a slot holds a real key, -inf elsewhere.
        self._mask = torch.full(
            (1, 1, 1, self.max_cache_len), float("-inf"), dtype=self.dtype, device=self.device
        )
        self._seen = torch.zeros(self._row_to_id.size, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ LoRA

    @property
    def lora(self) -> Path | None:
        """Directory of the adapter currently in effect, if any."""
        return self._lora.active_path if self._lora is not None else None

    def load_lora(self, adapter: str | os.PathLike[str], *, merge: bool = True) -> None:
        """Apply a peft LoRA adapter to the LM.

        With ``merge=True`` (the default) the adapter is folded into the base weights in place:
        the decode step runs no extra kernels at all, and an already-captured CUDA graph stays
        valid, because it reads the same weight tensors and those tensors now hold the merged
        values. Measured **340 codes/second, against 276 unmerged -- 1.23x**. Peft's
        ``merge_adapter`` is used rather than ``merge_and_unload`` for one reason: it is
        reversible, so switching voices unmerges and re-merges (283 ms) instead of reloading
        2.5 GB of weights. The arithmetic folded in is identical either way. Unmerging in
        bfloat16 does not land exactly back on the original weights -- measured max drift
        2.2e-3, 0.3% of the largest weight -- but it is bf16 rounding, not accumulation: it
        stops growing after the first round trip and is unchanged after fifty.

        With ``merge=False`` the adapter stays a separate pair of matrices applied at every
        attention projection, and switching is instant (15 ms) because nothing is written. That
        is the mode for a demo that changes voice constantly; each adapter gets its own captured
        graph, sharing one memory pool.
        """
        if self._lora is None:
            self._lora = _LoraState(self.model)
        # Unmerged adapters bring their own tensors, so a graph captured against a different
        # adapter is stale; graphs are keyed by adapter and recaptured lazily on first use.
        self._lora.activate(Path(adapter).expanduser(), merge=merge)

    def unload_lora(self) -> None:
        """Return to the base voice."""
        if self._lora is not None:
            self._lora.deactivate()

    @property
    def _graph_key(self) -> str:
        """Which captured graph is valid right now.

        Merged adapters mutate the base weights in place, so one graph serves every voice.
        Unmerged adapters bring their own tensors and need their own capture.
        """
        if self._lora is None or self._lora.merged or self._lora.active is None:
            return ""
        return self._lora.active

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

        Stops on ``<|speech_end|>`` or after ``params.max_tokens`` codes, whichever comes first.
        `greedy` forces argmax, which no preset can express (``SamplingParams`` requires a
        positive temperature) and which the tests use to compare the decode paths exactly.
        """
        params = params or TTS_SAMPLING
        ids = self.encode(prompt)
        return self.stream_ids(ids, params, greedy=greedy)

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
        generator = None
        if params.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(params.seed)
        temperature = 0.0 if greedy else params.temperature

        with torch.inference_mode():
            self._ensure_graph()
            logits = self._prefill(ids)
            for step in range(budget):
                row = int(
                    sampling.sample(
                        logits,
                        temperature=temperature,
                        top_p=params.top_p,
                        top_k=params.top_k,
                        repetition_penalty=params.repetition_penalty,
                        previous=self._seen,
                        generator=generator,
                    )
                )
                code = int(self._row_to_code[row])
                if code < 0:  # <|speech_end|>
                    return
                self._seen[row] = True
                yield code
                if step + 1 == budget:
                    return  # the budget is spent; do not pay for a step nobody will sample
                self._in_ids.fill_(int(self._row_to_id[row]))
                logits = self._step()

    # ------------------------------------------------------------------ steps

    def _prefill(self, ids: Sequence[int]) -> torch.Tensor:
        """Run the prompt through the model; logits for the first token to generate."""
        return self._logits(self._prefill_hidden(ids))

    def _prefill_hidden(self, ids: Sequence[int]) -> torch.Tensor:
        """Prefill, returning the raw hidden state of the last prompt position.

        No attention mask is passed: with an empty static cache and positions starting at 0,
        ``transformers`` takes its own fast path -- it slices the cache to the prompt length and
        lets SDPA apply causality itself, which dispatches to the flash kernel.
        """
        self._cache.reset()
        n = len(ids)
        self._mask.fill_(float("-inf"))
        self._mask[..., :n] = 0.0
        self._position.fill_(n)
        self._seen.zero_()
        # Every audio token in the prompt counts as "already seen" for the repetition penalty:
        # the reference clip's codes and any carried context are prompt tokens, and the penalty
        # is defined over prompt and output alike (see :mod:`kova_tts.engine.sampling`).
        prompt = np.asarray(ids, dtype=np.int64)
        audio = prompt[self.vocab.is_audio_id(prompt)]
        if audio.size:
            rows = np.searchsorted(self._row_to_id, audio)
            self._seen[torch.from_numpy(rows).to(self.device)] = True

        input_ids = torch.tensor([list(ids)], dtype=torch.long, device=self.device)
        out = self._lm(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=torch.arange(n, device=self.device).unsqueeze(0),
            past_key_values=self._cache,
            use_cache=True,
        )
        return out.last_hidden_state[:, -1]

    def _step(self) -> torch.Tensor:
        """One decode step, through the captured graph when there is one."""
        graph = self._graph.get(self._graph_key) if self.cuda_graph else None
        if graph is None:
            # Nothing captured for these weights yet -- only reachable when the step is driven
            # directly rather than through a generation, which calls _ensure_graph() first.
            return self._decode_body()
        graph.replay()
        return self._graph_logits[self._graph_key]

    def _ensure_graph(self) -> None:
        """Capture the decode step for the current weights, if it has not been captured yet.

        Deliberately done *before* the prefill rather than lazily inside the loop: capturing
        runs four real decode steps, which would write four junk tokens into the KV cache of a
        generation already in flight. Called here, the prefill that follows resets all of it.
        """
        if not self.cuda_graph or self._graph_key in self._graph:
            return
        # A short dummy prefill, both to allocate the cache's backing tensors and to leave the
        # model in the state a decode step expects.
        self._prefill([int(self._row_to_id[0])] * 4)
        self._capture(self._graph_key)

    def _decode_body(self) -> torch.Tensor:
        """The region the CUDA graph captures: one step, and the logits it produces."""
        return self._logits(self._decode_hidden())

    def _decode_hidden(self) -> torch.Tensor:
        """One decode step: open this position in the mask, step, advance the position.

        Nothing here touches the host. The mask update and the position increment index
        themselves with device tensors, and the static cache writes at its own device-side
        ``cumulative_length``, so a replay of this graph is correct at every position rather
        than only at the one it was captured at.
        """
        self._mask.view(-1).index_fill_(0, self._position.view(-1), 0.0)
        out = self._lm(
            input_ids=self._in_ids,
            attention_mask=self._mask,
            position_ids=self._position,
            past_key_values=self._cache,
            use_cache=True,
        )
        self._position.add_(1)
        return out.last_hidden_state[:, -1]

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project one hidden state onto the narrowed head. Returns float32 ``[8193]``."""
        return F.linear(hidden, self._head).float()[0]

    def _capture(self, key: str) -> None:
        """Warm up on a side stream, then capture one decode step.

        Capture runs a real step, so it advances the cache and the position; the caller is a
        fresh generation away from a prefill, which resets both.

        Both streams are created explicitly on this generator's device, and the capture stream
        is passed to ``torch.cuda.graph`` rather than left to default. That is not belt and
        braces: ``torch.cuda.graph`` lazily builds **one process-wide capture stream** on
        whichever device happened to be current at the first capture in the process, and reuses
        it forever. A second generator on a second GPU would capture onto a stream belonging to
        the first, record zero kernels, and replay as a silent no-op -- leaving the logits
        frozen at their captured values and the decode loop generating nonsense.
        """
        if not self._cache.layers[0].is_initialized:
            raise RuntimeError("Capture needs an initialised KV cache; prefill first.")
        with torch.cuda.device(self.device):
            warmup = torch.cuda.Stream(device=self.device)
            warmup.wait_stream(torch.cuda.current_stream(self.device))
            with torch.cuda.stream(warmup):
                for _ in range(3):
                    self._decode_body()
            torch.cuda.current_stream(self.device).wait_stream(warmup)
            torch.cuda.synchronize(self.device)

            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=self.device)
            graph = torch.cuda.CUDAGraph()
            pool = {} if self._graph_pool is None else {"pool": self._graph_pool}
            with torch.cuda.graph(graph, stream=self._capture_stream, **pool):
                self._graph_logits[key] = self._decode_body()
            self._assert_records(graph, self._graph_logits[key])
        if self._graph_pool is None:
            self._graph_pool = graph.pool()
        self._graph[key] = graph
        log.debug("Captured the decode step for adapter %r", key or "base")

    def _assert_records(self, graph: torch.cuda.CUDAGraph, logits: torch.Tensor) -> None:
        """Prove the graph replays into the output buffer before trusting it.

        An empty capture is not an error in torch -- it warns and hands back a graph whose
        ``replay()`` does nothing at all. That failure is invisible from the outside: generation
        keeps running, on frozen logits. One probe replay at capture time is cheap insurance
        against ever shipping that silently.
        """
        logits.zero_()
        graph.replay()
        torch.cuda.synchronize(self.device)
        if not bool(logits.any()):
            raise RuntimeError(
                f"The captured decode step on {self.device} replays as a no-op, so the graph "
                f"recorded no work. Generation would run on frozen logits. Disable graphs with "
                f"{ENV_DISABLE_CUDA_GRAPH}=1 (or cuda_graph=False) and please report this."
            )


class _LoraState:
    """The peft adapters loaded into one model, and which of them is in effect."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.peft: Any = None
        self.loaded: dict[str, Path] = {}
        self.active: str | None = None
        self.merged = False

    @property
    def active_path(self) -> Path | None:
        return self.loaded.get(self.active) if self.active else None

    def activate(self, adapter: Path, *, merge: bool) -> None:
        config = adapter / "adapter_config.json"
        if not config.is_file():
            raise paths.MissingArtifact(
                f"{adapter} is not a peft adapter directory: no adapter_config.json in it."
            )
        _reject_retrained_embeddings(config)
        name = _adapter_name(adapter)
        if self.active == name and self.merged == merge:
            return
        self._unmerge()
        self._load(name, adapter)
        self.peft.base_model.enable_adapter_layers()
        self.peft.set_adapter(name)
        self.active = name
        if merge:
            self.peft.merge_adapter()
            self.merged = True

    def deactivate(self) -> None:
        """Fall back to the base voice, leaving the adapters loaded for a later switch."""
        self._unmerge()
        if self.peft is not None:
            self.peft.base_model.disable_adapter_layers()
        self.active = None

    def _unmerge(self) -> None:
        if self.merged and self.peft is not None:
            self.peft.unmerge_adapter()
        self.merged = False

    def _load(self, name: str, adapter: Path) -> None:
        if name in self.loaded:
            return
        try:
            from peft import PeftModel
        except ImportError as exc:  # pragma: no cover - only a broken install gets here
            raise ImportError(
                "LoRA voices need peft, which is a base dependency of kova-tts. Reinstall with "
                "`uv sync` or `pip install kova-tts`."
            ) from exc

        if self.peft is None:
            self.peft = PeftModel.from_pretrained(self.model, str(adapter), adapter_name=name)
        else:
            self.peft.load_adapter(str(adapter), adapter_name=name)
        self.loaded[name] = adapter


def _reject_retrained_embeddings(config: Path) -> None:
    """Refuse an adapter that retrains the embedding or the LM head.

    The narrowed head is a copy of the embedding rows taken at load time, so an adapter that
    changes those rows would be applied to the model's input side and silently ignored on its
    output side. No shipped Kova voice does this -- they are attention-only -- but failing
    loudly beats generating subtly wrong audio.
    """
    import json

    data = json.loads(config.read_text(encoding="utf-8"))
    retrained = data.get("modules_to_save") or []
    offenders = [m for m in retrained if "embed" in m or "lm_head" in m]
    if offenders:
        raise ValueError(
            f"Adapter {config.parent} retrains {', '.join(offenders)}. This engine narrows the "
            f"LM head to a copy of the embedding taken at load time and cannot apply an adapter "
            f"to it. Merge the adapter into the base model first with `kova-tts merge`."
        )


def _adapter_name(adapter: Path) -> str:
    """A stable peft adapter name for a directory. peft rejects '.' in adapter names."""
    return adapter.name or "default"


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
