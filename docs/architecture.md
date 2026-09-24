# Architecture

This page is for someone who wants to change the code. It explains the decisions that are not
obvious from reading it, and the ones that will bite you if you do not know about them.

## Two stages

```
text ──▶ Llama-3.2-1B backbone ──▶ audio codes ──▶ neural codec ──▶ 48 kHz waveform
             (kova-tts)             80 per second      (kova-codec)
```

The language model does not produce audio. It produces integers in `[0, 8191]`, 80 per second of
speech, which are indices into the codec's single codebook. The codec turns those into samples.
The two halves are independently loadable, and a TTS process only ever needs the codec's
*decoder* — which is why decoding never touches WavLM.

| | |
|---|---|
| Backbone | Llama 3.2, 1B class: 16 layers, hidden 2048, 32 query heads over 8 KV heads, head dim 64 |
| Vocabulary | 136576 tokens, `tie_word_embeddings: true` |
| Audio tokens | 8192, ids 128256–136447 |
| Precision | bfloat16 for the LM; the codec defaults to float16 for decode-only CUDA, float32 otherwise |
| Sample rate | 32000 Hz mono in, 48000 Hz mono out |
| Hop | 400 input samples per code, so exactly 80 codes/second; 600 output samples per code |
| Codebook | One quantizer, 8192 entries, codebook dim 8 into a 1024-dim VQ space |

## The token layout

The most important thing on this page.

The 8192 audio tokens `<|s_0|>` … `<|s_8191|>` **do** occupy a contiguous id block, 128256 to
136447. But they were added to the tokenizer in *lexicographic string order*, so the block runs
`<|s_0|>, <|s_1000|>, <|s_1001|>, …` rather than `<|s_0|>, <|s_1|>, <|s_2|>, …`:

| Token | Id | `128256 + code` would give |
|---|---|---|
| `<|s_0|>` | 128256 | 128256 ✓ |
| `<|s_1|>` | 129367 | 128257 ✗ |
| `<|s_2|>` | 130478 | 128258 ✗ |
| `<|s_1000|>` | 128257 | 129256 ✗ |
| `<|s_8191|>` | 136245 | 136447 ✗ |

!!! danger "`128256 + code` is wrong and fails silently"

    It produces a valid token id in the audio block, so nothing raises. You get audio. It is
    just the wrong audio. This is the single easiest way to break this codebase.

Every conversion goes through `kova_tts.tokens.VocabMap`, which derives the mapping from the
tokenizer rather than assuming anything:

```python
from kova_tts import vocab_map

vm = vocab_map()                     # cached per model directory
vm.codes_to_ids([0, 1, 8191])        # array([128256, 129367, 136245])
vm.ids_to_codes(ids)                 # strict: raises on a non-audio id
vm.decode_codes(raw_lm_output)       # forgiving: drops EOS and text tokens
vm.is_audio_id(ids)                  # element-wise mask
```

Both directions are dense numpy tables, so conversion in the decode loop is fancy-indexing with
no Python-level loop. `VocabMap.from_tokenizer` validates the shape it depends on and refuses a
tokenizer that does not match: wrong number of audio tokens, an id block with a hole in it, or
no `<|speech_end|>`. Each of those raises with a message naming `KOVA_MODEL_PATH`, because in
practice they all mean the same thing — that is not a Kova checkpoint.

### Structural tokens

Immediately above the audio block:

| Token | Id | |
|---|---|---|
| `<|speech_end|>` | 136450 | Also the model's `eos_token_id`. Generation stops here |
| `<|speech_start|>` | 136451 | Opens the audio section; the LM continues from here |
| `<|text_prompt_end|>` | 136452 | Closes the text section |
| `<|text_prompt_start|>` | 136453 | Opens the text section |
| `<|begin_of_text|>` | 128000 | Llama's BOS, well below the audio block |

Those numbers are documentation. Nothing in the code hardcodes them; `tokens.py` names the
strings and `VocabMap` looks them up.

Alongside the audio codes there are the non-verbal tags — `[laugh]`, `[sigh]`, and 21 others in
`tokens.NON_VERBAL_TAGS`. They are single tokens, so they only work written verbatim, brackets
included, inside the transcript.

## The prompt format

Three shapes, all pure string manipulation in `kova_tts.prompt`. No tokenizer, no weights. The
model only ever sees one flat string, always tokenized with `add_special_tokens=False` so BOS
appears exactly where these functions put it and nowhere else.

```python
tts_prompt("Hello world.")
# <|text_prompt_start|>Hello world.<|text_prompt_end|><|speech_start|>

clone_prompt("what the clip says", "Say something new.", [12, 7, 4095])
# <|begin_of_text|><|text_prompt_start|>what the clip says Say something new.<|text_prompt_end|><|speech_start|><|s_12|><|s_7|><|s_4095|>

training_example("Hello world.", [12, 7])
# <|begin_of_text|><|text_prompt_start|>Hello world.<|text_prompt_end|><|speech_start|><|s_12|><|s_7|><|speech_end|>
```

`tts_prompt` deliberately omits BOS, so a caller that already emits one is not forced to strip it
back off; `KovaTTS` adds `BEGIN_OF_TEXT` itself. The other two include it, because a continuation
has to be tokenized as one string.

The exact byte layout matters. These strings are what the checkpoint saw in training, so a stray
space or a reordered tag puts the prompt off the distribution the model was fitted to.

## Voice cloning as continuation

There is no cloning mechanism. There is only continuation.

`clone_prompt` puts the reference transcript **in front of** the target text, and the reference
clip's codes at the **front of the continuation**. From the model's point of view it is halfway
through an utterance, already speaking in that voice, and it simply carries on.

Everything about cloning follows from that:

| Behaviour | Because |
|---|---|
| The transcript must match the clip word for word | It is the first half of one sentence the model is reading |
| The output starts with a re-rendering of the reference | Those codes lead the continuation, so their audio is produced first. `KovaTTS` trims `voice.ref_seconds` off the front |
| The repetition penalty is low (1.1) | The reference codes at the front *are* repetition; penalising them makes the model drift off the voice |
| `max_tokens` is higher for cloning (3500, against 2048) | The reference is generated before a word of your text |
| A long reference costs prompt, KV cache and decode time | It is all real prompt, and it is all decoded before your text |

The reference is also pushed through the codec before the new speech, so the decoder's LSTM and
first convolution start from real context rather than from silence. `clone_preroll` bounds how
much: `None` uses the whole clip (which is what makes `ref_seconds` the exact amount to trim),
and the server uses 80 codes — one second — because preroll is time to first audio spent
re-rendering audio that is thrown away again. The trim then follows the preroll rather than the
clip.

## The batch-1 decode loop

`kova_tts.engine.generator`. One request at a time, no scheduler, no padding, no batching —
which is exactly why a plain torch loop can win here. Two things dominate a 1B decode step at
batch 1, and both are dealt with.

There are in fact two of these loops, and `kova_tts.engine.backends` picks between them by
looking at the checkpoint rather than at the machine. The torch loop below is the general one.
The other, `kova_tts.engine.mlx_generator`, exists because on Apple Silicon neither of the two
fixes below is available — there are no CUDA graphs, and the traffic problem needs quantized
weights that torch cannot read — so the same two problems are solved with different tools:
4-bit weights, an output head sliced on disk instead of at load time, and a sampler that stays
inside the MLX graph so the host never stalls the loop. Same contract, same sampling order,
same audio. [Apple Silicon](apple-silicon.md) has the numbers.

### Python, and the CUDA graph

An eager `transformers` forward costs about 15 ms of host time per step while the GPU work is
under 3 ms: the loop is launch-bound by a factor of five. A preallocated static KV cache plus a
CUDA graph capture of the single-token step takes the host out of the inner loop entirely.

Batch 1, bf16, RTX 5090, 4096-token cache:

| | codes/second | real time |
|---|---|---|
| CUDA graph | ~330 | 4.1x |
| Eager (`KOVA_DISABLE_CUDA_GRAPH=1`) | ~58 | 0.72x |

**5.7x.** Prefill stays eager: it happens once, its shape changes per request, and capturing it
would buy nothing.

The decode step is memory-bound, so throughput tracks memory bandwidth rather than compute: the
same loop on an RTX 3090 runs at ~208 codes/second (2.6x real time).

For the graph to be replayable, every buffer the step reads or writes must keep its address for
the life of the generator — the input id, the position, the additive mask over the whole cache,
the `seen` mask for the repetition penalty. Nothing in the captured region touches the host: the
mask update and the position increment index themselves with *device* tensors, and the static
cache writes at its own device-side `cumulative_length`. That is what makes a replay correct at
every position rather than only at the one it was captured at.

Two subtleties in `_capture` worth not undoing:

- **The capture stream is created explicitly on the generator's device.** `torch.cuda.graph`
  lazily builds one process-wide capture stream on whichever device happened to be current at
  the first capture, and reuses it forever. A second generator on a second GPU would capture
  onto the first device's stream, record zero kernels, and replay as a silent no-op — leaving
  logits frozen at their captured values and the loop generating nonsense. A related trap sits
  one level up, in deciding *which* device you meant: torch's `cuda:N` is not `nvidia-smi`'s GPU
  N unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. Both failures are silent, and both are worth
  asserting against rather than trusting.
- **The capture is probed before it is trusted.** An empty capture is not an error in torch: it
  warns and hands back a graph whose `replay()` does nothing. `_assert_records` zeroes the
  output buffer, replays once, and raises if nothing was written.

Capture also runs real decode steps, which is why `_ensure_graph` is called *before* the
prefill rather than lazily inside the loop — otherwise it would write junk tokens into the KV
cache of a generation already in flight.

Both paths run the same arithmetic and produce the same tokens at `temperature=0`, which is what
makes the eager fallback testable; `test_the_cuda_graph_and_the_eager_loop_agree` does exactly
that.

### Memory traffic

Two fixes, both measured on the RTX 5090 above.

**The LM head.** `tie_word_embeddings` is true, so `lm_head.weight` *is* the 136576 × 2048
embedding matrix: 559 MB of the 2.51 GB read per step, 22% of the traffic, to produce logits for
128k text tokens the model must never emit mid-utterance. The 8193 rows that matter — 8192 audio
tokens plus `<|speech_end|>` — are gathered once at load into their own 34 MB buffer. The input
side keeps the full embedding. Worth **1.12x** (3.04 → 2.72 ms per step), against 1.27x if the
step were purely bandwidth-bound.

This is a traffic saving, not an approximation: gathering rows of the matrix and multiplying
reduces over the same 2048 values. Only the GEMM's tiling differs, which cannot move an argmax
on this checkpoint.

**Attention.** `transformers` materialises the GQA expansion (`repeat_kv`) before calling SDPA,
because SDPA's native GQA path refuses any attention mask. At batch 1 with one query that means
writing and re-reading 4x the KV cache every layer: 224 µs per layer at a 4096 cache, against
30 µs for the obvious hand-rolled gemv. `decode_attention` does the gemv and registers itself
with `transformers` as an attention implementation; anything that is not a single-token step
falls through to stock SDPA, so prefill is unchanged. Worth **2.14x** on the decode step, and it
is what keeps the step almost flat in cache length.

### Row space

The generator works in indices into `output_ids` — the sorted 8193 ids the model may legally
emit — rather than in token ids, because that is what the narrowed head produces. The mapping
back is `searchsorted`, not arithmetic, so nothing assumes where `<|speech_end|>` sits relative
to the audio block.

The repetition penalty is seeded from the prompt: every audio token already in the prompt
counts as seen, so a reference clip's codes and any carried context are penalised the same way
generated tokens are.

The MLX backend works in the same idea and different arithmetic: its head is a *contiguous
slice* of the id space rather than a gather of scattered rows, so a row is `id - offset`, and
the two ids inside that slice which are not emittable are biased to `-inf` instead of being
left out. The offset and the width come from the artifact's own `config.json`, and a head that
does not cover every emittable id is rejected at load.

### Not reentrant

One static KV cache, one set of graph input buffers. Starting a second generation while one is
running raises rather than interleaving. This propagates all the way up: the
[server](server.md#concurrency) refuses with a 409, the demo serialises, ComfyUI runs one node
at a time.

## Streaming decode

`kova_tts.engine.decoder`. The codec decoder has exactly two sources of cross-frame context, and
streaming means reproducing both by hand.

**An LSTM, unbounded in the past.** Carried window to window as an explicit state.
`decode_with_lstm` returns the state as of a chosen frame and takes it back on the next call.

**A first convolution reaching 3 frames either side.** Each window is fed `CONV_PADDING = 3`
extra codes of real context on each end; the codec trims exactly those frames off again after
the convolution, so what remains is bit-for-bit what a whole-utterance decode produced there.

Everything after the LSTM has a finite receptive field to the right, so each window is also
decoded with `LOOKAHEAD = 9` frames of future codes that are computed and thrown away. What is
left is `WINDOW = 31` frames — **387.5 ms at 80 codes/second**, which is where the "about 390 ms
per chunk" everywhere else comes from.

`plan_window` is pure arithmetic — no codec, no torch — so the window layout is testable on its
own. Two failure modes are baked into it, and both are silent if you get them wrong:

1. **Never emit audio the LSTM produced from padding.** If the final window runs past the last
   real code and that audio is emitted, the last ~20 frames drift from a whole-utterance decode
   by up to 0.065 — an order of magnitude worse than the ~6e-3 the seams cost otherwise. The
   final window is sized to end exactly at the last real code, and asks for no state back.
2. **Never ask for the LSTM state at the very end of a window.** `return_lstm_state=k` splits
   the window at `x[:k]` / `x[k:]`, so a window whose conv-trimmed length is exactly `k` makes
   the second half empty and torch raises *Expected sequence length to be larger than 0 in RNN*.
   The invariant that avoids it — a non-final window always has more than `WINDOW` frames of
   output — is asserted rather than left to luck, because it only holds while `lookahead >= 1`.

There is a third, subtler one: a steady-state window must not fire before every real code it
reads exists. The padding on the right of a *finished* stream is real padding; the end of an
unfinished one is not, and firing early puts up to 0.05 of error on the window's last frames.

Streaming against `decode_all` on the same codes, max absolute difference. The figure depends on
the signal, so both the source and the length are worth stating:

| Signal | Codec dtype | max abs difference |
|---|---|---|
| A real 304-code generation | float32 | 2.6e-3 |
| A real 304-code generation | float16 (the decode-only CUDA default) | 4.0e-3 |
| The tests' synthetic waveform, 400 and 2400 codes | float32 | 5.5e-3 and 6.3e-3, ~60 dB SNR |

Dominated by the seams, which is why the figure grows with the number of windows rather than
with the length of the audio. The tests assert `atol=1e-2`, a margin of roughly 1.6x over the
worst of these. `decode_all` is still the right call when the codes are already complete
— one kernel launch per layer instead of one per window, and no seams at all.

`StreamingDecoder.prime` is the cloning hook: reference codes are pushed through so the LSTM and
convolution start warm, and the audio they produce is discarded by sample count.

## Long text: segment carry

`KovaTTS.split_sentences` packs whole sentences into chunks of up to
`MAX_SEGMENT_CHARS = 160` characters, so two or three short sentences are generated together
rather than one at a time. Only a sentence longer than that on its own is broken internally, at
a clause boundary first and whitespace second, because a break mid-phrase is audible. A trailing
chunk under `MIN_SEGMENT_CHARS = 24` is merged *backwards* — "Yes." alone gives the model too
little to work with — which is the one case a chunk exceeds the target, and it is safe because
the last chunk is never carried.

Each chunk is then generated with the previous one threaded into its prompt: **its text in front
of the new text, its codes in front of the continuation.** Without that, every chunk restarts the
model's prosody from nothing. With no LoRA and no reference to anchor it, the model picks a
different speaker each time and a paragraph audibly changes voice partway through.

Text and codes have to travel *together*. Carrying codes alone reliably makes the model emit
`<|speech_end|>` on the first step: it has been handed several seconds of speech for a sentence
it has not started, so as far as it can tell the sentence is already finished.

The carry is therefore a matched pair, and cannot be truncated to a tail without the two
drifting apart. The unit of carry is a whole chunk, which is why the chunk size is the lever:

```python
CODES_PER_CHAR    = 6.0                                       # an upper bound, not an average
MAX_SEGMENT_CHARS = 160
MAX_CARRY_CODES   = ceil(MAX_SEGMENT_CHARS * CODES_PER_CHAR)  # 960 codes, about 12 seconds
```

The carry limit is *derived* from the chunk size rather than chosen independently, so a chunk the
splitter emits always fits. That invariant is asserted in the tests. It has to be: when the two
were independent numbers, 300-character chunks against a 640-code carry meant the carry was
silently discarded for any sentence over ~140 characters, and ordinary prose lost it on two
joins out of three.

Sizing the chunk down beats sizing the carry up, because the prompt also has to hold a cloned
voice's reference. A 160-character chunk leaves room for a reference of about 24 seconds before
anything has to give; keeping 300-character chunks and growing the carry to match would overflow
the cache on *every* cloned voice with a 10-second reference. `_prompt_ids` checks the real
tokenized length against `max_cache_len` and drops the carry if the generation would not fit, so
the failure mode is a lost carry rather than a crash.

Carried codes are prompt-only. They are never decoded twice, so the audio runs straight through
the boundary.

## LoRA voices

Adapters are peft LoRAs on the attention projections only (`q_proj`, `k_proj`, `v_proj`,
`o_proj`), rank 64 with alpha 64. About 55 MB each, 13.6 M trainable parameters.

Two modes, and the difference is measurable. The absolute figures below are from the same
RTX 5090 as the head and attention numbers above; the ratios are what to rely on:

| | Throughput | Voice switch | Graph |
|---|---|---|---|
| `merge_lora=True` (default) | 340 codes/s | 283 ms (unmerge + re-merge) | One graph serves every voice |
| `merge_lora=False` | 276 codes/s | 15 ms (nothing is written) | One capture per adapter, sharing a memory pool |

Merged is **1.23x** faster because the decode step runs no extra kernels at all, and an already
captured graph stays valid — it reads the same weight tensors, and those tensors now hold the
merged values.

peft's `merge_adapter` is used rather than `merge_and_unload` for one reason: it is reversible,
so switching voices unmerges and re-merges instead of reloading 2.5 GB of weights. Unmerging in
bfloat16 does not land exactly back on the original weights — measured max drift 2.2e-3, 0.3% of
the largest weight — but it is bf16 rounding, not accumulation: it stops growing after the first
round trip and is unchanged after fifty.

**An adapter may not retrain the embedding or the LM head.** The narrowed head is a copy of the
embedding rows taken at load time, so an adapter with `embed_tokens` or `lm_head` in
`modules_to_save` would be applied on the input side and silently ignored on the output side.
Loading one raises and tells you to merge it first.

## Finetuning

`kova_tts.finetune`. A corpus is a JSONL file of `{"text": ...}` rows built by
`prompt.training_example` — a readable text file you can diff and grep, tokenized at load time
rather than stored as ids.

**Loss starts after `<|speech_start|>`.** Every label up to and including that tag is `-100`;
supervision runs from the first audio token through `<|speech_end|>` inclusive.

**Rows over `max_length` are skipped, not truncated.** Truncating would cut off `<|speech_end|>`
and teach the model that utterances never end.

The ending-weighted loss is the quality lever. An ending is a handful of token positions out of
thousands, so under plain cross entropy the signal that decides "stop here" is drowned out by
the signal that decides "keep talking in this voice", and adapters trail off or overrun the
transcript. The fix reweights: the last `ramp_tokens` positions before `<|speech_end|>` get a
multiplier rising 1.0 → `ramp_max`, and the `<|speech_end|>` label gets `eos_token_weight`.

The alignment is the thing to get right. A causal LM predicts token `i` from position `i-1`, so
labels are shifted left by one against logits. The weight channel lives in *label* space —
`ending_w[i]` scales the cost of predicting `labels[i]` — and is therefore shifted with the
labels, not with the logits. An off-by-one here would quietly weight the wrong tokens and
degrade a run without ever failing a test, which is why `weighted_lm_loss` is tested directly
against hand-computed numbers.

The weights must stay anchored on `<|speech_end|>`. Weighting silence instead teaches "any quiet
stretch means stop", and the model truncates mid-sentence.

## The codec

`kova_codec`. Encode and decode are asymmetric, and the asymmetry is the point.

**Encode** resamples 32 kHz to 16 kHz, takes WavLM-large layer 23 hidden states, stretches them
from WavLM's 50 Hz onto the codec's 80 Hz frame grid with linear interpolation, concatenates
them with the acoustic encoder's output, projects through `fc_prior`, and quantizes to one
codebook of 8192 entries. Layer 23 is what the codec was trained against and is not a knob.

**Decode** needs none of that: codebook lookup, a first convolution, an LSTM, then upsampling by
`prod(up_ratios) = 600` samples per frame, which is what makes the output 48 kHz. The decoder was
trained on the same frozen encoder and codebook as the older 32 kHz one (`up_ratios` ending in 2
rather than 3, 400 samples per frame), so the two are interchangeable on the same codes; the
checkpoint's config picks the ratios, and `KovaCodec.sample_rate` and `hop_length` report them.
Nothing downstream assumes a rate — it reads `sample_rate` off the codec. So `decode_only=True` skips WavLM entirely — about
1.2 GB lighter, several seconds faster to start, and it never imports `transformers` inside the
codec. Plain TTS and LoRA voices only ever decode.

Weight norm is a training-time reparameterisation and is folded once at load. Codes below zero
are treated as padding — their embeddings are zeroed, which is exactly what a whole-utterance
decode sees beyond the ends of the sequence, and is what makes the streaming windows line up.

### Decode speed on CUDA

Three things keep the codec a small share of generation time on a GPU. Measured on an RTX 3090,
they take a steady streaming window from 23 ms to 5 ms, and streaming decode from ~200 ms to
~18 ms per second of audio.

- **cuDNN only where its plans are reused.** cuDNN builds an execution plan for every
  convolution shape it has not seen, about 1.2 s for the codec's dozens of shapes. The steady
  window is one shape, planned once; the last window and a whole-utterance decode are a new
  length almost every time, so they run on torch's own kernels instead
  (`kova_tts.engine.decoder.cudnn`). That plan cache is also per thread, which is why all model
  work runs on one long-lived engine thread (`kova_tts.server.engine.engine_thread`).
- **The steady window is a CUDA graph**, captured once per codec during warm-up and replayed
  for every window, the way the LM's decode step is (`kova_tts.engine.decoder.WindowGraph`).
  It is bit-identical to the eager decode. `KOVA_DISABLE_CUDA_GRAPH=1` turns off both graphs.
- **The anti-aliased activation is fused in Triton** (`kova_codec.triton_kernels`): upsample,
  SnakeBeta and downsample in two kernels instead of about eight, the counterpart of the Metal
  kernel on Apple Silicon. Triton compiles for whatever GPU it runs on. The result matches the
  torch path to 1e-6 in float32, and float16 decode fidelity is unchanged.
  `KOVA_CODEC_TRITON=0` forces the torch path.

## Deliberate non-features

Each of these is a decision. None of them is a bug.

### No text normalization

`1997`, `Dr.`, `$40` and `10:30` reach the model exactly as typed. There is no number expander,
no abbreviation table, no G2P front end. The model was trained on text, and a normalizer that
guesses wrong ("Dr." → "doctor" in "Dr. Martin Luther King Dr.") is worse than no normalizer,
because it is invisible from the outside. Normalize in your own layer, where you know the domain.

### No word or phoneme timestamps

The model emits audio codes; there is no alignment anywhere in the pipeline to read one off. You
know the total duration (`len(codes) / 80`) and, when streaming, when each 390 ms window landed
— and that is all. Force-align the output with a separate tool if you need word times.

### No concurrency in the server

One static KV cache, one set of CUDA graph buffers, batch size 1. A second caller waits up to
`--busy-timeout` seconds and is then refused with a 409. Scale with more processes, one per GPU.

### Plain PyTorch, batch 1

No separate inference runtime, and nothing here writes a kernel. At batch 1 the two things a
serving stack is built to do — schedule across requests, and page a large KV cache — have
nothing to schedule and one sequence to page, so what is left is the cost of the step itself.
The preallocated static cache, the CUDA graph over the single-token step, the narrowed head and
the single-query attention gemv are what address that, and they are the whole optimisation
story: four changes, all of them ordinary torch, all of them in `engine/generator.py`, which you
can step through.

The MLX backend is the one exception, and it earns it: on Apple Silicon the equivalent changes
are not expressible in torch at all. It is still a plain loop in one file, reading a checkpoint
converted ahead of time rather than reshaping one at load.

### No sample-audio evaluation during training

Metrics are loss only. Listening to a few fixed prompts every few hundred steps says more about
a voice than `eval_loss` does, and nothing here does it. `finetune.train.run(callbacks=...)`
takes `transformers.TrainerCallback` objects, so anything beyond loss metrics can be attached
from outside without editing the trainer.

## Testing

```bash
uv run pytest -m "not gpu and not weights"    # what CI runs
uv run pytest                                 # adds GPU and weight-backed tests
```

Two markers: `gpu` needs a CUDA device, `weights` needs real checkpoints. CI runs neither, and
sets `KOVA_DISABLE_DOTENV=1` so a developer's `.env` can never change what CI does.

The shape worth copying when you add tests: expensive dependencies are declared structurally and
substituted. `data.encode.Encoder` is a `Protocol` with one method so a deterministic fake gives
a sharper assertion than 1.2 GB of WavLM would; `server.create_app(tts=...)` drives every
endpoint against a stub without a GPU, a checkpoint, or torch; `plan_window` is pure arithmetic
so window layout is tested without a codec.

No audio is committed, ever — including test fixtures. `.gitignore` blocks every audio extension
the pipeline accepts, and a test enforces that those two lists agree. Tests that need real audio
read a path you supply through `KOVA_TEST_AUDIO` and skip when it is unset.
