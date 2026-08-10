# Apple Silicon

Kova runs on an M-series Mac. Both halves of the model do, but not the same way: the codec is
torch on Metal, and the language model is [MLX](https://github.com/ml-explore/mlx) reading a
checkpoint converted for it.

Everything on this page was measured on the smallest machine worth trying — a **base M1
MacBook Pro, 16 GB** — with mlx 0.32.0, mlx-lm 0.31.3 and torch 2.13. A larger M-series chip has
several times the memory bandwidth and every number below is bandwidth-bound, so treat these as
a floor.

## Install

```bash
uv sync --extra mlx
```

The `mlx` extra is the MLX runtime and `mlx-lm`, both macOS/arm64 wheels. It is an extra rather
than a platform-marked dependency on purpose: `uv sync --extra mlx` on Linux should fail loudly
rather than install nothing and leave `--backend mlx` mysteriously absent.

Add whatever else you want in the same command — `uv sync --extra mlx --extra server --extra
demo` — because `uv sync` prunes extras it is not asked for.

## The two artifacts

Point `.env` at them and nothing else needs configuring:

```ini
KOVA_MODEL_PATH=/path/to/kova-1b-mlx-4bit-g128
KOVA_CODEC_PATH=/path/to/kova-codec.pt
```

```bash
uv run kova-tts paths
```

`paths` prints which decode loop those artifacts resolve to, before anything is loaded:

```
model     /Users/you/models/kova-1b-mlx-4bit-g128
codec     /Users/you/models/kova-codec.pt
backend   mlx
```

### The model

An MLX artifact, not the bf16 checkpoint. Three things were done to it, and the backend checks
for all three:

* **Quantized**, 4 bits at group 128. 0.63 GB against 2.35.
* **Untied and output-sliced.** `lm_head` is a copy of only the rows generation can emit — the
  8192 audio codes, `<|speech_end|>`, and the handful of unused ids between them — while the
  input embedding keeps all 136576 rows, because prompts contain text and reference audio. That
  takes per-step traffic from 1253M parameters to 990M. The input table costs footprint and
  nothing per step: it is read as a gather, not in bulk.
* **Labelled.** `config.json` gains `head_vocab_size` and `head_vocab_offset`, which is how the
  runtime maps a head row back to a token id without hardcoding where the audio block starts.

A directory without `head_vocab_size` is refused rather than run. mlx-lm would otherwise build a
full-width head, find no weights for it, and generate noise.

### The codec

The same codec as everywhere else: encoder, decoder, semantic encoder and prior, and the config
that shapes them — 0.7 GB, all of it read. Weights are matched by prefix, so a bundle carrying
extra tensors loads too; the 0.7 GB file is what you want to move to a laptop.

## Which backend runs

The artifact decides. `KovaTTS.from_pretrained()` reads the model directory and picks the loop
that can load it, so on a converted checkpoint you get MLX without asking.

| | |
|---|---|
| `--backend auto` (default) | MLX for a converted directory, torch for anything else |
| `--backend torch` | Refuses a converted directory rather than mis-loading it |
| `--backend mlx` | Requires the `mlx` extra |

The two artifacts are not interchangeable. torch cannot read MLX's quantized tensors, and the
sliced head is a shape `transformers` would get wrong, so asking for the backend the checkpoint
is not for is an error with a sentence explaining which file you want.

`GET /health` reports both, because on this hardware they are not the same question:

```json
{"status": "ok", "device": "mps", "backend": "mlx", "sample_rate": 32000}
```

The device is the codec's, and the codec is torch on Metal under either backend.

## Speed

Warm, batch 1, on the base M1.

| Stage | Rate |
|---|---|
| LM decode, MLX 4-bit g128, sliced head | ~77 codes/second, 0.96x real time |
| LM decode, MLX 3-bit g128, sliced head | ~96 codes/second, 1.20x real time |
| LM decode, torch bf16 on Metal | ~19 codes/second, 0.24x real time |
| Codec, whole utterance, fp16 | **10.2x real time** (was 4.6x) |
| Codec, streaming at the default 31-frame window | **5.0x real time** (was 2.3x) |
| **End to end, `stream()`**, 4-bit | **~0.77x real time** |
| Time to first audio, warm | ~0.6–0.7 s |
| Load: LM + codec | ~4 s + ~4 s |

Read those loosely. Run-to-run spread on this machine is about ±8% — bigger than most of the
differences anyone will want to draw from the table — and the end-to-end row is the median of
two texts at two seeds. **A base M1 does not clear real time.** The LM rows are the opposite
kind of number: measured warm with a short KV cache and no prefill, which is the *best* a step
ever goes, and a real utterance pays both.

The two halves take turns on one GPU, so the rates add as reciprocals: `1/(1/0.86 + 1/5.0)` is
0.73. Running them on separate threads does **not** help — measured at 0.98x, i.e. slightly
worse — because both are memory-bound and the GPU is already saturated, so their times are
simply additive.

**The LM is the whole story.** It is ~85% of end-to-end time, and it is not inefficient. A
decode step reads ~526 MB of weights at 4 bits, and this machine's *achievable* read bandwidth
is about 53 GB/s (the 68.25 GB/s figure is the spec, not what a benchmark sees). MLX's quantized
matvec kernels run at 79–82% of spec — at or above what a plain strided read achieves — and a
whole transformer layer lands at 69% of that. Of a decode step, ~71% is weight streaming that
cannot be avoided and ~29% is dispatch and small ops; `mx.compile` on a decoder layer measured
*slower* (0.82x), so collecting that 29% would mean hand-written Metal kernels for the whole
transformer.

The codec, having been the other half of the problem, is now ~15% of the time: making it
infinitely fast would buy about 15%. Nothing here is compute-bound; it is all bandwidth, and
every other M-series chip has 3–6x more of it.

### Sampling stays on the GPU

The decode loop samples inside the MLX graph rather than copying logits to the host: penalty,
temperature, top-k, top-p, the draw and the update to the seen mask are all one unevaluated
expression, and the loop queues the next step before it reads this one's token. That is worth
8% — 74.9 codes/second host-side against 81.6 in-graph.

The cost is that `kova_tts.engine.mlx_sampling` is a second implementation of a sampler that has
to mean exactly what the first one means, since the checkpoint's presets were tuned under that
exact order of operations. `tests/test_mlx_sampling.py` asserts the two produce the same scores
stage by stage.

### The streaming window is worth raising

Every streamed chunk is one pass through the whole decoder stack, and it carries
`2 * LOOKAHEAD + 2 * CONV_PADDING` frames of context it computes and throws away. A 31-frame
window therefore does 47.6 frames of work to emit 31, on tensors small enough that the GPU is
launch-bound on top of that:

| `--decode-window` | Chunk spacing | Codec |
|---|---|---|
| 31 (default) | 388 ms | 5.0x |
| 63 | 788 ms | 5.9x |
| 127 | 1588 ms | 8.0x |
| 191 | 2388 ms | 8.9x |

It changes no audio at any size — windows are bit-comparable with a whole-utterance decode — so
this is purely frame latency traded for throughput. 63 is a reasonable knee if you are rendering
rather than conversing.

```bash
uv run kova-tts serve --decode-window 63
```

### The decoder runs fused Metal kernels

Two changes in `kova-codec` are Metal-only — CUDA and the CPU take the pure-torch path
unchanged, and both are guarded on `x.device.type == "mps"`:

* **`Activation1d` is one kernel.** Upsample 2x, SnakeBeta, downsample was 43 instances of an
  eleven-pass read/modify/write over tensors twice the width of their layer — half of decode
  time. `kova_codec.mps_kernels` fuses the whole sandwich into a single pass that evaluates each
  upsampled sample (and each sine) exactly once, cooperatively per threadgroup, four outputs to
  a thread: **11.4x** on that module, 123 ms down to 11 ms over the decoder's shapes.
* **`ConvTranspose1d` runs as a `conv1d`.** For stride `S` and kernel `2S`, each output sample is
  fed by exactly two input samples, so all `S` phases fall out of one 3-tap forward convolution.
  Bit-for-bit identical, **2.4x** on Metal, where `conv_transpose1d` is poorly served.

Together: whole-utterance decode 4.8x → 10.2x, streaming 2.7x → 8.3x. Accuracy is unchanged —
against a float32 CPU decode the fused float16 path measures 39.6 dB where the unfused one
measures 39.4, and in float32 the two Metal paths agree to 117 dB.

### Getting above real time

**The short answer for a base M1 is: get a bigger chip.** Everything here is bandwidth-bound, an
M1 Pro has ~3x the bandwidth, and it clears real time on the same code and the same artifacts.

On a base M1 the one lever in the artifacts is the **3-bit build**: ~96 codes/second against
~77, because a decode step reads 402 MB where 4-bit reads 526. It is not free. Scored against
the 8-bit reference over 120 teacher-forced steps it drifts about twice as far as 4-bit does
(mean KL 1.65 against 0.89; top-1 agreement 27% against 49%), and it rendered a test sentence as
8.1 s of audio where 4-bit rendered 6.0 s — so the *same sentence* can take longer in wall-clock
even though codes-per-second is higher. Listen before adopting it; 4-bit g128 is what the
defaults assume.

Below 3 bits falls off a cliff. A 2.6-bit build (2-bit MLP, 3-bit attention) is 1.16x faster and
unusable: KL 3.34, top-5 agreement 27%, and it renders a single sentence as 25 s of rambling.

What does **not** reduce the bytes, since that is the question this comes back to: the output
head is already sliced to the emittable rows (6.8 MB, 1.7% of a step), the input embedding is
genuinely a gather (23.9 µs, against the 2.1 ms a bulk read would cost), and MLX caps
`group_size` at 128 so coarser grouping is unavailable. 98% of a decode step is transformer-layer
weights, which leaves bit-width as the only lever.

Things that were tried and did **not** help, so nobody pays for them twice:

* **Threading the codec against the LM**: 0.98x, marginally *worse*. Both are memory-bound and
  the GPU is already saturated, so the two times are simply additive.
* **Quantizing the KV cache**: 15.7 ms/step against 13.9 at 3072 tokens. Dequantization costs
  more than the traffic it saves at this size.
* **Longer chunks.** `MAX_SEGMENT_CHARS` is not a throughput knob: at 240 characters the model
  starts emitting `<|speech_end|>` almost immediately and produces a fraction of the audio.
* **fp16 prefill.** Dequantizing a layer and running an fp16 GEMM is 1.4x faster at prefill
  widths, but dequantization costs 3.4 ms per matrix, which eats the gain.

### Float16 is the codec default, and it is checked

A decode-only codec on any accelerator runs in fp16. On Metal that is 1.3x the speed of fp32 for
39 dB SNR against it — the same trade the CUDA path has always made, and it is the dtype rather
than the device: Metal's fp32 decode tracks the CPU's to 105 dB. Pass `dtype=torch.float32` to
`KovaCodec.from_checkpoint` if you would rather have the last 3 dB than the 30%.

Encoding stays fp32 everywhere, cloning included.

## What does not work here

**LoRA voices.** A peft adapter is bf16 matrices; the MLX model's projections are 4-bit tensors,
and there is nothing to add them to without dequantizing a layer per step. Merge the adapter
into the bf16 checkpoint and convert *that* — one model directory per voice, and point
`KOVA_MODEL_PATH` at the one you want. `--voice` raises a message saying so rather than
producing the base voice quietly.

Cloning from a reference clip is unaffected and works exactly as it does elsewhere: it needs no
adapter, only WavLM, which runs on Metal in fp32.

**Finetuning.** `kova-tts finetune` needs a CUDA box. Nothing stops the Trainer from placing a
1B model plus LoRA optimizer state on a 16 GB Mac, but it is not a configuration anyone should
wait on, and the end-to-end training test skips itself when there is no CUDA device.

**ASR on the GPU.** faster-whisper is CTranslate2, which has CUDA and CPU backends and no Metal
one, so `--asr-device` is `cpu` here. Transcribing a reference clip takes a few seconds rather
than a fraction of one; give `--clone-text` and it never runs.

## Reproducing the numbers

```bash
uv run kova-tts paths                       # which artifacts, which backend
uv run kova-tts generate "..." --out a.wav  # prints seconds of audio and the ratio
uv run python scripts/ws_latency.py --server ws://127.0.0.1:8000
```

`ws_latency.py` reports a **buffer floor**: how far ahead of the player the stream stayed. On
this hardware it is negative, which is the same fact as the table above said in a different
unit — a player that started on the first chunk would run dry before the end.

One measurement caveat worth knowing: **the GPU clocks up over a session**, by around 5%, and
relaxes when it idles. A single pass down a list of variants measures the first cold and the
last warm. Warm up, then interleave.
