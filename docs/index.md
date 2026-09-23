# Kova TTS

Expressive text-to-speech with voice cloning, LoRA finetuning, and a local server.

A Llama-3.2-1B backbone generates discrete audio codes at 80 per second. A neural codec decodes
those codes to 32 kHz mono speech. Both halves live in this repository.

| Package | Role |
|---|---|
| [`kova-tts`](https://github.com/evalabs-ai/kova-tts/tree/main/packages/kova-tts) | The model: inference, voice cloning, server, LoRA finetuning |
| [`kova-codec`](https://github.com/evalabs-ai/kova-tts/tree/main/packages/kova-codec) | The codec: 32 kHz waveforms ↔ codes at 80 tokens/second |

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("Hello world.")
tts.save(wav, "out.wav")
```

## Read this before you start

Two things will bite a first-time reader, so they are stated here rather than buried.

**Model downloads.** The [base model and codec](https://huggingface.co/kova-ai/kova-tts-1)
download automatically on first use. The
[five LoRA voices](https://huggingface.co/kova-ai/kova-tts-1-voices) are a separate download;
set `KOVA_LORA_DIR` to their local directory to enable them.
[Installation](installation.md) covers downloads, local paths, and offline use.

**Research and non-commercial use.** Kova-provided code, base weights, and documentation
are governed by the [main license](https://github.com/evalabs-ai/kova-tts/blob/main/LICENSE).
The pretrained LoRA voices also require their
[Voice Package Supplement](https://huggingface.co/kova-ai/kova-tts-1-voices/blob/main/LICENSE-SUPPLEMENT)
and [NOTICE](https://huggingface.co/kova-ai/kova-tts-1-voices/blob/main/NOTICE).
Commercial use requires a separate written license. Third-party components retain their
own terms; see [component notices](https://github.com/evalabs-ai/kova-tts/blob/main/NOTICE).

## What it does

| Capability | Where |
|---|---|
| Synthesize text to a 32 kHz WAV | [Quickstart](quickstart.md) |
| Stream audio while it is still being generated | [Quickstart](quickstart.md#streaming) |
| Clone a voice from a few seconds of reference audio | [Voice cloning](voice-cloning.md) |
| Train a per-voice LoRA adapter | [Finetuning](finetuning.md) |
| Turn a folder of recordings into a training corpus | [Dataset preparation](dataset-preparation.md) |
| Serve HTTP, Server-Sent Events and a WebSocket session | [Server](server.md) |
| Browser demo with streaming playback | [Demo](demo.md) |
| ComfyUI nodes | [ComfyUI](comfyui.md) |

## What it deliberately does not do

None of these are missing features. Each is a decision with a reason, and
[Architecture](architecture.md#deliberate-non-features) gives the reasons.

- **No text normalization.** `1997` and `Dr.` reach the model exactly as you typed them.
- **No word or phoneme timestamps.** The model emits audio codes, not alignments.
- **One request at a time.** The server generates one utterance at a time and refuses a
  concurrent request with a 409 rather than queueing it.
- **No separate inference runtime.** Plain PyTorch: a preallocated static KV cache and a
  CUDA-graph capture of the single-token decode step, at batch 1. That is the whole optimisation
  story, and it lives in one readable file.

## Performance

Measured on this repository, batch 1, bfloat16, on an RTX 5090 with a 4096-token KV cache.
Reproduce them before relying on them; hardware and driver versions move these numbers a lot.
On an RTX 3090 the same decode loop runs at ~208 codes/second (2.6x real time) — the step is
memory-bound, so it tracks bandwidth closely.

| Measurement | Value |
|---|---|
| Decode throughput, CUDA graph | ~330 codes/second (~4.1x real time) |
| Decode throughput, eager fallback | ~58 codes/second (~0.72x real time) |
| Speedup from the CUDA graph | 5.7x |
| Time to first audio, warm, in-process | ~190 ms |
| Time to first audio, warm, over WebSocket, text already sent | ~200 ms |
| Time to first audio, warm, over SSE | ~230 ms |

Over a WebSocket the figure above is a floor, not a forecast: a session speaks as its text
arrives, so a client that is still receiving the text waits for the text as much as for the GPU.
[When to flush](server.md#when-to-flush) has that measured end to end.

The first generation in a process is much slower than these: the codec is loaded lazily on first
use and cuDNN autotunes its convolutions then. See [Quickstart](quickstart.md#what-warm-means).

On an Apple Silicon Mac the shape is different enough to need its own page. The base M1 figures,
against a 4-bit MLX checkpoint: the LM decodes at ~82 codes/second (1.02x real time), the codec
takes its own share of the same GPU, and the two together land at 0.7–0.8x real time.
[Apple Silicon](apple-silicon.md) has the rest, including what the torch fallback costs and what
does not work there.

## Hosted

Everything here runs locally, on your own GPU. The same model, run by the people who wrote this
repository, is also hosted at [kova.ai](https://kova.ai): a demo you can use in the browser
without setting any of this up, and an API for running it in production, including commercially.

Those are the terms of that service. The code and model downloads are governed separately
by the Research and Non-Commercial Model License and, for pretrained voices, the supplement.

Built with [Kova TTS](https://kova.ai/text-to-speech). Built with Llama.

## Where to go next

- New here: [Installation](installation.md), then [Quickstart](quickstart.md).
- Building on it: [Python API](python-api.md), [Server](server.md).
- Contributing, or wondering why something is the way it is: [Architecture](architecture.md).
