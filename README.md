<div align="center">

# Kova TTS

Expressive text-to-speech with voice cloning, LoRA finetuning, and a local server.

</div>

A Llama-3.2-1B backbone generates discrete audio codes at 80 per second; a neural codec decodes
them to 48 kHz speech. Both halves live here.

| Package | Role |
|---|---|
| [`kova-tts`](packages/kova-tts) | The model: inference, voice cloning, server, LoRA finetuning |
| [`kova-codec`](packages/kova-codec) | The codec: 32 kHz waveforms -> codes at 80 tokens/second -> 48 kHz waveforms |

**Documentation: [`docs/`](docs/index.md)**, or `uv run mkdocs serve` for the rendered site.

Model files are hosted separately on Hugging Face:
[base model and codec](https://huggingface.co/kova-ai/kova-tts-1) and
[five LoRA voices](https://huggingface.co/kova-ai/kova-tts-1-voices).

## Install

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for anything
interactive — it runs on CPU, far slower than real time. No GPU handy? There is a hosted demo of
the same model at [kova.ai](https://kova.ai).

```bash
git clone https://github.com/evalabs-ai/kova-tts
cd kova-tts
uv sync                   # core inference; add extras for the demo, server or training
```

On an Apple Silicon Mac, add `--extra mlx` and use a checkpoint converted for it: a base M1
lands at 0.7–0.8x real time end to end, against several times real time on a recent NVIDIA card.
See [docs/apple-silicon.md](docs/apple-silicon.md).

The base model and codec download automatically on first use. To download them ahead of time:

```bash
uv run kova-tts download
```

For the five pretrained voices, download the separate adapter package and set its local path:

```bash
uv run hf download kova-ai/kova-tts-1-voices --local-dir ./voices
export KOVA_LORA_DIR="$PWD/voices"
```

The available voice names are `bdl`, `slt`, `jmk`, `awb`, and `Kathleen`.
See [Installation](docs/installation.md) for local checkpoint paths, authentication,
and offline use.

## Quickstart

Start with the demo — it is the fastest way to hear the model, and it exercises everything:
streaming playback, the installed voices, and cloning.

```bash
uv sync --extra demo
uv run kova-tts demo --preload
```

```
Kova TTS on http://127.0.0.1:7860
```

Open it, type something, press **Speak**. Audio starts playing while the rest is still being
generated — about a quarter of a second in on a warm CUDA box, not after the clip is finished.
`--preload` loads the weights at startup so the first press pays none of that; without it the
first generation takes a few seconds longer and every one after is the same.

One prompt box does everything. **Random** fills it with an example, the voice picker sits in its
toolbar, and the sampling controls are behind a settings button you should not need to press.
**New voice** opens the cloning panel right under the text: a short passage to read aloud with
the transcript already filled in — record yourself reading it, press Clone, and the new voice is
selected. Bring your own recording instead if you prefer; it is transcribed for you to check.

## From Python

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("Hello world.")        # float32 mono numpy at 48 kHz
tts.save(wav, "out.wav")
```

A fresh install has no LoRA voices — `tts.voices()` is empty until you install or train one.
Cloning needs no training at all, but the transcript has to match the clip word for word,
because cloning continues the recording:

```python
voice = tts.clone("reference.wav", transcript="exactly what the clip says")
wav = tts.generate("Say something new.", voice)
```

Don't have the transcript? Install the `data` extra and hand `KovaTTS` a transcriber — there is
no ASR in the package itself, only the seam for one:

```python
from kova_tts.cli import asr_transcriber          # needs: uv sync --extra data

tts = KovaTTS.from_pretrained(transcriber=asr_transcriber())
voice = tts.clone("reference.wav")        # transcribed automatically
```

> A permissive license on a recording does not grant the right to reproduce the speaker's voice.
> You are responsible for having the rights to the voice you clone.

Both of these are also commands — see [`examples/`](examples) for the longer versions:

```bash
uv run kova-tts generate "Hello world." --out out.wav
uv run kova-tts generate "Say something new." --clone-audio reference.wav --out cloned.wav
uv run kova-tts --help    # paths, generate, prepare-data, finetune, merge, serve, demo, download
```

The first generation in a process is slow: the codec loads lazily on it. Warm, generation ran at
about 4x real time on an RTX 5090 and 2.6x on an RTX 3090, with audio starting about 190 ms in,
and about 0.75x on a base M1. The decode step is memory-bound, so it tracks bandwidth; measure
your own hardware before relying on a number. [Details](docs/index.md#performance).

## What else is here

| | |
|---|---|
| [Local server](docs/server.md) | HTTP, Server-Sent Events, and a streaming WebSocket session |
| [Browser demo](apps/demo/README.md) | Gradio, streaming playback, voice cloning |
| [ComfyUI nodes](apps/comfyui/README.md) | Load once, generate, clone |
| [Dataset preparation](docs/dataset-preparation.md) | A folder of your recordings -> a JSONL corpus |
| [LoRA finetuning](docs/finetuning.md) | A per-voice adapter, ~55 MB, minutes on one GPU |
| [Architecture](docs/architecture.md) | Token layout, prompt format, the decode loop, streaming |

## Deliberate non-features

This is a model and local inference release, not a production server release. The included
server is intended for local use, experimentation, and integration development. Production
deployments need their own access control, concurrency management, and serving optimizations.
With production serving optimizations on NVIDIA H100 GPUs, Kova achieves under 100 ms time
to first audio in our optimized serving system. The local runtime included here is a
separate implementation with the measurements reported above.

No text normalization, no word timestamps, one request at a time. Plain PyTorch — a static KV
cache and a CUDA-graph decode step at batch 1. Each has a reason — see
[Architecture](docs/architecture.md#deliberate-non-features) before filing a bug.

## Hosted

Everything in this repository runs locally. The same model, run by the people who wrote this
repo, is also hosted at [kova.ai](https://kova.ai): a demo you can use in the browser, and an
API for running it in production, including commercially.

That is a statement about the terms of that service, and only about that service. It grants
nothing here — see [License](#license).

## Development

```bash
uv sync --extra server --extra demo --extra data --extra finetune
uv run pytest                                 # everything, including GPU and weight-backed tests
uv run pytest -m "not gpu and not weights"    # what CI runs
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict                  # the docs site
```

Tests that need a CUDA device are marked `gpu`; tests that need real checkpoints are marked
`weights`. CI runs neither.

## License

Kova-provided source code, base-model weights, and documentation are governed by the
[Research and Non-Commercial Model License](LICENSE). Commercial use of these materials,
derived models, or generated output requires a separate written commercial license.

The five pretrained LoRA voices also require the
[Voice Package Supplement](https://huggingface.co/kova-ai/kova-tts-1-voices/blob/main/LICENSE-SUPPLEMENT)
and [voice NOTICE](https://huggingface.co/kova-ai/kova-tts-1-voices/blob/main/NOTICE).
See [LICENSE-WEIGHTS](LICENSE-WEIGHTS) for the model repository links.
Third-party components retain their own terms; their notices and full license texts are in
[NOTICE](NOTICE) and [licenses/third-party](licenses/third-party/README.md).

Built with [Kova TTS](https://kova.ai/text-to-speech). Built with Llama.
License questions: legal@evalabs.ai.
