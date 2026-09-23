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

## Install

Requires Python 3.10+, [uv](https://docs.astral.sh/uv/), and a CUDA GPU for anything
interactive — it runs on CPU, far slower than real time. No GPU handy? There is a hosted demo of
the same model at [kova.ai](https://kova.ai).

```bash
git clone https://github.com/evalabs-ai/kova-tts
cd kova-tts
uv sync                   # add --all-extras for the server, demo, ASR and finetuning
```

On an Apple Silicon Mac, add `--extra mlx` and use a checkpoint converted for it: a base M1
lands at 0.7–0.8x real time end to end, against several times real time on a recent NVIDIA card.
See [docs/apple-silicon.md](docs/apple-silicon.md).

> [!IMPORTANT]
> **The Hugging Face repository is not published yet.** With no local checkpoints configured,
> `KovaTTS.from_pretrained()` falls through to the Hub and raises a 404 — there is nothing to
> download, and `kova-tts download` cannot help either. Point the paths at checkpoints you have:
>
> ```bash
> cp .env.example .env     # fill in KOVA_MODEL_PATH and KOVA_CODEC_PATH
> uv run kova-tts paths    # shows what resolved, and what didn't
> ```
>
> `.env` is found by walking up from your working directory, so run commands from the checkout
> or export the variables yourself. See [docs/installation.md](docs/installation.md).

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

Two tabs. **Speak** has the text box, a voice picker, a seed, and the sampling controls behind an
accordion you should not need to open. **Clone a voice** shows a short passage to read aloud with
the transcript already filled in — record yourself reading it, press Clone, and the voice is on
the picker. Bring your own recording instead if you prefer; then the transcript is yours to type.

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
uv sync --all-extras
uv run pytest                                 # everything, including GPU and weight-backed tests
uv run pytest -m "not gpu and not weights"    # what CI runs
uv run ruff check . && uv run ruff format .
uv run mkdocs build --strict                  # the docs site
```

Tests that need a CUDA device are marked `gpu`; tests that need real checkpoints are marked
`weights`. CI runs neither.

## License

Not yet chosen. [LICENSE](LICENSE) and [LICENSE-WEIGHTS](LICENSE-WEIGHTS) are placeholders that
grant nothing; assume no license is granted until they are replaced.
