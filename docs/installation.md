# Installation

## Requirements

- Python 3.10 or newer.
- [uv](https://docs.astral.sh/uv/). Everything below assumes it; `pip` works too, but uv is what
  the lockfile and the CI job use.
- A CUDA GPU for anything interactive, or an Apple Silicon Mac — see
  [Apple Silicon](apple-silicon.md). A Mac runs the published checkpoint as it is, under torch
  on Metal, but well below real time; the faster MLX build is not published yet. The code runs on CPU too — the tests do — but
  generation is far slower than real time there, so a CPU box is for development, not for
  listening.
- Model weights from Hugging Face, downloaded automatically or provided locally.
  See [Getting the weights](#getting-the-weights).

### Choosing a GPU on a multi-GPU machine

Everything that takes a device — `--device cuda:1`, `CUDA_VISIBLE_DEVICES`, `KovaTTS(device=...)`
— names a **torch** device index. `CUDA_VISIBLE_DEVICES` picks the same card `cuda:1` would, but
it also renumbers what is left, so the process then sees that card as `cuda:0`.

!!! warning "`nvidia-smi`'s GPU number is not torch's `cuda:N`"

    CUDA enumerates devices `FASTEST_FIRST` by default, while `nvidia-smi` reports them in PCI
    bus order. On a mixed-generation box the two disagree, and nothing warns you: you simply run
    on a different card than you meant to. Check, do not assume:

    ```bash
    python -c "import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    ```

    Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` to make torch agree with `nvidia-smi`. Do it in the same
    environment as the run, not just in the shell you checked from.

## Install from a checkout

```bash
git clone https://github.com/evalabs-ai/kova-tts
cd kova-tts
uv sync
```

`uv sync` gives you the two packages, their runtime dependencies, and the `dev` and `docs`
dependency groups. It does **not** install the optional extras.

## Extras

Each extra is a feature you may not want. Install the ones you need:

| Extra | Installs | Needed for |
|---|---|---|
| `server` | fastapi, uvicorn, pydantic, websockets, httpx | `kova-tts serve`, and the streaming examples |
| `demo` | gradio | `kova-tts demo` |
| `data` | faster-whisper | Transcribing reference clips and recordings without transcripts |
| `finetune` | accelerate, pyyaml | `kova-tts finetune` and `kova-tts merge` |
| `mlx` | mlx, mlx-lm | The MLX decode loop on [Apple Silicon](apple-silicon.md). macOS/arm64 wheels only |

```bash
uv sync --extra server            # one extra
uv sync --extra server --extra demo --extra data --extra finetune
```

On Apple Silicon no extra is needed for the published checkpoint, which runs under torch. Add
`--extra mlx` only for an MLX-converted checkpoint. This extra is not available on Linux.
`uv sync` prunes as well as installs: switching to `uv sync --extra demo` removes other extras.
Name every extra you want in one command.

`peft` is a base dependency rather than part of `finetune`, because loading a LoRA voice is an
inference feature — `--voice` works on a plain `uv sync`.

## Getting the weights

Four artifacts. Only the first two are always needed.

| Artifact | What it is | Environment variable | Needed for |
|---|---|---|---|
| Model | Directory with `config.json`, weights and `tokenizer.json` | `KOVA_MODEL_PATH` | Everything |
| Codec | A single checkpoint file | `KOVA_CODEC_PATH` | Everything |
| WavLM | A `microsoft/wavlm-large` directory or repo id | `KOVA_WAVLM_PATH` | Encoding audio: cloning, dataset prep |
| LoRA voices | Directory with one subdirectory per voice | `KOVA_LORA_DIR` | `--voice` |

The model, codec, and WavLM resolve in the same order: the value you passed in code or on the
command line, then the environment variable (read from the nearest `.env`), then the Hugging
Face Hub. LoRA voices are resolved from a configured local directory.

The base model and codec are hosted in
[`kova-ai/kova-tts-1`](https://huggingface.co/kova-ai/kova-tts-1). They download automatically
on first use. To prefetch them, including the repository's legal documents:

```bash
uv run kova-tts download
```

WavLM downloads separately from `microsoft/wavlm-large` when encoding audio is needed.
Use `uv run kova-tts download --wavlm` to prefetch it as well.

For a private or gated repository, authenticate with `uv run hf auth login` using an account
that has access. Public, ungated downloads do not require a login.

Download the voice package separately, retaining its license files:

```bash
uv run hf download kova-ai/kova-tts-1-voices --local-dir ./voices
export KOVA_LORA_DIR="$PWD/voices"
```

The voice folders are `bdl`, `slt`, `jmk`, `awb`, and `Kathleen`. The base model speaks
without an adapter; use one of these names with `--voice` after installing the voices.
Do not copy an adapter into a standalone distribution without the voice package's
`LICENSE`, `LICENSE-SUPPLEMENT`, and complete `NOTICE`.

### Using local checkpoints

To use files already on disk, copy the template and set their paths:

```bash
cp .env.example .env
```

```bash title=".env"
KOVA_MODEL_PATH=/models/kova-tts-1
KOVA_CODEC_PATH=/models/kova/codec.pt
KOVA_WAVLM_PATH=/models/wavlm-large
KOVA_LORA_DIR=/models/kova/voices
```

`.env` is gitignored. The model, codec, and WavLM fall back to their Hub repositories
when unset. LoRA voices need a local `KOVA_LORA_DIR`; they are not downloaded automatically.

## Check that it worked

```bash
uv run kova-tts paths
```

```
config    /home/you/kova-tts/.env
model     /models/kova-tts-1
wavlm     /models/wavlm-large
codec     /models/kova/codec.pt
loras     /models/kova/voices
voices    my_voice
```

It exits `0` when everything resolved and `1` when anything did not, so it is usable in a
script. Run it first whenever something cannot find a checkpoint — it is the cheapest command in
the project. It loads no model, and drags in neither transformers, peft, gradio nor
faster-whisper; the CLI imports each subcommand's dependencies inside that subcommand.

!!! note "`.env` is found relative to your working directory"

    The lookup walks up from the directory you are standing in, so a command run from inside the
    checkout finds it and the same command run from `/tmp` does not. If you work outside the
    repository — a finetuning run beside its corpus, for instance — export `KOVA_MODEL_PATH` and
    friends in your shell, or set the paths explicitly (`--model`, `--codec`, or `model:` in a
    finetuning config). Set `KOVA_DISABLE_DOTENV=1` to skip the `.env` lookup entirely.

    Real environment variables always win over `.env`.

## Offline and container installs

`kova-tts download` prefetches into the local Hub cache so a later run can use the
cached files:

```bash
uv run kova-tts download            # LM and codec
uv run kova-tts download --wavlm    # and WavLM, ~1.2 GB, only needed to encode audio
```

Artifacts already pointed at a local path by `.env` are reported and skipped rather than
downloaded a second time. `--cache-dir` fills a specific cache, `--token` authenticates against
a private or gated repository, and `--repo` overrides the source.

For a container image, see [Docker](docker.md).

## Development

```bash
uv sync --extra server --extra demo --extra data --extra finetune
uv run pytest -m "not gpu and not weights"    # no GPU and no checkpoints needed
uv run pytest                                 # everything, including GPU and weight-backed tests
uv run ruff check . && uv run ruff format .
```

Tests that need a CUDA device are marked `gpu`; tests that need real checkpoints are marked
`weights`. CI runs neither, and runs with `KOVA_DISABLE_DOTENV=1` so a developer's `.env` can
never change what CI does.

A few tests read audio from your machine when you point them at some — `KOVA_TEST_AUDIO`,
`KOVA_TEST_AUDIO_DIR`, `KOVA_TEST_AUDIO_METADATA` in `.env.example`. They are skipped when
unset. No audio is committed to this repository; `.gitignore` blocks every audio extension the
pipeline accepts, and a test enforces that the two lists agree.

To build these docs:

```bash
uv run mkdocs serve             # live preview on http://127.0.0.1:8000
uv run mkdocs build --strict    # what CI would run
```
