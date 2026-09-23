# Kova TTS in Docker

One image. It runs the server, the demo, or a one-shot `generate` — whichever the command
says. There is no second container, no codec service, no queue: this is a single-user,
single-request TTS engine that holds one model on one GPU and answers one caller at a time,
and a compose stack would only be a more elaborate way to say that.

```bash
docker build -t kova-tts:latest -f docker/Dockerfile .
```

Roughly 10 minutes on the first build, and the result is in the region of 7 GB unpacked. Note
that `docker images` may print a much larger figure, because Docker's containerd store keeps the
compressed blobs alongside the unpacked snapshot and reports the two added together. Almost all
of the size is torch and its bundled CUDA libraries; `docker history kova-tts:latest` shows
where it went on your build. No weights are in there; see
[what is in the image](#what-is-in-the-image).

## Run it

Nothing works until the model can be found, so start there. Weights are never baked into the
image — you either mount local checkpoints or let them download into a cached volume.

### With local checkpoints

Put the artifacts in one directory and mount it read-only:

```
/path/to/checkpoints
├── model/          # config.json, model.safetensors, tokenizer.json
├── codec.pt        # the codec checkpoint
├── wavlm-large/    # optional; only read when encoding audio
└── loras/          # optional; one subdirectory per voice
    └── <voice>/adapter_config.json
```

```bash
docker run --rm --gpus all -p 8000:8000 \
  -v /path/to/checkpoints:/weights:ro \
  -e KOVA_MODEL_PATH=/weights/model \
  -e KOVA_CODEC_PATH=/weights/codec.pt \
  -e KOVA_WAVLM_PATH=/weights/wavlm-large \
  -e KOVA_LORA_DIR=/weights/loras \
  kova-tts:latest
```

The layout is a convention, not a requirement — the four `KOVA_*` variables are what actually
decide, and they can point anywhere you have mounted. The model, codec, and WavLM fall back to
the Hugging Face Hub when unset. LoRA voices require a local directory mounted into the container.

### From the Hub

Leave the `KOVA_*` variables unset and mount a cache instead, so the download happens once for
the machine rather than once per container:

```bash
docker volume create kova-hf-cache
docker run --rm --gpus all -p 8000:8000 \
  -v kova-hf-cache:/home/kova/.cache/huggingface \
  kova-tts:latest
```

Prefetch it deliberately, rather than discovering the download during your first request:

```bash
docker run --rm -v kova-hf-cache:/home/kova/.cache/huggingface kova-tts:latest download --wavlm
```

The default base-model repository is `kova-ai/kova-tts-1`. Set `KOVA_HUB_REPO` to use another
repository. For private or gated access, pass `HF_TOKEN` from your environment with
`-e HF_TOKEN` on each Docker command that downloads model files.

The five pretrained LoRA voices are a separate download from
[`kova-ai/kova-tts-1-voices`](https://huggingface.co/kova-ai/kova-tts-1-voices).
From the code checkout on the host:

```bash
uv run hf download kova-ai/kova-tts-1-voices --local-dir ./voices
docker run --rm --gpus all -p 8000:8000 \
  -v kova-hf-cache:/home/kova/.cache/huggingface \
  -v "$PWD/voices:/voices:ro" \
  -e KOVA_LORA_DIR=/voices \
  kova-tts:latest
```

Keep the voice package's legal documents alongside the adapter folders.

### With compose

```bash
cd docker
cp env.example .env          # set KOVA_WEIGHTS_DIR to your checkpoint directory
docker compose up            # server on http://localhost:8000
```

`docker/.env` is compose's own configuration — which host directory to mount, which ports to
publish. It is **not** the repository's `.env`, and the image deliberately ignores `.env` files
altogether (see [configuration](#configuration)).

Other commands go through the same service:

```bash
docker compose run --rm kova paths
docker compose run --rm kova generate "Hello there." -o /out/hello.wav   # lands in docker/out/
docker compose run --rm --service-ports kova demo                        # port 7860
```

## Commands

| Command | What it does |
|---|---|
| `serve` *(default)* | FastAPI + WebSocket server, bound to `0.0.0.0:8000` |
| `demo` | Gradio app, bound to `0.0.0.0:7860` |
| `generate "text" -o /out/x.wav` | one-shot synthesis |
| `paths` | where every artifact resolved — run this first when anything is wrong |
| `download` | prefetch weights into the mounted Hugging Face cache |
| anything else | run verbatim: `bash`, `python`, a script |

Both servers default to `127.0.0.1` in the source, which is right on a workstation and useless
in a container — a published port forwards to the container's external interface, and a process
on loopback never sees it. The entrypoint supplies `--host 0.0.0.0` for `serve` and `demo`
unless you name a host yourself, so `serve --host 127.0.0.1` still means what it says for anyone
running `--network host` on purpose. Everything else is passed through untouched:

```bash
docker run ... kova-tts:latest serve --busy-timeout 0 --no-warmup --log-level debug
```

## GPU

You need an NVIDIA GPU, a recent driver, and the [NVIDIA container
toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. One line tells you whether the toolkit is wired up:

```bash
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

The image itself carries no CUDA toolkit. The locked torch is a cu13 wheel that brings its own
CUDA libraries, so the base image is `nvidia/cuda:...-base` rather than `-runtime`: a `-runtime`
base would add ~2.5 GB of a second copy of libraries torch never opens, because it loads the
ones next to itself in `site-packages`. What the CUDA base is there for is the driver contract
that makes `--gpus all` inject the host's driver.

About 6 GB of VRAM covers the 1B model, the codec and a generation, so any 8 GB card is enough.

## Configuration

| Variable | Meaning |
|---|---|
| `KOVA_MODEL_PATH` | LM directory, or a Hub repo id |
| `KOVA_CODEC_PATH` | codec checkpoint file |
| `KOVA_WAVLM_PATH` | WavLM-large directory; only read when encoding audio |
| `KOVA_LORA_DIR` | directory of LoRA voices, one subdirectory each |
| `KOVA_HUB_REPO` | Hub repository anything unset falls back to |
| `HF_TOKEN` | for a gated or unpublished repository |
| `KOVA_PORT`, `KOVA_DEMO_PORT` | ports the entrypoint binds (default 8000, 7860) |

**The image ignores `.env` files**, via `KOVA_DISABLE_DOTENV=1`. This is deliberate: a `.env`
reaching a container is almost always the wrong one — it names checkpoint paths on somebody's
workstation, and inside the container those paths do not exist, so what you get is a confusing
`MissingArtifact` for a file you can see with your own eyes on the host. Configuration comes
from the environment, which is the language `docker run -e` and compose already speak. If you
bind-mount a checkout at `/app` and genuinely want its `.env` read, pass an **empty** value:
`-e KOVA_DISABLE_DOTENV=`. Any non-empty value — `0` included — still means disabled.

## Verifying it works

```bash
curl -s localhost:8000/health
# {"status":"ok","model_loaded":true,"device":"cuda","sample_rate":48000,"voices":1,...}

curl -s localhost:8000/v1/voices

curl -s -X POST localhost:8000/v1/tts \
  -H 'content-type: application/json' \
  -d '{"text":"The quick brown fox jumps over the lazy dog.","seed":1234}' \
  -o out.wav
```

On an RTX 3090, measured from the host against a container started exactly as above:

| | |
|---|---|
| Startup to `model_loaded: true` | 18–62 s (weights, codec, CUDA graph capture, warmup generation; the spread is the host's page cache) |
| `POST /v1/tts`, 7.4 s of speech | 1.9 s — about 3.8x realtime |
| `POST /v1/tts`, 1.2 s of speech | 0.4 s |
| Cold `generate` in a fresh container | ~23 s wall, most of it model load |

The first number is why the compose healthcheck has a five-minute `start_period`: a container
is not unhealthy while it is legitimately loading.

## When it cannot find a model

`kova-tts paths` is the whole diagnostic. It resolves every artifact and prints where each one
landed, without loading anything:

```bash
docker run --rm -v /path/to/checkpoints:/weights:ro \
  -e KOVA_MODEL_PATH=/weights/model -e KOVA_CODEC_PATH=/weights/codec.pt \
  kova-tts:latest paths
```

```
config    no .env found
model     /weights/model
wavlm     microsoft/wavlm-large
codec     /weights/codec.pt
loras     not configured
voices    none
```

Read it against what you expected:

- **`ERROR: Model directory not found at /weights/model`** — the path is right in your shell and
  wrong in the container. `KOVA_*` must name the path *inside* the container, under whatever
  you mounted at `/weights`. Check with `docker run --rm -v /path/to/checkpoints:/weights:ro
  kova-tts:latest ls -la /weights`.
- **`config  /app/.env`** — you are bind-mounting a checkout and it brought its own `.env`. See
  [configuration](#configuration).
- **A repo id where you expected a path** (`kova-ai/kova-tts-1`) — that variable is unset, so
  it fell through to the Hub. Empty strings count as unset, which is why compose passes
  `KOVA_MODEL_PATH: ""` by default rather than a placeholder that would fail.
- **`Could not download ...`** — check network access and the repository ID. For a private or
  gated repository, pass `HF_TOKEN` for an account with access. You can also mount local checkpoints.
- **`voices  none` with `loras` set** — a voice is a directory containing `adapter_config.json`.
  A directory of `.safetensors` files with no config is not one.
- **Permission denied writing the output wav** — the container runs as uid 1000. Either create
  the output directory before mounting it, or add `--user "$(id -u):$(id -g)"`.

## CPU

It runs, and you should not use it. Measured in this image on 24 cores, with no `--gpus`:

```
1.61 s of audio in 34.6 s
```

That is 0.05x realtime — roughly **21x slower than the audio is long**, and about 80x slower
than the same container on a 3090. A one-sentence request takes over half a minute; the demo is
unusable; the server will time a second caller out before it finishes the first. It is fine for
checking that a container starts, that `paths` resolves and that the API shape is what you
expected, and that is the only thing it is for.

There is deliberately **no CPU-only image**. A separate Dockerfile would save about 3.4 GB by
dropping the bundled CUDA libraries, and in exchange it would double the build surface, need
its own testing, and exist to serve a mode that is too slow to serve anybody. The one image
already runs on CPU when you omit `--gpus`; that is the entire CPU story, and it does not need
a second artifact to tell it.

## What is in the image

Dependencies and source, and nothing else. No weights, no `.env`, no audio:

```bash
docker run --rm kova-tts:latest ls -la /app
docker run --rm kova-tts:latest du -sh /app/.venv /app/packages
```

Nearly all of the size is the installed dependency tree; after that, in descending order, the
CUDA base image, the compiler and certificates, Python itself, uv, and — last by a wide margin,
a few megabytes — this repository. Four optional extras are installed:

| Extra | Why it is in |
|---|---|
| `server` | FastAPI, uvicorn, websockets. The default command. |
| `demo` | Gradio 6. The other command the image advertises. |
| `data` | faster-whisper, so `--clone-audio` works without a transcript. ~350 MB for the feature most people try first. |
| `finetune` | accelerate and pyyaml, for `finetune` and `merge`. Both arrive transitively with `peft`, which is a base dependency, so this extra adds almost nothing — and it is not needed to *use* a LoRA voice. |

Build a smaller image by naming fewer:

```bash
docker build -t kova-tts:server -f docker/Dockerfile \
  --build-arg KOVA_EXTRAS="--extra server --extra finetune" .
```

Dropping `demo` and `data` removes Gradio, faster-whisper and their native halves, which is
where the savings are. Dropping `finetune` saves next to nothing now that it is only accelerate
and pyyaml, and costs you the `finetune` and `merge` commands; LoRA voices keep working either
way, since `peft` is a base dependency. Dropping torch's CUDA libraries is not something an
extra can do.

Development tooling — pytest, ruff, the docs toolchain — is excluded by `--no-default-groups`,
so the image cannot run the test suite. That is on purpose; run tests on the host with
`uv run pytest`.

## Notes

- **Layer ordering.** Dependencies install from `uv.lock` before any source is copied, so
  editing a `.py` rebuilds in ~28 s instead of ~10 min. Changing `pyproject.toml` or `uv.lock`
  correctly invalidates the dependency layer.
- **Non-root.** Everything runs as uid 1000, and the venv is *built* by that user, so the image
  never pays for a `chown -R` of a 6 GB directory.
- **The workspace is installed editable at `/app`.** Not stylistic: `kova-tts demo` locates
  `apps/demo/app.py` by walking up from `kova_tts/cli.py`, which only lands on the checkout
  when the package is imported from `packages/kova-tts/src`.
- **A C compiler is installed, and is not optional.** Triton JIT-compiles a launcher for every
  kernel it runs, at run time — and in torch 2.13 that is not only a `torch.compile` concern,
  because `torch._native` routes ordinary eager operations through Triton. Without `gcc` the
  model dies during warmup with `Failed to find C compiler`.
- **One request at a time.** The server serializes on the model and answers a second concurrent
  caller with `409`; `--busy-timeout` decides how long it waits first. Running two containers
  against one GPU is not a way around this — it is two copies of the weights in VRAM.

## License files in the image

The image includes `/app/LICENSE`, `/app/NOTICE`, and `/app/licenses/third-party/`.
The Python package builds automatically include the same legal files in their installed
metadata. The Docker build verifies those installed copies against the root files and
fails if any are missing or different.

Pretrained voices are downloaded or mounted separately. Keep their `LICENSE`,
`LICENSE-SUPPLEMENT`, dataset `NOTICE`, and component notices with that voice package.
