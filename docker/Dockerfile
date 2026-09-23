# syntax=docker/dockerfile:1.7
#
# One image, one process, one request at a time. It runs the server, the demo, or a one-shot
# `generate`; which of those is decided by the command, not by a second image.
#
#   docker build -t kova-tts:latest -f docker/Dockerfile .
#
# Three things about this file are load-bearing.
#
# **The base is `-base`, not `-runtime`.** The locked torch is a cu13 wheel from PyPI, and it
# brings its own CUDA libraries with it -- nvidia-cudnn-cu13, nvidia-nccl-cu13, cusparselt,
# nvshmem, all pinned in `uv.lock`. A `-runtime` base would add ~2.5 GB of a *second* copy of
# those libraries that torch will never open, because it loads the ones next to itself in
# site-packages. What `-base` is actually here for is the driver contract: the
# `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` labels that tell the NVIDIA container
# runtime to inject the host's driver.
#
# **Dependencies install before the source is copied.** Only the manifests and the lockfile go
# in first, so editing a `.py` re-runs the second `uv sync` (a second or two) instead of
# redownloading 5 GB of torch.
#
# **The workspace is installed editable, at /app.** Not a stylistic choice: `kova-tts demo`
# finds `apps/demo/app.py` by walking up from `kova_tts/cli.py`, which only lands on the
# checkout when the package is imported from `packages/kova-tts/src`. A non-editable install
# puts `cli.py` under `site-packages` and the demo cannot find itself.

ARG CUDA_IMAGE=nvidia/cuda:13.0.1-base-ubuntu24.04
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.10.7

FROM ${UV_IMAGE} AS uv

FROM ${CUDA_IMAGE}

# Which optional dependency groups land in the image. Override to slim it down, e.g.
# `--build-arg KOVA_EXTRAS="--extra server"` for a server-only image with no voice cloning.
#
#   server    FastAPI + uvicorn + websockets. The default command.
#   demo      Gradio 6. The other command this image advertises.
#   data      faster-whisper, so `--clone-audio` works without a transcript. An image that can
#             serve a voice but cannot clone one is a poor first five minutes, and this extra
#             is ~350 MB against a 6 GB image.
#   finetune  accelerate and pyyaml, so the image can also *train* a LoRA. Both already arrive
#             transitively with peft, which is a base dependency, so this extra adds almost
#             nothing. Not needed to *use* a LoRA voice either -- `--voice` works without it.
ARG KOVA_EXTRAS="--extra server --extra demo --extra data --extra finetune"

ARG KOVA_UID=1000
ARG KOVA_GID=1000

# ARG, not ENV: this is a build-time concern, and baking it in would silently change how apt
# behaves for anyone who later opens a shell in the container.
ARG DEBIAN_FRONTEND=noninteractive

# Three packages, and each earns its place:
#
#   ca-certificates  Without it every https call the Hugging Face Hub makes fails certificate
#                    verification. Everything else native arrives inside a wheel -- libsndfile
#                    inside soundfile, ffmpeg inside av, the CUDA libraries inside torch.
#   gcc, libc6-dev   Triton JIT-compiles a small C launcher for every kernel it runs, at run
#                    time, and shells out to a C compiler to do it. In torch 2.13 that is not
#                    only a `torch.compile` concern: `torch._native` routes ordinary eager ops
#                    (a batched matmul inside Llama's rotary embedding, for one) through Triton
#                    kernels, so a GPU image without a compiler dies during model warmup with
#                    "Failed to find C compiler". The three together cost ~215 MB, and none of
#                    them is optional.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /usr/local/bin/

# Ubuntu 24.04 ships a stock `ubuntu` account on uid 1000, and 1000 is exactly the uid a host
# bind mount usually wants to be readable by. Reclaim it under a name that says what it is.
RUN userdel --remove ubuntu 2>/dev/null || true; \
    groupadd --gid ${KOVA_GID} kova \
    && useradd --create-home --uid ${KOVA_UID} --gid ${KOVA_GID} --shell /bin/bash kova

# UV_LINK_MODE=copy because the wheel cache lives on a cache mount and the venv on an overlay:
# hardlinking across that boundary fails, and uv would warn on every install.
# UV_COMPILE_BYTECODE=1 moves .pyc generation into the build, where it is paid once, instead of
# into the first import of a 5 GB dependency tree at container start.
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_PYTHON=3.12 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_CACHE_DIR=/tmp/uv-cache \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

RUN uv python install 3.12 && chmod -R a+rX /opt/python

# Everything from here runs unprivileged. The venv is created by `kova`, so no `chown -R` of a
# 5 GB directory is ever needed -- that would double the image.
RUN mkdir -p /app /out && chown kova:kova /app /out
WORKDIR /app
USER kova

# Layer 1: dependencies. `--no-install-workspace` resolves and installs everything in the
# lockfile *except* kova-codec and kova-tts themselves, which is what lets this layer survive
# any change to the source.
COPY --chown=kova:kova pyproject.toml uv.lock ./
COPY --chown=kova:kova packages/kova-codec/pyproject.toml packages/kova-codec/
COPY --chown=kova:kova packages/kova-tts/pyproject.toml packages/kova-tts/
# `--no-default-groups` keeps pytest, ruff and the docs toolchain out: they are declared as
# default dependency groups so that a plain `uv sync` on a workstation is ready to develop
# with, and none of them has any business in a runtime image.
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${KOVA_UID},gid=${KOVA_GID},sharing=locked \
    uv sync --frozen --no-default-groups --no-install-workspace ${KOVA_EXTRAS}

# Layer 2: the project. Cheap, because layer 1 already installed everything it depends on.
COPY --chown=kova:kova . /app
# The full copy above includes LICENSE, NOTICE, licenses/, and the shared build hook.
# Verify the installed package metadata and legal files against those root copies.
RUN --mount=type=cache,target=/tmp/uv-cache,uid=${KOVA_UID},gid=${KOVA_GID},sharing=locked \
    uv sync --frozen --no-default-groups ${KOVA_EXTRAS} \
    && /app/.venv/bin/python scripts/check_package_licenses.py --installed

USER root
COPY docker/entrypoint.sh /usr/local/bin/kova-entrypoint
RUN chmod 0755 /usr/local/bin/kova-entrypoint
USER kova

# HF_HOME sits under $HOME so that one `-v kova-hf-cache:/home/kova/.cache/huggingface` catches
# every download the Hub makes. Weights are never baked in; they are mounted, or fetched once
# into that cache and kept there across container restarts.
#
# KOVA_DISABLE_DOTENV=1 because a `.env` reaching this container is always the wrong `.env`:
# it names checkpoint paths on somebody's workstation, and inside here those paths do not
# exist. Configuration comes from KOVA_* in the environment, which is the language `docker run
# -e` and compose already speak. To opt back in -- a checkout bind-mounted at /app with a
# container-correct `.env` -- pass an *empty* value, `-e KOVA_DISABLE_DOTENV=`; any non-empty
# value, "0" included, still means disabled.
#
# TRITON_CACHE_DIR under /tmp rather than under $HOME, because Triton compiles kernels on the
# first generation and must have somewhere to put them. $HOME only works while the container
# runs as uid 1000; `docker run --user 1234` gets HOME=/ and a permission error deep inside a
# JIT. /tmp is writable by everyone, in every one of those cases.
ENV PATH=/app/.venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/kova/.cache/huggingface \
    TRITON_CACHE_DIR=/tmp/triton-cache \
    KOVA_DISABLE_DOTENV=1

# HTTP + WebSocket server, and the Gradio demo. Documentation only -- publishing is `-p`.
EXPOSE 8000 7860

ENTRYPOINT ["/usr/local/bin/kova-entrypoint"]
CMD ["serve"]
