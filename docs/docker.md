# Docker

The container build lives in `docker/`, and
[`docker/README.md`](https://github.com/evalabs-ai/kova-tts/blob/main/docker/README.md) is its
reference — it is written and maintained alongside the Dockerfile, so it is the only place that
can say what the image is called and how to run it. What is worth knowing before you open it is
the shape of the thing: it is one CUDA image running one process, which is `kova-tts serve` by
default and can be the demo or a one-shot `generate` instead. **Weights are never baked in.**
They are mounted, or fetched once into a Hugging Face cache volume that survives restarts, and
every path is configured the way the rest of the project configures them — `KOVA_MODEL_PATH`,
`KOVA_CODEC_PATH`, `KOVA_WAVLM_PATH`, `KOVA_LORA_DIR`, here supplied as ordinary environment
variables, since the image sets `KOVA_DISABLE_DOTENV=1` on the grounds that a `.env` reaching a
container is always the wrong `.env`: it names checkpoint paths on somebody's workstation.

Two constraints carry into any container you build. The entrypoint binds `0.0.0.0`, because a
published port forwards to the container's external interface and a process listening only on
loopback never sees it — and the server has no authentication, so publishing that port means
deciding who can reach it. And one container serves one generation at a time: scaling means more
containers, each pinned to its own GPU, not a bigger one.
