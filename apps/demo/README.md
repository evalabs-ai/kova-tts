# Kova TTS demo

A single-page Gradio app: type something, pick a voice, hear it start playing before it has
finished generating. Voice cloning is on the second tab.

## Run it

```bash
uv sync --extra demo
uv run kova-tts demo                       # or: uv run python apps/demo/app.py
```

Then open <http://127.0.0.1:7860>.

| Flag | Meaning |
|---|---|
| `--host` / `--port` | where to bind (default `127.0.0.1:7860`) |
| `--share` | expose a temporary public `gradio.live` link |
| `--open` | open a browser window on startup |
| `--preload` | load the weights at startup rather than on the first generation |
| `--model` / `--codec` / `--wavlm` / `--lora-dir` | override the configured paths |
| `--device` | e.g. `cuda:1` |
| `-v` | log what the engine is doing |

Anything left unset resolves through `KOVA_*` in your `.env`, and then through the Hugging Face
Hub. If the page says it cannot find the weights, `uv run kova-tts paths` shows why.

## What is on the page

**Speak.** Text box, voice picker, seed, and a Speak button. Audio appears in two players: the
top one streams — it starts playing after the first ~390 ms frame is decoded, while the rest of
the sentence is still being generated — and the lower one holds the finished clip for scrubbing
and downloading. The line underneath reports time to first audio, how much speech was produced
in how long, and the seed that produced it, so a result you like can be reproduced by typing
that seed back in.

**Advanced** (collapsed) holds temperature, top-p, top-k, repetition penalty and the token
budget. They start at the tuned preset, and switching voice resets them to the preset that voice
calls for — cloning wants a hotter, much less penalised setting than plain synthesis. You should
not have to open this accordion at all.

**Clone a voice.** Upload or record five to twenty seconds of clean speech, optionally type the
transcript, and press Clone. The voice appears in the picker on the Speak tab immediately. Left
without a transcript the recording is transcribed automatically, which needs the `data` extra
(`uv sync --extra data`); the demo says so plainly if it is missing. Clones live in memory for
the life of the process and are never written to disk.

## Notes

- **One generation at a time.** The model has a single KV cache and refuses to interleave
  requests, so the demo serializes them and tells a second visitor to wait rather than showing
  them a traceback.
- **On a CPU it is very slow.** The banner says so; a CUDA device is what this is for.
- Long text is generated sentence by sentence with the previous sentence carried into the next
  prompt, which is why the joins hold together. The demo accepts about 1200 characters at a
  time; use the Python API for anything longer.

## Embedding it

`build_ui()` returns a `gradio.Blocks` without launching it, and `DemoSession` holds all the
state, so the app can be mounted inside another server or driven in a test:

```python
from apps.demo.app import DemoSession, build_ui

ui = build_ui(DemoSession(tts=my_engine))     # any object with the KovaTTS surface
```
