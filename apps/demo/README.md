# Kova TTS demo

A single-page Gradio app: type something, pick a voice, hear it start playing before it has
finished generating. Voice cloning is one button away, in the same prompt box.

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
| `--model` / `--codec` / `--wavlm` | override the configured paths |
| `--lora-dir` | LoRA voices for professional cloning: a directory or a Hub repo id (default: `KOVA_LORA_DIR`, else the published [`kova-ai/kova-tts-1-voices`](https://huggingface.co/kova-ai/kova-tts-1-voices), adapters only); `''` for none |
| `--zero-shot-dir` | zero-shot presets to offer (default: the bundled [`zero_shot_voices/`](zero_shot_voices/README.md)); `''` for none |
| `--device` | e.g. `cuda:1` or `mps` |
| `--backend` | `auto`, `torch` or `mlx`; the default reads it off the checkpoint |
| `--decode-window` | Codec frames per streamed chunk. Worth raising to 62 on Apple Silicon |
| `-v` | log what the engine is doing |

Anything left unset resolves through `KOVA_*` in your `.env`, and then through the Hugging Face
Hub. If the page says it cannot find the weights, `uv run kova-tts paths` shows why.

## What is on the page

One page, one prompt box, styled after [kova.ai](https://kova.ai). The header says what the
machine is running (device, backend, installed voices) and links to the endpoint's API docs;
anything that needs fixing — a CPU-only machine, missing weights — shows as a banner above the
box.

**The prompt box.** Type, or press **Random** for one of the example prompts. The character
count sits beside it. The toolbar underneath holds a four-way switch for the voice source, the
settings button and **Generate**, and keeps that shape whatever is picked. The voice is chosen in two
steps: a source on the switch, then a voice within it on the line below —

| Source | What it is |
|---|---|
| Base model | the checkpoint with no adapter and no reference |
| Zero-shot preset | one of the bundled reference clips, previewed under the toolbar with its transcript; encoded on first use (shown when there are presets) |
| Your recording | anything cloned this session; with nothing cloned yet, picking it opens the cloning panel |
| Professional cloning | a LoRA adapter from `--lora-dir` / `KOVA_LORA_DIR`; with none installed, picking it says how to add one |

 Sound starts after the first ~390 ms frame is decoded and runs continuously to the end
of the utterance, while the later sentences are still being generated. **Generate** always
starts over: pressing it mid-run drops the current clip and synthesizes the box's text afresh,
with whatever voice and settings are picked now.

**The player** draws the clip as a waveform: bars fill in as audio arrives, turn teal as they are
heard, and the rest of the bar stays flat until it has been generated. Once the generation
finishes, the same bar scrubs the finished clip and a download button appears — the clip is
built in the browser from the frames it already received, so nothing is generated or encoded
twice. Underneath: time to first audio, how much speech was produced in how long, and how that
compares to real time.

**Settings** (the sliders button) holds temperature, top-p, top-k, repetition penalty and the
token budget. They start at the tuned preset, and switching voice resets them to the preset that
voice calls for — the cloning preset differs only in giving the model a larger token budget.
"Reset to preset" puts them back. You should not have to open this panel at all.

**Cloning** happens in a panel inside the prompt box. Picking **Your recording** opens it while
nothing has been cloned; after that, **New recording** beside the voice picker opens it again. A
switch at the top of the panel picks how the reference comes in:

| Mode | What you do | Where the transcript comes from |
|---|---|---|
| Read aloud | record yourself reading the passage shown; "Another" swaps it | the passage itself — nothing to type, nothing transcribed |
| Freestyle or upload | say anything into the microphone, or upload a clip | transcribed the moment the clip arrives, into a box you can edit |

Freestyle transcription is NVIDIA Parakeet TDT (`nvidia/parakeet-tdt-0.6b-v3`, via transformers;
override with `KOVA_DEMO_ASR_MODEL`). Check the transcript before cloning: cloning *continues*
the reference, and a transcript that does not match garbles the output. Name the voice, press
Clone, and the panel closes with the new voice already selected. Clones live in memory for the
life of the process and are never written to disk.

Uploads land in a per-user `GRADIO_TEMP_DIR` (`$TMPDIR/gradio-<user>`) rather than Gradio's
shared `/tmp/gradio`: on a machine where another user ran Gradio first, that directory is theirs
and every upload fails with a permission error.

## Notes

- **One generation at a time.** The model has a single KV cache and refuses to interleave
  requests, so the demo serializes them and tells a second visitor to wait rather than showing
  them a traceback.
- **On a CPU it is very slow.** The banner says so; a CUDA device is what this is for. On Apple
  Silicon the published checkpoint runs under torch on Metal, well below real time, so expect
  audio to stall as it plays; the banner says which backend it picked.
- Long text is generated chunk by chunk — a few short sentences at a time — with the previous
  chunk carried into the next prompt, which is why the joins hold together. The demo accepts
  about 1200 characters at a time; use the Python API for anything longer.

## How the audio reaches your speakers

Worth knowing, because it is the one thing in here that is not stock Gradio.

Gradio's streaming audio component does not send the samples it is given. It re-encodes every
yielded frame to AAC with ffmpeg and serves the result as HLS segments — a lossy codec applied
2.5 times a second, each segment carrying its own encoder priming. Streamed through it, a
generation clicks and rasps; the finished clip of that same generation is clean.

So the page does not use it. `apps/demo/player.js` POSTs to `/v1/tts/stream` — the same
server-sent-event protocol [the server](../../docs/server.md) documents, `chunk` events carrying
base64 16-bit PCM — decodes each frame to a `Float32Array`, and hands it to Web Audio as an
`AudioBufferSourceNode` scheduled at a running cursor. Frames abut to the sample, nothing is
re-encoded between the codec and the speakers, and the wav you download is those same frames
with a 44-byte header in front.

The endpoint lives beside the page, in the same process, holding the same model: a Gradio app
is a FastAPI app underneath, so `build_app()` mounts the Blocks inside one of its own. Open
<http://127.0.0.1:7860/docs> while the demo is running to see it.

## The files

| File | What is in it |
|---|---|
| `app.py` | `build_app()`, the argument parsing, and `main()`. Composes the rest. |
| `content.py` | Everything the page says and how it looks: prose, example prompts, the stylesheet, the header and the player's markup. |
| `session.py` | `DemoSession` — the engine, the lock that serializes it, and cloned voices. |
| `streaming.py` | The `/v1/tts/stream` endpoint the player pulls audio from. |
| `ui.py` | `build_ui()` — the prompt box, its panels and callbacks — and `build_theme()`. |
| `player.js` | The browser half: SSE in, Web Audio out, the waveform, and the finished clip and the wav. |
| `switch.js` | Slides the voice-source switch's highlight onto the picked source. |
| `kova-logo.svg` | The Kova wordmark, inlined into the header. |

`app.py` puts its own directory on `sys.path` when it is imported, so the modules beside it are
imported by plain name. That is what lets `kova-tts demo` load `app.py` straight from its path
in a checkout where `apps/` is not an importable package.

## Embedding it

`build_app()` returns the whole thing — page plus streaming endpoint — as a FastAPI
application, and `DemoSession` holds all the state:

```python
import uvicorn
from apps.demo.app import DemoSession, build_app

app = build_app(DemoSession(tts=my_engine))   # any object with the KovaTTS surface
uvicorn.run(app, port=7860)
```

`build_ui()` still returns a bare `gradio.Blocks` without launching it, for a caller that wants
to place the page itself — but the page is only half a demo without something serving
`STREAM_PATH`, so pass `stream_path=` if you mount it somewhere other than the root.
