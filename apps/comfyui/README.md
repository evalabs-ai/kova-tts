# Kova TTS nodes for ComfyUI

Three nodes: load the model once, generate speech from text, and clone a voice from a recording.

## Install

The nodes need `kova-tts` importable **from the Python that runs ComfyUI**, and the directory
itself inside `custom_nodes`.

```bash
# 1. install kova-tts into ComfyUI's environment
/path/to/ComfyUI/venv/bin/pip install "kova-tts[data]"     # 'data' is optional: see below

# 2. link this directory into custom_nodes
ln -s /path/to/kova-tts/apps/comfyui /path/to/ComfyUI/custom_nodes/kova_tts

# 3. restart ComfyUI
```

Copy the directory instead of symlinking it if you prefer; nothing here writes to it.

The `data` extra installs faster-whisper, which lets the clone node transcribe a reference
recording for you. Without it, cloning still works — you type the transcript.

### Weights

Everything resolves the same way the rest of the project does: the value typed into the node,
then the matching `KOVA_*` environment variable, then the Hugging Face Hub. To point at local
checkpoints, set them in the environment ComfyUI starts in:

```bash
export KOVA_MODEL_PATH=/models/kova-tts-1b
export KOVA_CODEC_PATH=/models/kova/codec.pt
export KOVA_LORA_DIR=/models/kova/voices
```

`kova-tts paths` prints what resolved and what did not.

## Nodes

All three live under **audio → Kova TTS**.

| Node | Inputs | Outputs |
|---|---|---|
| **Kova TTS Loader** | `device`, `precision`, and optional `model` / `codec` / `lora_dir` paths | `KOVA_TTS` |
| **Kova TTS Generate** | `KOVA_TTS`, `text`, `seed`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens`, optional `KOVA_VOICE` or `voice_name` | `AUDIO` |
| **Kova TTS Clone Voice** | `KOVA_TTS`, `AUDIO`, `name`, optional `transcript` | `KOVA_VOICE` |

`KOVA_TTS` and `KOVA_VOICE` are opaque links between these nodes. `AUDIO` is ComfyUI's own
format, so the generate node's output goes straight into **Save Audio** or **Preview Audio**,
and the clone node's input comes straight from **Load Audio**.

The loader keeps the model in memory across executions, so only the first run of a workflow
pays to load it. Changing one of its settings loads the new model and releases the old one —
one engine at a time, because two is two copies of the weights in VRAM.

## Workflows

**Plain synthesis**

```
Kova TTS Loader ──▶ Kova TTS Generate ──▶ Save Audio
                       text: "..."
```

Leave `voice` unconnected and `voice_name` empty for the model's own voice, or type the name of
an installed LoRA voice into `voice_name`.

**Voice cloning**

```
Load Audio ─────────▶ Kova TTS Clone Voice ──▶ Kova TTS Generate ──▶ Save Audio
Kova TTS Loader ──┬─▶
                  └────────────────────────────▶
```

Five to twenty seconds of clean speech clones well; less than a second is refused. If you leave
`transcript` empty the recording is transcribed for you, and if the transcript is typed it has
to match the recording word for word — cloning continues the reference, so a wrong transcript
makes the model try to speak words the audio does not contain.

## Notes

- Generation is **one clip at a time**. The model has a single KV cache and refuses to
  interleave two requests, so run one generate node at a time rather than batching a queue of
  them in parallel.
- Sampling defaults are the tuned preset. Cloned voices sample the same way, but the reference
  clip is generated before your text, so give `max_tokens` room (3500 is the preset).
- Output audio is 48 kHz mono.
