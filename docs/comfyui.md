# ComfyUI

A node pack with three nodes: load the model once, generate speech from text, clone a voice from
a recording. They live under **audio → Kova TTS** in the node menu.

| Node | In | Out |
|---|---|---|
| **Kova TTS Loader** | `device`, `precision`, optional `model` / `codec` / `lora_dir` paths | `KOVA_TTS` |
| **Kova TTS Generate** | `KOVA_TTS`, `text`, sampling knobs, optional `KOVA_VOICE` or `voice_name` | `AUDIO` |
| **Kova TTS Clone Voice** | `KOVA_TTS`, `AUDIO`, `name`, optional `transcript` | `KOVA_VOICE` |

`AUDIO` is ComfyUI's own format, so the generate node feeds **Save Audio** or **Preview Audio**
directly, and the clone node takes its input straight from **Load Audio**. `KOVA_TTS` and
`KOVA_VOICE` are opaque links between these nodes.

**Its own README is the reference**, and it is kept current with the nodes:
[`apps/comfyui/README.md`](https://github.com/evalabs-ai/kova-tts/blob/main/apps/comfyui/README.md)
— installation into ComfyUI's environment, workflow diagrams, and where the weights come from.

The install is two steps, and the first is the one people get wrong: `kova-tts` has to be
importable **from the Python that runs ComfyUI**, not from this checkout's virtualenv. The
directory then goes into `custom_nodes`, by symlink or by copy.

Two things carry over from the rest of the project:

- **One clip at a time.** The engine has a single KV cache and refuses to interleave two
  requests, so run one generate node at a time rather than batching a queue of them.
- **Cloned voices need `max_tokens` room** for the reference clip, which is generated before your
  text. The sampling defaults are otherwise the same for both. See [Voice
  cloning](voice-cloning.md#sampling).

The loader keeps the model in memory across executions, so only the first run of a workflow pays
to load it. Changing one of its settings loads the new model and releases the old one — one
engine at a time, because two is two copies of the weights in VRAM.
