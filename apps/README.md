# Apps

Two things you can point at the engine. Neither is part of the installed package: both live here
in the source checkout, and both are ordinary callers of
[`KovaTTS`](../packages/kova-tts/src/kova_tts/engine/tts.py) with no privileged access to it.

| | | Needs |
|---|---|---|
| [`demo/`](demo/README.md) | A Gradio page: type, pick a voice, hear it start playing before it has finished generating. Voice cloning on the second tab. | `uv sync --extra demo` |
| [`comfyui/`](comfyui/README.md) | Three ComfyUI nodes: load once, generate, clone. | ComfyUI, plus `kova-tts` installed into *its* Python |

```bash
uv sync --extra demo
uv run kova-tts demo            # http://127.0.0.1:7860
```

Each directory's README is the reference for that app — flags, what is on the page, how to
embed it. Start there.

Two constraints they share with everything else in this repository, worth knowing before you
file a bug against either:

- **One generation at a time.** The engine holds a single KV cache and refuses to interleave two
  requests. The demo serialises and tells the second visitor to wait; in ComfyUI, run one
  generate node at a time.
- **Weights resolve the same way everywhere.** The value you typed in, then the matching `KOVA_*`
  environment variable, then the Hugging Face Hub. `kova-tts paths` prints what resolved and what
  did not.

For a headless deployment, use the [server](../packages/kova-tts/src/kova_tts/server) instead —
`kova-tts serve` gives you HTTP, Server-Sent Events and a streaming WebSocket session.
