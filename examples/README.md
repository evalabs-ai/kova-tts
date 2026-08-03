# Examples

Short, runnable scripts. Run them from a checkout with `uv run python examples/<name>.py`, and
run `uv run kova-tts paths` first if anything cannot find a checkpoint.

| Script | What it shows | Needs |
|---|---|---|
| [`quickstart.py`](quickstart.py) | Load the model, synthesize a line, write a WAV. | — |
| [`voice_clone.py`](voice_clone.py) | Clone a voice from your own reference recording, transcribing it automatically. | `--extra data` |
| [`stream_sse.py`](stream_sse.py) | Streaming synthesis over Server-Sent Events. | `--extra server` |
| [`stream_ws.py`](stream_ws.py) | Streaming synthesis over a WebSocket. | `--extra server` |

No audio ships with this repository, so the cloning example needs a recording of your own:
five to twenty seconds of one person speaking clearly.

Everything these scripts do is also a command:

```bash
uv run kova-tts generate "Something to say." --out out.wav
uv run kova-tts generate "Something new." --clone-audio your_reference.wav --out cloned.wav
```
