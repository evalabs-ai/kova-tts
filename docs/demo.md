# Demo

A single-page Gradio app: type something, pick a voice, hear it start playing before it has
finished generating. Voice cloning is one button away, in the same prompt box.

```bash
uv sync --extra demo
uv run kova-tts demo
```

Then open <http://127.0.0.1:7860>.

The demo ships in `apps/demo/` in the source checkout, not in the installed package, so
`kova-tts demo` needs a clone of the repository. It says so plainly if you run it from a wheel
install. `uv run python apps/demo/app.py` is the same thing.

**Its own README is the reference**, and it is kept current with the app:
[`apps/demo/README.md`](https://github.com/evalabs-ai/kova-tts/blob/main/apps/demo/README.md) —
every flag, what is on the page, and how to mount `build_app()` inside another server.

Three things to expect before you open it:

- **The first generation is slow**, because the weights load lazily on it. `--preload` moves
  that to startup so the first visitor waits for none of it.
- **One generation at a time.** Two browser tabs will not run in parallel; the second is told to
  wait rather than shown a traceback. Same reason as [the server's](server.md#concurrency).
- **On CPU it is very slow.** The page says so. A CUDA device is what this is for.

If the page reports that it cannot find the weights, `uv run kova-tts paths` shows why — see
[Installation](installation.md#getting-the-weights).
