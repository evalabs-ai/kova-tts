# Quickstart

Assumes [Installation](installation.md) is done and `uv run kova-tts paths` shows a model and a
codec. Run everything from the repository checkout, so `.env` is found.

## Say something

=== "Python"

    ```python
    from kova_tts import KovaTTS

    tts = KovaTTS.from_pretrained()
    wav = tts.generate("Hello world.")
    tts.save(wav, "out.wav")
    ```

=== "Command line"

    ```bash
    uv run kova-tts generate "Hello world." --out out.wav
    ```

    ```
    out.wav  1.38 s of audio in 5.4 s (0.3x)
    ```

`wav` is a 1-D float32 numpy array, mono, 32 kHz. That is the only audio format anything in this
project hands you. `save` writes 16-bit WAV and creates the parent directory if it is missing.

There is a longer version of this in
[`examples/quickstart.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/quickstart.py):

```bash
uv run python examples/quickstart.py
```

## What "warm" means

The 0.3x above is not the model's speed. `from_pretrained` loads the language model and captures
its CUDA graph, but the **codec is loaded lazily on the first generation**, and cuDNN autotunes
its convolutions there too. A one-shot CLI run pays all of that inside the one generation it
measures.

Generate twice in one process and the second is the real number:

```python
import time
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
text = "The kettle had just boiled, and the rain was still going at the window."
for _ in range(3):
    started = time.perf_counter()
    wav = tts.generate(text, seed=7)
    elapsed = time.perf_counter() - started
    seconds = wav.size / tts.sample_rate
    print(f"{seconds:.2f}s audio in {elapsed:.2f}s -> {seconds / elapsed:.2f}x")
```

```
4.03s audio in 5.91s -> 0.68x        <- codec load lands here
4.03s audio in 0.98s -> 4.11x
4.03s audio in 0.98s -> 4.13x
```

Anything long-lived — the server, the demo, the ComfyUI loader node — pays this once at startup
and never again. The server does it explicitly with a warmup generation; `--no-warmup` moves the
cost onto the first request instead.

## Streaming

`stream` yields audio as the codec produces it, roughly every 390 ms, instead of waiting for the
whole utterance. This is what you want whenever a person is listening.

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()

for frame in tts.stream("The first words play while the last are still being written."):
    if frame.samples.size:
        ...            # frame.samples: float32 mono at frame.sample_rate
    if frame.is_final:
        break
```

The last frame always has `is_final=True`, even when it carries no samples, so a consumer can
close cleanly. Concatenating every frame's samples gives the same waveform `generate` returns.

From the command line, `--stream` decodes as it generates and reports time to first audio:

```bash
uv run kova-tts generate "A longer line, so there is something to stream." --stream --out out.wav
```

```
out.wav  4.71 s of audio in 6.1 s, first audio after 4.89 s (0.8x)
```

Again: cold. Warm, first audio lands around 190 ms in-process and around 200 ms over the
[server's WebSocket](server.md).

Streaming is not faster overall — it is the same work, reported earlier.

## Reproducing a result

Pass a `seed` and the same request gives the same audio:

```bash
uv run kova-tts generate "A fixed seed makes this reproducible." --seed 7 --out seed.wav
```

```python
wav = tts.generate("A fixed seed makes this reproducible.", seed=7)
```

Without a seed, sampling is random each time — the presets use `temperature=0.9` for plain
synthesis. Greedy decoding is not available: `SamplingParams` requires a positive temperature.

## Voices

A voice is either a **LoRA adapter** installed on this machine, or a **clone** built from a
reference recording.

```python
print(tts.voices())          # names from KOVA_LORA_DIR; [] on a fresh install
wav = tts.generate("Hello.", "my_voice")
```

```bash
uv run kova-tts paths                                  # the last line lists installed voices
uv run kova-tts generate "Hello." --voice my_voice --out out.wav
```

A fresh install has no voices at all: `KOVA_LORA_DIR` is unset, `tts.voices()` is empty, and
`--voice` fails with a message saying so. Either train one — [Finetuning](finetuning.md) — or
clone one from a recording, which needs no training at all:
[Voice cloning](voice-cloning.md).

Passing a name that does not resolve names the ones that do:

```
error: No voice named 'nope'. Available: no LoRA directory configured (set KOVA_LORA_DIR).
Run `kova-tts paths` to see what resolved.
```

## Longer text

Text longer than a sentence is split and generated segment by segment, with the previous
segment's text and its codes threaded into the next prompt so prosody carries across the join.
You do not have to do anything for this; it is what `generate` and `stream` already do. See
[Architecture](architecture.md#long-text-segment-carry) for how.

Two things worth knowing before you paste a chapter in:

- **Nothing is normalized.** `1997`, `Dr.`, `$40` and `10:30` reach the model exactly as typed.
  If you want "nineteen ninety-seven", write that.
- **A generation budget applies per segment**, not per document: 2048 codes for plain synthesis,
  about 25 seconds of speech. Prompt plus generation must also fit the 4096-token KV cache.

## Non-verbal tags

The model was trained on a small set of bracketed tags, which are single tokens and must be
written verbatim inside the text:

```python
wav = tts.generate("That is the funniest thing I have heard all week. [laugh]")
```

The full list lives in `kova_tts.tokens.NON_VERBAL_TAGS`:

```python
from kova_tts.tokens import NON_VERBAL_TAGS
print(NON_VERBAL_TAGS)
```

```
('[laugh]', '[exhale]', '[chuckle]', '[sigh]', '[clear_throat]', '[grunt]', '[gasp]',
 '[giggle]', '[sniff]', '[groan]', '[cough]', '[sing]', '[quote]', '[yawn]', '[inhale]',
 '[snort]', '[shush]', '[pause]', '[stutter]', '[gulp]', '[hum]', '[cry]', '[smack]')
```

They are a hint, not a command — the model decides whether to act on one.

## Next

- [Voice cloning](voice-cloning.md) — sound like a specific person, with no training.
- [Python API](python-api.md) — the full surface of `KovaTTS`.
- [Server](server.md) — HTTP, SSE and WebSocket.
