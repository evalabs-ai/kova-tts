# Command line

```bash
uv run kova-tts --help
```

```
usage: kova-tts [-h] [--version]
                {paths,generate,prepare-data,finetune,merge,serve,demo,download}
                ...

Kova TTS: synthesize, clone, prepare data, finetune, serve.

positional arguments:
  {paths,generate,prepare-data,finetune,merge,serve,demo,download}
    paths               show resolved model artifact locations
    generate            synthesize text to a WAV file
    prepare-data        turn a folder of recordings into a JSONL finetuning
                        corpus
    finetune            train a per-voice LoRA adapter (--config CONFIG.yaml)
    merge               merge a LoRA adapter into its base model (ADAPTER
                        OUTPUT)
    serve               run the HTTP + streaming server
    demo                run the browser demo
    download            prefetch weights from the Hub for offline use
```

Every subcommand imports only what it needs, so `kova-tts paths` answers without loading
transformers, peft, gradio or faster-whisper. `finetune`, `merge`, `serve` and `demo` forward
their arguments untouched to the parser that owns them, which is why `kova-tts merge --help`
prints that parser's real help rather than a paraphrase that could drift. The cost: `--help` for
one of those reports a missing extra instead of printing help, if the extra is not installed.

Errors are one line, prefixed `error:`, with exit code 2. They are expected to name the fix.

---

## `paths`

Where each artifact resolved to. The first thing to run when something cannot find a checkpoint.

```bash
uv run kova-tts paths
```

```
config    /home/you/kova-tts/.env
model     /models/kova-tts-1b
wavlm     /models/wavlm-large
codec     /models/kova/codec.pt
loras     /models/kova/voices
voices    my_voice
```

Exits `0` when everything resolved, `1` when anything did not. `config` names the `.env` that
was loaded, or says none was found. See [Installation](installation.md#check-that-it-worked).

---

## `generate`

Synthesize to a WAV file. The everyday command.

```bash
uv run kova-tts generate "Hello world." --out out.wav
```

```
out.wav  1.38 s of audio in 5.4 s (0.3x)
```

The reported rate is generation only — but the *first* generation in a process also loads the
codec, so a one-shot run always looks slow. See
[Quickstart](quickstart.md#what-warm-means).

### Text

```bash
uv run kova-tts generate "Text as an argument."          --out out.wav
uv run kova-tts generate --text-file script.txt          --out out.wav
echo "From stdin." | uv run kova-tts generate --text-file - --out out.wav
```

Argument and `--text-file` are mutually exclusive. Empty text is an error, not a silent
zero-length file.

### Who to speak as

```bash
uv run kova-tts generate "Hello." --voice my_voice --out out.wav

uv run kova-tts generate "Hello." \
    --clone-audio reference.wav \
    --clone-text "exactly what the clip says" \
    --out out.wav
```

`--voice` names a LoRA adapter from the configured directory; `kova-tts paths` lists what is
installed. `--clone-audio` clones from a recording — drop `--clone-text` and the clip is
transcribed with ASR, which needs `uv sync --extra data`. The two flags are two ways of saying
who, so passing both is an error, and `--clone-text` without `--clone-audio` is too.

### Streaming and seeds

```bash
uv run kova-tts generate "A longer line." --stream --out out.wav
uv run kova-tts generate "Reproducible." --seed 7 --out out.wav
```

`--stream` decodes as the model generates and reports time to first audio. It is not faster
overall; it is here because time to first audio is the number a streaming deployment lives on.

### All flags

| Flag | |
|---|---|
| `text` (positional) | What to say |
| `--text-file FILE` | Read the text from a file, or from `-` for stdin |
| `-o`, `--out FILE` | Output WAV (default `out.wav`) |
| `--voice NAME` | LoRA voice name |
| `--stream` | Decode as it generates, and report time to first audio |
| `--seed N` | Make the sampling reproducible |
| `--clone-audio FILE` | Reference recording to clone |
| `--clone-text TEXT` | What the reference says, word for word |
| `--temperature`, `--top-p`, `--top-k`, `--repetition-penalty`, `--max-tokens` | Sampling; unset flags keep the preset the voice implies |
| `--model`, `--codec`, `--wavlm`, `--lora-dir` | Override the configured `KOVA_*` path |
| `--device` | Torch device, e.g. `cuda:1` |
| `--asr-model`, `--asr-language`, `--asr-device` | Transcription, used only when cloning without `--clone-text` |

Sampling flags are all-or-nothing per preset: leave them alone and the engine picks the preset
the voice calls for, which is what you want. Setting even one switches to explicit overrides on
top of that preset.

---

## `prepare-data`

A folder of recordings in, a JSONL corpus out. Full page: [Dataset
preparation](dataset-preparation.md).

```bash
uv run kova-tts prepare-data recordings/ --out data/train.jsonl --val-split 0.1
```

```
source       recordings (4 audio files)
transcripts  4 from .txt/.lab sidecars
clips        4 encoded
duration     0:13 total, longest 4.0 s / 398 tokens
written      data/train.jsonl (3 rows), data/val.jsonl (1 rows)
```

`--dry-run` prints the same summary without loading the codec or writing anything. Re-running is
safe and resumes: rows carry the clip they came from, so a second run only encodes what is new.

---

## `finetune`

```bash
uv run kova-tts finetune --config my_voice.yaml
```

Prints the adapter directory on success. Full page: [Finetuning](finetuning.md).

| Flag | |
|---|---|
| `--config FILE` | YAML run config. Required |
| `--dataset`, `--output-dir`, `--model`, `--voice`, `--epochs`, `--lr` | Override the config |
| `--resume-from DIR` | Resume from a Trainer checkpoint directory |
| `--no-wandb` | Disable wandb even if the config enables it |

Command-line paths resolve against your working directory; paths *inside* the config resolve
against the config file's own directory.

---

## `merge`

Fold a LoRA adapter into its base model and write a standalone checkpoint.

```bash
uv run kova-tts merge runs/ft_001_my_voice_2026-08-03_11-04-22/final merged/
uv run kova-tts generate "The merged checkpoint works on its own." --model merged/ --out out.wav
```

| Flag | |
|---|---|
| `adapter` (positional) | Directory holding `adapter_config.json` |
| `output` (positional) | Directory to write the merged model to |
| `--base-model` | Override the base checkpoint recorded in the adapter |
| `--dtype` | `bfloat16` (default), `float16`, `float32` |

The merge is exact for a plain LoRA. Merge in the dtype you will serve in — merging in float32
and casting afterwards is not the same arithmetic.

---

## `serve`

```bash
uv run kova-tts serve                      # http://127.0.0.1:8000
uv run kova-tts serve --port 8123 --device cuda:1
```

Needs `uv sync --extra server`. Full page: [Server](server.md).

| Flag | |
|---|---|
| `--host`, `--port` | Where to bind. Loopback by default, deliberately: there is no authentication |
| `--model`, `--codec`, `--wavlm`, `--lora-dir`, `--device` | Override the configured paths |
| `--clone-preroll N` | Reference codes decoded to warm a cloned generation (default 80, one second) |
| `--voice-alias NAME=VOICE` | Point one of OpenAI's stock voice names at a real one, e.g. `alloy=my_voice`. Repeatable |
| `--busy-timeout SECONDS` | How long a second caller waits before a 409 (default 5; `0` refuses at once) |
| `--no-warmup` | Skip the startup generation; the first request pays for it instead |
| `--log-level` | `critical`, `error`, `warning`, `info`, `debug`, `trace` |

---

## `demo`

```bash
uv run kova-tts demo
```

Needs `uv sync --extra demo`, and a source checkout — the demo ships in `apps/`, not in the
installed package. Full page: [Demo](demo.md).

| Flag | |
|---|---|
| `--host`, `--port` | Where to bind (default `127.0.0.1:7860`) |
| `--share` | Expose a temporary public `gradio.live` link |
| `--open` | Open a browser window on startup |
| `--preload` | Load and warm up at startup, so the first visitor waits for none of it |
| `--model`, `--codec`, `--wavlm`, `--lora-dir`, `--device` | Override the configured paths |
| `-v`, `--verbose` | Log what the engine is doing |

---

## `download`

Prefetch weights into the local Hub cache, so a later run needs no network.

```bash
uv run kova-tts download
uv run kova-tts download --wavlm --cache-dir /models/hub
```

| Flag | |
|---|---|
| `--repo` | Hub repo holding the LM and codec |
| `--wavlm` | Also fetch WavLM, needed only to encode audio |
| `--cache-dir` | Hub cache directory to fill |
| `--token` | Hub token, for a gated repository |
| `--force` | Re-download even when the cache already has it |

Artifacts already pointed at a local path by `.env` are reported and skipped — downloading a
second copy would be a surprise measured in gigabytes.

!!! warning "This cannot work yet"

    The published repository does not exist. Today the command fails with a 404 and prints what
    to do instead:

    ```
    error: Could not download the model repository 'kova-ai/kova-tts-1b': RepositoryNotFoundError: 404 Client Error.
      - the released weights may not be public yet; set KOVA_HUB_REPO to the repository you have access to,
      - or run `huggingface-cli login` if it is gated,
      - or point KOVA_MODEL_PATH / KOVA_CODEC_PATH at local checkpoints and skip the download entirely (`kova-tts paths` shows what resolved).
    ```

The same flags are available as a standalone script,
[`scripts/download_weights.py`](https://github.com/evalabs-ai/kova-tts/blob/main/scripts/download_weights.py),
for a container build that has the repository checked out but not installed.
