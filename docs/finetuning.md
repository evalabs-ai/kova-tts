# Finetuning

Train a per-voice LoRA adapter. Higher fidelity than [cloning](voice-cloning.md), and a voice
you can keep: an adapter is ~55 MB on disk and loads by name.

```bash
uv sync --extra finetune
```

Single GPU by design. A 1B backbone with rank-64 attention adapters and gradient checkpointing
trains a voice in minutes on one consumer card; distributed training would add a launcher, a
sharding story, and a class of bugs this workload does not need. Pin the card with
`CUDA_VISIBLE_DEVICES` if the machine has more than one — on a multi-GPU box the trainer loads
onto device 0 and explicitly refuses to become a `DataParallel` run, which would quietly
multiply your effective batch size by the device count.

Check which card that index actually is before you rely on it: `nvidia-smi`'s GPU number is not
torch's, unless `CUDA_DEVICE_ORDER=PCI_BUS_ID` is set. See
[Choosing a GPU](installation.md#choosing-a-gpu-on-a-multi-gpu-machine).

!!! warning "Rights"

    Point this at your own recordings. A permissive license on a recording does not grant the
    right to reproduce the speaker's voice, and you are responsible for having permission to
    train on the voice you are training on.

## 1. Build a corpus

[Dataset preparation](dataset-preparation.md) covers this in full.

```bash
uv run kova-tts prepare-data recordings/ --out data/train.jsonl --val-split 0.05
```

200–800 clips of clean, consistently recorded speech is the useful range. More helps, but
inconsistent recording conditions hurt more than extra data helps: if you have hours of
material, a few hundred well-chosen clips beat all of it.

## 2. Write a config

A usable config is three lines. Every other key defaults to something that trains a good
adapter, so a config file is a diff against the defaults rather than a wall of values to check.

```yaml title="my_voice.yaml"
dataset: data/train.jsonl
voice: my_voice
output_dir: runs
```

**Relative paths resolve against the config file's own directory**, not your shell's — so a
config can sit next to its corpus and the pair moves between machines intact. That includes the
default `output_dir: runs`.

Unknown keys are an error with a spelling suggestion, not a warning: a typo in `max_lenght` that
silently trained at the default would cost a GPU-hour to notice.

The fully commented template is
[`packages/kova-tts/src/kova_tts/finetune/configs/example_voice.yaml`](https://github.com/evalabs-ai/kova-tts/blob/main/packages/kova-tts/src/kova_tts/finetune/configs/example_voice.yaml).
Copy it next to your data and delete the lines you are not changing.

### The keys worth knowing

| Key | Default | |
|---|---|---|
| `dataset` | — | Required. JSONL corpus |
| `val_dataset` | unset | Held-out corpus. Unset means `val_split` carves one out |
| `val_split` | 0.05 | Fraction held out. `0` trains without evaluation |
| `max_length` | 4096 | Examples longer than this are **skipped, not truncated** |
| `model` | unset | Base checkpoint. Unset means whatever `KOVA_MODEL_PATH` resolves to |
| `init_from_adapter` | unset | Continue training an existing adapter |
| `lora.r` / `lora.alpha` | 64 / 64 | `alpha == r` means a scaling of 1.0 |
| `lora.target_modules` | `[q_proj, k_proj, v_proj, o_proj]` | Attention only |
| `epochs` | 4 | |
| `batch_size` / `grad_accum` | 2 / 4 | Effective batch of 8 sequences |
| `lr` | 0.00015 | |
| `gradient_checkpointing` | true | ~30% slower per step, and what makes 4096-token rows fit |
| `output_dir` | `runs` | Parent directory; each run gets a fresh numbered subdirectory |
| `voice` | dataset stem | Named in the run directory |
| `wandb.enabled` | false | And no project name is assumed |

Two of those are traps worth spelling out:

- **`max_length` skips, it does not truncate.** Truncating would cut off `<|speech_end|>` and
  teach the model that utterances never end. Rows over the limit are dropped in
  `prepare-data`, where the reason can be reported, and dropped again here if any survived.
- **`lr: 1.5e-4` parses as a *string* in YAML** unless the exponent is signed. Write `0.00015`.
  The validator says so if you get it wrong.

Leaving the MLP and embedding layers frozen is what keeps an adapter at ~55 MB and keeps a voice
from dragging the base model's pronunciation with it. Raising `r` much past 64 buys little for a
single voice; 16–32 is a reasonable trade if you are shipping many.

### The ending-weight recipe

The single biggest quality lever, and it is on by default.

```yaml
ending_weight:
  enabled: true
  ramp_tokens: 10
  ramp_max: 4.5
  eos_token_weight: 3.0
```

A LoRA trained with plain cross entropy learns a voice quickly but keeps *ending* the utterance
badly — trailing off, repeating the last syllable, running past the transcript before finally
emitting `<|speech_end|>`. The cause is arithmetic: an ending is a handful of token positions
out of thousands, so the signal that decides "stop here" is drowned out by the signal that
decides "keep talking in this voice".

So the loss is reweighted rather than the data resampled. Cross entropy on the last
`ramp_tokens` positions before `<|speech_end|>` is multiplied by a ramp rising 1.0 → `ramp_max`,
and the `<|speech_end|>` label itself by `eos_token_weight`.

The weights land on the ending **pattern** — the last real tokens of speech. If you extend this
recipe, keep it anchored on `<|speech_end|>`: weighting silence instead teaches "any quiet
stretch means stop", and the model truncates mid-sentence.

`ramp_tokens: 0` keeps only the flat EOS upweight. `enabled: false` is plain unweighted cross
entropy.

## 3. Train

```bash
uv run kova-tts finetune --config my_voice.yaml
```

Every line you will see, from a deliberately tiny run — a real corpus puts thousands in place of
the `3/3`, and many more steps between the first line and the last:

```
INFO Run directory: /path/to/runs/ft_001_my_voice_2026-08-03_15-23-55
INFO train.jsonl: 3/3 examples; longest 282
INFO val.jsonl: 1/1 examples; longest 343
INFO Training on 3 examples, validating on 1 (longest 282 of max_length 4096)
INFO 2 CUDA devices visible; training on device 0 only. Set CUDA_VISIBLE_DEVICES to pick a different one.
INFO Ending-weight loss on: EndingWeight(enabled=True, ramp_tokens=10, ramp_max=4.5, eos_token_weight=3.0)
trainable params: 13,631,488 || all params: 1,266,485,248 || trainable%: 1.0763
INFO LoRA adapter saved to /path/to/runs/ft_001_my_voice_2026-08-03_15-23-55/final
/path/to/runs/ft_001_my_voice_2026-08-03_15-23-55/final
```

`n/m examples` is the corpus load report: rows that could not be used are skipped and counted
rather than raised on, so a badly formed line does not lose a run at line 9,000. The adapter
directory is printed on stdout, alone on the last line, so a script can capture it. The example
config estimates ~15 minutes on one consumer GPU for a few hundred clips.

Each run creates a fresh numbered subdirectory inside `output_dir`:

```
runs/ft_001_my_voice_2026-08-03_15-23-55/
  config.yaml          <- the resolved config, defaults filled in, paths absolute
  checkpoint-*/        <- Trainer checkpoints
  final/               <- the adapter you want
```

`config.yaml` is the record of what actually ran. A run whose hyperparameters are only knowable
from a config file that has since been edited is a run that cannot be reproduced.

Overrides that do not need a config edit:

```bash
uv run kova-tts finetune --config my_voice.yaml --epochs 6 --lr 0.0001 --no-wandb
uv run kova-tts finetune --config my_voice.yaml --resume-from runs/ft_001_.../checkpoint-500
```

Command-line paths resolve against your working directory, not the config's.

The config file is optional. Without `--config`, the defaults plus your flags are the whole
config, which is the convenient form for sweeping one recipe over several corpora and base
checkpoints without writing a file or editing `.env` per run:

```bash
uv run kova-tts finetune --dataset data/a/train.jsonl --val-dataset data/a/val.jsonl \
    --model /path/to/base --voice a --output-dir runs
```

`--model` wins over `KOVA_MODEL_PATH`; the full set is `--dataset`, `--val-dataset`,
`--val-split`, `--model`, `--voice`, `--output-dir`, `--epochs`, `--lr`, `--resume-from` and
`--no-wandb`. Flags are applied before the config is validated, so `--dataset` also rescues a
config whose own `dataset:` no longer exists. Any `KOVA_*` variable can likewise be set in the
environment for a single command -- a real environment variable always beats `.env`.

!!! note "Running outside the checkout"

    `.env` is found by walking up from your working directory, so a run launched from beside its
    corpus will not see the checkout's `.env` and will fall through to the Hub. Set `model:` in
    the config, or export `KOVA_MODEL_PATH`. See
    [Installation](installation.md#check-that-it-worked).

Metrics are loss only — nothing here listens to the model while it trains. `run()` takes a
`callbacks` argument of `transformers.TrainerCallback` objects, so anything beyond loss metrics
can be attached from outside without editing the trainer.

## 4. Use the voice

A voice is a directory named after the voice, inside `KOVA_LORA_DIR`. Copy or symlink `final/`
into place under that name:

```bash
cp -r runs/ft_001_my_voice_2026-08-03_15-23-55/final "$KOVA_LORA_DIR/my_voice"
uv run kova-tts paths                 # the voice now shows on the last line
uv run kova-tts generate "A voice trained from my own recordings." --voice my_voice --out out.wav
```

The directory name *is* the voice name — a directory called `final` gives you a voice called
`final`. Any subdirectory holding an `adapter_config.json` counts.

`--lora-dir` overrides the configured directory for one command, which is convenient while you
are still deciding whether you like a run:

```bash
uv run kova-tts generate "Testing." --voice final --lora-dir runs/ft_001_my_voice_2026-08-03_15-23-55 --out out.wav
```

In Python it is just a name:

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("A voice trained from my own recordings.", "my_voice")
```

!!! note "Adapters may not retrain the embedding or the LM head"

    The engine narrows the LM head to a copy of the embedding rows taken at load time, so an
    adapter with `embed_tokens` or `lm_head` in `modules_to_save` would be applied on the input
    side and silently ignored on the output side. Loading one raises instead, and tells you to
    merge it first. No default config produces such an adapter — `modules_to_save` is unset.

## 5. Merging, if you need it

An adapter is the right thing to ship when one process serves many voices: the base model loads
once and adapters swap per request. Merging is right when a deployment serves exactly one voice,
or when a runtime cannot apply adapters at all.

```bash
uv run kova-tts merge runs/ft_001_my_voice_2026-08-03_15-23-55/final merged/
uv run kova-tts generate "The merged checkpoint works on its own." --model merged/ --out out.wav
```

The output is an ordinary causal-LM directory any tool can load, at the cost of a full copy of
the weights per voice. The merge is exact for a plain LoRA — `W + (alpha/r) * B @ A` folded into
`W`, no approximation — but do it in the dtype you will serve in. Merging in float32 and casting
afterwards is not the same arithmetic.

The tokenizer travels with the adapter when one was saved beside it, so a merged voice keeps the
vocabulary it was actually trained with.

## Continuing a run

```yaml
init_from_adapter: runs/ft_001_my_voice_2026-08-03_15-23-55/final
```

Trains on top of an existing adapter instead of a fresh one. The existing adapter's own
`adapter_config.json` then defines `r`, `alpha` and `target_modules`, so the `lora:` section is
deliberately ignored rather than half-applied.

This is different from `--resume-from`, which picks an interrupted run back up from a Trainer
checkpoint — same run, same schedule, same optimizer state.

## What is happening under the hood

Worth knowing, because it explains the failure modes:

- **The corpus is tokenized here**, at load time, not stored as ids. That is why a corpus is a
  readable text file you can diff and grep, and why a badly formed line is skipped and counted
  rather than crashing a run at line 9,000.
- **Loss starts after `<|speech_start|>`.** Every label up to and including that tag is `-100`;
  supervision runs from the first audio token through `<|speech_end|>` inclusive. The model is
  being taught to speak the given text in a voice, not to reproduce the prompt.
- **`enable_input_require_grads` is on with gradient checkpointing.** Without it the
  checkpointed base layers produce activations with no `grad_fn`, the adapter receives no
  gradient at all, and you get a run that trains for an hour and changes nothing.
- **Padding is masked out of the loss**, and the pad token is EOS, which avoids resizing the
  embeddings.

[Architecture](architecture.md) has the rest.
