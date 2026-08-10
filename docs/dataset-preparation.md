# Dataset preparation

`kova-tts prepare-data` turns a folder of your own recordings into the JSONL corpus
[finetuning](finetuning.md) expects.

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

Point it at your own recordings. Nothing in this project ships audio, and nothing here assumes a
particular corpus.

## What it does

```
discover -> load, downmix, resample to 32 kHz -> split on silence -> trim -> -23 LUFS
         -> encode -> {"text": ...} JSONL
```

Three things about that order are load-bearing:

- **Trim before normalise.** Leading silence drags a clip's integrated loudness down, and the
  gain applied would then be wrong.
- **Normalise before encode.** Every clip in the corpus should reach the codec at the same
  level, whatever it was recorded at.
- **Transcribe after segmenting.** A recording cut into six clips needs six transcripts, and
  there is no honest way to divide one transcript across the cuts.

That last one has a consequence worth stating plainly: **a clip that already has a transcript is
never split.** If it is too long it is reported as too long, and you either cut it yourself or
re-run with `--transcribe`, which re-transcribes everything and therefore makes splitting safe.

## Input layouts

Two, because these are the two ways people actually have their data.

**Sidecars.** `clip.wav` beside `clip.txt`. Nothing to configure. `.lab` also works — that is
what forced aligners emit.

```
recordings/
  clip_00.wav
  clip_00.txt
  clip_01.wav
  clip_01.txt
```

**A metadata file.** One row per recording, naming the file and its text. CSV, TSV, JSONL, a
JSON array of objects, or a `{filename: text}` mapping. It is auto-detected in the source
directory when named `metadata.csv`, `transcripts.tsv`, or one of the other usual spellings;
otherwise pass `--metadata`.

```csv
file_name,text
clip_00.wav,The kettle had just boiled.
clip_01.wav,She read the last page twice.
```

Column names are the awkward part, so the recognised ones include `file_name`, `filename`,
`file`, `audio`, `path`, `wav`, `id` for the filename, and `text`, `transcript`, `sentence`,
`normalized_text`, `caption` for the text. A headerless two-column table is read positionally.
Whatever it matched is printed back in the summary, so a mis-read column is visible without
opening the file:

```
transcripts  metadata.csv: 'file_name' -> 'text', ',' delimited (412 rows)
```

Metadata wins over sidecars: an explicit manifest is a deliberate act, a stray `.txt` next to a
clip may be anything. Filenames are matched by relative path, then basename, then stem, so an
absolute path exported on another machine still lines up. A name matching two different
recordings is **dropped as ambiguous** rather than guessed at.

**No transcripts at all** is also a layout. Every clip is transcribed with ASR, and long
recordings can be split, because there is no transcript to divide.

Recognised audio extensions: `.wav`, `.flac`, `.ogg`, `.opus`, `.mp3`, `.m4a`. mp3 and m4a
depend on your libsndfile; a file that cannot be decoded is reported as a skip, not assumed
absent. Hidden files and directories are ignored.

## Transcription

| Flag | |
|---|---|
| *(default)* | Transcribe only what has no transcript |
| `--transcribe` | Re-transcribe everything, replacing supplied transcripts. Also what allows long recordings to be split |
| `--no-transcribe` | Never transcribe; recordings without a transcript are skipped |

ASR is faster-whisper, behind the `data` extra:

```bash
uv sync --extra data
uv run kova-tts prepare-data recordings/ --asr-model large-v3 --asr-language en
```

Transcription quality is a ceiling on the adapter — the model is being taught to say *this text*
in this voice, so a wrong word teaches a wrong pronunciation. The default is `small` rather than
`tiny` for that reason, and `large-v3` is worth the extra minutes on a corpus you will train on
repeatedly. `--asr-language` skips per-clip detection, which is right for a single-language
corpus.

`--asr-device` and `--asr-compute-type` control CTranslate2: `float16` on CUDA, `int8` on CPU by
default.

## Segmentation

Defaults suit narration recorded in one take.

| Flag | Default | |
|---|---|---|
| `--max-seconds` | 30 | Clips longer than this are split, or reported as too long |
| `--min-seconds` | 0.5 | Shorter clips are dropped |
| `--silence-db` | 40 | Silence threshold, dB below the recording's loudest frame |
| `--min-silence` | 0.4 | How much quiet counts as a boundary |
| `--pad` | 0.05 | Silence kept at each end after trimming. A hard cut on the first sample of speech clips a plosive's attack |
| `--max-tokens` | 4096 | Row length limit, matching the trainer's `max_length` |

Why splitting exists at all: audio costs 80 tokens per second, so a 4096-token row holds about
51 seconds *including* the text prompt — and rows over the limit are dropped by the trainer, not
truncated. A half-hour recording has to become clips first. Cuts land in the middle of a
silence, so no word is ever sliced in half; a stretch with no silence in it is not cut, and the
clip is reported.

Why trimming exists: leading and trailing silence is tokens the model is taught to emit before
speaking and after finishing, which is exactly how an adapter learns to trail off instead of
stopping.

The threshold is relative to the loudest frame, not absolute, because trimming happens *before*
loudness normalisation and the input level is whatever your recording chain produced.

## Output

One JSON object per line. The field that matters is `text`, built by
`kova_tts.prompt.training_example` and nothing else — the JSONL format is defined in exactly one
place:

```json
{"id": "clip_00.wav#0", "audio": "clip_00.wav", "segment": 0, "seconds": 4.34,
 "tokens": 398, "text": "<|begin_of_text|><|text_prompt_start|>...<|speech_end|>"}
```

Every other field is provenance. The trainer ignores them; `prepare-data` uses `id` and `audio`
to resume.

`--val-split 0.1` writes a held-out corpus beside the training one — `val.jsonl` next to
`train.jsonl` — which is what a finetuning config expects to find. The split is deterministic
for a given corpus and `--seed`, whatever order the clips were encoded in.

## Re-running is safe

Rows carry the clip they came from, so a second run over the same folder reuses what is already
in the output file and only encodes what is new. A run interrupted at clip 3000 of 4000 picks up
where it stopped. Both halves of an existing corpus are read back, held-out rows included —
leaving them out would re-encode every one of them and write a second copy into the training
file.

`--overwrite` ignores an existing corpus instead of adding to it. Corpus files are written
atomically, via a temporary file, because the usual call rewrites a file it just read.

## Dry runs and the report

```bash
uv run kova-tts prepare-data recordings/ --dry-run
```

`--dry-run` does everything except loading the codec and writing files, so it costs no GPU and
no 1.2 GB WavLM load. Use it to check that transcripts matched before committing to a long run.

Nothing here raises for a clip it cannot use. Every one becomes a skip with a reason, and the
summary groups them:

| Reason | |
|---|---|
| `unreadable` | The file could not be decoded |
| `no transcript` | No sidecar, no metadata row, and transcription was off or unavailable |
| `empty transcript` | ASR heard no speech |
| `silent` | No signal above the noise floor, or silent after trimming |
| `too short` | Under `--min-seconds` |
| `too long` | Over `--max-seconds` with no silence to split on, or over the token limit |

The summary warns when the longest rows sit close to `--max-tokens`: a corpus like that starts
losing rows the moment anything about it changes.

## Encoding

Each clip is encoded on its own. The encoder runs in float32 rather than fp16, because fp16
encode flips a small fraction of VQ codes.

## From Python

```python
from kova_tts.data.prepare import prepare

report = prepare(
    "recordings/",
    "data/train.jsonl",
    val_split=0.1,
    transcribe=None,        # None: only what has no transcript. True: everything. False: never
    max_tokens=4096,
)
print(report.summary())
```

`prepare` also takes an already-loaded `encoder=` and `transcriber=`, which is how you prepare
several folders without reloading WavLM and Whisper each time.

## Next

[Finetuning](finetuning.md) — turning the corpus into a voice.
