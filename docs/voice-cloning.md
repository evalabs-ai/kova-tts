# Voice cloning

Clone a voice from a few seconds of reference audio, with no training and no adapter. It works
because the model is simply **continuing a recording it has already been given**: the reference
clip's transcript goes in front of your text, the clip's codes go at the front of the
continuation, and the model carries on in the voice it is already speaking in.

That one sentence explains every rule on this page.

!!! warning "Rights"

    A permissive license on a recording does not grant the right to reproduce the speaker's
    voice. You are responsible for having permission to clone the voice you clone.

## The transcript has to be exact

The reference transcript is concatenated in front of your target text. If it does not match what
the clip actually says, the model tries to speak words the reference codes do not contain, and
the output garbles. Word for word, including the filler words.

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
voice = tts.clone("reference.wav", transcript="exactly what the clip says")
wav = tts.generate("Say something new in the same voice.", voice)
tts.save(wav, "cloned.wav")
```

Do not have the transcript? Hand `KovaTTS` a transcriber. There is no ASR in the package — only
the seam for one, which the `data` extra fills:

```bash
uv sync --extra data
```

```python
from kova_tts import KovaTTS
from kova_tts.cli import asr_transcriber

tts = KovaTTS.from_pretrained(transcriber=asr_transcriber())
voice = tts.clone("reference.wav")            # transcribed automatically
print(f'reference heard as: "{voice.ref_text}"')
```

Always check what it heard. A wrong transcript is the single most common cause of garbled
cloning, which is why the CLI prints it unprompted.

## From the command line

```bash
# with the transcript
uv run kova-tts generate "Say something new in this voice." \
    --clone-audio reference.wav \
    --clone-text "exactly what the clip says" \
    --out cloned.wav

# without it, transcribed by ASR (needs --extra data)
uv run kova-tts generate "Now without a transcript." \
    --clone-audio reference.wav \
    --out cloned.wav
```

```
reference heard as: "exactly what the clip says"
cloned.wav  1.66 s of audio in 4.5 s (0.4x)
```

`--voice` and `--clone-audio` are two ways of saying who, and passing both is an error.

## Choosing a reference clip

| | |
|---|---|
| Length | 5–20 seconds is the useful range |
| Minimum | 1 second, enforced — below that the model ignores the clip and falls back to its base speaker |
| Maximum | 20 seconds, enforced by trimming; more costs prompt, cache and decode time with no measured gain |
| Content | One speaker, no music, no second voice, no heavy room |
| Format | Anything `soundfile` reads; downmixed and resampled for you — see below for 16 kHz |

Loudness is normalised to −23 LUFS before encoding, so that every reference clip reaches the
codec at the same level regardless of how it was recorded. You do not do this yourself —
`clone` does it, whether you pass a path or an already-loaded waveform.

## 16 kHz recordings

The encoder takes 32 kHz, and also 16 kHz natively, through a small trained front end onto the
same codes. That is what a reference recorded at 16 kHz or below wants: a phone call, a dataset
built for ASR. Such a recording has nothing above 8 kHz either way.
Upsampled to 32 kHz it reaches the decoder looking like dull audio, and the clone comes out
dull; encoded natively, the 48 kHz decoder fills the top octave in with plausible high
frequencies. It cannot recover the real ones.

There is nothing to configure. `clone` picks the path from the recording itself: a file's header
says its rate, and an in-memory waveform's is `clone(..., sample_rate=)` (32000 when omitted). At
or below 16 kHz it is encoded at 16 kHz; anything wider goes in at 32 kHz.

## Sampling

Both presets sample identically; only the token budget differs:

| | Plain synthesis | Cloning |
|---|---|---|
| `temperature` | 1.1 | 1.1 |
| `top_p` | 0.9 | 0.9 |
| `top_k` | 75 | 75 |
| `repetition_penalty` | 1.1 | 1.1 |
| `max_tokens` | 2048 | 3500 |

`generate` picks the right one from the voice you pass, so normally you do nothing. The budget is
larger because the reference clip is generated before your text is, and a clone that runs out of
tokens stops mid-sentence.

To override, pass `params=`:

```python
from kova_tts import CLONE_SAMPLING, KovaTTS

tts = KovaTTS.from_pretrained()
voice = tts.clone("reference.wav", transcript="exactly what the clip says")
wav = tts.generate("Something else.", voice, params=CLONE_SAMPLING.replace(temperature=1.0))
```

## Reusing a clone

`clone` returns a `Voice`. Encoding the reference is the expensive part, so do it once and pass
the result around:

```python
voice = tts.clone("reference.wav", transcript="exactly what the clip says", name="my_voice")
for line in lines:
    tts.save(tts.generate(line, voice), f"{line[:8]}.wav")
```

```python
voice.name           # 'my_voice', or the file's stem if you did not name it
voice.is_clone       # True
voice.ref_seconds    # duration of the reference, in seconds
voice.ref_codes      # tuple of ints, 80 per second
voice.ref_text       # the transcript
```

A `Voice` is a frozen dataclass of plain data — a name, a transcript and a tuple of ints — so it
pickles, and it can be rebuilt directly if you have already encoded a clip yourself.

Clones are never written to disk by anything in this project. They live for the life of the
process. A voice you want to keep is a [LoRA adapter](finetuning.md), not a clone.

## What comes out

The generated audio **begins with a re-rendering of the reference clip**, because the reference
codes lead the continuation. `generate` and `stream` trim that for you. If you drive the
generator yourself, you have to trim `voice.ref_seconds` off the front — the arithmetic is
invisible in the code and very audible in the output.

Two other consequences of cloning being a continuation:

- The whole reference is pushed through the codec before the new speech, so the decoder's LSTM
  and first convolution start from real context rather than from silence. That preroll is what
  keeps the first frames free of a discontinuity.
- Prerolling the *whole* clip costs time to first audio, since it happens before a word of your
  text is decoded. The server prerolls one second instead (`--clone-preroll 80`), and trims to
  match. `KovaTTS(clone_preroll=...)` is the same knob in Python.

## Cloning fails when

| Symptom | Cause |
|---|---|
| Garbled speech, wrong words | The transcript does not match the clip |
| Sounds like the base voice, not the reference | Clip too short, or too quiet, or several speakers |
| `ValueError: Reference audio is 0.42 s` | Under the 1-second minimum |
| `ValueError: The reference clip encoded to no codes` | The clip is silent |
| `RuntimeError: Encoding is disabled` | The codec was built decode-only; use `KovaTTS`, which loads an encoding codec for `clone` |
| `MissingArtifact` naming WavLM | Encoding needs WavLM; set `KOVA_WAVLM_PATH` or let it fall back to `microsoft/wavlm-large` |

Cloning is the only thing in the project that loads WavLM (~1.2 GB). Plain synthesis and LoRA
voices never do.

## See also

- [`examples/voice_clone.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/voice_clone.py)
  — the whole flow, with ASR, in 30 lines.
- [Architecture](architecture.md#voice-cloning-as-continuation) — the prompt layout and why
  there is no dedicated zero-shot marker.
- [Finetuning](finetuning.md) — for a voice you want to keep, and for higher fidelity.
