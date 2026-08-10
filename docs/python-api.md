# Python API

Everything goes through `KovaTTS`. The server, the demo, the ComfyUI nodes and the CLI are all
callers of the same class, so its surface is small on purpose.

```python
from kova_tts import KovaTTS

tts = KovaTTS.from_pretrained()
wav = tts.generate("Hello world.")
tts.save(wav, "out.wav")
```

## Conventions

Two, and they hold everywhere.

**Audio is a 1-D float32 numpy array, mono, nominally in [-1, 1], at 32 kHz.** Not a tensor, not
a tuple, not interleaved stereo. `kova_tts.audio.as_waveform` coerces anything close to that
form; the codec's rate is `tts.sample_rate`, and it is always 32000. `generate` and `stream`
take `sample_rate=` when you need something else — see
[Output sample rates](#output-sample-rates).

**Codes are Python ints in `[0, 8191]`.** 80 of them per second of audio.

## `KovaTTS`

### Loading

```python
KovaTTS.from_pretrained(
    model=None,              # LM directory or Hub repo id; None resolves through KOVA_MODEL_PATH
    *,
    codec=None,              # codec checkpoint;      None resolves through KOVA_CODEC_PATH
    wavlm=None,              # WavLM directory;       None resolves through KOVA_WAVLM_PATH
    device=None,             # torch device; None picks cuda when available
    dtype=torch.bfloat16,
    max_cache_len=4096,      # prompt + generation must fit
    cuda_graph=None,         # None = on when the device is CUDA
    lora_root=None,          # directory of LoRA voices; None resolves through KOVA_LORA_DIR
    merge_lora=True,         # fold adapter weights into the base model
    clone_preroll=None,      # reference codes decoded to warm the codec; None = the whole clip
    transcriber=None,        # callable(path) -> str, for clone() without a transcript
)
```

The language model loads now; the codec loads on first use, and an *encoding* codec (the one
that needs WavLM) only if you call `clone`. `from_pretrained` also warms up: it captures the
decode step's CUDA graph and runs one prefill, so the first request does not pay ~1.2 s for it.

`merge_lora=True` folds an adapter into the base weights — faster per step (measured 1.23x), and
voices can still be switched because peft's merge is reversible. `merge_lora=False` keeps the
adapter separate and switches instantly, which is what a demo changing voices constantly wants.

There is also a process-wide singleton for scripts:

```python
from kova_tts import generate
generate("Hello world.", out="out.wav")
```

Keyword arguments are honoured only on the call that actually loads the model.

### Synthesis

```python
generate(text, voice=None, *, params=None, seed=None, sample_rate=None) -> np.ndarray
stream(text, voice=None, *, params=None, seed=None, sample_rate=None) -> Iterator[AudioFrame]
save(wav, path, sample_rate=None) -> Path
```

`voice` is `None` (the base voice), a LoRA voice name, or a `Voice` from `clone`. `params` is a
`SamplingParams`; leaving it `None` lets the voice pick its own preset, which is what you want.
`seed` fixes the sampler. `sample_rate` defaults to the model's own 32 kHz.

`stream` yields `AudioFrame` about every 390 ms and always ends with `is_final=True`.
Concatenating every frame's samples reproduces `generate`'s output.

```python
import numpy as np
frames = [f.samples for f in tts.stream("Two sentences. Streamed as they decode.")]
wav = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
```

### Output sample rates

The model is native 32 kHz. Voice-agent pipelines usually run at 16 kHz and telephony at 8 kHz,
so pass the rate you want and let the library convert:

```python
wav = tts.generate("Hello world.", sample_rate=16_000)

for frame in tts.stream("Hello world.", sample_rate=8_000):
    send(frame.samples, frame.sample_rate)        # sample_rate is the rate delivered
```

Both paths use one windowed-sinc polyphase filter, and `stream` carries its state across frames.
That matters more than it sounds: resampling each frame on its own restarts the filter at every
frame boundary, and the step it leaves behind is an audible click two or three times a second.
Here `stream(sample_rate=r)` concatenated is equal to `generate(sample_rate=r)`, sample for
sample, and both are equal to `torchaudio.functional.resample` of the whole 32 kHz waveform.

For a cloned voice the reference trim happens before the conversion, so the output starts at the
same word whatever rate you ask for.

If you have a finished waveform and want the rate changed, `kova_tts.audio.resample` is the same
filter. If you are converting a stream that did not come from `stream()`, use
`kova_tts.audio.StreamingResampler` rather than calling `resample` per chunk.

!!! warning "One generation at a time"

    The generator holds a single static KV cache and one set of CUDA graph input buffers.
    Starting a second generation while a `stream` is still running raises rather than silently
    interleaving. Serialise with a lock, or run one process per GPU.

    A `stream` you abandon halfway stays in flight until it is closed. Use
    `contextlib.closing`, or exhaust it.

### Voices

```python
voices() -> list[str]                                 # LoRA voice names on this machine
voice(name) -> Voice                                  # resolve one, or raise MissingArtifact
clone(audio, transcript=None, *, name=None) -> Voice  # build a cloned voice
```

`clone` takes a path or an already-loaded waveform. Without `transcript` it needs the
`transcriber` you passed to the constructor; without that it raises with the fix in the message.
See [Voice cloning](voice-cloning.md).

## Types

All of these live in `kova_tts.engine.types` and are re-exported from `kova_tts`. They are
framework-free: importing them pulls in neither torch nor transformers.

### `Voice`

```python
Voice(name, lora_path=None, ref_codes=(), ref_text="")
```

| Field / property | |
|---|---|
| `name` | Required and non-empty; it is how the voice is referred to later |
| `lora_path` | A peft adapter directory. Changes the weights |
| `ref_codes` | Encoded reference clip. Changes nothing about the weights |
| `ref_text` | The reference transcript. Required whenever `ref_codes` is set |
| `is_clone` | True when `ref_codes` is non-empty |
| `ref_seconds` | Duration of the reference — exactly how much of a cloned output is a re-rendering of it |

A `Voice` with neither a `lora_path` nor `ref_codes` cannot change how the model sounds, and is
rejected in `__post_init__` rather than silently doing nothing.

### `SamplingParams`

```python
SamplingParams(temperature=1.1, top_p=0.9, top_k=75, repetition_penalty=1.1,
               max_tokens=2048, seed=None)
```

Frozen and validated: `temperature > 0`, `0 < top_p <= 1`, `top_k >= 0` (0 disables it),
`repetition_penalty > 0`, `max_tokens > 0`. `replace(**overrides)` returns a re-validated copy.

Two presets ship. They sample identically; `CLONE_SAMPLING` differs only in `max_tokens` (3500
against 2048), because a clone prompt spends part of its budget on the reference clip before it
reaches your text:

```python
from kova_tts import CLONE_SAMPLING, TTS_SAMPLING
TTS_SAMPLING.replace(temperature=0.8)
SamplingParams.for_cloning(top_k=30)      # same thing, spelled as a classmethod
```

The model is sensitive to these values. Change one knob at a time.

Greedy decoding is not expressible — temperature must be positive. The generator has a `greedy=`
flag used by tests to compare the CUDA graph and eager paths exactly.

### `AudioFrame`

```python
AudioFrame(samples, sample_rate=32000, is_final=False)
frame.duration_seconds
```

`samples` must be 1-D; a 2-D array is rejected with a message pointing at `as_waveform`.

## Audio helpers

`kova_tts.audio`, all numpy in and numpy out:

| Function | |
|---|---|
| `load_audio(path, sample_rate=32000)` | Read any `soundfile` format as mono float32 at a rate |
| `as_waveform(wav)` | Coerce to 1-D float32 mono, downmixing 2-D input |
| `resample(wav, orig_rate, target_rate)` | Polyphase resampling of a finished waveform |
| `StreamingResampler(orig_rate, target_rate)` | The same filter, for a stream |
| `normalize_loudness(wav, sample_rate, target_lufs=-23.0)` | ITU-R BS.1770 with a peak limiter |
| `trim_leading(wav, seconds, sample_rate)` | Drop the first N seconds |
| `to_pcm_bytes(wav)` | 16-bit little-endian PCM, what the streaming server sends |
| `to_wav_bytes(wav, sample_rate)` | A complete WAV file in memory |
| `save_wav(path, wav, sample_rate)` | Write 16-bit WAV, creating the directory |
| `encode_audio(wav, sample_rate, fmt)` | Encode a whole waveform into a container |
| `StreamingEncoder(fmt, sample_rate)` | The same, incrementally |
| `content_type(fmt)` | Media type to send in `Content-Type` |
| `container_rate(fmt, sample_rate)` | The rate a format will actually be written at |
| `streaming_wav_header(sample_rate)` | 44-byte RIFF header with placeholder sizes |

`normalize_loudness` puts every clip at the same level before it reaches the codec, which is why
cloning applies it whatever you hand it. It leaves
silent, unmeasurable and sub-400 ms clips alone rather than applying an infinite gain.

`to_pcm_bytes` scales negative samples by 32768 and non-negative ones by 32767, because
full-scale `int16` is asymmetric. One shared factor either leaves −1.0 a step short of the rail
or wraps +1.0 round to −32768.

### Encoding

```python
from kova_tts.audio import SUPPORTED_FORMATS, content_type, encode_audio

encode_audio(wav, 32_000, "mp3")     # -> bytes
content_type("mp3")                  # 'audio/mpeg'
```

| `fmt` | | `content_type` |
|---|---|---|
| `pcm` | Raw 16-bit little-endian samples, no container and no stated rate | `application/octet-stream` |
| `wav` | 16-bit RIFF | `audio/wav` |
| `mp3` | MPEG-1 layer III | `audio/mpeg` |
| `flac` | Lossless | `audio/flac` |
| `opus` | Opus in an Ogg container | `audio/ogg` |

All five come from the libsndfile that `soundfile` already bundles — no ffmpeg, no extra
dependency. **Check `SUPPORTED_FORMATS` rather than that table**: it is probed at import from
what this install can actually write, since MPEG support only arrived in libsndfile 1.1 and a
wheel may carry an older one. An unavailable or unknown format raises with the available list in
the message; `aac` is not something libsndfile writes at all and is refused.

MP3 and Opus are each defined only for a fixed set of sample rates, and 32 kHz is not one of
Opus's. `container_rate(fmt, rate)` reports what a format will really be written at — audio is
resampled to it, and the container states it, so the result plays at the right speed either way.

### Encoding a stream

```python
from kova_tts.audio import STREAMING_FORMATS, StreamingEncoder

encoder = StreamingEncoder("mp3", 32_000)
for frame in tts.stream(text):
    if data := encoder.encode(frame.samples):
        send(data)
send(encoder.finish())
```

`encode` returns whatever bytes the codec has produced, which may be nothing on the first call —
a frame-based codec cannot emit anything until it has a frame. It never holds the utterance.

`pcm` and `wav` add zero latency (`wav` is `streaming_wav_header` followed by the samples). `mp3`
and `flac` produce bytes on the first chunk at the streaming decoder's cadence. **`opus` does
not**, and is deliberately absent from `STREAMING_FORMATS`: libsndfile emits an Ogg page only
once the page is full, about a second of speech, so opus bytes arrive in bursts a second apart
however finely audio is fed in. Encode opus with `encode_audio` when the audio is complete.

What a streaming encoder produces is a *stream*, not a saved file. Both `wav` and `flac` state
"length unknown" at the front, because the length is not known when those bytes go out — legal,
and what a stream is supposed to say, but a file written from it will not report its duration and
may not seek. MP3 additionally carries the codec's 1105-sample priming delay (35 ms at 32 kHz),
since the tag that tells a decoder to drop it is written at close. Every encoded frame is
otherwise identical to `encode_audio`'s. When the audio is already in hand, use `encode_audio`.

## Prompts and tokens

Useful if you are driving the language model yourself, or building training data.

```python
from kova_tts import tts_prompt, clone_prompt, training_example, parse_audio_tokens

tts_prompt("Hello world.")
# '<|text_prompt_start|>Hello world.<|text_prompt_end|><|speech_start|>'

training_example("Hello world.", [12, 7])
# '<|begin_of_text|><|text_prompt_start|>Hello world.<|text_prompt_end|><|speech_start|><|s_12|><|s_7|><|speech_end|>'
```

Note which of these include `<|begin_of_text|>` and which do not — `tts_prompt` leaves it out for
engines that prepend BOS themselves. Prompts are always tokenized with
`add_special_tokens=False`, so BOS appears exactly where these functions put it.

```python
from kova_tts import vocab_map
vm = vocab_map()               # cached per model directory
vm.codes_to_ids([0, 1, 8191])  # array([128256, 129367, 136245])
vm.decode_codes(raw_ids)       # ids -> codes, dropping EOS and text tokens
```

!!! danger "`128256 + code` is wrong"

    The 8192 audio tokens occupy a contiguous id block, but they were added to the tokenizer in
    *lexicographic* string order: `<|s_0|>` is 128256 and `<|s_1|>` is 129367. Computing the id
    arithmetically produces the wrong token and therefore silently wrong audio. Every conversion
    must go through `VocabMap`. [Architecture](architecture.md#the-token-layout) has the details.

## Paths

`kova_tts.paths` is the resolver everything else uses, and it is worth calling directly when you
are building tooling:

```python
from kova_tts import paths

paths.model_path()          # str: local directory, or a Hub repo id
paths.codec_path()          # Path: downloads from the Hub if no local path is configured
paths.wavlm_path()
paths.lora_dir()            # Path | None
paths.available_loras()     # list[str]
paths.load_dotenv()         # Path | None: which .env was loaded
```

A value that looks like a filesystem path must exist, or `MissingArtifact` is raised naming the
variable that set it — this keeps a typo in `.env` from turning into a multi-gigabyte download.

## Module map

| Module | |
|---|---|
| `kova_tts.engine.tts` | `KovaTTS`, sentence splitting, the segment carry |
| `kova_tts.engine.generator` | The batch-1 decode loop: static cache, CUDA graph, narrowed head, LoRA |
| `kova_tts.engine.decoder` | Windowed streaming decode, and the whole-utterance path |
| `kova_tts.engine.sampling` | Temperature, top-p, top-k, repetition penalty |
| `kova_tts.engine.types` | `Voice`, `SamplingParams`, `AudioFrame`, the presets |
| `kova_tts.tokens` | Structural tokens, `VocabMap` |
| `kova_tts.prompt` | The three prompt shapes |
| `kova_tts.voices` | Name → `Voice`, and clip → `Voice` |
| `kova_tts.audio` | Waveform I/O and conditioning |
| `kova_tts.paths` | Artifact resolution |
| `kova_tts.data` | Folder of recordings → JSONL corpus |
| `kova_tts.finetune` | LoRA training, the ending-weighted loss, merging |
| `kova_tts.server` | FastAPI app, SSE, WebSocket |
| `kova_codec` | `KovaCodec`: `encode`, `decode`, `decode_with_lstm` |

Importing `kova_tts` is cheap: `KovaTTS` and `Generator` are loaded lazily on first attribute
access, so tooling that only wants prompts or paths never pays for transformers.

## The codec directly

```python
from kova_codec import KovaCodec

codec = KovaCodec.from_checkpoint("codec.pt", device="cuda", decode_only=True)
wav = codec.decode(codes)                  # [T] codes -> [T * 400] samples
```

`decode_only=True` skips WavLM entirely — about 1.2 GB lighter and several seconds faster to
start — and `encode` then raises. For encoding, pass `wavlm=` instead. `decode_with_lstm` is the
streaming primitive; `kova_tts.engine.decoder.StreamingDecoder` is the window bookkeeping built
on top of it, and getting that bookkeeping right is fiddly enough that you should use it rather
than reimplement it.
