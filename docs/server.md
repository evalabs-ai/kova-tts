# Server

A local HTTP server with three ways to get audio out — one request/one file, Server-Sent Events,
and an incremental WebSocket session — plus an [OpenAI-compatible
endpoint](#openai-compatible-api) for tools that already speak that API.

```bash
uv sync --extra server
uv run kova-tts serve
```

```
kova-tts 0.1.0 serving on http://127.0.0.1:8000
```

Interactive API docs are at `/docs`, and the OpenAPI schema at `/openapi.json`.

!!! warning "No authentication, and one request at a time"

    It binds loopback by default and it has no auth, because the answer to "who may use my GPU"
    is "whoever is on this machine". Pass `--host 0.0.0.0` knowingly.

    It also generates exactly one utterance at a time. A second concurrent caller waits up to
    `--busy-timeout` seconds (default 5) and is then refused with `409`. There is no queue.
    [Why](#concurrency).

## Endpoints

| Method | Path | |
|---|---|---|
| `GET` | `/health` | Is the model up, where, and how many voices |
| `GET` | `/v1/voices` | The LoRA voices installed on this machine |
| `POST` | `/v1/tts` | Synthesize the whole utterance, return audio bytes |
| `POST` | `/v1/tts/stream` | Synthesize, streaming Server-Sent Events |
| `WS` | `/v1/ws` | Incremental session: push text, flush when you want audio |
| `POST` | `/v1/audio/speech` | [OpenAI-compatible](#openai-compatible-api) speech, audio bytes back |
| `GET` | `/v1/models` | [OpenAI-compatible](#openai-compatible-api) model list, for client dropdowns |

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"device":"cuda","backend":"torch","sample_rate":48000,
 "voices":1,"busy":false,"version":"0.1.0"}
```

`backend` is which decode loop the LM runs on, `torch` or `mlx`. It is reported separately from
the device because on Apple Silicon both backends say `mps` — the codec is torch on Metal either
way — and the difference between them is a factor of four. See
[Apple Silicon](apple-silicon.md).

`model_loaded` is always true in a served response — loading happens in the application
lifespan, so the server does not accept connections until it has finished. The field exists so a
client can tell this server apart from a proxy that answers `/health`.

### `GET /v1/voices`

```bash
curl -s http://127.0.0.1:8000/v1/voices
```

```json
{"voices":[{"name":"my_voice","kind":"lora"}],"lora_dir":"/models/kova/voices"}
```

`lora_dir` is echoed because an empty list with no explanation is a poor way to discover that
your adapter directory is pointed somewhere wrong. `"lora_dir": null` means none is configured.

### `POST /v1/tts`

```bash
curl -X POST http://127.0.0.1:8000/v1/tts \
     -H 'Content-Type: application/json' \
     -d '{"text": "Hello from the local server."}' \
     -o speech.wav
```

Request body:

| Field | | |
|---|---|---|
| `text` | required | Up to 5000 characters |
| `voice` | optional | A name from `GET /v1/voices`. Omit for the base voice |
| `sampling` | optional | `temperature`, `top_p`, `top_k`, `repetition_penalty`, `max_tokens` — send only what you want changed |
| `seed` | optional | Fixes the sampler |
| `response_format` | optional | `"wav"` (default) or `"pcm"` |

Response headers carry `X-Sample-Rate` and `X-Duration-Seconds`. `wav` comes back as `audio/wav`
with a `Content-Disposition` filename; `pcm` comes back as `audio/pcm` — headerless 16-bit
little-endian mono at 48 kHz.

```bash
curl -sD - -o out.pcm -X POST http://127.0.0.1:8000/v1/tts \
     -H 'Content-Type: application/json' \
     -d '{"text": "Raw PCM this time.", "response_format": "pcm"}' | grep -i '^x-\|content-type'
```

```
x-sample-rate: 48000
x-duration-seconds: 2.237
content-type: audio/pcm
```

Requests are strict — an unknown field is a `422` naming the field, not a silently ignored
setting:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/tts \
     -H 'Content-Type: application/json' \
     -d '{"text": "hi", "temperature": 0.5}'
```

```json
{"error":"invalid_request","message":"temperature: Extra inputs are not permitted"}
```

(`temperature` belongs inside `sampling`.)

### `POST /v1/tts/stream`

Same body as `/v1/tts`, minus `response_format`. The response is `text/event-stream`: a `chunk`
event per decoded frame, then exactly one terminal event — `done` on success, `error` on
failure.

```
event: chunk
data: {"index": 0, "audio": "<base64>", "sample_rate": 48000}

event: done
data: {"chunks": 6, "samples": 123840, "duration_seconds": 2.58, "sample_rate": 48000}
```

`audio` is base64 of raw 16-bit little-endian mono PCM. Concatenating every chunk's decoded
bytes gives exactly the PCM `/v1/tts` would have returned with `"response_format": "pcm"`.

A chunk is one codec window, about 390 ms. On a warm server the first one arrived in roughly
230 ms on an RTX 5090 and 0.45–0.75 s on an RTX 3090.

Anything the server refuses outright — a bad body, an unknown voice, a `409` — arrives as an
HTTP status with the JSON error envelope, because the reservation happens before the response
starts. A failure *after* the stream has opened cannot change the status code, so it arrives as
an `error` event instead. Either way a stream ends with exactly one terminal event.

The response sets `X-Accel-Buffering: no`, which stops an intermediate proxy holding every chunk
back until the generation finishes.

[`examples/stream_sse.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/stream_sse.py)
implements the whole protocol in the standard library, which makes it the specification:

```bash
uv run python examples/stream_sse.py "Streaming over server sent events works." --out sse.wav
```

```
first audio after 231 ms
8 chunks, 3.04 s of audio
wrote sse.wav
```

### `WS /v1/ws`

An incremental session, for a caller that does not have the whole text yet — an LLM writing a
reply a token at a time, a chat UI. **Send text as you get it and the session speaks as it
arrives**, generating only as far as the text it has been given. A flush does not start
generation; it ends the turn, letting the model finish naturally.

So a client has nothing to schedule: stream text, flush when the turn is over. An extra flush is
not a free head start — it declares the turn over that many times, and every turn gets its own
ending. [When to flush](#when-to-flush) measures what that costs.

```
-> {"start_context": {"voice": "my_voice"}}
<- {"context_started": {...}}
-> {"send_text": "The first half of a sentence "}
-> {"send_text": "and the second half."}
-> {"flush": true, "flush_id": "a"}
<- {"audio_chunk": "<base64 audio>"}       (repeated)
<- {"flush_completed": true, "flush_id": "a"}
-> {"close_context": true, "flush_id": "b"}
<- {"audio_chunk": "<base64 audio>"}       (the rest, then the container's own tail)
<- {"flush_completed": true, "flush_id": "b"}
<- {"context_closed": true}
```

Client frames, discriminated by their key:

| Frame | |
|---|---|
| `{"start_context": {...}}` | Settings for the session, fixed for its lifetime: `voice` **or** `reference` (a clip to clone), `sampling`, `seed`, `response_format` |
| `{"send_text": "..."}` | Add text. Send as often as you like; the session speaks it as it arrives |
| `{"flush": true, "flush_id": "..."}` | End the turn — finish what is left, stay open |
| `{"close_context": true, "flush_id": "..."}` | End the turn, then end the session |

Server frames:

| Frame | |
|---|---|
| `{"context_started": {...}}` | The accepted configuration, with defaults filled in and `response_format` holding what will really be sent |
| `{"audio_chunk": "<base64>"}` | The next bytes of the session's audio stream |
| `{"flush_completed": true, "flush_id": "..."}` | Terminates every flush |
| `{"context_closed": true}` | The last frame of the session |
| `{"error": "...", "flush_id": "..."}` | `flush_id` only when one flush is to blame |

Unset optionals are omitted rather than sent as `null`.

Guarantees worth relying on:

- **No boundary is audible.** Text is split into chunks the model renders in one generation, and
  each chunk continues the one before it — its words *and* the codes they became. That chain runs
  through turn boundaries as readily as through sentence boundaries, and each turn's decoder
  resumes from the audio the last one ended on, so nothing in the stream carries a seam. What a
  turn boundary does cost is an *ending*, which is why [where you flush](#when-to-flush) decides
  how much speech comes out.
- **`flush_completed` always arrives**, even when a flush produced no audio or failed, so a
  client that waits for it after every flush can never hang.
- **Frames for two flushes never interleave**, however fast you send. Inbound frames are read by
  one task, flushes execute one at a time in arrival order by a second, and every outgoing frame
  goes through a single queue drained by a third.
- **One bad frame does not drop the session.** It comes back as an `error` frame and the socket
  stays open.
- **The model is held only for the duration of a flush**, not the life of the connection. An
  idle session costs nothing.

Generation and decoding run at the same time inside a flush, so the audio for its first chunk is
on the wire while its last chunk is still being generated.

#### When to flush

Once per turn, when the text is finished. There is no schedule to tune: the session already
speaks as text arrives, and the only thing an extra flush adds is a place for the voice to stop.

Measured with
[`scripts/ws_latency.py`](https://github.com/evalabs-ai/kova-tts/blob/main/scripts/ws_latency.py),
which feeds a 32-word paragraph at 5 words a second — roughly what a fast LLM emits — and times
from the **first word sent**, on an RTX 3090 at 16 kHz out:

| Flush on | First audio | Speech produced | Delivery | Worst join |
|---|---|---|---|---|
| the end of the turn | 2.63 s | 10.71 s | 1.5× realtime | — |
| a sentence | 2.17 s | 10.61 s | 1.8× realtime | 0.00003 |
| a 500 ms timer | 0.98 s | 38.64 s | 1.8× realtime | 0.02795 |
| every word | 0.44 s | 40.88 s | 1.4× realtime | 0.09601 |

The first row is the intended usage, and the 2.63 s is mostly text: the session waits for enough
characters to be worth generating, which at 5 words a second is about two of them. It is speaking
long before the flush arrives.

The lower first-audio numbers below it are real, and they are not a win. A flush ignores the
buffering threshold and runs the model to its own end, so an early one does start the audio
sooner — by asking for a complete little utterance instead of the first part of a long one.
Flushing every word asks thirty-two times and gets thirty-two endings: the same thirty-two words,
stretched to nearly four times the duration. Sentence boundaries are real turn boundaries, which
is why that row costs nothing.

Continuity holds either way: the worst step at any boundary stays inside the waveform's own
99.9th-percentile step, which is the session carrying the voice across it.

One caveat about playback. Delivery ran 1.4–1.8× realtime on the 3090, but the margin a player
would have had in hand went **negative** on two of these runs — −0.06 s flushing at the end of the
turn, −0.01 s flushing every word — so a player that starts on the very first chunk can briefly
run dry. On an RTX 5090 the same runs never dropped below 0.33 s. Buffer a few hundred
milliseconds before starting playback rather than playing the first chunk on arrival.

#### Cloning a voice

`start_context` takes a `reference` instead of a `voice`, and the session speaks as the voice in
that recording — no adapter, no training. Send the clip either way round:

```json
{"start_context": {"reference": {"transcript": "exactly what the clip says",
                                 "audio": "<base64 of an audio file>"}}}
```

| Field | | |
|---|---|---|
| `transcript` | required | What the clip says, word for word |
| `audio` | either | Base64 of a file — anything `soundfile` reads. Resampled and loudness-normalised for you |
| `codes` | or | That clip already encoded, 80 per second, as `Voice.ref_codes` holds it |

The transcript is not optional and not approximate. Cloning is a **continuation**: it goes in
front of your text and the clip's codes lead the generation, so the model reads the transcript as
words it has already spoken. One that does not match leaves it speaking words the audio does not
contain. There is no speech recognition in the server to fall back on.

`audio` is self-contained — one frame and the session can speak — but it is the only thing in this
server that loads WavLM (~1.2 GB), so the *first* cloned session of a process pays for that before
anything else happens: 8.8 s to first audio against 1.2 s for the ones after it, on an RTX 3090.
`codes` never touches WavLM, which is what a client speaking as the same voice all day should
send — encode the clip once with `KovaTTS.clone`, keep `voice.ref_codes`, and start every session
from it.

Five to twenty seconds of one speaker clones well. Under a second is refused, and so is anything
over twenty — the reference leads *every* prompt the session builds rather than costing once, so a
long clip is refused rather than trimmed, with a message saying how many seconds would fit.
`voice` and `reference` are two ways of saying who speaks, so sending both is an error.

Everything is validated when `start_context` arrives, before a word of text: an unusable clip
fails immediately, no session is created, and you can correct it and send `start_context` again on
the same connection. `context_started` reports what the clip came to, which is what a client
cannot work out for itself:

```json
{"context_started": {"reference": {"seconds": 8.5, "codes": 680}, "response_format": {...}}}
```

The clip itself does not come back. See [Voice cloning](voice-cloning.md) for choosing a
recording and for what goes wrong when one is unsuitable.

#### `response_format`

`start_context` chooses the container and the sample rate, both fixed for the session:

```json
{"start_context": {"response_format": {"encoding": "wav", "sample_rate": 16000}}}
```

| `encoding` | | |
|---|---|---|
| `pcm` | default | Raw 16-bit little-endian mono. No header, so nothing states the rate — `context_started` does |
| `wav` | | Header on the first chunk, samples after it |
| `mp3` | | Defined for a fixed set of rates; anything else is snapped up to the nearest one |
| `flac` | | Streamed form states "length unknown", which the format allows |
| `opus` | refused | Its Ogg pages only leave once they are full — about a second of speech — so a flush's last words would still be inside the encoder when its `flush_completed` went out. Ask `POST /v1/audio/speech` for opus |

`sample_rate` is anything from 8000 to 48000 and defaults to the model's own 48000, so a client
that sends no `response_format` is unaffected. **Realtime voice pipelines run at 16 kHz**, and
asking for it here is better than resampling the output yourself: the conversion uses the same
windowed-sinc filter every other path in this project uses, and its state crosses chunk joins and
flush boundaries alike, so nothing anywhere in the stream carries a step.

`context_started` echoes the format **that is really in effect**, including a rate a container had
to snap, so a client never has to guess what its bytes are:

```json
{"context_started": {"response_format": {"encoding": "mp3", "sample_rate": 22050}}}
```

Every `audio_chunk` of a session is a slice of one stream, not a file of its own: concatenate them
in order and you have a single `wav`, `mp3` or `flac`. The same caveat as the HTTP path applies —
[a stream is not a file](#a-stream-is-not-a-file) — so a container that states a length at the
front states an unknown one, and `pcm` or `wav` are the two a strict reader will take back
unaided.

One consequence of the container spanning the session: `flush_completed` means every sample of
that flush has reached the encoder, which for `pcm` and `wav` also means every byte has reached
you. `mp3` and `flac` are block codecs and may still be holding a partial block — up to 36 ms and
128 ms of audio respectively — which leaves with the next flush's bytes or with the trailer at
`close_context`. Nothing is lost; it simply arrives a fraction of a block late.

An encoding this server will not stream, one your `soundfile` build cannot write, or a rate
outside the range is refused at `start_context` with a message naming what does work. No session
is created, so you can correct it and send `start_context` again on the same connection.

[`examples/stream_ws.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/stream_ws.py)
drives a session from text it already has, treating each sentence as a turn. A client that is
still receiving its text sends it with `send_text` as it arrives and flushes once, at the end:

```bash
uv run python examples/stream_ws.py "Hello. This is the second sentence." --out ws.wav
uv run python examples/stream_ws.py "For a voice agent." --rate 16000 --out agent.wav
```

```
context started: {'response_format': {'encoding': 'pcm', 'sample_rate': 48000}}
first audio after 265 ms
flush s0 done, 110400 bytes so far
flush s1 done, 313200 bytes so far
wrote ws.wav: 3.26 s at 48000 Hz
```

## OpenAI-compatible API

`POST /v1/audio/speech` answers OpenAI's audio-speech request, so anything already built against
that API — Open WebUI, SillyTavern, LibreChat, AnythingLLM, Home Assistant, the `openai` Python
and Node SDKs — can use this server by changing a base URL and nothing else.

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"model": "tts-1", "input": "Hello from the local server.", "voice": "default"}' \
     -o speech.mp3
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="not-needed")
client.audio.speech.create(
    model="tts-1", voice="default", input="Hello from the local server."
).stream_to_file("speech.mp3")
```

The response is **audio bytes with the matching `Content-Type`**, not a JSON envelope — that is
what those clients read. There is no authentication, so the API key is ignored; send anything.

!!! note "An addition, not a replacement"

    `/v1/tts`, `/v1/tts/stream` and `/v1/ws` are unchanged and are not deprecated. They are this
    server's own API and they keep everything this one has no field for: `sampling`, `seed`,
    strict rejection of unknown fields, and an incremental session that speaks one continuous
    utterance however the text arrives. Use the
    compatible endpoint to plug into a tool you did not write; use the native API when you are
    writing the client.

### Request

| Field | | |
|---|---|---|
| `input` | required | The text to speak, up to 5000 characters |
| `model` | optional | Accepted whatever it says — there is one model here. `kova-tts:<voice>` [picks a voice](#choosing-a-voice-from-a-model-dropdown) |
| `voice` | optional | A name from `GET /v1/voices`; OpenAI's stock names are [accepted too](#voices). Omit it, or send `"default"`, for the base voice |
| `response_format` | optional | `mp3` (default), `opus`, `flac`, `wav`, `pcm`. `aac` is refused |
| `speed` | optional | Only `1.0`. Anything else is a `422` — [why](#speed-and-instructions) |
| `instructions` | optional | Only empty. Anything else is a `422` — [why](#speed-and-instructions) |
| `stream_format` | optional | `"audio"` (default) or `"sse"` — [OpenAI's audio events](#stream_format-sse) |
| `sample_rate` | optional | **Extension.** Output rate; default is the model's 48 kHz |
| `stream` | optional | **Extension.** `false` for a file with exact headers instead of a stream |

Unknown fields are **ignored** rather than rejected, which is the opposite of `/v1/tts`. OpenAI
adds fields to this body — `instructions` and `stream_format` both arrived after the endpoint
shipped — and every client that follows sends them to whatever server it is pointed at.
Refusing a field this server has never heard of would break working clients on someone else's
release schedule. The cost is real: a misspelled `respones_format` here is silently the default,
where on `/v1/tts` it would be a `422` naming the field. Use `/v1/tts` when you want the strict
door.

### Voices

`voice` names a LoRA voice from `GET /v1/voices`. OpenAI's own voice names — `alloy`, `nova`,
`onyx` and the rest — do not exist here, but **they are accepted and speak in the base voice**
rather than failing:

```bash
# what every SDK example, and the stock config of most clients, sends
curl -s http://127.0.0.1:8000/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"model": "tts-1", "input": "It just works.", "voice": "alloy"}' \
     -o speech.mp3
```

OpenAI's schema accepts nothing but its own names, so a client's voice field arrives holding
`alloy` whether the user chose it or not — several hardcode it. Refusing that would fail the
first request anybody makes after changing a base URL, which is the exact moment this endpoint
exists to make easy. The full stock set is accepted: `alloy`, `ash`, `ballad`, `cedar`, `coral`,
`echo`, `fable`, `marin`, `nova`, `onyx`, `sage`, `shimmer`, `verse`.

A name that is neither installed here nor one of those is a typo, and typos still get a `404`
that says what to use:

```json
{"error":"not_found","message":"no voice named 'my_voise' on this server. Here `voice` names one of this machine's LoRA voices (GET /v1/voices), not an OpenAI voice like 'alloy' -- those are accepted and speak in the base voice. Available: my_voice. Omit voice, or send \"default\", for the base voice."}
```

Every response says which voice actually spoke, so nothing has to be guessed at:

```
X-Voice: my_voice        # or "base" for the base model's own voice
```

#### Pointing a stock name at a real voice

If this machine has LoRA voices, map them, and a client that only ever sends `alloy` gets one:

```bash
uv run kova-tts serve --voice-alias alloy=my_voice --voice-alias nova=another_voice
```

```bash
KOVA_VOICE_ALIASES="alloy=my_voice,nova=another_voice" uv run kova-tts serve
```

The flag is repeatable, the environment variable is comma-separated, and the flag wins. An
alias that names a voice which is not installed **stops the server at startup** rather than
failing one client later:

```
ValueError: voice alias points at a voice this machine does not have: alloy=ghost.
Installed voices: my_voice.
```

Unmapped stock names keep speaking in the base voice. `GET /v1/voices` remains the list of real
ones, and a LoRA voice actually named `alloy` — or `default` — always wins over any alias: the
adapter directory is consulted first.

#### Choosing a voice from a model dropdown

Some clients expose a model field and no voice field. `GET /v1/models` lists one entry per
installed voice for exactly that case:

```json
{"object":"list","data":[
  {"id":"kova-tts","object":"model","created":1751328000,"owned_by":"kova-tts"},
  {"id":"kova-tts:my_voice","object":"model","created":1751328000,"owned_by":"kova-tts"}]}
```

Sending `"model": "kova-tts:my_voice"` speaks in that voice. It is still not a gate — any other
model name is accepted and ignored — and an explicit `voice` that names a real voice wins over
the model id, because it is the more specific instrument. A hardcoded `alloy` in the voice field
does *not* override a voice chosen this way.

### Formats, and which of them stream

Every format is encoded **as the codec decodes it**, so audio starts long before the utterance
is finished. Measured on an RTX 5090 with a warm server, on one sentence of about nine seconds of
speech:

| `response_format` | `Content-Type` | Time to first byte | Notes |
|---|---|---|---|
| `pcm` | `application/octet-stream` | 0.27 s | Raw 16-bit little-endian mono. No header, so nothing states the rate |
| `wav` | `audio/wav` | 0.21 s | Streamed header has placeholder sizes; a player reads to the end |
| `mp3` | `audio/mpeg` | 0.23 s | OpenAI's default, and this endpoint's |
| `flac` | `audio/flac` | 0.22 s | Streamed form states "length unknown", which the format allows |
| `opus` | `audio/ogg` | 0.29 s | Ogg/Opus at 48 kHz. Bytes arrive in page-sized bursts, up to ~0.47 s apart |
| `aac` | — | — | Refused: every encoder for it is ffmpeg or a licensed library |

Encoding costs essentially nothing on top of raw PCM: the first byte of every container arrives
within ~30 ms of the first PCM byte, and the same request encoded whole instead of streamed
answers in 2.2–3.5 s, the full generation time. `opus` is the one with a caveat — its Ogg pages
only leave once they are full, so bytes come in bursts rather than steadily. Its *first* bytes
are as prompt as anything else's.

Everything is written by the libsndfile that `soundfile` already bundles, so no ffmpeg and no
extra dependency. An installation whose libsndfile is too old to write MPEG says so, and names
what it can produce instead; a request that named no format at all still gets audio.

#### A stream is not a file

`mp3` and `flac` both keep a length in a header at the *front*, which cannot be correct in a
stream: by the time the last sample exists those bytes are long gone. Players and ffmpeg read to
the end regardless, but a strict reader — `soundfile`, for one — will truncate a streamed MP3 and
refuse a streamed FLAC. Send `"stream": false` when you want a file with exact sizes, duration
metadata and seek tables:

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"input": "A file, not a stream.", "response_format": "flac", "stream": false}' \
     -o speech.flac
```

The trade is exactly the head start: the first byte then arrives when the last one does.
`X-Duration-Seconds` is sent on that path, since by then the duration is known. A streamed
response sends `X-Sample-Rate`, no `Content-Length`, and `X-Accel-Buffering: no` so an
intermediate proxy does not hold the chunks back. Every response carries `X-Voice`.

#### `sample_rate`

Not part of OpenAI's schema. Voice-agent pipelines run at 16 kHz and telephony at 8 kHz, and an
integrator who has to resample themselves usually reaches for linear interpolation and gets
aliasing:

```bash
curl -s http://127.0.0.1:8000/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"input": "Sixteen kilohertz, for a voice agent.", "response_format": "wav", "sample_rate": 16000}' \
     -o speech.wav
```

Absent, the response carries the model's own 48 kHz, so a stock OpenAI client is unaffected.
MP3 and Opus are each defined for a fixed set of rates and anything else is snapped *up* to the
nearest one they accept — Opus has no 32 kHz mode, so asking for `opus` at 32000 is carried at
48 kHz.
`X-Sample-Rate` always reports the rate the bytes are really at.

That matters most for `pcm`, which has no header to state it. **OpenAI documents `pcm` as
24 kHz**; this server sends the model's 48 kHz unless you ask otherwise, so a client that assumes
the documented rate should say so:

```json
{"input": "...", "response_format": "pcm", "sample_rate": 24000}
```

#### `stream_format: "sse"`

OpenAI's newer clients can ask for the audio as Server-Sent Events. The same bytes arrive
base64-encoded in `speech.audio.delta` events, followed by one `speech.audio.done`:

```
data: {"type":"speech.audio.delta","audio":"<base64 of the container's bytes>"}

data: {"type":"speech.audio.done","usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}
```

Concatenating every delta gives byte-for-byte what the plain response would have sent. Usage is
zeroed because this server meters nothing, and an invented token count would be worse than an
honest zero. A generation that fails halfway sends `{"type":"error","error":{...}}` — this
server's own addition, since OpenAI has no failure event here and a stream that simply stops is
worse than one that says why.

This is *not* the same thing as `/v1/tts/stream`, which is this server's own SSE protocol with
`chunk`/`done`/`error` events and raw PCM. Use whichever your client already speaks.

### `speed` and `instructions`

Both are **refused** with a `422` unless they are the no-op value — `speed: 1.0`, empty
`instructions`:

```json
{"error":"invalid_request","message":"speed: Value error, this model has no speed control, so only speed=1.0 is accepted, not 1.5; change the tempo after the fact instead (ffmpeg's atempo filter, or your player's playback rate)"}
```

This model has neither control. Accepting `speed: 1.5` and returning audio at normal tempo would
look like success and sound wrong, and nothing in the response would say so. A refusal is
something a client can see and act on; silence is not.

### `GET /v1/models`

```bash
curl -s http://127.0.0.1:8000/v1/models
```

```json
{"object":"list","data":[
  {"id":"kova-tts","object":"model","created":1751328000,"owned_by":"kova-tts"},
  {"id":"kova-tts:my_voice","object":"model","created":1751328000,"owned_by":"kova-tts"}]}
```

There is one model on this machine, so `model` in a request is accepted whatever it says — a
client hardcoding `tts-1` or `gpt-4o-mini-tts` works. This endpoint exists because several
clients fill a dropdown from it, and some refuse to send a request until it answers. The
`kova-tts:<voice>` entries are for the clients whose only dropdown is that one; see
[Choosing a voice from a model dropdown](#choosing-a-voice-from-a-model-dropdown).

### Everything else is the same server

The single-flight lock is shared: a second caller during a generation waits `--busy-timeout`
seconds and then gets the same `409` with the same envelope. Errors use the same envelope as
every other endpoint, so a client that gets a `404` or a `422` here gets `{"error", "message"}`
and a message that says what to change.

[`examples/openai_client.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/openai_client.py)
does the whole thing in the standard library — no SDK — and prints the time to first byte:

```bash
uv run python examples/openai_client.py "Pointing an OpenAI client at a local server." --out speech.mp3
```

```
POST /v1/audio/speech  response_format=mp3
first byte after 224 ms
wrote speech.mp3: 30348 bytes, audio/mpeg
whole utterance in 1.10 s
```

## Errors

Every failure comes back in the same envelope, whatever the status code:

```json
{"error": "busy", "message": "..."}
```

`error` is a short stable code to branch on; `message` is a sentence for a human, and it is
expected to say what to change. FastAPI's own validation errors are reshaped into the same
envelope, so a caller never has to parse two formats.

| Status | `error` | |
|---|---|---|
| 404 | `not_found` | A voice name that does not resolve, or a checkpoint that moved |
| 409 | `busy` | A generation is already in flight |
| 422 | `invalid_request` | Body did not validate, a sampling value is out of range, or an audio format this installation cannot produce |

```bash
curl -s -X POST http://127.0.0.1:8000/v1/tts \
     -H 'Content-Type: application/json' -d '{"text": "hi", "voice": "nope"}'
```

```json
{"error":"not_found","message":"No voice named 'nope'. Available: my_voice (in /models/kova/voices)."}
```

## Concurrency

There is none, by design. The generator holds a single static KV cache and one set of CUDA graph
input buffers; batch size is 1 in the decode loop. Overlapping two generations would raise from
three layers down, and that error would look like a `500` to a client — which is a lie, because
nothing is broken.

So every path that touches the model goes through one lock, and refusal is part of the API:

```json
{"error":"busy","message":"This server generates one utterance at a time -- the model holds a single KV cache, so there is nothing to parallelise onto. Wait for the request in flight to finish and send this one again."}
```

The refusal is not instant. `--busy-timeout` seconds of waiting come first (default 5), because
the overwhelmingly common "concurrent" request on a local server is not concurrent at all — it
is a page reload, a client retry, or a second script started a moment early, landing while the
previous generation's last frame is still being written. A few seconds of waiting turns that
race into a request that simply works, while a genuinely parallel caller still gets a clean 409
inside the timeout rather than being silently queued behind a thirty-second generation.

`--busy-timeout 0` refuses the instant the model is busy.

To serve more than one stream at a time, run more than one process, each pinned to its own GPU
with `--device`. `--device cuda:1` is a *torch* index, which is not necessarily the card
`nvidia-smi` calls GPU 1 — see
[Choosing a GPU](installation.md#choosing-a-gpu-on-a-multi-gpu-machine).

## Startup

The model loads once, in the application lifespan, off the event loop. By default the server
then synthesizes one short sentence before accepting connections: that forces the codec load and
the CUDA graph capture the first real request would otherwise pay for. `--no-warmup` skips it.

A warmup that fails is logged and does not stop the server booting.

## Options

See [`kova-tts serve`](cli.md#serve) for the full flag list.

`--clone-preroll` configures the loaded engine: a cloned voice normally pushes its whole reference
clip through the codec before generating, and the server sets that to one second instead, because
preroll is time to first audio spent re-rendering audio that is thrown away again. It matters if
you embed `create_app` in something that clones; the endpoints here do not read it.

!!! note "Cloning is a WebSocket session, not an HTTP request"

    `start_context` takes a [`reference`](#cloning-a-voice) — a recording, or one you encoded
    earlier — and the session speaks as that voice. Nothing on `/v1/tts`, `/v1/tts/stream` or
    `/v1/audio/speech` does: `voice` there names a LoRA adapter and nothing else, and there is no
    endpoint that uploads a clip.

    To clone outside a session, do it in Python and hand the resulting `Voice` to your own
    `KovaTTS`, or embed the app:

    ```python
    from kova_tts import KovaTTS
    from kova_tts.server.app import create_app

    tts = KovaTTS.from_pretrained()
    app = create_app(tts=tts)      # nothing is loaded; this engine is used as-is
    ```

    `create_app(tts=...)` is also how the test suite drives every endpoint against a stub with no
    GPU, no checkpoint and no torch.
