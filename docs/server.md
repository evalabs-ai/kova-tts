# Server

A local HTTP server with three ways to get audio out: one request/one file, Server-Sent Events,
and an incremental WebSocket session.

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

### `GET /health`

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status":"ok","model_loaded":true,"device":"cuda","sample_rate":32000,
 "voices":1,"busy":false,"version":"0.1.0"}
```

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
little-endian mono at 32 kHz.

```bash
curl -sD - -o out.pcm -X POST http://127.0.0.1:8000/v1/tts \
     -H 'Content-Type: application/json' \
     -d '{"text": "Raw PCM this time.", "response_format": "pcm"}' | grep -i '^x-\|content-type'
```

```
x-sample-rate: 32000
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
data: {"index": 0, "audio": "<base64>", "sample_rate": 32000}

event: done
data: {"chunks": 6, "samples": 82560, "duration_seconds": 2.58, "sample_rate": 32000}
```

`audio` is base64 of raw 16-bit little-endian mono PCM. Concatenating every chunk's decoded
bytes gives exactly the PCM `/v1/tts` would have returned with `"response_format": "pcm"`.

A chunk is one codec window, about 390 ms. On a warm server the first one arrives in roughly
230 ms.

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
reply a token at a time, a chat UI. Text is **buffered and never spoken until you flush**, so
you choose the sentence boundaries and therefore where the latency goes.

```
-> {"start_context": {"voice": "my_voice"}}
<- {"context_started": {...}}
-> {"send_text": "The first half of a sentence "}
-> {"send_text": "and the second half."}
-> {"flush": true, "flush_id": "a"}
<- {"audio_chunk": "<base64 pcm>"}         (repeated)
<- {"flush_completed": true, "flush_id": "a"}
-> {"close_context": true, "flush_id": "b"}
<- {"audio_chunk": "<base64 pcm>"}         (anything still buffered)
<- {"flush_completed": true, "flush_id": "b"}
<- {"context_closed": true}
```

Client frames, discriminated by their key:

| Frame | |
|---|---|
| `{"start_context": {...}}` | Settings for the session, fixed for its lifetime: `voice`, `sampling`, `seed`, `response_format` |
| `{"send_text": "..."}` | Add to the buffer. Send as often as you like |
| `{"flush": true, "flush_id": "..."}` | Speak everything buffered, stay open |
| `{"close_context": true, "flush_id": "..."}` | Speak the rest, then end the session |

Server frames: `context_started`, `audio_chunk`, `flush_completed`, `context_closed`, `error`.
Unset optionals are omitted rather than sent as `null`.

Guarantees worth relying on:

- **`flush_completed` always arrives**, even when a flush produced no audio or failed, so a
  client that waits for it after every flush can never hang.
- **Frames for two flushes never interleave**, however fast you send. Inbound frames are read by
  one task, flushes execute one at a time in arrival order by a second, and every outgoing frame
  goes through a single queue drained by a third.
- **One bad frame does not drop the session.** It comes back as an `error` frame and the socket
  stays open.
- **The model is held only for the duration of a flush**, not the life of the connection. An
  idle session costs nothing.

And two things it deliberately does not do:

- **Prosody does not carry across a flush boundary.** Each flush is its own generation, so a
  flush per word sounds like a flush per word. Flush on sentences.
- **`response_format.sample_rate` must be 32000.** The server streams the codec's own rate and
  does not resample. The field exists so a client written against the production protocol works
  here unchanged.

[`examples/stream_ws.py`](https://github.com/evalabs-ai/kova-tts/blob/main/examples/stream_ws.py)
drives a session a sentence at a time, which is the pattern worth copying:

```bash
uv run python examples/stream_ws.py "Hello. This is the second sentence." --out ws.wav
```

```
context started: {'response_format': {'encoding': 'pcm', 'sample_rate': 32000}}
first audio after 200 ms
flush s0 done, 78400 bytes so far
flush s1 done, 208000 bytes so far
wrote ws.wav: 3.25 s
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
| 422 | `invalid_request` | Body did not validate, or a sampling value is out of range |

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

`--clone-preroll` configures the loaded engine: a cloned voice normally pushes its whole
reference clip through the codec before generating, and the server sets that to one second
instead, because preroll is time to first audio spent re-rendering audio that is thrown away
again. It matters if you embed `create_app` in something that clones — the HTTP API itself does
not.

!!! note "No cloning over HTTP"

    `voice` names a LoRA adapter and nothing else. There is no endpoint that uploads a reference
    clip, so there is no way to clone a voice through this server. Clone in Python and pass the
    resulting `Voice` to your own `KovaTTS`, or embed the app:

    ```python
    from kova_tts import KovaTTS
    from kova_tts.server.app import create_app

    tts = KovaTTS.from_pretrained(clone_preroll=80)
    app = create_app(tts=tts)      # nothing is loaded; this engine is used as-is
    ```

    `create_app(tts=...)` is also how the test suite drives every endpoint against a stub with no
    GPU, no checkpoint and no torch.
