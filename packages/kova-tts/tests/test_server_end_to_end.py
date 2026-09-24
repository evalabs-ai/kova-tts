"""The server on the real model: real weights, real codec, real audio out.

Everything in ``test_server.py`` runs against a stub, which proves the plumbing and nothing
about the speech. This file closes that gap: it boots the application exactly as
``kova-tts serve`` does, asks all three endpoints for the same sentence, and checks that what
comes back is audio of a plausible length -- and that the two streaming paths agree with each
other sample for sample.

Determinism comes from putting torch's random state back to one fixed point before each run,
not from a zero temperature: :class:`~kova_tts.engine.types.SamplingParams` rejects
``temperature <= 0`` on purpose, because greedy decoding is not a supported operating point for
this model. ``TestClient`` runs the application in this process, on the same device generator,
so resetting it here pins the sampler just as firmly and stays on the sampling path the model
was tuned for.

The two streaming endpoints are compared **byte for byte**; the synchronous one is compared
with a tolerance. That asymmetry is not slack, it is the design:
:class:`~kova_tts.engine.decoder.StreamingDecoder` decodes in windows and
:func:`~kova_tts.engine.decoder.decode_all` decodes the utterance in one pass, which its own
docstring measures as agreeing to ~2.3e-3 at the seams. Byte equality there would be a bug in
this test, not a feature of the server.

Needs a GPU and local weights; skipped otherwise.
"""

from __future__ import annotations

import base64
import io
import json
import time

import numpy as np
import pytest
import soundfile as sf
import torch

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

#: One sentence: long enough to span several frames and to be recognisably speech, short enough
#: that the whole file finishes while somebody is still watching it.
TEXT = "The quick brown fox jumps over the lazy dog."

#: The same words, delivered a piece at a time the way a client with an LLM behind it delivers
#: them: three flushes, none of them a complete thought on its own.
FLUSH_TEXTS = ("The quick brown fox ", "jumps over ", "the lazy dog.")

#: Spoken by the model itself to make a reference clip to clone from. No recording ships with
#: this repository, and one rendered here is real speech on the model's own distribution, which
#: is what a reference has to be for the checks below to mean anything. Long enough to clear the
#: one-second minimum with room to spare.
REFERENCE_TEXT = "Pack my box with five dozen liquor jugs, and then take it away again."

#: A sentence to speak in the cloned voice: nothing like the reference, so audio that came back
#: as a re-rendering of the clip would be obvious rather than plausible.
CLONE_TEXT = "How quickly daft zebras jump over the lazy dog."

#: A rate a realtime voice pipeline runs at, and not the codec's own.
AGENT_RATE = 16_000

#: The random state every compared run starts from, so they all sample the same codes.
RANDOM_STATE = 20240917

#: Anything shorter than this for the sentence above is a failure, not a fast model.
MIN_SECONDS = 0.5

#: The server's busy timeout for this run. Generous, because other work may be sharing the GPU
#: and a request that waits is better than one that fails. Nothing here needs an HTTP timeout:
#: TestClient calls the application in-process, with no socket to time out on.
BUSY_TIMEOUT = 300.0


def freest_cuda_device() -> str:
    """The CUDA device with the most free memory, so a parallel run is not fought over.

    Chosen by index rather than by setting ``CUDA_VISIBLE_DEVICES``: torch caches the device
    list at first use, and by the time a test runs another fixture may already have opened a
    context. Naming the device at construction sidesteps that entirely.
    """
    try:
        import torch

        # torch.cuda.mem_get_info, not nvidia-smi: CUDA defaults to FASTEST_FIRST device
        # ordering, so nvidia-smi's GPU 0 is not necessarily torch's cuda:0. Indexing one tool
        # by the other's order silently selects the wrong card.
        free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
        return f"cuda:{max(range(len(free)), key=free.__getitem__)}" if free else "cuda"
    except Exception:  # noqa: BLE001 - any failure just means "let torch choose"
        return "cuda"


@pytest.fixture(scope="module")
def client():
    """The real application, booted once for the whole file."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device available")

    from fastapi.testclient import TestClient

    from kova_tts import paths
    from kova_tts.server.app import create_app

    paths.load_dotenv()
    try:
        paths.model_path()
    except paths.MissingArtifact as exc:
        pytest.skip(f"no local model configured: {exc}")

    app = create_app(device=freest_cuda_device(), busy_timeout=BUSY_TIMEOUT)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def tts(client):
    """The loaded model itself, for comparing a session against a whole-utterance render."""
    return client.app.state.engine.tts


def _session(client, texts, **config) -> tuple[bytes, list[int], float]:
    """Drive a session that flushes once per entry of `texts` and closes on the last.

    Returns every audio byte it sent, the offset into those bytes at which each flush was
    acknowledged -- which is where a discontinuity would be if a flush boundary left one -- and
    the time to the first frame.
    """
    payload = bytearray()
    boundaries: list[int] = []
    started = time.perf_counter()
    first: float | None = None
    with client.websocket_connect("/v1/ws") as ws:
        _same_draws()
        ws.send_json({"start_context": config})
        assert "context_started" in ws.receive_json()
        for index, text in enumerate(texts):
            last = index == len(texts) - 1
            ws.send_json({"send_text": text})
            ws.send_json({"close_context" if last else "flush": True, "flush_id": str(index)})
            while True:
                frame = ws.receive_json()
                if "audio_chunk" in frame:
                    if first is None:
                        first = time.perf_counter() - started
                    payload += base64.b64decode(frame["audio_chunk"])
                elif "error" in frame:
                    pytest.fail(f"session failed: {frame['error']}")
                else:
                    assert frame == {"flush_completed": True, "flush_id": str(index)}
                    boundaries.append(len(payload))
                    break
        assert ws.receive_json() == {"context_closed": True}
    assert first is not None, "the session produced no audio at all"
    return bytes(payload), boundaries, first


def _wav_samples(payload: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
    assert rate == OUTPUT_SAMPLE_RATE
    return audio


def _pcm_samples(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0


def _body() -> dict:
    return {"text": TEXT}


def _same_draws() -> None:
    """Start the next generation from :data:`RANDOM_STATE`, like every other compared run."""
    torch.manual_seed(RANDOM_STATE)


# ------------------------------------------------------------------------------------ the model


def test_health_reports_a_cuda_device(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["device"].startswith("cuda")
    assert body["sample_rate"] == OUTPUT_SAMPLE_RATE


def test_synthesis_returns_real_audio(client):
    started = time.perf_counter()
    response = client.post("/v1/tts", json=_body())
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    audio = _wav_samples(response.content)
    seconds = audio.size / OUTPUT_SAMPLE_RATE
    print(f"\n[server] POST /v1/tts: {seconds:.2f} s of audio in {elapsed:.2f} s")

    assert seconds > MIN_SECONDS
    assert float(np.max(np.abs(audio))) > 0.01  # speech, not a buffer of zeros
    assert float(np.sqrt(np.mean(audio**2))) > 0.001


# ------------------------------------------------------------------- the three paths must agree


@pytest.fixture(scope="module")
def sync_audio(client) -> np.ndarray:
    _same_draws()
    response = client.post("/v1/tts", json={**_body(), "response_format": "pcm"})
    assert response.status_code == 200
    return _pcm_samples(response.content)


@pytest.fixture(scope="module")
def sse_audio(client) -> bytes:
    """The SSE stream's PCM. Its time to first chunk is printed, not returned."""
    chunks: list[bytes] = []
    started = time.perf_counter()
    first: float | None = None
    _same_draws()
    with client.stream("POST", "/v1/tts/stream", json=_body()) as response:
        assert response.status_code == 200
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line[len("event: ") :]
            elif line.startswith("data: "):
                payload = json.loads(line[len("data: ") :])
                if event == "chunk":
                    if first is None:
                        first = time.perf_counter() - started
                    chunks.append(base64.b64decode(payload["audio"]))
                elif event == "error":
                    pytest.fail(f"stream failed: {payload['message']}")
    print(f"[server] SSE time-to-first-audio: {first:.2f} s over {len(chunks)} chunks")
    assert chunks
    return b"".join(chunks)


@pytest.fixture(scope="module")
def ws_audio(client) -> bytes:
    """The WebSocket session's PCM. Its time to first chunk is printed, not returned."""
    chunks: list[bytes] = []
    started = time.perf_counter()
    first: float | None = None
    with client.websocket_connect("/v1/ws") as ws:
        _same_draws()
        ws.send_json({"start_context": {}})
        assert "context_started" in ws.receive_json()
        ws.send_json({"send_text": TEXT})
        ws.send_json({"close_context": True, "flush_id": "only"})
        while True:
            frame = ws.receive_json()
            if "audio_chunk" in frame:
                if first is None:
                    first = time.perf_counter() - started
                chunks.append(base64.b64decode(frame["audio_chunk"]))
            elif "error" in frame:
                pytest.fail(f"session failed: {frame['error']}")
            elif "context_closed" in frame:
                break
    print(f"[server] WS time-to-first-audio: {first:.2f} s over {len(chunks)} chunks")
    assert chunks
    return b"".join(chunks)


def test_the_two_streaming_paths_are_byte_identical(sse_audio, ws_audio):
    """Same draws, same decoder, same frames: the transports must not change the audio."""
    assert sse_audio == ws_audio


def test_the_stream_matches_the_synchronous_endpoint(sync_audio, sse_audio):
    streamed = _pcm_samples(sse_audio)
    assert streamed.size == sync_audio.size, (
        "the streaming and whole-utterance decoders produced different lengths, which means "
        "the two runs did not generate the same codes -- check that nothing else draws from "
        "the random state between runs"
    )
    # The seam tolerance from StreamingDecoder's own docstring, with room for 16-bit rounding.
    assert float(np.max(np.abs(streamed - sync_audio))) < 5e-3


def test_the_stream_is_the_same_speech(sync_audio, sse_audio):
    """A correlation check, because a length match alone would survive a shifted stream."""
    streamed = _pcm_samples(sse_audio)
    correlation = float(np.corrcoef(streamed, sync_audio)[0, 1])
    print(f"[server] streamed vs whole-utterance correlation: {correlation:.6f}")
    assert correlation > 0.999


# ------------------------------------------------------------ the session, as one utterance


def test_a_session_at_a_realtime_rate_matches_the_model(client, tts):
    """One flush at 16 kHz, against the same text rendered whole at 16 kHz.

    The session converts as it streams, with the filter's state crossing every join; the facade
    converts the finished waveform in one pass. They are the same audio, which is what makes the
    rate a delivery detail rather than a different render.
    """
    payload, _, first = _session(client, [TEXT], response_format={"sample_rate": AGENT_RATE})
    streamed = _pcm_samples(payload)
    _same_draws()
    whole = tts.generate(TEXT, None, params=None, sample_rate=AGENT_RATE)
    seconds = streamed.size / AGENT_RATE
    print(f"\n[server] WS {AGENT_RATE} Hz: {seconds:.2f} s of audio, first frame at {first:.2f} s")

    assert seconds > MIN_SECONDS
    assert streamed.size == whole.size, (
        "the session and the whole-utterance render produced different lengths, which means "
        "they did not generate the same codes -- check that nothing else draws from the "
        "random state between runs"
    )
    # The seam tolerance StreamingDecoder's own docstring measures, with room for 16-bit
    # rounding: the difference here is the windowed decode, not the rate conversion.
    assert float(np.max(np.abs(streamed - whole))) < 5e-3
    assert float(np.corrcoef(streamed, whole)[0, 1]) > 0.999


def test_the_flush_joins_carry_no_step(client):
    """Three flushes, none of them a whole sentence, at a rate that is not the codec's.

    Measured against the waveform's own step distribution, the way the resampler's tests measure
    it: a boundary that restarted anything would sit outside it.
    """
    payload, boundaries, first = _session(
        client, FLUSH_TEXTS, response_format={"sample_rate": AGENT_RATE}
    )
    streamed = _pcm_samples(payload)
    joins = np.array([offset // 2 for offset in boundaries[:-1]])
    joins = joins[(joins > 0) & (joins < streamed.size)]
    assert joins.size >= 2, "this text no longer produces enough flush boundaries to check"

    steps = np.abs(np.diff(streamed))
    worst = float(steps[joins - 1].max())
    ceiling = float(np.percentile(steps, 99.9))
    print(
        f"[server] {len(FLUSH_TEXTS)} flushes, first frame at {first:.2f} s; worst join step "
        f"{worst:.5f} against a 99.9th percentile of {ceiling:.5f}"
    )
    assert worst <= ceiling


def test_a_session_can_stream_a_container(client):
    """wav over the socket: one header, then the samples, at the rate that was asked for."""
    payload, _, _ = _session(
        client, FLUSH_TEXTS, response_format={"encoding": "wav", "sample_rate": AGENT_RATE}
    )
    assert payload[:4] == b"RIFF" and payload[8:12] == b"WAVE"
    assert payload.count(b"RIFF") == 1, "three flushes must not be three files"
    with sf.SoundFile(io.BytesIO(payload)) as handle:
        assert handle.samplerate == AGENT_RATE
        assert handle.channels == 1
        assert handle.read(dtype="float32").size / AGENT_RATE > MIN_SECONDS


def test_a_voice_that_does_not_exist_is_a_404(client):
    body = client.post("/v1/tts", json={**_body(), "voice": "definitely-not-a-voice"})
    assert body.status_code == 404
    assert body.json()["error"] == "not_found"


# ------------------------------------------------------------------ the session, in a cloned voice


@pytest.fixture(scope="module")
def reference(tts) -> np.ndarray:
    """The reference clip, rendered by the model itself so that nothing has to be committed."""
    wav = tts.generate(REFERENCE_TEXT, None, params=None)
    assert wav.size / OUTPUT_SAMPLE_RATE > 1.0, "the reference came out too short to clone from"
    return wav


@pytest.fixture(scope="module")
def reference_audio(reference) -> dict:
    """That clip as a client with only a recording would send it."""
    from kova_tts.audio import to_wav_bytes

    encoded = base64.b64encode(to_wav_bytes(reference, OUTPUT_SAMPLE_RATE)).decode("ascii")
    return {"transcript": REFERENCE_TEXT, "audio": encoded}


@pytest.fixture(scope="module")
def reference_encoded(tts, reference) -> dict:
    """The same clip as a client that encoded it once already would send it."""
    voice = tts.clone(reference, transcript=REFERENCE_TEXT, sample_rate=tts.sample_rate)
    return {"transcript": REFERENCE_TEXT, "codes": list(voice.ref_codes)}


def _cloned(client, reference: dict, texts=(CLONE_TEXT,)) -> tuple[bytes, float, dict]:
    """Drive a cloned session and return its audio, its time to first frame, and the echo."""
    payload = bytearray()
    started = time.perf_counter()
    first: float | None = None
    with client.websocket_connect("/v1/ws") as ws:
        ws.send_json({"start_context": {"reference": reference}})
        opening = ws.receive_json()
        assert "context_started" in opening, opening
        opened = time.perf_counter() - started
        for index, text in enumerate(texts):
            last = index == len(texts) - 1
            ws.send_json({"send_text": text})
            ws.send_json({"close_context" if last else "flush": True, "flush_id": str(index)})
            while True:
                frame = ws.receive_json()
                if "audio_chunk" in frame:
                    if first is None:
                        first = time.perf_counter() - started
                    payload += base64.b64decode(frame["audio_chunk"])
                elif "error" in frame:
                    pytest.fail(f"cloned session failed: {frame['error']}")
                elif "flush_completed" in frame:
                    break
        assert ws.receive_json() == {"context_closed": True}
    assert first is not None, "the cloned session produced no audio at all"
    return bytes(payload), opened, {**opening["context_started"], "first_audio": first}


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    n = min(left.size, right.size)
    a, b = left[:n] - left[:n].mean(), right[:n] - right[:n].mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def test_a_cloned_session_says_the_text_and_not_the_reference(client, reference, reference_audio):
    """The failure this design could produce: a session that opens by re-reading the clip.

    The reference leads every prompt, so the audio that follows it is the model continuing --
    but a session that decoded the reference into the stream, or primed the decoder without
    dropping what priming produced, would start with the clip instead. Measured as a
    correlation, because a length check alone would survive either mistake.
    """
    payload, opened, echo = _cloned(client, reference_audio)
    streamed = _pcm_samples(payload)
    seconds = streamed.size / OUTPUT_SAMPLE_RATE
    print(
        f"\n[server] WS cloned from audio: opened in {opened:.2f} s, first frame at "
        f"{echo['first_audio']:.2f} s, {seconds:.2f} s of audio"
    )

    assert seconds > MIN_SECONDS
    assert echo["reference"]["codes"] > 0
    head = min(
        int(echo["reference"]["seconds"] * OUTPUT_SAMPLE_RATE), streamed.size, reference.size
    )
    correlation = _correlation(streamed[:head], reference[:head])
    print(f"[server] cloned head vs the reference clip: {correlation:.3f}")
    assert abs(correlation) < 0.2, (
        "the session began with a re-rendering of the reference clip rather than with the text "
        "it was asked for"
    )


def test_both_forms_of_reference_describe_the_same_clip(client, reference_audio, reference_encoded):
    """One clip, two ways to send it: the codes are the reference either way.

    The times are the reason both exist -- encoding here needs WavLM loaded and a pass over the
    audio, and a client that speaks as one voice all day should pay for that once.
    """
    _, encoding, from_audio = _cloned(client, reference_audio)
    _, reusing, from_codes = _cloned(client, reference_encoded)
    print(
        f"[server] start_context: {encoding:.2f} s encoding the clip, {reusing:.2f} s from codes "
        f"already encoded"
    )
    assert from_audio["reference"] == from_codes["reference"]
    assert reusing < encoding


def test_a_cloned_session_joins_its_flushes_without_a_step(client, reference_encoded):
    """Three flushes in a cloned voice, at a rate that is not the codec's."""
    payload, _, _ = _cloned(client, reference_encoded, FLUSH_TEXTS)
    streamed = _pcm_samples(payload)
    assert streamed.size / OUTPUT_SAMPLE_RATE > MIN_SECONDS
    steps = np.abs(np.diff(streamed))
    assert float(steps.max()) <= 10 * float(np.percentile(steps, 99.9))


def test_a_reference_recording_of_your_own(client, tts):
    """The same path on a real recording, for anyone who has one to point at.

    Skipped unless ``KOVA_TEST_AUDIO`` names a clip and ``KOVA_TEST_TRANSCRIPT`` says what it
    says: nothing here ships one, a tone is not a voice, and with words that do not match the
    clip the model often stops at once -- a property of cloning, not a fault in the session, and
    one that would make this pass or fail by the luck of the draw. What is asserted is the trim.
    """
    import os

    from kova_tts import audio as audio_io

    path = os.environ.get("KOVA_TEST_AUDIO", "").strip()
    transcript = os.environ.get("KOVA_TEST_TRANSCRIPT", "").strip()
    if not path or not transcript:
        pytest.skip(
            "set KOVA_TEST_AUDIO to a local speech file and KOVA_TEST_TRANSCRIPT to what it says"
        )
    clip = audio_io.load_audio(path)[: int(10 * SAMPLE_RATE)]
    voice = tts.clone(clip, transcript=transcript)

    payload, _, echo = _cloned(client, {"transcript": transcript, "codes": list(voice.ref_codes)})
    streamed = _pcm_samples(payload)
    correlation = _correlation(streamed, clip)
    print(
        f"[server] cloned from a local recording: {echo['reference']['seconds']:.1f} s reference, "
        f"{streamed.size / OUTPUT_SAMPLE_RATE:.2f} s of audio, correlating {correlation:.3f} "
        "with the clip"
    )
    assert streamed.size > 0
    assert abs(correlation) < 0.2


def test_the_openai_endpoint_returns_real_audio(client):
    """``POST /v1/audio/speech`` on the real model, in the shape a real client sends.

    One request, driven from the body the OpenAI Python SDK puts on the wire, decoded with a
    library that was told only what ``Content-Type`` said. The mp3 is asked for as a file --
    ``stream: false`` -- because a streamed mp3 states no length and libsndfile believes the
    placeholder; the streaming path is covered by the wav below, whose bytes are the same
    samples either way.
    """
    started = time.perf_counter()
    response = client.post(
        "/v1/audio/speech",
        headers={"Accept": "application/octet-stream", "Content-Type": "application/json"},
        json={
            "model": "tts-1",
            "input": TEXT,
            "voice": "default",
            "response_format": "mp3",
            "speed": 1.0,
            "stream": False,
        },
    )
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"

    audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
    seconds = audio.size / rate
    print(f"\n[server] POST /v1/audio/speech mp3: {seconds:.2f} s of audio in {elapsed:.2f} s")

    assert rate == OUTPUT_SAMPLE_RATE
    assert seconds > MIN_SECONDS
    assert float(np.sqrt(np.mean(audio**2))) > 0.001  # speech, not a buffer of zeros

    streamed = client.post("/v1/audio/speech", json={"input": TEXT, "response_format": "wav"})
    assert streamed.status_code == 200
    assert streamed.headers["content-type"] == "audio/wav"
    wav, wav_rate = sf.read(io.BytesIO(streamed.content), dtype="float32")
    assert wav_rate == OUTPUT_SAMPLE_RATE
    assert wav.size / wav_rate > MIN_SECONDS
