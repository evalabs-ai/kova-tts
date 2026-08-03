"""The server on the real model: real weights, real codec, real audio out.

Everything in ``test_server.py`` runs against a stub, which proves the plumbing and nothing
about the speech. This file closes that gap: it boots the application exactly as
``kova-tts serve`` does, asks all three endpoints for the same sentence, and checks that what
comes back is audio of a plausible length -- and that the two streaming paths agree with each
other sample for sample.

Determinism comes from ``seed=``, not from a zero temperature:
:class:`~kova_tts.engine.types.SamplingParams` rejects ``temperature <= 0`` on purpose, because
greedy decoding is not a supported operating point for this model. A fixed seed pins the
sampler just as firmly and stays on the sampling path the model was tuned for.

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

from kova_codec.constants import SAMPLE_RATE

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

#: One sentence: long enough to span several frames and to be recognisably speech, short enough
#: that the whole file finishes while somebody is still watching it.
TEXT = "The quick brown fox jumps over the lazy dog."

#: Pins the sampler so the three endpoints are generating the same thing.
SEED = 20240917

#: Anything shorter than this for the sentence above is a failure, not a fast model.
MIN_SECONDS = 0.5

#: The server's busy timeout for this run. Generous, because up to three other jobs may be
#: sharing these GPUs and a request that waits is better than one that fails. Nothing here
#: needs an HTTP timeout: TestClient calls the application in-process, with no socket to
#: time out on.
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


def _wav_samples(payload: bytes) -> np.ndarray:
    audio, rate = sf.read(io.BytesIO(payload), dtype="float32")
    assert rate == SAMPLE_RATE
    return audio


def _pcm_samples(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0


def _body() -> dict:
    return {"text": TEXT, "seed": SEED}


# ------------------------------------------------------------------------------------ the model


def test_health_reports_a_cuda_device(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["device"].startswith("cuda")
    assert body["sample_rate"] == SAMPLE_RATE


def test_synthesis_returns_real_audio(client):
    started = time.perf_counter()
    response = client.post("/v1/tts", json=_body())
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"

    audio = _wav_samples(response.content)
    seconds = audio.size / SAMPLE_RATE
    print(f"\n[server] POST /v1/tts: {seconds:.2f} s of audio in {elapsed:.2f} s")

    assert seconds > MIN_SECONDS
    assert float(np.max(np.abs(audio))) > 0.01  # speech, not a buffer of zeros
    assert float(np.sqrt(np.mean(audio**2))) > 0.001


# ------------------------------------------------------------------- the three paths must agree


@pytest.fixture(scope="module")
def sync_audio(client) -> np.ndarray:
    response = client.post("/v1/tts", json={**_body(), "response_format": "pcm"})
    assert response.status_code == 200
    return _pcm_samples(response.content)


@pytest.fixture(scope="module")
def sse_audio(client) -> bytes:
    """The SSE stream's PCM, and the time to its first chunk."""
    chunks: list[bytes] = []
    started = time.perf_counter()
    first: float | None = None
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
    """The WebSocket session's PCM, and the time to its first chunk."""
    chunks: list[bytes] = []
    started = time.perf_counter()
    first: float | None = None
    with client.websocket_connect("/v1/ws") as ws:
        ws.send_json({"start_context": {"seed": SEED}})
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
    """Same seed, same decoder, same frames: the transports must not change the audio."""
    assert sse_audio == ws_audio


def test_the_stream_matches_the_synchronous_endpoint(sync_audio, sse_audio):
    streamed = _pcm_samples(sse_audio)
    assert streamed.size == sync_audio.size, (
        "the streaming and whole-utterance decoders produced different lengths, which means "
        "the two runs did not generate the same codes -- check that the seed is being applied"
    )
    # The seam tolerance from StreamingDecoder's own docstring, with room for 16-bit rounding.
    assert float(np.max(np.abs(streamed - sync_audio))) < 5e-3


def test_the_stream_is_the_same_speech(sync_audio, sse_audio):
    """A correlation check, because a length match alone would survive a shifted stream."""
    streamed = _pcm_samples(sse_audio)
    correlation = float(np.corrcoef(streamed, sync_audio)[0, 1])
    print(f"[server] streamed vs whole-utterance correlation: {correlation:.6f}")
    assert correlation > 0.999


def test_a_voice_that_does_not_exist_is_a_404(client):
    body = client.post("/v1/tts", json={**_body(), "voice": "definitely-not-a-voice"})
    assert body.status_code == 404
    assert body.json()["error"] == "not_found"
