"""The server, driven end to end against a stub model: no GPU, no weights, no torch.

Everything here goes through :class:`fastapi.testclient.TestClient`, so the ASGI stack, the
validators, the SSE framing and the WebSocket handshake are all real; only the model is not.
That is the whole reason :func:`~kova_tts.server.app.create_app` takes a ``tts=`` argument.

The stub speaks the two methods the server uses -- ``generate`` and ``stream`` -- and returns a
tone whose length is a known function of the text, so a test can assert on the duration of the
audio that comes back rather than merely that some bytes arrived.
"""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from contextlib import aclosing
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from kova_codec.constants import SAMPLE_RATE
from kova_tts import paths
from kova_tts.engine.types import CLONE_SAMPLING, TTS_SAMPLING, AudioFrame, Voice
from kova_tts.paths import MissingArtifact
from kova_tts.server import protocol as wire
from kova_tts.server.app import create_app
from kova_tts.server.engine import Engine, aiter_frames

#: Samples of tone the stub produces per character of text. Small enough that a test finishes
#: instantly, large enough that a request spans several ~390 ms frames.
SAMPLES_PER_CHAR = 4000

#: What the stub cuts its output into, matching the real decoder's steady-state window.
FRAME_SAMPLES = 12400  # 31 codes at 80 Hz -> 387.5 ms at 32 kHz


class StubGenerator:
    """Stands in for the LM, purely so ``/health`` has a device to report."""

    device = "cpu"


class StubTTS:
    """A :class:`~kova_tts.engine.tts.KovaTTS` that makes tones instead of speech.

    Records the arguments of every call, so a test can prove that a request's voice, sampling
    overrides and seed reached the model rather than being dropped on the way.
    """

    def __init__(self, *, voices: list[str] | None = None, fail: Exception | None = None) -> None:
        self.generator = StubGenerator()
        self.sample_rate = SAMPLE_RATE
        self.lora_root = None
        self._voices = list(voices or ["some-voice", "another-voice"])
        self._fail = fail
        self.calls: list[dict] = []
        #: Set while a generation is running, cleared when it ends; the concurrency tests wait
        #: on this to know the model is genuinely occupied before sending a second request.
        self.entered = threading.Event()
        #: Held closed to stall a generation for as long as a test needs it stalled.
        self.release = threading.Event()
        self.release.set()

    # -- the surface the server uses ------------------------------------------------------

    def voices(self) -> list[str]:
        return list(self._voices)

    def voice(self, name: str) -> Voice:
        if name not in self._voices:
            raise MissingArtifact(f"No voice named {name!r}. Available: {', '.join(self._voices)}.")
        return Voice(name=name, lora_path=Path("adapters") / name)

    def generate(self, text, voice=None, *, params=None, seed=None) -> np.ndarray:
        self._record(text, voice, params, seed)
        self.entered.set()
        self.release.wait(timeout=10)
        if self._fail is not None:
            raise self._fail
        return _tone(len(text) * SAMPLES_PER_CHAR)

    def stream(self, text, voice=None, *, params=None, seed=None):
        self._record(text, voice, params, seed)
        wav = _tone(len(text) * SAMPLES_PER_CHAR)
        self.entered.set()
        self.release.wait(timeout=10)
        if self._fail is not None:
            raise self._fail
        for start in range(0, wav.size, FRAME_SAMPLES):
            yield AudioFrame(wav[start : start + FRAME_SAMPLES], SAMPLE_RATE)
        # The real stream always ends with a final frame, empty or not.
        yield AudioFrame(np.zeros(0, dtype=np.float32), SAMPLE_RATE, is_final=True)

    def _record(self, text, voice, params, seed) -> None:
        if voice is not None:
            self.voice(voice)  # the real facade resolves the voice before it generates anything
        self.calls.append({"text": text, "voice": voice, "params": params, "seed": seed})


def _tone(samples: int) -> np.ndarray:
    """A quiet 220 Hz sine, so a decoded wav is checkable as a real signal and not silence."""
    t = np.arange(samples, dtype=np.float32) / SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


@pytest.fixture
def tts() -> StubTTS:
    return StubTTS()


@pytest.fixture
def client(tts: StubTTS):
    with TestClient(create_app(tts=tts)) as client:
        yield client


def _pcm_samples(payload: bytes) -> np.ndarray:
    return np.frombuffer(payload, dtype="<i2")


def _sse_events(payload: bytes) -> list[tuple[str, dict]]:
    """Parse a whole ``text/event-stream`` body into (event name, data) pairs."""
    events: list[tuple[str, dict]] = []
    for block in payload.decode("utf-8").split("\n\n"):
        if not block.strip():
            continue
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: ") :]
            elif line.startswith("data: "):
                data += line[len("data: ") :]
        events.append((name, json.loads(data)))
    return events


# ------------------------------------------------------------------------------------ inspection


class TestHealth:
    def test_reports_the_loaded_model(self, client):
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["device"] == "cpu"
        assert body["sample_rate"] == SAMPLE_RATE
        assert body["voices"] == 2
        assert body["busy"] is False

    def test_reports_the_version(self, client):
        from kova_tts import __version__

        assert client.get("/health").json()["version"] == __version__


class TestVoices:
    def test_lists_the_voices_the_model_has(self, client, tts):
        body = client.get("/v1/voices").json()
        assert [v["name"] for v in body["voices"]] == tts.voices()
        assert {v["kind"] for v in body["voices"]} == {"lora"}

    def test_reports_the_adapter_directory(self, client, monkeypatch, tmp_path):
        monkeypatch.setenv("KOVA_LORA_DIR", str(tmp_path))
        assert client.get("/v1/voices").json()["lora_dir"] == str(tmp_path)

    def test_no_adapter_directory_is_null_rather_than_missing(self, client, monkeypatch):
        monkeypatch.setenv("KOVA_DISABLE_DOTENV", "1")
        monkeypatch.delenv("KOVA_LORA_DIR", raising=False)
        # Forget any .env already read, or the developer's own adapter directory answers here.
        paths.reset_dotenv_cache()
        assert client.get("/v1/voices").json()["lora_dir"] is None

    def test_a_broken_adapter_directory_does_not_fail_the_listing(self, client, monkeypatch):
        monkeypatch.setenv("KOVA_LORA_DIR", "/does/not/exist")
        body = client.get("/v1/voices")
        assert body.status_code == 200
        assert body.json()["lora_dir"] is None


# -------------------------------------------------------------------------------- POST /v1/tts


class TestSynthesis:
    def test_returns_a_wav_of_the_expected_duration(self, client):
        text = "Hello there, this is a test."
        response = client.post("/v1/tts", json={"text": text})

        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"

        audio, rate = sf.read(io.BytesIO(response.content), dtype="float32")
        assert rate == SAMPLE_RATE
        assert audio.ndim == 1
        assert audio.size == len(text) * SAMPLES_PER_CHAR
        # A real signal, not a buffer of zeros.
        assert float(np.max(np.abs(audio))) > 0.1

    def test_reports_the_duration_in_a_header(self, client):
        text = "Hello there, this is a test."
        response = client.post("/v1/tts", json={"text": text})
        expected = len(text) * SAMPLES_PER_CHAR / SAMPLE_RATE
        assert response.headers["x-sample-rate"] == str(SAMPLE_RATE)
        assert float(response.headers["x-duration-seconds"]) == pytest.approx(expected, abs=1e-3)

    def test_raw_pcm_when_asked_for(self, client):
        text = "Raw samples, please."
        response = client.post("/v1/tts", json={"text": text, "response_format": "pcm"})

        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == len(text) * SAMPLES_PER_CHAR * 2
        assert np.abs(_pcm_samples(response.content)).max() > 1000

    def test_pcm_is_the_same_audio_as_the_wav(self, client):
        text = "The two formats must agree."
        wav = client.post("/v1/tts", json={"text": text}).content
        pcm = client.post("/v1/tts", json={"text": text, "response_format": "pcm"}).content

        inside, rate = sf.read(io.BytesIO(wav), dtype="float32")
        assert rate == SAMPLE_RATE
        # Compared as floats, not bytes: soundfile scales to 16-bit against 32768 and rounds,
        # kova_tts.audio.to_pcm_bytes scales against 32767 and truncates, so the two encodings
        # of the same waveform differ in the bottom bit or two and never byte for byte.
        recovered = _pcm_samples(pcm).astype(np.float32) / 32767.0
        assert inside.size == recovered.size
        assert np.allclose(inside, recovered, atol=1e-4)

    def test_the_voice_and_seed_reach_the_model(self, client, tts):
        client.post("/v1/tts", json={"text": "Speak.", "voice": "some-voice", "seed": 7})
        assert tts.calls[-1]["voice"] == "some-voice"
        assert tts.calls[-1]["seed"] == 7

    def test_no_overrides_leaves_the_preset_to_the_model(self, client, tts):
        client.post("/v1/tts", json={"text": "Speak."})
        assert tts.calls[-1]["params"] is None

    def test_overrides_are_applied_to_the_tts_preset(self, client, tts):
        client.post("/v1/tts", json={"text": "Speak.", "sampling": {"temperature": 0.5}})
        params = tts.calls[-1]["params"]
        assert params.temperature == 0.5
        assert params.top_k == TTS_SAMPLING.top_k
        assert params.repetition_penalty == TTS_SAMPLING.repetition_penalty


class TestSynthesisValidation:
    def test_empty_text_is_refused_with_a_reason(self, client):
        body = client.post("/v1/tts", json={"text": "   "})
        assert body.status_code == 422
        assert body.json()["error"] == "invalid_request"
        assert "text is empty" in body.json()["message"]

    def test_missing_text_names_the_field(self, client):
        body = client.post("/v1/tts", json={})
        assert body.status_code == 422
        assert "text" in body.json()["message"]

    def test_text_over_the_limit_is_refused(self, client):
        body = client.post("/v1/tts", json={"text": "a" * (wire.MAX_TEXT_CHARS + 1)})
        assert body.status_code == 422
        assert "text" in body.json()["message"]

    def test_an_unknown_field_is_reported_not_ignored(self, client):
        body = client.post("/v1/tts", json={"text": "Hello.", "temprature": 0.4})
        assert body.status_code == 422
        assert "temprature" in body.json()["message"]

    def test_an_impossible_sampling_value_explains_itself(self, client):
        body = client.post("/v1/tts", json={"text": "Hello.", "sampling": {"temperature": 0}})
        assert body.status_code == 422
        assert body.json()["error"] == "invalid_request"
        assert "temperature" in body.json()["message"]

    def test_an_unknown_response_format_is_refused(self, client):
        body = client.post("/v1/tts", json={"text": "Hello.", "response_format": "mp3"})
        assert body.status_code == 422
        assert "response_format" in body.json()["message"]

    def test_an_unknown_voice_is_a_404_naming_what_is_available(self, client):
        body = client.post("/v1/tts", json={"text": "Hello.", "voice": "nope"})
        assert body.status_code == 404
        assert body.json()["error"] == "not_found"
        assert "some-voice" in body.json()["message"]

    def test_an_unknown_voice_is_a_404_on_the_stream_too(self, client):
        """Checked before the stream opens, so it is a status code and not an error event."""
        body = client.post("/v1/tts/stream", json={"text": "Hello.", "voice": "nope"})
        assert body.status_code == 404
        assert body.headers["content-type"].startswith("application/json")


# ------------------------------------------------------------------------- POST /v1/tts/stream


class TestServerSentEvents:
    def test_chunks_then_a_done_event(self, client):
        text = "This sentence is long enough to span several frames of audio."
        response = client.post("/v1/tts/stream", json={"text": text})

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = _sse_events(response.content)
        names = [name for name, _ in events]
        assert names[-1] == "done"
        assert set(names[:-1]) == {"chunk"}
        assert len(names) > 2  # several frames, not one lump

    def test_chunks_are_numbered_from_zero(self, client):
        events = _sse_events(
            client.post("/v1/tts/stream", json={"text": "A sentence with several frames."}).content
        )
        chunks = [data for name, data in events if name == "chunk"]
        assert [c["index"] for c in chunks] == list(range(len(chunks)))
        assert {c["sample_rate"] for c in chunks} == {SAMPLE_RATE}

    def test_the_concatenated_chunks_are_the_whole_utterance(self, client):
        text = "The stream and the file have to agree, sample for sample."
        events = _sse_events(client.post("/v1/tts/stream", json={"text": text}).content)
        streamed = b"".join(
            base64.b64decode(data["audio"]) for name, data in events if name == "chunk"
        )
        whole = client.post("/v1/tts", json={"text": text, "response_format": "pcm"}).content
        assert streamed == whole

    def test_the_done_event_totals_the_stream(self, client):
        text = "Count everything that went out."
        events = _sse_events(client.post("/v1/tts/stream", json={"text": text}).content)
        chunks = [data for name, data in events if name == "chunk"]
        done = events[-1][1]
        assert done["chunks"] == len(chunks)
        assert done["samples"] == len(text) * SAMPLES_PER_CHAR
        assert done["duration_seconds"] == pytest.approx(done["samples"] / SAMPLE_RATE, abs=1e-3)

    def test_empty_final_frames_are_not_sent_as_chunks(self, client):
        events = _sse_events(client.post("/v1/tts/stream", json={"text": "Short."}).content)
        for name, data in events:
            if name == "chunk":
                assert base64.b64decode(data["audio"])

    def test_a_failure_mid_stream_arrives_as_a_terminal_error_event(self, client, tts):
        tts._fail = RuntimeError("the codec fell over")
        events = _sse_events(client.post("/v1/tts/stream", json={"text": "Hello."}).content)
        assert len(events) == 1
        name, data = events[0]
        assert name == "error"
        assert data["error"] == "RuntimeError"
        assert "codec fell over" in data["message"]

    def test_validation_still_happens_before_the_stream_starts(self, client):
        body = client.post("/v1/tts/stream", json={"text": ""})
        assert body.status_code == 422
        assert body.headers["content-type"].startswith("application/json")

    def test_response_format_is_not_a_streaming_option(self, client):
        # PCM is the only thing a stream can carry; asking for wav should say so, not be ignored.
        body = client.post("/v1/tts/stream", json={"text": "Hello.", "response_format": "wav"})
        assert body.status_code == 422
        assert "response_format" in body.json()["message"]


# ------------------------------------------------------------------------ the one-at-a-time rule


def _post_in_background(client, path, payload):
    result: dict = {}
    thread = threading.Thread(
        target=lambda: result.update(response=client.post(path, json=payload)), daemon=True
    )
    thread.start()
    return thread, result


class TestOneGenerationAtATime:
    """A second caller is refused, not queued and not allowed to corrupt the first."""

    @pytest.fixture
    def client(self, tts):
        # busy_timeout=0 so the refusal is immediate and the test does not sleep for it.
        with TestClient(create_app(tts=tts, busy_timeout=0.0)) as client:
            yield client

    def test_a_concurrent_request_is_refused_with_409(self, client, tts):
        tts.release.clear()
        thread, first = _post_in_background(client, "/v1/tts", {"text": "The first request."})
        assert tts.entered.wait(timeout=5)

        second = client.post("/v1/tts", json={"text": "The second request."})
        assert second.status_code == 409
        assert second.json()["error"] == "busy"
        assert "one utterance at a time" in second.json()["message"]

        tts.release.set()
        thread.join(timeout=10)
        assert first["response"].status_code == 200

    def test_health_reports_the_model_as_busy(self, client, tts):
        tts.release.clear()
        thread, _ = _post_in_background(client, "/v1/tts", {"text": "Occupying the model."})
        assert tts.entered.wait(timeout=5)

        assert client.get("/health").json()["busy"] is True

        tts.release.set()
        thread.join(timeout=10)
        assert client.get("/health").json()["busy"] is False

    def test_a_stream_refused_mid_flight_is_a_status_not_an_event(self, client, tts):
        tts.release.clear()
        thread, _ = _post_in_background(client, "/v1/tts", {"text": "Occupying the model."})
        assert tts.entered.wait(timeout=5)

        second = client.post("/v1/tts/stream", json={"text": "Also wants the model."})
        assert second.status_code == 409
        assert second.headers["content-type"].startswith("application/json")

        tts.release.set()
        thread.join(timeout=10)

    def test_the_model_is_free_again_afterwards(self, client, tts):
        assert client.post("/v1/tts", json={"text": "One."}).status_code == 200
        assert client.post("/v1/tts", json={"text": "Two."}).status_code == 200


class TestBusyTimeout:
    def test_a_caller_waits_for_the_configured_window(self, tts):
        """The default is a short wait, so a back-to-back retry succeeds instead of 409-ing."""
        with TestClient(create_app(tts=tts, busy_timeout=5.0)) as client:
            tts.release.clear()
            thread, _ = _post_in_background(client, "/v1/tts", {"text": "First."})
            assert tts.entered.wait(timeout=5)

            def unblock() -> None:
                time.sleep(0.2)
                tts.release.set()

            threading.Thread(target=unblock, daemon=True).start()
            second = client.post("/v1/tts", json={"text": "Second."})
            assert second.status_code == 200
            thread.join(timeout=10)


# ------------------------------------------------------------------------------------- WS /v1/ws


class TestWebSocketSession:
    def test_the_whole_frame_sequence(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            assert "context_started" in ws.receive_json()

            ws.send_json({"send_text": "The first half of a sentence "})
            ws.send_json({"send_text": "and the second half of it."})
            ws.send_json({"flush": True, "flush_id": "one"})

            chunks = []
            while True:
                frame = ws.receive_json()
                if "flush_completed" in frame:
                    assert frame["flush_id"] == "one"
                    break
                assert "audio_chunk" in frame
                chunks.append(frame["audio_chunk"])
            assert chunks

            ws.send_json({"close_context": True, "flush_id": "two"})
            assert ws.receive_json() == {"flush_completed": True, "flush_id": "two"}
            assert ws.receive_json() == {"context_closed": True}

    def test_the_echoed_configuration_fills_in_defaults(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"voice": "some-voice", "seed": 3}})
            started = ws.receive_json()["context_started"]
            assert started["voice"] == "some-voice"
            assert started["seed"] == 3
            assert started["response_format"] == {"encoding": "pcm", "sample_rate": SAMPLE_RATE}
            ws.send_json({"close_context": True, "flush_id": "x"})

    def test_the_configuration_reaches_the_model(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json(
                {
                    "start_context": {
                        "voice": "some-voice",
                        "seed": 11,
                        "sampling": {"top_k": 5},
                    }
                }
            )
            ws.receive_json()
            ws.send_json({"send_text": "Say something."})
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)

        call = tts.calls[-1]
        assert call["voice"] == "some-voice"
        assert call["seed"] == 11
        assert call["params"].top_k == 5

    def test_text_is_buffered_until_a_flush(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": "One. "})
            ws.send_json({"send_text": "Two."})
            ws.send_json({"flush": True, "flush_id": "a"})
            _until_flush(ws, "a")
            ws.send_json({"close_context": True, "flush_id": "b"})
            _drain(ws)

        # One generation for the flush, and none for the empty close: the buffer was drained.
        assert [call["text"] for call in tts.calls] == ["One. Two."]

    def test_a_flush_with_nothing_buffered_is_still_acknowledged(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"flush": True, "flush_id": "empty"})
            assert ws.receive_json() == {"flush_completed": True, "flush_id": "empty"}
            ws.send_json({"close_context": True, "flush_id": "x"})
        assert tts.calls == []

    def test_two_flushes_keep_their_order_and_do_not_interleave(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": "The first flush of text."})
            ws.send_json({"flush": True, "flush_id": "a"})
            ws.send_json({"send_text": "The second flush of text."})
            ws.send_json({"flush": True, "flush_id": "b"})

            order = []
            seen = 0
            while seen < 2:
                frame = ws.receive_json()
                if "flush_completed" in frame:
                    order.append(frame["flush_id"])
                    seen += 1
                else:
                    order.append("audio")
            assert order.index("a") < order.index("b")
            assert order[order.index("a") - 1] == "audio"

            ws.send_json({"close_context": True, "flush_id": "c"})
            _drain(ws)
        assert [call["text"] for call in tts.calls] == [
            "The first flush of text.",
            "The second flush of text.",
        ]

    def test_the_chunks_decode_to_the_same_audio_as_the_sync_endpoint(self, client):
        text = "The socket and the file have to agree, sample for sample."
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": text})
            ws.send_json({"close_context": True, "flush_id": "x"})
            chunks = []
            while True:
                frame = ws.receive_json()
                if "audio_chunk" in frame:
                    chunks.append(base64.b64decode(frame["audio_chunk"]))
                elif "context_closed" in frame:
                    break

        whole = client.post("/v1/tts", json={"text": text, "response_format": "pcm"}).content
        assert b"".join(chunks) == whole


class TestWebSocketErrors:
    def test_text_before_a_context_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"send_text": "Too early."})
            frame = ws.receive_json()
            assert "no active context" in frame["error"]

    def test_a_flush_before_a_context_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"flush": True})
            assert "no active context" in ws.receive_json()["error"]

    def test_the_session_survives_an_out_of_order_frame(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"flush": True})
            ws.receive_json()
            ws.send_json({"start_context": {}})
            assert "context_started" in ws.receive_json()
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)

    def test_starting_twice_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"start_context": {}})
            assert ws.receive_json()["error"] == "context already started"
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)

    def test_an_unknown_frame_names_the_frames_that_exist(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"speak_now": "please"})
            message = ws.receive_json()["error"]
            assert "unknown frame" in message
            for key in wire.INCOMING_KEYS:
                assert key in message

    def test_a_frame_that_is_not_an_object_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json(["send_text", "hello"])
            assert "bad frame" in ws.receive_json()["error"]

    def test_an_unknown_field_inside_a_frame_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"voise": "some-voice"}})
            assert "bad frame" in ws.receive_json()["error"]

    def test_an_unknown_voice_is_reported_at_start(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"voice": "nope"}})
            assert "No voice named 'nope'" in ws.receive_json()["error"]
            # No session was created, so the client can correct itself and carry on.
            ws.send_json({"start_context": {"voice": "some-voice"}})
            assert "context_started" in ws.receive_json()
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)

    def test_an_impossible_sampling_value_is_reported_at_start(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"sampling": {"top_p": 2.0}}})
            assert "top_p" in ws.receive_json()["error"]

    def test_a_sample_rate_this_server_cannot_produce_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"sample_rate": 24000}}})
            message = ws.receive_json()["error"]
            assert "does not resample" in message
            assert str(SAMPLE_RATE) in message

    def test_a_failing_flush_is_an_error_frame_and_the_flush_still_completes(self, client, tts):
        tts._fail = RuntimeError("the codec fell over")
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": "Say something."})
            ws.send_json({"flush": True, "flush_id": "a"})
            failure = ws.receive_json()
            assert "codec fell over" in failure["error"]
            assert failure["flush_id"] == "a"
            assert ws.receive_json() == {"flush_completed": True, "flush_id": "a"}
            ws.send_json({"close_context": True, "flush_id": "b"})
            _drain(ws)

    def test_a_flush_that_collides_with_a_request_is_told_the_model_is_busy(self, tts):
        with TestClient(create_app(tts=tts, busy_timeout=0.0)) as client:
            tts.release.clear()
            thread, _ = _post_in_background(client, "/v1/tts", {"text": "Occupying the model."})
            assert tts.entered.wait(timeout=5)

            with client.websocket_connect("/v1/ws") as ws:
                ws.send_json({"start_context": {}})
                ws.receive_json()
                ws.send_json({"send_text": "Also wants the model."})
                ws.send_json({"flush": True, "flush_id": "a"})
                failure = ws.receive_json()
                assert "one utterance at a time" in failure["error"]
                assert ws.receive_json() == {"flush_completed": True, "flush_id": "a"}

            tts.release.set()
            thread.join(timeout=10)

    def test_frames_never_carry_nulls(self, client):
        """An unset optional is left out entirely, so a client never has to filter nulls."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"flush": True})
            assert "flush_id" not in ws.receive_json()
            ws.send_json({"start_context": {}})
            started = ws.receive_json()["context_started"]
            assert "voice" not in started
            assert "seed" not in started
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)


def _until_flush(ws, flush_id: str) -> None:
    while True:
        frame = ws.receive_json()
        if frame.get("flush_completed") and frame["flush_id"] == flush_id:
            return


def _drain(ws) -> None:
    """Read frames until the session closes."""
    while True:
        frame = ws.receive_json()
        if "context_closed" in frame:
            return


# ---------------------------------------------------------------------------------- unit corners


class TestProtocol:
    def test_overrides_only_replace_what_was_sent(self):
        params = wire.SamplingOverrides(top_p=0.5).apply(CLONE_SAMPLING)
        assert params.top_p == 0.5
        assert params.temperature == CLONE_SAMPLING.temperature

    def test_no_overrides_returns_the_preset_unchanged(self):
        assert wire.SamplingOverrides().apply(TTS_SAMPLING) is TTS_SAMPLING

    def test_parse_incoming_rejects_a_bare_value(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            wire.parse_incoming("start_context")

    def test_parse_incoming_reports_an_empty_object(self):
        with pytest.raises(ValueError, match="no keys at all"):
            wire.parse_incoming({})

    def test_to_wire_drops_unset_optionals(self):
        assert wire.to_wire(wire.Error(error="nope")) == {"error": "nope"}


class TestEngineInternals:
    @pytest.mark.asyncio
    async def test_an_abandoned_stream_closes_the_generator(self):
        """Bailing out of a stream must release the model, not leave it marked busy."""
        closed = []

        def frames():
            try:
                yield AudioFrame(_tone(10), SAMPLE_RATE)
                yield AudioFrame(_tone(10), SAMPLE_RATE)
            finally:
                closed.append(True)

        async with aclosing(aiter_frames(frames())) as stream:
            async for _ in stream:
                break
        assert closed == [True]

    @pytest.mark.asyncio
    async def test_reserve_releases_even_when_the_body_raises(self):
        engine = Engine(StubTTS(), busy_timeout=0.0)
        with pytest.raises(RuntimeError):
            async with engine.reserve():
                raise RuntimeError("boom")
        assert engine.busy is False


class TestCreateApp:
    def test_an_injected_model_is_never_replaced(self, tts):
        app = create_app(tts=tts)
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert app.state.engine.tts is tts

    def test_the_busy_timeout_is_configurable(self, tts):
        app = create_app(tts=tts, busy_timeout=1.5)
        with TestClient(app):
            assert app.state.engine.busy_timeout == 1.5

    def test_the_openapi_schema_builds(self, client):
        schema = client.get("/openapi.json").json()
        assert "/v1/tts" in schema["paths"]
        assert "/v1/tts/stream" in schema["paths"]
