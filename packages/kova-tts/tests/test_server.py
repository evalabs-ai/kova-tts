"""The server, driven end to end against a stub model: no GPU and no weights.

Everything here goes through :class:`fastapi.testclient.TestClient`, so the ASGI stack, the
validators, the SSE framing and the WebSocket handshake are all real; only the model is not.
That is the whole reason :func:`~kova_tts.server.app.create_app` takes a ``tts=`` argument.

The stub has two faces, because the server uses the model two ways. ``generate`` and ``stream``
return a tone whose length is a known function of the text, so a test can assert on the duration
of the audio that comes back rather than merely that some bytes arrived. Underneath them sit a
stand-in LM and a stand-in codec, which is what the WebSocket session drives: the LM hands out
code numbers that count up over the life of the stub, and the codec decodes code ``c`` to the
tone that begins at sample ``c * HOP_LENGTH``. A run of generations therefore decodes to one
continuous waveform, which is what makes a discontinuity in a session's audio provably the
session's doing and not the stand-in's.

Two parts of the model itself are borrowed rather than imitated, because the session's contract
with them is what these tests are about: :class:`~kova_tts.engine.tts.KovaTTS` builds the prompts
and decides when a carry will not fit the KV cache, and the stand-in LM answers whatever prompt
it is handed with **the rest of the utterance that prompt describes** -- every character of text
is :data:`CODES_PER_CHAR` codes of speech, so a prompt already carrying codes for some of its own
text is answered with only what is left of it. A session that restarted a generation instead of
continuing it would therefore produce measurably more audio than its text calls for.
"""

from __future__ import annotations

import base64
import io
import json
import re
import threading
import time
from contextlib import aclosing
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from fastapi.testclient import TestClient

from kova_codec.constants import HOP_LENGTH, SAMPLE_RATE, TOKEN_RATE
from kova_tts import audio, paths
from kova_tts.engine.decoder import decode_all
from kova_tts.engine.generator import DEFAULT_MAX_CACHE_LEN
from kova_tts.engine.tts import MAX_CARRY_CODES, MAX_SEGMENT_CHARS, KovaTTS
from kova_tts.engine.types import CLONE_SAMPLING, TTS_SAMPLING, AudioFrame, Voice
from kova_tts.paths import MissingArtifact
from kova_tts.prompt import parse_audio_tokens
from kova_tts.server import protocol as wire
from kova_tts.server import ws as ws_module
from kova_tts.server.app import create_app
from kova_tts.server.engine import Engine, aiter_frames

#: Codes the stub LM emits per character of text: how fast it speaks. Inside the range the model
#: measures for real prose, so that a full chunk's codes fit a carry exactly as
#: :data:`~kova_tts.engine.tts.MAX_CARRY_CODES` promises they will, and above the session's own
#: :data:`~kova_tts.server.ws.CODES_PER_CHAR` estimate, which is deliberately below any real rate
#: of speech. Small enough that a test finishes instantly, large enough that a chunk spans
#: several of the decoder's ~390 ms windows.
CODES_PER_CHAR = 5

#: Samples of tone per character, which is the same number seen from the other end: one code is
#: ``HOP_LENGTH`` samples, so the two faces of the stub agree on how long a text sounds.
SAMPLES_PER_CHAR = CODES_PER_CHAR * HOP_LENGTH

#: Whitespace, which the stub does not speak. Leaving it out is what makes the rate additive
#: across a join: the prompt layer puts a space between the text a prompt carries and the text
#: that follows it, and a space nobody uttered is not speech that has to be accounted for.
_SPACE = re.compile(r"\s+")


def speech_codes(text: str) -> int:
    """Codes the stub renders `text` as."""
    return CODES_PER_CHAR * len(_SPACE.sub("", text))


def speech_samples(text: str) -> int:
    """Samples the stub renders `text` as, from either of its two faces."""
    return speech_codes(text) * HOP_LENGTH


#: What the stub cuts its output into, matching the real decoder's steady-state window.
FRAME_SAMPLES = 12400  # 31 codes at 80 Hz -> 387.5 ms at 32 kHz

#: The tone both faces of the stub produce. Quiet, so a decoded wav is checkable as a real
#: signal and not silence, and low enough that consecutive samples differ by very little --
#: which is what makes a seam stand out against the signal's own step distribution. The
#: frequency is deliberately incommensurate with :data:`SAMPLES_PER_CHAR`, so a chunk or flush
#: boundary lands at an arbitrary phase rather than always at a zero crossing, where a step
#: would hide however large it was.
TONE_HZ = 193.0
TONE_AMPLITUDE = 0.2

#: The head of a decode that begins without recurrent state, and the gain it comes out at. A
#: decoder handed no history produces its first moments from silence rather than from the audio
#: before them, and the stand-in reproduces that: it is what makes a session that fails to hand
#: its next decoder the codes it left off at show up as a step rather than as clean audio.
COLD_START_SAMPLES = 800
COLD_START_GAIN = 0.5


class StubCodec:
    """Decodes code `c` to the ``HOP_LENGTH`` samples of a tone beginning at sample ``c * HOP``.

    Position-aware, so consecutive codes decode to one continuous waveform however they are cut
    into windows, spread over chunks, or split across flushes -- and cold at the head of a decode
    that was given no state to resume from.
    """

    device = torch.device("cpu")
    sample_rate = SAMPLE_RATE

    @staticmethod
    def _tone(codes: torch.Tensor) -> torch.Tensor:
        # The phase is accumulated in double and only then narrowed: several seconds in, a
        # float32 argument to sine has drifted far enough to show up against the 16-bit
        # quantisation the tests compare through.
        offsets = (codes[..., None] * HOP_LENGTH + torch.arange(HOP_LENGTH)).double()
        tone = TONE_AMPLITUDE * torch.sin(2 * np.pi * TONE_HZ * offsets / SAMPLE_RATE)
        return tone.flatten(-2).float()

    @staticmethod
    def _cold(wav: torch.Tensor) -> torch.Tensor:
        chilled = wav.clone()
        chilled[..., :COLD_START_SAMPLES] *= COLD_START_GAIN
        return chilled

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._cold(self._tone(codes))

    def decode_with_lstm(self, codes, state=None, return_lstm_state=None, conv_padding=None):
        trimmed = codes[:, conv_padding:-conv_padding] if conv_padding else codes
        wav = self._tone(trimmed)
        if state is None:
            wav = self._cold(wav)
        return wav, (None if return_lstm_state is None else torch.zeros(1))


#: Everything the tokenizer would see as one token on its own, which is every tag the prompt
#: layer writes -- the markers around the text and one per audio code.
_TAG = re.compile(r"<\|[^|]*\|>")


class StubGenerator:
    """Stands in for the LM: tokenizes a prompt, and continues the utterance it describes.

    One id per character of text and one per tag, so a prompt costs the KV cache roughly what a
    real one of its length would and the rule that drops an oversized carry is exercised for
    real. What comes back is what is *left* of the utterance: every character of text is
    :data:`CODES_PER_CHAR` codes of speech, and the codes already in the prompt -- an earlier
    burst of this same chunk, the chunk before it, a reference clip in front of both -- are
    speech that has already been produced and are not produced again.

    Codes count up over the life of the stub instead of restarting per generation, so what the
    codec makes of a whole session is one continuous tone.
    """

    device = "cpu"

    def __init__(self, tts: StubTTS | None = None) -> None:
        self.tts = tts
        #: How many codes have been handed out, and therefore the next code's value.
        self.emitted = 0
        #: What the last encoded prompt still has to say. Read by the generation that follows
        #: it, exactly as a real model reads the prompt it was just given.
        self.owed = 0
        self.max_cache_len = DEFAULT_MAX_CACHE_LEN

    def encode(self, prompt: str) -> list[int]:
        text = _TAG.sub("", prompt)
        codes = parse_audio_tokens(prompt)
        self.owed = max(speech_codes(text) - len(codes), 0)
        return [0] * (len(text) + len(codes))

    def stream_ids(self, ids, params=None):
        if self.tts is not None and self.tts._fail is not None:
            raise self.tts._fail
        for _ in range(self.owed):
            code = self.emitted
            # Counted before it is handed over, so that a consumer which stops early -- a burst
            # that has reached its cap -- leaves `emitted` naming exactly the codes it received.
            self.emitted += 1
            yield code


class StubEncoder:
    """The encoding half of the codec, as voice cloning uses it.

    One code per :data:`HOP_LENGTH` samples, which is the rate the real codec works out to, and
    nothing at all for a clip with no signal in it -- silence is the one input whose encoding a
    caller has to be told about rather than handed.
    """

    device = torch.device("cpu")
    sample_rate = SAMPLE_RATE

    def __init__(self) -> None:
        #: Clips encoded so far, so a test can show that a pre-encoded reference loads nothing.
        self.encoded = 0

    def encode(self, wav) -> torch.Tensor:
        self.encoded += 1
        samples = np.asarray(wav, dtype=np.float32)
        if float(np.max(np.abs(samples))) < 1e-3:
            return torch.zeros(0, dtype=torch.long)
        return torch.arange(samples.size // HOP_LENGTH, dtype=torch.long) % 8192


class StubTTS:
    """A :class:`~kova_tts.engine.tts.KovaTTS` that makes tones instead of speech.

    Records the arguments of every call, so a test can prove that a request's voice, sampling
    overrides and seed reached the model rather than being dropped on the way. ``calls`` covers
    the whole-text methods; ``prompts`` covers the session path, one entry per burst, including
    what it was given to continue from and the prompt string that was really encoded.

    The prompt builder, the fit rule that drops a carry too big for the cache, the sampling
    presets, the preroll and cloning itself are the model's own methods bound to this stub: what
    the session sends the LM is then the layout the checkpoint was trained on, not a second
    implementation of it that could agree with the session while both were wrong.
    """

    _prompt_builder = KovaTTS._prompt
    _sampling = staticmethod(KovaTTS._sampling)
    _preroll = KovaTTS._preroll
    clone = KovaTTS.clone

    def __init__(self, *, voices: list[str] | None = None, fail: Exception | None = None) -> None:
        self.generator = StubGenerator(self)
        self.codec = StubCodec()
        self.encoding_codec = StubEncoder()
        self.sample_rate = SAMPLE_RATE
        self.lora_root = None
        self.transcriber = None
        self.clone_preroll = ws_module.PREROLL_CODES
        self._voices = list(voices or ["some-voice", "another-voice"])
        self._fail = fail
        self.calls: list[dict] = []
        self.prompts: list[dict] = []
        self._built = ""
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
        return _tone(speech_samples(text))

    def stream(self, text, voice=None, *, params=None, seed=None):
        self._record(text, voice, params, seed)
        wav = _tone(speech_samples(text))
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

    # -- the surface a session drives ----------------------------------------------------

    def _prepare(self, voice):
        """Resolve a voice; an unknown name fails here, before anything is generated.

        A :class:`~kova_tts.engine.types.Voice` passes through untouched, which is how a cloned
        session hands back the voice it built at ``start_context``.
        """
        if voice is None or isinstance(voice, Voice):
            return voice
        return self.voice(voice)

    def _prompt(self, text, voice, prior=None) -> str:
        """The model's own prompt, kept so a test can assert on what the LM was really sent."""
        prompt = self._prompt_builder(text, voice, prior)
        self._built = prompt
        return prompt

    def _prompt_ids(self, chunk, voice, carry, params) -> list[int]:
        """The model's own tokens for one burst, and a record of what it was continuing."""
        ids = KovaTTS._prompt_ids(self, chunk, voice, carry, params)
        self.prompts.append(
            {
                "chunk": chunk,
                "carry": carry,
                "voice": voice,
                "params": params,
                # The prompt built last is the one these ids came from: a carry that would not
                # have fitted the cache is dropped by building the prompt a second time without.
                "prompt": self._built,
                "ids": ids,
            }
        )
        return ids


def _tone(samples: int) -> np.ndarray:
    """A quiet sine, so a decoded wav is checkable as a real signal and not silence.

    Cold at the head, exactly as :class:`StubCodec` is: this is the same audio seen from the
    other end, and the two have to agree sample for sample.
    """
    t = np.arange(samples, dtype=np.float64) / SAMPLE_RATE
    wav = (TONE_AMPLITUDE * np.sin(2 * np.pi * TONE_HZ * t)).astype(np.float32)
    wav[:COLD_START_SAMPLES] *= COLD_START_GAIN
    return wav


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
        assert audio.size == speech_samples(text)
        # A real signal, not a buffer of zeros.
        assert float(np.max(np.abs(audio))) > 0.1

    def test_reports_the_duration_in_a_header(self, client):
        text = "Hello there, this is a test."
        response = client.post("/v1/tts", json={"text": text})
        expected = speech_samples(text) / SAMPLE_RATE
        assert response.headers["x-sample-rate"] == str(SAMPLE_RATE)
        assert float(response.headers["x-duration-seconds"]) == pytest.approx(expected, abs=1e-3)

    def test_raw_pcm_when_asked_for(self, client):
        text = "Raw samples, please."
        response = client.post("/v1/tts", json={"text": text, "response_format": "pcm"})

        assert response.headers["content-type"] == "audio/pcm"
        assert len(response.content) == speech_samples(text) * 2
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
        assert done["samples"] == speech_samples(text)
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

    def test_the_model_is_free_again_afterwards(self, client):
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

#: One sentence per flush, for the tests that need a session with boundaries inside it. Each is
#: its own chunk: long enough that the splitter does not merge it backwards, short enough that it
#: is never broken up.
FLUSH_TEXTS = (
    "The first sentence of a longer answer.",
    "Here is the second one, arriving later.",
    "And a third to finish the thought.",
)

#: Enough text to make the session speak without being asked, in one send_text. Long enough that
#: the burst it triggers spans several decoder windows, so a test can wait for audio rather than
#: for a timeout.
UNPROMPTED_TEXT = "The session speaks this much without being asked for it first."

#: What a reference clip says, for the tests that clone.
REFERENCE_TEXT = "This is the reference recording, spoken plainly."

#: How long that clip is: exactly what the stub would take to say its transcript. A reference is
#: speech the model has already produced, and making the stand-in agree with that is what keeps a
#: cloned session's chunks the same length as any other session's.
REFERENCE_SECONDS = speech_codes(REFERENCE_TEXT) / TOKEN_RATE

#: The first bytes of each container a session can send, so a stream can be identified without
#: decoding it. The mp3 entry is a frame sync word (11 set bits); mp3 has no magic number.
MAGIC = {
    "wav": (b"RIFF",),
    "flac": (b"fLaC",),
    "mp3": (b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3", b"ID3"),
}


def run_session(client, texts, **config) -> tuple[dict, bytes, list[int]]:
    """Drive a session that flushes once per entry of `texts` and closes on the last.

    Returns the echoed configuration, every audio byte the session sent, and the offset into
    those bytes at which each flush was acknowledged -- which is where a discontinuity would be
    if a flush boundary left one.
    """
    payload = bytearray()
    boundaries: list[int] = []
    with client.websocket_connect("/v1/ws") as ws:
        ws.send_json({"start_context": config})
        opening = ws.receive_json()
        assert "context_started" in opening, opening
        for index, text in enumerate(texts):
            last = index == len(texts) - 1
            ws.send_json({"send_text": text})
            ws.send_json({"close_context" if last else "flush": True, "flush_id": str(index)})
            while True:
                frame = ws.receive_json()
                if "audio_chunk" in frame:
                    payload += base64.b64decode(frame["audio_chunk"])
                elif "error" in frame:
                    raise AssertionError(frame["error"])
                else:
                    assert frame == {"flush_completed": True, "flush_id": str(index)}
                    boundaries.append(len(payload))
                    break
        assert ws.receive_json() == {"context_closed": True}
    return opening["context_started"], bytes(payload), boundaries


def read_stream(payload: bytes, encoding: str = "pcm") -> np.ndarray:
    """A streamed container's samples.

    ``pcm`` is the samples themselves. Everything else states a length in a header written
    before the length exists, so it is read block by block to the end rather than trusted.
    """
    if encoding == "pcm":
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    blocks = []
    with sf.SoundFile(io.BytesIO(payload)) as handle:
        while len(block := handle.read(8192, dtype="float32")):
            blocks.append(block)
    return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)


def assert_joins_are_invisible(samples: np.ndarray, boundaries: list[int]) -> None:
    """No flush boundary may leave a step the signal itself would not have produced.

    `boundaries` are byte offsets, as :func:`run_session` reports them, in a stream of 16-bit
    samples. Measured the way the resampler's own tests measure it: against the distribution of
    every step in the waveform, a seam is an outlier, and there should be none.
    """
    joins = np.array([offset // 2 for offset in boundaries[:-1]])
    joins = joins[(joins > 0) & (joins < samples.size)]
    assert joins.size >= 2, "this session no longer has enough flush boundaries to check"
    steps = np.abs(np.diff(samples))
    assert steps[joins - 1].max() <= np.percentile(steps, 99.9)


def reference_audio(
    seconds: float = REFERENCE_SECONDS, amplitude: float = 0.4, rate: int = SAMPLE_RATE
) -> str:
    """A base64 wav for a session to clone from.

    Synthesized on the spot rather than read from a file: nothing in this repository ships a
    recording, and a tone exercises every step of the path a real clip takes -- the base64, the
    container, the rate conversion, the loudness normalisation and the encoder.
    """
    t = np.arange(int(seconds * rate), dtype=np.float64) / rate
    wav = (amplitude * np.sin(2 * np.pi * 190.0 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wav, rate, format="WAV", subtype="PCM_16")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def reference_codes(seconds: float = REFERENCE_SECONDS) -> list[int]:
    """The same clip, as a client that encoded it once would send it: 80 codes a second."""
    return [code % 8192 for code in range(int(seconds * TOKEN_RATE))]


def start_cloned(ws, *, codes: bool = True, **overrides) -> dict:
    """Open a cloned session and return the configuration it echoed back."""
    reference = {"transcript": REFERENCE_TEXT}
    reference.update({"codes": reference_codes()} if codes else {"audio": reference_audio()})
    ws.send_json({"start_context": {"reference": reference, **overrides}})
    opening = ws.receive_json()
    assert "context_started" in opening, opening
    return opening["context_started"]


def refuse_start(client, config) -> str:
    """Send one ``start_context`` that should be refused, and return the message it came back
    with.

    The connection is used again afterwards, because a refusal that leaves no session behind is
    half of what makes it recoverable: the client corrects the frame and sends it again.
    """
    with client.websocket_connect("/v1/ws") as ws:
        ws.send_json({"start_context": config})
        frame = ws.receive_json()
        assert "error" in frame, frame
        ws.send_json({"start_context": {}})
        assert "context_started" in ws.receive_json()
        ws.send_json({"close_context": True, "flush_id": "x"})
        _drain(ws)
    return frame["error"]


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

        prompt = tts.prompts[-1]
        assert prompt["voice"].name == "some-voice"
        assert prompt["params"].seed == 11
        assert prompt["params"].top_k == 5

    def test_text_below_the_threshold_is_buffered_until_a_flush(self, client, tts):
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
        assert [prompt["chunk"] for prompt in tts.prompts] == ["One. Two."]

    def test_a_flush_with_nothing_buffered_is_still_acknowledged(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"flush": True, "flush_id": "empty"})
            assert ws.receive_json() == {"flush_completed": True, "flush_id": "empty"}
            ws.send_json({"close_context": True, "flush_id": "x"})
        assert tts.prompts == []

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
        assert [prompt["chunk"] for prompt in tts.prompts] == [
            "The first flush of text.",
            "The first flush of text. The second flush of text.",
        ]

    def test_the_chunks_decode_to_the_same_audio_as_the_sync_endpoint(self, client):
        text = "The socket and the file have to agree, sample for sample."
        _, payload, _ = run_session(client, [text])

        whole = client.post("/v1/tts", json={"text": text, "response_format": "pcm"}).content
        streamed = read_stream(payload)
        assert streamed.size == len(whole) // 2
        # Compared as floats, not bytes: the two faces of the stub compute the same tone from
        # opposite ends -- sample index against code number -- and agree to well inside the
        # bottom bit of the 16-bit encoding.
        assert np.allclose(streamed, read_stream(whole), atol=1e-4)


class TestWebSocketFormats:
    """``response_format`` is honoured, and what it settled on comes back in the echo."""

    @pytest.mark.parametrize("encoding", audio.STREAMING_FORMATS)
    def test_every_streamable_container_arrives_as_itself(self, client, encoding):
        started, payload, _ = run_session(
            client, FLUSH_TEXTS, response_format={"encoding": encoding}
        )
        assert started["response_format"]["encoding"] == encoding
        assert payload
        prefixes = MAGIC.get(encoding)
        assert prefixes is None or payload.startswith(prefixes)

    def test_the_container_is_opened_once_for_the_whole_session(self, client):
        """Three flushes, one file: a header per flush would be three files spliced together."""
        _, payload, _ = run_session(client, FLUSH_TEXTS, response_format={"encoding": "wav"})
        assert payload.count(b"RIFF") == 1
        samples = read_stream(payload, "wav")
        assert samples.size == sum(speech_samples(text) for text in FLUSH_TEXTS)

    def test_a_container_that_cannot_keep_up_is_refused_at_start(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"encoding": "opus"}}})
            message = ws.receive_json()["error"]
            assert "opus" in message
            for name in audio.STREAMING_FORMATS:
                assert name in message

    def test_an_unknown_container_names_what_this_install_can_stream(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"encoding": "aac"}}})
            message = ws.receive_json()["error"]
            assert "aac" in message
            for name in audio.STREAMING_FORMATS:
                assert name in message

    def test_a_refused_format_leaves_the_connection_usable(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"encoding": "aac"}}})
            assert "error" in ws.receive_json()
            ws.send_json({"start_context": {"response_format": {"encoding": "wav"}}})
            assert "context_started" in ws.receive_json()
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)

    @pytest.mark.parametrize("rate", [16_000, 24_000, 8_000])
    def test_a_rate_other_than_the_codec_s_is_delivered(self, client, rate):
        _, payload, _ = run_session(client, [FLUSH_TEXTS[0]], response_format={"sample_rate": rate})
        native = speech_samples(FLUSH_TEXTS[0])
        assert read_stream(payload).size == -(-native * rate // SAMPLE_RATE)

    def test_a_container_states_the_rate_it_was_asked_for(self, client):
        started, payload, _ = run_session(
            client, [FLUSH_TEXTS[0]], response_format={"encoding": "wav", "sample_rate": 16_000}
        )
        assert started["response_format"] == {"encoding": "wav", "sample_rate": 16_000}
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            assert handle.samplerate == 16_000

    def test_the_echo_reports_a_rate_the_container_had_to_snap(self, client):
        """mp3 is defined for a fixed set of rates; the echo says which one it landed on."""
        if "mp3" not in audio.STREAMING_FORMATS:
            pytest.skip("this libsndfile cannot write mp3")
        started, _, _ = run_session(
            client, [FLUSH_TEXTS[0]], response_format={"encoding": "mp3", "sample_rate": 20_000}
        )
        assert started["response_format"]["sample_rate"] == 22_050

    def test_a_rate_outside_the_bounds_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"sample_rate": 4000}}})
            assert "bad frame" in ws.receive_json()["error"]


class TestWebSocketContinuity:
    """A session is one utterance, however the client chops it up."""

    def test_each_turn_continues_the_one_before_it(self, client, tts):
        """A flush ends a turn, not the utterance: the next one goes on from where it stopped."""
        run_session(client, FLUSH_TEXTS)
        chunks = [prompt["chunk"] for prompt in tts.prompts]
        assert chunks == [" ".join(FLUSH_TEXTS[: n + 1]) for n in range(len(FLUSH_TEXTS))]

        carried = [prompt["carry"] for prompt in tts.prompts]
        assert carried[0] is None, "the first turn of a session has nothing to continue"
        for chunk, carry in zip(chunks[:-1], carried[1:], strict=True):
            # Every code produced for the chunk so far leads the continuation, and the text they
            # were produced for is the text the prompt states again in front of the new words.
            assert carry.text == ""
            assert carry.codes == tuple(range(speech_codes(chunk)))

    def test_the_whole_session_is_one_decode(self, client, tts):
        """Concatenated, every flush's audio is what decoding the session's codes in one pass
        gives: nothing dropped, nothing sent twice, nothing restarted at a boundary."""
        _, payload, _ = run_session(client, FLUSH_TEXTS)
        whole = decode_all(tts.codec, list(range(tts.generator.emitted)))
        streamed = read_stream(payload)
        assert streamed.size == whole.size
        assert np.allclose(streamed, whole, atol=1e-4)

    @pytest.mark.parametrize("rate", [16_000, 8_000])
    def test_a_converted_session_is_the_whole_signal_converted(self, client, tts, rate):
        """The filter state has to cross flush boundaries as well as chunk ones: converting each
        flush on its own gives the right length and a step at every join."""
        _, payload, _ = run_session(client, FLUSH_TEXTS, response_format={"sample_rate": rate})
        codes = list(range(tts.generator.emitted))
        whole = audio.resample(decode_all(tts.codec, codes), SAMPLE_RATE, rate)
        streamed = read_stream(payload)
        assert streamed.size == whole.size
        assert np.allclose(streamed, whole, atol=1e-4)

    @pytest.mark.parametrize("rate", [SAMPLE_RATE, 16_000, 8_000])
    def test_the_flush_joins_carry_no_step(self, client, rate):
        _, payload, boundaries = run_session(
            client, FLUSH_TEXTS, response_format={"sample_rate": rate}
        )
        assert_joins_are_invisible(read_stream(payload), boundaries)

    def test_flushing_word_by_word_is_as_continuous_as_flushing_once(self, client):
        """The case a client that flushes on a timer produces: it cannot see the sentence end
        until the punctuation arrives, so it hands over a word at a time."""
        words = [f"{word} " for word in " ".join(FLUSH_TEXTS).split()]
        _, payload, boundaries = run_session(client, words, response_format={"sample_rate": 16_000})
        assert len(boundaries) == len(words)
        assert_joins_are_invisible(read_stream(payload), boundaries)

    def test_word_by_word_flushes_decode_as_one_signal(self, client, tts):
        words = [f"{word} " for word in " ".join(FLUSH_TEXTS).split()]
        _, payload, _ = run_session(client, words)
        whole = decode_all(tts.codec, list(range(tts.generator.emitted)))
        streamed = read_stream(payload)
        assert streamed.size == whole.size
        assert np.allclose(streamed, whole, atol=1e-4)

    def test_the_delivery_of_the_text_does_not_change_how_much_speech_it_becomes(self, client):
        """The property that makes the flush strategy the client's business and nobody else's.

        The same paragraph, handed over three ways -- all at once, a sentence at a time, a word
        at a time -- comes back as the same amount of speech, because a burst extends the
        utterance in progress rather than beginning one that has to be led into and led out of.
        """
        words = [f"{word} " for word in " ".join(FLUSH_TEXTS).split()]
        one = read_stream(run_session(client, [" ".join(FLUSH_TEXTS)])[1]).size
        sentences = read_stream(run_session(client, FLUSH_TEXTS)[1]).size
        per_word = read_stream(run_session(client, words)[1]).size
        assert one == sentences == per_word


def speak_unprompted(client) -> None:
    """Drive a session that bursts on :data:`UNPROMPTED_TEXT` alone, then closes.

    The one ``send_text`` is past the threshold, so audio comes back before any flush. Asserting
    that here makes it a precondition of the tests below rather than something each rediscovers.
    """
    with client.websocket_connect("/v1/ws") as ws:
        ws.send_json({"start_context": {}})
        ws.receive_json()
        ws.send_json({"send_text": UNPROMPTED_TEXT})
        assert "audio_chunk" in ws.receive_json()
        ws.send_json({"close_context": True, "flush_id": "x"})
        _drain(ws)


class TestWebSocketScheduling:
    """Generation follows the text, not the flushes: how far it goes, and when it stops."""

    def test_the_session_speaks_before_it_is_asked_to(self, client, tts):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": UNPROMPTED_TEXT})
            assert "audio_chunk" in ws.receive_json()
            ws.send_json({"close_context": True, "flush_id": "x"})
            _drain(ws)
        assert tts.prompts, "no generation ran until the session was flushed"

    def test_a_handful_of_characters_waits_for_a_flush(self, client, tts):
        """Below the threshold there is too little to render well, so nothing is rendered."""
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": "Too little."})
            ws.send_json({"flush": True, "flush_id": "a"})
            frame = ws.receive_json()
            # The first frame back is audio for the flush, and there was none before it.
            assert "audio_chunk" in frame
            assert len(tts.prompts) == 1
            _until_flush(ws, "a")
            ws.send_json({"close_context": True, "flush_id": "b"})
            _drain(ws)

    def test_an_unfinished_burst_stops_at_the_text_it_was_given(self, client, tts):
        """The cap, in the two places it shows: the tokens asked for, and the codes kept."""
        speak_unprompted(client)

        target = ws_module.CODES_PER_CHAR * len(UNPROMPTED_TEXT.strip())
        assert tts.prompts[0]["params"].max_tokens == target
        # The stub speaks faster than the estimate, so the burst was stopped by the cap and not
        # by the model running out of things to say.
        assert speech_codes(UNPROMPTED_TEXT) > target
        assert len(tts.prompts[1]["carry"].codes) == target

    def test_a_flush_lets_the_model_finish(self, client, tts):
        """Nothing caps the last burst of a turn, or a turn could end mid-word."""
        speak_unprompted(client)
        assert tts.prompts[-1]["params"].max_tokens == TTS_SAMPLING.max_tokens

    def test_a_second_burst_continues_the_first_rather_than_repeating_it(self, client, tts):
        """The whole design in one assertion: the codes of the burst before it lead the prompt,
        and what comes back is the rest of the utterance rather than another copy of it."""
        payload = bytearray()
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": UNPROMPTED_TEXT})
            frame = ws.receive_json()
            assert "audio_chunk" in frame
            payload += base64.b64decode(frame["audio_chunk"])
            ws.send_json({"send_text": " And a little more of it."})
            ws.send_json({"close_context": True, "flush_id": "x"})
            payload += _collect(ws)

        first, second = tts.prompts[0], tts.prompts[1]
        assert first["carry"] is None
        assert second["chunk"].startswith(first["chunk"])
        assert second["carry"].codes == tuple(range(ws_module.CODES_PER_CHAR * len(first["chunk"])))
        assert parse_audio_tokens(second["prompt"]) == list(second["carry"].codes)
        # One utterance's worth of speech, not one and a bit.
        assert read_stream(bytes(payload)).size == speech_samples(second["chunk"])

    def test_text_longer_than_a_chunk_is_spoken_as_several(self, client, tts):
        """A chunk cannot grow forever: it is finished, and the next one carries it."""
        paragraph = " ".join(FLUSH_TEXTS * 2)
        assert len(paragraph) > 2 * ws_module.MIN_BUFFER_CHARS
        _, payload, _ = run_session(client, [paragraph])

        chunks = [prompt["chunk"] for prompt in tts.prompts]
        assert len(chunks) > 1
        assert all(len(chunk) <= 200 for chunk in chunks), chunks
        assert " ".join(chunks) == paragraph
        assert read_stream(payload).size == speech_samples(paragraph)

    def test_a_model_that_stops_early_is_taken_up_again_by_the_next_text(
        self, client, tts, monkeypatch
    ):
        """A burst that comes back short is the model saying it has read everything it was given.

        Where the two are level is then here rather than where the estimate put it, and the next
        burst measures its cap from this point -- otherwise a chunk that overshot the estimate
        once would never be spoken from again.
        """
        # An estimate above the stub's own rate of speech, so a capped burst runs out of text
        # rather than out of budget.
        monkeypatch.setattr(ws_module, "CODES_PER_CHAR", 2 * CODES_PER_CHAR)
        payload = bytearray()
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {}})
            ws.receive_json()
            ws.send_json({"send_text": UNPROMPTED_TEXT})
            frame = ws.receive_json()
            assert "audio_chunk" in frame
            payload += base64.b64decode(frame["audio_chunk"])
            ws.send_json({"send_text": f" {FLUSH_TEXTS[0]}"})
            ws.send_json({"close_context": True, "flush_id": "x"})
            payload += _collect(ws)

        assert len(tts.prompts) == 2
        first, second = tts.prompts
        assert first["chunk"] == UNPROMPTED_TEXT
        assert second["chunk"] == f"{UNPROMPTED_TEXT} {FLUSH_TEXTS[0]}"
        assert second["carry"].codes == tuple(range(speech_codes(first["chunk"])))
        assert read_stream(bytes(payload)).size == speech_samples(second["chunk"])


class TestWebSocketCloning:
    """``start_context`` takes a reference clip, and every prompt after it leads with one."""

    def test_a_reference_of_codes_needs_nothing_loaded(self, client, tts):
        started = start_cloned_session(client, tts, codes=True)
        assert started["reference"] == {
            "seconds": REFERENCE_SECONDS,
            "codes": len(reference_codes()),
        }
        assert tts.encoding_codec.encoded == 0, "codes were re-encoded rather than used"

    def test_a_reference_of_audio_is_encoded_here(self, client, tts):
        started = start_cloned_session(client, tts, codes=False)
        assert tts.encoding_codec.encoded == 1
        assert started["reference"] == {
            "seconds": REFERENCE_SECONDS,
            "codes": len(reference_codes()),
        }

    def test_the_echo_does_not_send_the_clip_back(self, client, tts):
        started = start_cloned_session(client, tts, codes=True)
        assert set(started["reference"]) == {"seconds", "codes"}
        assert "transcript" not in json.dumps(started)

    def test_the_prompt_leads_with_the_reference_and_then_the_carry(self, client, tts):
        """The layout is the requirement, so the assertion is on the prompt itself."""
        with client.websocket_connect("/v1/ws") as ws:
            start_cloned(ws, codes=True)
            ws.send_json({"send_text": UNPROMPTED_TEXT})
            assert "audio_chunk" in ws.receive_json()
            ws.send_json({"send_text": f" {FLUSH_TEXTS[0]}"})
            ws.send_json({"flush": True, "flush_id": "a"})
            _until_flush(ws, "a")
            # Enough to fill the chunk, so the next burst is the first of a new one.
            ws.send_json({"send_text": f" {' '.join(FLUSH_TEXTS[1:])}"})
            ws.send_json({"close_context": True, "flush_id": "b"})
            _drain(ws)

        reference = reference_codes()
        opening, second = tts.prompts[0], tts.prompts[1]
        # The reference leads the continuation of every prompt, including the first, where it is
        # the only thing in front of the text.
        assert parse_audio_tokens(opening["prompt"]) == reference
        assert opening["prompt"].index(REFERENCE_TEXT) < opening["prompt"].index(opening["chunk"])

        # A burst inside the same chunk: the reference, then the codes the chunk is up to.
        assert parse_audio_tokens(second["prompt"]) == reference + list(second["carry"].codes)

        # A new chunk: the reference, then the chunk before it, then this chunk's text.
        last = tts.prompts[-1]
        carried = last["carry"]
        assert parse_audio_tokens(last["prompt"]) == reference + list(carried.codes)
        text = last["prompt"]
        assert text.index(REFERENCE_TEXT) < text.index(carried.text) < text.index(last["chunk"])

    def test_the_reference_is_sent_again_with_every_chunk(self, client, tts):
        run_session(
            client,
            FLUSH_TEXTS,
            reference={"transcript": REFERENCE_TEXT, "codes": reference_codes()},
        )
        assert len(tts.prompts) == len(FLUSH_TEXTS)
        for prompt in tts.prompts:
            assert (
                parse_audio_tokens(prompt["prompt"])[: len(reference_codes())] == reference_codes()
            )
            assert REFERENCE_TEXT in prompt["prompt"]

    def test_a_cloned_session_speaks_as_the_cloned_voice(self, client, tts):
        run_session(
            client,
            [FLUSH_TEXTS[0]],
            reference={"transcript": REFERENCE_TEXT, "codes": reference_codes()},
        )
        voice = tts.prompts[0]["voice"]
        assert voice.is_clone and voice.ref_text == REFERENCE_TEXT
        assert tts.prompts[0]["params"].max_tokens == CLONE_SAMPLING.max_tokens

    def test_the_first_decoder_starts_from_the_reference(self, client, tts):
        """There is no previous flush to prime from, so the tail of the clip stands in for one.

        Primed, the decoder's first samples are the audio the codes call for; unprimed they are
        the attenuated head of a decode that began from silence.
        """
        _, payload, _ = run_session(
            client,
            [FLUSH_TEXTS[0]],
            reference={"transcript": REFERENCE_TEXT, "codes": reference_codes()},
        )
        streamed = read_stream(payload)
        warm = StubCodec._tone(torch.arange(tts.generator.emitted)).numpy()
        assert streamed.size == warm.size
        assert np.allclose(streamed[:COLD_START_SAMPLES], warm[:COLD_START_SAMPLES], atol=1e-4)

    def test_a_cloned_session_joins_its_flushes_without_a_step(self, client):
        _, payload, boundaries = run_session(
            client,
            FLUSH_TEXTS,
            reference={"transcript": REFERENCE_TEXT, "codes": reference_codes()},
            response_format={"sample_rate": 16_000},
        )
        assert_joins_are_invisible(read_stream(payload), boundaries)


def start_cloned_session(client, tts, *, codes: bool) -> dict:
    """Open a cloned session, close it again, and return what ``context_started`` said."""
    with client.websocket_connect("/v1/ws") as ws:
        started = start_cloned(ws, codes=codes)
        ws.send_json({"close_context": True, "flush_id": "x"})
        _drain(ws)
    return started


class TestWebSocketReferenceRefusals:
    """An unusable reference is refused while the client is still setting up."""

    def test_a_reference_without_a_transcript_says_why_it_is_needed(self, client):
        message = refuse_start(client, {"reference": {"codes": reference_codes()}})
        assert "transcript" in message
        assert "speech recognition" in message

    def test_a_blank_transcript_is_the_same_refusal(self, client):
        message = refuse_start(
            client, {"reference": {"codes": reference_codes(), "transcript": " "}}
        )
        assert "transcript" in message

    def test_a_reference_with_no_clip_at_all_is_refused(self, client):
        message = refuse_start(client, {"reference": {"transcript": REFERENCE_TEXT}})
        assert "neither" in message

    def test_both_forms_at_once_is_refused(self, client):
        message = refuse_start(
            client,
            {
                "reference": {
                    "transcript": REFERENCE_TEXT,
                    "codes": reference_codes(),
                    "audio": reference_audio(),
                }
            },
        )
        assert "both" in message

    def test_a_voice_and_a_reference_together_is_refused(self, client):
        message = refuse_start(
            client,
            {
                "voice": "some-voice",
                "reference": {"transcript": REFERENCE_TEXT, "codes": reference_codes()},
            },
        )
        assert "who speaks" in message

    def test_a_clip_that_is_too_short_is_refused(self, client):
        message = refuse_start(
            client,
            {"reference": {"transcript": REFERENCE_TEXT, "audio": reference_audio(seconds=0.5)}},
        )
        assert "0.50 s" in message and "at least" in message

    def test_a_clip_that_is_too_long_is_refused_rather_than_trimmed(self, client):
        message = refuse_start(
            client,
            {"reference": {"transcript": REFERENCE_TEXT, "audio": reference_audio(seconds=25.0)}},
        )
        assert "25.0 s" in message and "at most" in message

    def test_a_silent_clip_is_refused(self, client):
        message = refuse_start(
            client,
            {"reference": {"transcript": REFERENCE_TEXT, "audio": reference_audio(amplitude=0.0)}},
        )
        assert "silent" in message

    def test_audio_that_is_not_base64_is_refused(self, client):
        message = refuse_start(
            client, {"reference": {"transcript": REFERENCE_TEXT, "audio": "%%%%"}}
        )
        assert "base64" in message

    def test_audio_that_is_not_a_file_is_refused(self, client):
        raw = base64.b64encode(np.zeros(64_000, dtype="<i2").tobytes()).decode("ascii")
        message = refuse_start(client, {"reference": {"transcript": REFERENCE_TEXT, "audio": raw}})
        assert "decoded" in message and "soundfile reads" in message

    def test_codes_outside_the_codebook_are_refused(self, client):
        message = refuse_start(
            client,
            {"reference": {"transcript": REFERENCE_TEXT, "codes": [1, 2, 99_999] * 60}},
        )
        assert "codebook" in message

    def test_too_few_codes_to_clone_from_are_refused(self, client):
        message = refuse_start(
            client, {"reference": {"transcript": REFERENCE_TEXT, "codes": [1, 2]}}
        )
        assert "at least" in message

    def test_a_reference_that_would_not_fit_the_cache_is_refused(self, client, tts):
        tts.generator.max_cache_len = 300
        message = refuse_start(
            client, {"reference": {"transcript": REFERENCE_TEXT, "codes": reference_codes()}}
        )
        assert "KV cache" in message and "300" in message


class TestWebSocketCarryUnderPressure:
    """What gives way when a prompt will not fit, and what may never give way with it."""

    #: Exactly the room a cloned session asks for at ``start_context``: the reference and its
    #: transcript, one full chunk of text, the codes that chunk turns into and the codes a burst
    #: on it may still generate -- the last two being what
    #: :data:`~kova_tts.engine.tts.MAX_CARRY_CODES` counts, since a chunk is sized so that its
    #: codes fit a carry. Accepted at this size, and with nothing spare for the chunk before it.
    CACHE = len(reference_codes()) + len(REFERENCE_TEXT) + MAX_SEGMENT_CHARS + 2 * MAX_CARRY_CODES

    #: A paragraph in pieces, each enough to speak on. Together they overflow one chunk twice
    #: over, so a burst late in the second chunk has both a chunk to carry and codes of its own
    #: that it cannot lose -- which is the only shape in which the two compete for the cache.
    PIECES = (
        "The opening sentence of this session, which is long enough to speak on.",
        "A second sentence follows it, and the chunk still has room for that one.",
        "The third one does not fit, so the chunk before it is closed and carried.",
        "A fourth arrives while the new chunk is already part way through speaking.",
        "The fifth is what closes that chunk, with a whole reference in front of it.",
        "A sixth and last sentence, spoken from a chunk that begins after all of them.",
    )

    def test_the_carry_gives_way_before_the_chunk_or_the_reference_does(self, client, tts):
        tts.generator.max_cache_len = self.CACHE
        _, payload, _ = run_session(
            client,
            self.PIECES,
            reference={"transcript": REFERENCE_TEXT, "codes": reference_codes()},
        )

        for prompt in tts.prompts:
            assert REFERENCE_TEXT in prompt["prompt"], "the reference gave way"
            assert len(prompt["ids"]) < self.CACHE

        # A prompt built with a carry that is not in it is one the cache would not take.
        given_up = [
            index
            for index, prompt in enumerate(tts.prompts)
            if prompt["carry"] is not None
            and prompt["carry"].text
            and prompt["carry"].text not in prompt["prompt"]
        ]
        assert given_up, "no burst ever had to give up the chunk before it"

        kept = tts.prompts[given_up[-1] + 1]
        codes = parse_audio_tokens(kept["prompt"])
        assert codes[: len(reference_codes())] == reference_codes()
        own = codes[len(reference_codes()) :]
        assert own, "the chunk's own codes went with the carry, so it would be spoken twice"
        assert tuple(own) == kept["carry"].codes
        # And the session still says what it was given, once.
        assert read_stream(payload).size == speech_samples(" ".join(self.PIECES))


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

    def test_an_unknown_response_format_field_is_refused(self, client):
        with client.websocket_connect("/v1/ws") as ws:
            ws.send_json({"start_context": {"response_format": {"bitrate": 128}}})
            assert "bad frame" in ws.receive_json()["error"]

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


def _collect(ws) -> bytes:
    """Every audio byte from here to the end of the session."""
    payload = bytearray()
    while True:
        frame = ws.receive_json()
        if "audio_chunk" in frame:
            payload += base64.b64decode(frame["audio_chunk"])
        elif "context_closed" in frame:
            return bytes(payload)


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
