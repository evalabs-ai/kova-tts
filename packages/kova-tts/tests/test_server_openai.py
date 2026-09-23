"""The OpenAI-compatible endpoint, driven against the same stub model as ``test_server.py``.

Two things are being proved here, and they are different. One is the usual: the routing, the
validators and the error envelope. The other is the point of the endpoint -- that bytes coming
back really are the container named in ``Content-Type``, decodable by a library that was not
told what to expect. A test that only checked status codes would pass just as happily while
handing every client a wav labelled ``audio/mpeg``.

The request bodies in :class:`TestOpenAIClientShape` are the ones the OpenAI Python SDK puts on
the wire for ``client.audio.speech.create(...)``. They are written out by hand so the check
runs everywhere; where the SDK happens to be installed, it drives the app itself as well.

Containers are produced by :mod:`kova_tts.audio` -- :func:`~kova_tts.audio.encode_audio` for a
whole file, :class:`~kova_tts.audio.StreamingEncoder` for the streamed path. Where those are
not present the tests that need them skip rather than fail, so an installation without them
still gets the whole HTTP contract checked for what it can produce.
"""

from __future__ import annotations

import base64
import io
import json
import shutil
import struct
import subprocess
import threading

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from test_server import StubTTS, speech_samples  # the fake engine the server tests already use

from kova_codec.constants import OUTPUT_SAMPLE_RATE
from kova_tts import audio as audio_layer
from kova_tts.server import formats
from kova_tts.server.app import create_app
from kova_tts.server.engine import Engine
from kova_tts.server.openai_api import MODEL_ID, STOCK_VOICES, parse_aliases, validate_aliases

#: Long enough to span several decoder frames at the stub's rate, short enough to stay quick.
TEXT = "Point your client at this server and it just works."

#: What the stub produces for TEXT, in seconds. Every container must come back this long,
#: whatever it resamples to on the way.
EXPECTED_SECONDS = speech_samples(TEXT) / OUTPUT_SAMPLE_RATE

#: Containers this installation can actually produce: probed by the audio layer, because a
#: soundfile wheel older than libsndfile 1.1 cannot write MPEG at all.
AVAILABLE = list(formats.available())

#: Every name OpenAI defines, so the ones this build lacks are still checked -- they must be
#: refused with a message, not with a traceback.
UNAVAILABLE = [name for name in ("mp3", "opus", "flac", "wav", "pcm") if name not in AVAILABLE]

#: Whether ffmpeg is on this machine. Not a dependency of anything here -- it is simply the
#: decoder that reads a *stream* the way a media player does, so it is used when it is around.
FFMPEG = shutil.which("ffmpeg")

#: Containers whose streamed bytes libsndfile alone can read back in full. The other two state
#: a length in a header written before the length exists, and libsndfile believes it.
STREAMS_AS_A_FILE = {"wav", "pcm", "opus"}


@pytest.fixture
def tts() -> StubTTS:
    return StubTTS()


@pytest.fixture
def client(tts: StubTTS):
    with TestClient(create_app(tts=tts)) as client:
        yield client


def speak(client: TestClient, **overrides) -> object:
    """POST a valid speech request with `overrides` folded in."""
    return client.post("/v1/audio/speech", json={"input": TEXT, **overrides})


def rate_of(name: str, requested: int | None = None) -> int:
    """The rate a response in this container will really carry."""
    return formats.output_rate(name, requested)


def decoded(payload: bytes, name: str, requested: int | None = None) -> np.ndarray:
    """The response body as a waveform, decoded the way an unsuspecting client would.

    ``pcm`` is the one container that carries no header, so its rate has to be taken on trust
    -- which is exactly the promise the endpoint is making about those bytes.
    """
    expected = rate_of(name, requested)
    if name == "pcm":
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    samples, actual = sf.read(io.BytesIO(payload), dtype="float32")
    assert actual == expected, f"{name} decoded at {actual} Hz, not {expected}"
    return samples


def seconds_of(payload: bytes, name: str, requested: int | None = None) -> float:
    return decoded(payload, name, requested).size / rate_of(name, requested)


#: The first bytes of each container, so a streamed body can be identified without decoding it
#: -- which matters because a stream, unlike a file, may carry no length for a decoder to trust.
#: The mp3 entry is a frame sync word (11 set bits), not a magic number; mp3 has none.
MAGIC = {
    "wav": (b"RIFF",),
    "flac": (b"fLaC",),
    "opus": (b"OggS",),
    "mp3": (b"\xff\xfb", b"\xff\xfa", b"\xff\xf3", b"\xff\xf2", b"\xff\xe3", b"ID3"),
}


def starts_like(payload: bytes, name: str) -> bool:
    """Is this really `name`? True for pcm, which is samples and has nothing to check."""
    prefixes = MAGIC.get(name)
    return True if prefixes is None else payload.startswith(prefixes)


def read_stream(payload: bytes, name: str) -> np.ndarray | None:
    """Decode a *streamed* body, or ``None`` when nothing here can read it back.

    A stream is not a file. mp3 and flac both keep a length field at the front that is only
    correct once the audio has ended, and libsndfile trusts it: it truncates a streamed mp3 and
    refuses a streamed flac outright. ffmpeg -- and every media player -- reads to the end
    instead, so ffmpeg goes first here and libsndfile is the fallback. A caller that gets
    ``None`` back has learnt that nothing on this machine can check the contents, which is not
    the same as the bytes being wrong.
    """
    if name == "pcm":
        return np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32767.0
    if FFMPEG is not None:
        finished = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [FFMPEG, "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "pipe:1"],
            input=payload,
            capture_output=True,
            check=False,
        )
        if finished.returncode == 0 and finished.stdout:
            return np.frombuffer(finished.stdout, dtype="<f4")
    if name in STREAMS_AS_A_FILE:
        with sf.SoundFile(io.BytesIO(payload)) as handle:
            blocks = []
            while True:
                block = handle.read(8192, dtype="float32")
                if not len(block):
                    break
                blocks.append(block)
        return np.concatenate(blocks) if blocks else np.zeros(0, dtype=np.float32)
    return None


def stream_seconds(payload: bytes, name: str, requested: int | None = None) -> float | None:
    samples = read_stream(payload, name)
    return None if samples is None else samples.size / rate_of(name, requested)


def sse_events(payload: bytes) -> list[dict]:
    """Parse ``stream_format: "sse"`` output: one JSON object per ``data:`` line."""
    return [
        json.loads(line[len("data: ") :])
        for line in payload.decode("utf-8").splitlines()
        if line.startswith("data: ")
    ]


# ------------------------------------------------------------------------------ GET /v1/models


class TestModels:
    def test_lists_the_one_model_this_server_has(self, client):
        body = client.get("/v1/models").json()
        assert body["object"] == "list"
        assert body["data"][0]["id"] == MODEL_ID
        assert body["data"][0]["object"] == "model"

    def test_every_entry_has_the_fields_a_client_deserializes(self, client):
        entry = client.get("/v1/models").json()["data"][0]
        assert set(entry) == {"id", "object", "created", "owned_by"}

    def test_created_is_a_plausible_date_rather_than_the_epoch(self, client):
        """Some clients sort or display it, and 1970 looks like a bug."""
        assert client.get("/v1/models").json()["data"][0]["created"] > 1_600_000_000

    def test_each_installed_voice_is_listed_as_its_own_model(self, client, tts):
        """For clients whose only dropdown is a model dropdown."""
        ids = [entry["id"] for entry in client.get("/v1/models").json()["data"]]
        assert ids == [MODEL_ID] + [f"{MODEL_ID}:{name}" for name in tts.voices()]

    def test_a_voice_model_id_picks_that_voice(self, client, tts):
        assert speak(client, model=f"{MODEL_ID}:another-voice").status_code == 200
        assert tts.calls[-1]["voice"] == "another-voice"

    def test_the_voice_field_wins_over_the_model_id(self, client, tts):
        """`voice` is the more specific instrument, so it decides."""
        response = speak(client, model=f"{MODEL_ID}:another-voice", voice="some-voice")
        assert response.status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"

    def test_a_stock_voice_does_not_override_a_voice_model_id(self, client, tts):
        """A client sending its hardcoded `alloy` should still get the voice it selected."""
        assert speak(client, model=f"{MODEL_ID}:another-voice", voice="alloy").status_code == 200
        assert tts.calls[-1]["voice"] == "another-voice"

    def test_a_model_id_beats_an_alias_for_the_same_stock_name(self, tts):
        """The model id was chosen by somebody; the voice field is whatever shipped with it."""
        with TestClient(create_app(tts=tts, voice_aliases=["alloy=some-voice"])) as client:
            response = speak(client, model=f"{MODEL_ID}:another-voice", voice="alloy")
        assert response.status_code == 200
        assert tts.calls[-1]["voice"] == "another-voice"
        assert response.headers["x-voice"] == "another-voice"

    def test_a_model_id_naming_no_real_voice_is_still_accepted(self, client, tts):
        """`model` is never a gate: an unknown one is ignored, not refused."""
        assert speak(client, model=f"{MODEL_ID}:not-a-voice").status_code == 200
        assert tts.calls[-1]["voice"] is None


# ---------------------------------------------------------------------- POST /v1/audio/speech


class TestSpeech:
    @pytest.mark.parametrize("name", AVAILABLE)
    def test_every_format_returns_a_well_formed_file(self, client, name):
        """``stream: false`` -- the container is written in one pass, so it decodes exactly."""
        response = speak(client, response_format=name, stream=False)

        assert response.status_code == 200
        assert response.headers["content-type"].split(";")[0] == formats.media_type(name)
        assert response.headers["x-sample-rate"] == str(rate_of(name))

        samples = decoded(response.content, name)
        # Lossy containers pad to their frame size; a tenth of a second covers that and would
        # still catch a container that silently truncated or doubled the utterance.
        assert samples.size / rate_of(name) == pytest.approx(EXPECTED_SECONDS, abs=0.1)
        assert float(np.max(np.abs(samples))) > 0.01  # the stub's tone, not a buffer of zeros

    @pytest.mark.parametrize("name", AVAILABLE)
    def test_every_format_streams_by_default(self, client, name):
        """The default path: bytes leave as they are decoded, and are still that container."""
        response = speak(client, response_format=name)

        assert response.status_code == 200
        assert response.headers["content-type"].split(";")[0] == formats.media_type(name)
        assert "content-length" not in response.headers, "a streamed body has no known length"
        assert starts_like(response.content, name)

    @pytest.mark.parametrize("name", AVAILABLE)
    def test_a_streamed_body_holds_the_whole_utterance(self, client, name):
        """ "It streams" is worthless if what streams is not playable to the end.

        The tolerance is wider than the file path's: a streamed mp3 keeps the codec's 1105-sample
        priming delay, which a finished file can annotate away and a stream cannot.
        """
        seconds = stream_seconds(speak(client, response_format=name).content, name)
        if seconds is None:
            pytest.skip(f"nothing on this machine decodes a streamed {name} (install ffmpeg)")
        assert seconds == pytest.approx(EXPECTED_SECONDS, abs=0.15)

    @pytest.mark.parametrize("name", AVAILABLE)
    def test_the_filename_matches_the_container(self, client, name):
        response = speak(client, response_format=name)
        assert response.headers["content-disposition"].endswith(f'speech.{name}"')

    def test_no_response_format_is_mp3_like_openai(self, client):
        """A client that sends no format gets OpenAI's default, not an error."""
        response = speak(client)
        assert response.status_code == 200
        content_type = response.headers["content-type"].split(";")[0]
        assert content_type == formats.media_type(formats.default_format())
        if "mp3" in AVAILABLE:
            assert content_type == "audio/mpeg"

    def test_the_format_name_is_case_insensitive(self, client):
        assert speak(client, response_format="WAV").headers["content-type"] == "audio/wav"

    def test_a_streamed_wav_states_no_length(self, client):
        """A WAV written before its own length is known cannot claim one in its header."""
        body = speak(client, response_format="wav").content
        assert body[:4] == b"RIFF" and body[8:12] == b"WAVE"
        data_size = struct.unpack("<I", body[40:44])[0]
        assert data_size != len(body) - 44, "a streamed header cannot know the real data size"

    def test_a_whole_wav_states_its_real_length(self, client):
        body = speak(client, response_format="wav", stream=False).content
        assert struct.unpack("<I", body[40:44])[0] == len(body) - 44

    def test_pcm_is_headerless_at_the_model_rate(self, client):
        response = speak(client, response_format="pcm")
        assert response.content[:4] != b"RIFF"
        assert response.headers["x-sample-rate"] == str(OUTPUT_SAMPLE_RATE)
        assert len(response.content) // 2 == pytest.approx(
            EXPECTED_SECONDS * OUTPUT_SAMPLE_RATE, abs=2
        )

    def test_a_long_input_is_still_one_request(self, client, tts):
        response = speak(client, input="Hello. " * 200)
        assert response.status_code == 200
        assert len(tts.calls) == 1


class TestSampleRate:
    """The one field OpenAI's schema does not have, for pipelines fixed at another rate."""

    @pytest.mark.parametrize("rate", [8000, 16000, 24000])
    def test_wav_comes_back_at_the_rate_that_was_asked_for(self, client, rate):
        response = speak(client, response_format="wav", sample_rate=rate)
        assert response.status_code == 200
        assert response.headers["x-sample-rate"] == str(rate)
        assert seconds_of(response.content, "wav", rate) == pytest.approx(
            EXPECTED_SECONDS, abs=0.05
        )

    def test_pcm_at_the_rate_openai_documents(self, client):
        """OpenAI's pcm is 24 kHz; a client that assumes it says so asks, and gets it."""
        response = speak(client, response_format="pcm", sample_rate=24000)
        assert len(response.content) // 2 == pytest.approx(EXPECTED_SECONDS * 24000, abs=64)
        assert response.headers["x-sample-rate"] == "24000"

    def test_an_absent_rate_is_the_models_own(self, client):
        assert speak(client).headers["x-sample-rate"] == str(OUTPUT_SAMPLE_RATE)

    @pytest.mark.parametrize("rate", [7999, 96000, 0, -16000])
    def test_a_rate_outside_the_range_is_refused(self, client, rate):
        response = speak(client, response_format="wav", sample_rate=rate)
        assert response.status_code == 422
        assert "sample_rate" in response.json()["message"]

    @pytest.mark.skipif("opus" not in AVAILABLE, reason="this install cannot encode opus")
    def test_opus_reports_the_rate_it_was_really_written_at(self, client):
        """Opus takes 8/12/16/24/48 kHz, so 32 kHz is carried at 48 kHz instead."""
        response = speak(client, response_format="opus", sample_rate=32000)
        assert response.status_code == 200
        assert response.headers["x-sample-rate"] == "48000"
        assert seconds_of(response.content, "opus", 32000) == pytest.approx(
            EXPECTED_SECONDS, abs=0.1
        )


class TestVoice:
    def test_a_lora_voice_reaches_the_model(self, client, tts):
        assert speak(client, voice="some-voice").status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"

    @pytest.mark.parametrize("voice", [None, "", "   ", "default", "base", "DEFAULT"])
    def test_the_base_voice_is_what_an_empty_voice_means(self, client, tts, voice):
        assert speak(client, voice=voice).status_code == 200
        assert tts.calls[-1]["voice"] is None

    def test_a_voice_object_is_unwrapped(self, client, tts):
        """Newer clients may send ``{"id": "..."}`` instead of a bare name."""
        assert speak(client, voice={"id": "some-voice"}).status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"

    @pytest.mark.parametrize("voice", sorted(STOCK_VOICES))
    def test_every_openai_stock_voice_speaks_rather_than_404ing(self, client, tts, voice):
        """The whole point of the endpoint: a stock client works without being reconfigured.

        Its voice field arrives holding `alloy` whether the user chose it or not -- several
        clients hardcode it -- so refusing it would fail the first request anybody makes.
        """
        response = speak(client, voice=voice)
        assert response.status_code == 200
        assert tts.calls[-1]["voice"] is None
        assert response.headers["x-voice"] == "base"

    def test_a_typo_still_says_what_is_available(self, client):
        response = speak(client, voice="some-voise")
        assert response.status_code == 404
        body = response.json()
        assert body["error"] == "not_found"
        assert "some-voice" in body["message"] and "another-voice" in body["message"]
        assert "/v1/voices" in body["message"]

    def test_the_resolved_voice_is_reported_in_a_header(self, client):
        assert speak(client, voice="some-voice").headers["x-voice"] == "some-voice"
        assert speak(client).headers["x-voice"] == "base"

    def test_a_real_voice_named_like_a_stock_one_wins(self, tts):
        """An installed adapter is always the most specific answer."""
        tts._voices = ["alloy"]
        with TestClient(create_app(tts=tts, voice_aliases={})) as client:
            response = speak(client, voice="alloy")
        assert response.status_code == 200
        assert tts.calls[-1]["voice"] == "alloy"
        assert response.headers["x-voice"] == "alloy"

    def test_a_real_voice_named_default_wins_over_the_alias(self):
        """The adapter directory is consulted first, so the alias can never shadow a voice."""
        tts = StubTTS(voices=["default"])
        with TestClient(create_app(tts=tts)) as client:
            assert speak(client, voice="default").status_code == 200
        assert tts.calls[-1]["voice"] == "default"

    def test_a_server_with_no_voices_still_answers_a_stock_client(self):
        """The commonest configuration: the base model, with no adapter directory at all.

        Every voice a client could name is a stock one there, so every one of them has to work.
        """
        tts = StubTTS()
        tts._voices = []  # the stub reads an empty list as "unset", so it is emptied after
        with TestClient(create_app(tts=tts, voice_aliases={})) as client:
            assert speak(client, voice="alloy").status_code == 200
            typo = speak(client, voice="not-a-voice")
        assert tts.calls[-1]["voice"] is None
        assert typo.status_code == 404
        assert "none installed" in typo.json()["message"]


class TestVoiceAliases:
    """Where an operator points OpenAI's stock names, for a machine that has real voices."""

    @pytest.fixture
    def aliased(self, tts):
        with TestClient(create_app(tts=tts, voice_aliases=["alloy=some-voice"])) as client:
            yield client

    def test_an_aliased_stock_name_reaches_the_real_voice(self, aliased, tts):
        response = speak(aliased, voice="alloy")
        assert response.status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"
        assert response.headers["x-voice"] == "some-voice"

    def test_an_alias_may_name_something_that_is_not_a_stock_voice(self, tts):
        """Nothing stops an operator mapping their own shorthand, so nothing does."""
        with TestClient(create_app(tts=tts, voice_aliases=["house-voice=some-voice"])) as client:
            assert speak(client, voice="house-voice").status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"

    def test_an_unmapped_stock_name_still_speaks_in_the_base_voice(self, aliased, tts):
        assert speak(aliased, voice="nova").status_code == 200
        assert tts.calls[-1]["voice"] is None

    def test_the_alias_is_case_insensitive(self, aliased, tts):
        assert speak(aliased, voice="ALLOY").status_code == 200
        assert tts.calls[-1]["voice"] == "some-voice"

    def test_the_environment_configures_the_same_thing(self, tts, monkeypatch):
        monkeypatch.setenv("KOVA_VOICE_ALIASES", "alloy=some-voice,nova=another-voice")
        with TestClient(create_app(tts=tts)) as client:
            assert speak(client, voice="nova").status_code == 200
        assert tts.calls[-1]["voice"] == "another-voice"

    def test_an_alias_to_a_voice_that_is_not_installed_stops_the_server(self, tts):
        """Loudly, at startup: the alternative is one client quietly broken."""
        with pytest.raises(ValueError, match="does not have"):
            with TestClient(create_app(tts=tts, voice_aliases=["alloy=ghost"])):
                pass

    @pytest.mark.parametrize("entry", ["alloy", "alloy=", "=some-voice", "  "])
    def test_a_malformed_alias_is_refused_with_the_entry_named(self, entry):
        if not entry.strip():
            assert parse_aliases([entry]) == {}
            return
        with pytest.raises(ValueError, match="name=voice"):
            parse_aliases([entry])

    def test_entries_may_be_repeated_or_comma_separated(self):
        assert parse_aliases(["alloy=a", "nova=b"]) == {"alloy": "a", "nova": "b"}
        assert parse_aliases(["alloy=a,nova=b"]) == {"alloy": "a", "nova": "b"}

    def test_validation_names_every_bad_mapping_at_once(self, tts):
        with pytest.raises(ValueError) as failure:
            validate_aliases(Engine(tts), {"alloy": "ghost", "nova": "phantom"})
        assert "ghost" in str(failure.value) and "phantom" in str(failure.value)


class TestUnsupportedControls:
    @pytest.mark.parametrize("speed", [1.5, 0.5, 2, 0.25])
    def test_speed_is_refused_rather_than_ignored(self, client, tts, speed):
        response = speak(client, speed=speed)
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_request"
        assert "speed" in response.json()["message"]
        assert not tts.calls, "the model should not have run for a request that was refused"

    @pytest.mark.parametrize("speed", [1.0, 1])
    def test_the_only_speed_this_model_has_is_accepted(self, client, speed):
        assert speak(client, speed=speed).status_code == 200

    def test_instructions_are_refused_rather_than_ignored(self, client):
        response = speak(client, instructions="Read it like a weather forecast.")
        assert response.status_code == 422
        assert "instructions" in response.json()["message"]

    @pytest.mark.parametrize("instructions", [None, "", "  "])
    def test_empty_instructions_are_fine(self, client, instructions):
        assert speak(client, instructions=instructions).status_code == 200


class TestFormatRefusals:
    def test_aac_names_what_this_server_can_do_instead(self, client):
        response = speak(client, response_format="aac")
        assert response.status_code == 422
        message = response.json()["message"]
        assert "aac" in message
        for name in AVAILABLE:
            assert name in message

    def test_an_unknown_format_is_refused(self, client):
        response = speak(client, response_format="ogg-but-not-really")
        assert response.status_code == 422
        message = response.json()["message"]
        assert "ogg-but-not-really" in message
        for name in AVAILABLE:
            assert name in message

    @pytest.mark.parametrize("name", UNAVAILABLE)
    def test_a_container_this_install_cannot_encode_is_refused_by_name(self, client, name):
        """Only runs where an encoder really is missing; the message must name the fix."""
        response = speak(client, response_format=name)
        assert response.status_code == 422
        assert name in response.json()["message"]
        assert "wav" in response.json()["message"]

    def test_the_default_falls_back_when_mp3_cannot_be_encoded(self, client, monkeypatch):
        """A request that named no format has nothing wrong with it, so it must not 4xx.

        Simulated by hiding mp3 from the audio layer's format list, which is what an older
        soundfile wheel actually looks like.
        """
        monkeypatch.setattr(
            audio_layer, "SUPPORTED_FORMATS", tuple(f for f in AVAILABLE if f != "mp3")
        )
        assert formats.default_format() == "wav"
        response = speak(client)
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"


class TestModelField:
    @pytest.mark.parametrize(
        "model", ["tts-1", "tts-1-hd", "gpt-4o-mini-tts", MODEL_ID, "nonsense"]
    )
    def test_any_model_name_is_accepted(self, client, model):
        assert speak(client, model=model).status_code == 200

    def test_no_model_at_all_is_accepted(self, client):
        assert client.post("/v1/audio/speech", json={"input": TEXT}).status_code == 200


class TestMalformedRequests:
    def test_a_missing_input_is_a_422(self, client):
        response = client.post("/v1/audio/speech", json={"model": "tts-1", "voice": "some-voice"})
        assert response.status_code == 422
        assert "input" in response.json()["message"]

    def test_a_blank_input_says_what_to_send(self, client):
        response = speak(client, input="   ")
        assert response.status_code == 422
        assert "input is empty" in response.json()["message"]

    def test_an_over_long_input_is_refused(self, client):
        response = speak(client, input="a" * 20_000)
        assert response.status_code == 422
        assert response.json()["error"] == "invalid_request"

    def test_a_body_that_is_not_an_object_is_a_422(self, client):
        response = client.post("/v1/audio/speech", json=["hello"])
        assert response.status_code == 422

    def test_broken_json_is_a_422_in_the_usual_envelope(self, client):
        response = client.post(
            "/v1/audio/speech",
            content=b"{not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
        assert set(response.json()) == {"error", "message"}

    def test_a_wrongly_typed_field_is_a_422(self, client):
        response = speak(client, speed="fast")
        assert response.status_code == 422

    def test_a_field_this_server_has_never_heard_of_is_ignored(self, client):
        """Deliberately unlike /v1/tts: OpenAI adds fields, and clients send them everywhere."""
        assert speak(client, some_future_option={"nested": True}).status_code == 200


class TestOneGenerationAtATime:
    @pytest.mark.parametrize("response_format", ["wav", *[n for n in AVAILABLE if n != "wav"][:1]])
    def test_a_second_caller_gets_the_same_409(self, tts, response_format):
        """The compatibility surface reuses the engine's lock; it does not invent a queue.

        Run for a streamed container and a buffered one: they reserve the model by different
        routes, and only one of them was ever going to be tested by accident.
        """
        tts.release.clear()
        app = create_app(tts=tts, busy_timeout=0)
        with TestClient(app) as client:
            done = threading.Event()

            def occupy() -> None:
                speak(client, response_format=response_format)
                done.set()

            worker = threading.Thread(target=occupy, daemon=True)
            worker.start()
            assert tts.entered.wait(timeout=5)

            refused = speak(client, response_format=response_format)
            assert refused.status_code == 409
            assert refused.json()["error"] == "busy"

            tts.release.set()
            assert done.wait(timeout=10)
            worker.join(timeout=5)

    def test_the_model_is_released_after_a_failure(self, client, tts):
        tts._fail = RuntimeError("the GPU fell over")
        failed = speak(client, response_format="wav", stream_format="sse")
        assert failed.status_code == 200  # the status line was sent before generation began
        assert sse_events(failed.content)[-1]["type"] == "error"

        tts._fail = None
        assert speak(client, response_format="wav").status_code == 200


# -------------------------------------------------------------------- stream_format: "sse"


class TestServerSentEvents:
    def test_the_deltas_rebuild_the_same_file(self, client):
        streamed = speak(client, response_format="wav", stream_format="sse")
        assert streamed.status_code == 200
        assert streamed.headers["content-type"].startswith("text/event-stream")

        events = sse_events(streamed.content)
        deltas = [event for event in events if event["type"] == "speech.audio.delta"]
        assert len(deltas) > 1, "a streamed container should arrive as more than one delta"

        rebuilt = b"".join(base64.b64decode(event["audio"]) for event in deltas)
        assert seconds_of(rebuilt, "wav") == pytest.approx(EXPECTED_SECONDS, abs=0.1)

    def test_the_stream_ends_with_one_done_event(self, client):
        events = sse_events(speak(client, stream_format="sse").content)
        assert events[-1]["type"] == "speech.audio.done"
        assert set(events[-1]["usage"]) == {"input_tokens", "output_tokens", "total_tokens"}
        assert [event["type"] for event in events].count("speech.audio.done") == 1

    def test_a_whole_file_arrives_as_one_delta(self, client):
        """``stream: false`` still speaks SSE; it just has one thing to say."""
        events = sse_events(
            speak(client, response_format="wav", stream=False, stream_format="sse").content
        )
        assert [event["type"] for event in events] == ["speech.audio.delta", "speech.audio.done"]
        rebuilt = base64.b64decode(events[0]["audio"])
        assert seconds_of(rebuilt, "wav") == pytest.approx(EXPECTED_SECONDS, abs=0.1)


# ------------------------------------------------------------------- what a real client sends


class TestOpenAIClientShape:
    """The exact bodies ``openai.OpenAI().audio.speech.create(...)`` puts on the wire.

    The SDK sends only the arguments it was given, so both the full body and the three-field
    minimum are checked, along with the ``Accept: application/octet-stream`` header it sets for
    a binary response.
    """

    SDK_HEADERS = {"Accept": "application/octet-stream", "Content-Type": "application/json"}

    def test_the_full_body_works(self, client):
        response = client.post(
            "/v1/audio/speech",
            headers=self.SDK_HEADERS,
            content=json.dumps(
                {
                    "input": TEXT,
                    "model": "tts-1",
                    "voice": "some-voice",
                    "response_format": "wav",
                    "speed": 1.0,
                }
            ),
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "audio/wav"
        assert seconds_of(response.content, "wav") == pytest.approx(EXPECTED_SECONDS, abs=0.15)

    def test_the_minimal_body_works(self, client):
        """model, input and voice are the SDK's only required arguments."""
        response = client.post(
            "/v1/audio/speech",
            headers=self.SDK_HEADERS,
            content=json.dumps({"input": TEXT, "model": "tts-1", "voice": "default"}),
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == formats.media_type(formats.default_format())

    @pytest.mark.skipif("mp3" not in AVAILABLE, reason="this install cannot encode mp3")
    def test_the_default_container_is_real_mp3(self, client):
        """What a stock client gets when it names no format: bytes that really are mp3."""
        response = client.post(
            "/v1/audio/speech",
            headers=self.SDK_HEADERS,
            content=json.dumps({"input": TEXT, "model": "tts-1", "voice": "some-voice"}),
        )
        assert response.headers["content-type"] == "audio/mpeg"
        assert starts_like(response.content, "mp3")
        whole = client.post(
            "/v1/audio/speech",
            headers=self.SDK_HEADERS,
            content=json.dumps(
                {"input": TEXT, "model": "tts-1", "voice": "some-voice", "stream": False}
            ),
        )
        assert seconds_of(whole.content, "mp3") == pytest.approx(EXPECTED_SECONDS, abs=0.1)

    def test_the_streaming_body_works(self, client):
        """``with_streaming_response`` plus ``stream_format="sse"``, as newer clients send it."""
        response = client.post(
            "/v1/audio/speech",
            headers=self.SDK_HEADERS,
            content=json.dumps(
                {
                    "input": TEXT,
                    "model": "gpt-4o-mini-tts",
                    "voice": "some-voice",
                    "response_format": "pcm",
                    "stream_format": "sse",
                }
            ),
        )
        assert response.status_code == 200
        assert sse_events(response.content)[-1]["type"] == "speech.audio.done"

    def test_the_sdk_itself_when_it_is_installed(self, client):
        """The real client library, driving the application through the test transport.

        ``TestClient`` is an ``httpx.Client``, so the SDK can be handed it and will send
        whatever it would send to OpenAI -- no hand-written body in this one.
        """
        openai = pytest.importorskip("openai")

        sdk = openai.OpenAI(
            api_key="not-needed", base_url="http://testserver/v1", http_client=client
        )
        speech = sdk.audio.speech.create(
            model="tts-1", voice="some-voice", input=TEXT, response_format="wav"
        )
        assert seconds_of(speech.read(), "wav") == pytest.approx(EXPECTED_SECONDS, abs=0.15)
        assert sdk.models.list().data[0].id == MODEL_ID


# ------------------------------------------------------------------------------- unit corners


class TestFormatPolicy:
    def test_the_media_type_comes_from_the_module_that_writes_the_bytes(self):
        """One source for the header and the encoder, so the two cannot drift apart."""
        for name in AVAILABLE:
            assert formats.media_type(name) == audio_layer.content_type(name)

    def test_aac_is_refused_with_a_reason_rather_than_a_traceback(self):
        with pytest.raises(Exception) as failure:
            formats.resolve("aac")
        assert "ffmpeg" in str(failure.value)

    def test_the_default_is_openais_default_where_it_can_be_encoded(self):
        expected = "mp3" if "mp3" in AVAILABLE else "wav"
        assert formats.default_format() == expected

    def test_stream_false_always_takes_the_whole_file_path(self):
        for name in AVAILABLE:
            assert not formats.streamable(name, False)
            assert formats.streamable(name, True)

    def test_a_rate_the_codec_refuses_is_snapped_up_rather_than_rejected(self):
        """MP3 has no 20 kHz mode; the audio layer picks the next rate up and says so."""
        assert formats.output_rate("mp3", 20000) == 22050
        assert formats.output_rate("wav", 20000) == 20000
