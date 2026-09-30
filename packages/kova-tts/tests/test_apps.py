"""The demo and the ComfyUI pack, on a CPU with no weights.

Both live in ``apps/``, outside the installed package, because neither is a library: the demo
is a program and the node pack is a directory that gets linked into ``custom_nodes``. They are
loaded here by path, exactly the way their real callers load them -- ``kova-tts demo`` reaches
the demo through its file, and ComfyUI imports the node directory as a package.

Everything runs against :class:`FakeTTS`, which has the surface of
:class:`~kova_tts.engine.tts.KovaTTS` and a sine wave where the model would be. That is enough
to check what actually breaks in these two files: the demo's streaming endpoint handing the
browser the model's own samples and nothing else, the states a first-time user lands in, and
the ComfyUI type conversion at the boundary.

The demo is exercised through :func:`build_app`, over HTTP, because that is how the page uses
it: the browser POSTs to the streaming endpoint and plays the PCM itself. A test that called a
Python callback instead would prove nothing about the thing that was broken.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kova_codec.constants import OUTPUT_SAMPLE_RATE, SAMPLE_RATE
from kova_tts import AudioFrame, MissingArtifact, Voice
from kova_tts.audio import to_pcm_bytes

APPS = Path(__file__).resolve().parents[3] / "apps"


def _load(name: str, path: Path, *, package: bool = False) -> Any:
    """Import a module from its path, optionally as a package so relative imports resolve."""
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # a package's own submodules look themselves up here
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo() -> Any:
    """``apps/demo/app.py``, skipped when the ``demo`` extra is not installed."""
    pytest.importorskip("gradio")
    pytest.importorskip("httpx")
    return _load("kova_demo_app", APPS / "demo" / "app.py")


@pytest.fixture(scope="module")
def comfy() -> Any:
    """``apps/comfyui/`` as ComfyUI would import it: a package, and no ComfyUI in sight."""
    return _load("kova_comfyui_pack", APPS / "comfyui" / "__init__.py", package=True)


# ------------------------------------------------------------------------------- the fake model


def sine(seconds: float, *, freq: float = 220.0, rate: int = SAMPLE_RATE) -> np.ndarray:
    """A waveform to stand in for speech. No audio is committed to this repository."""
    t = np.arange(int(seconds * rate), dtype=np.float32) / rate
    return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class FakeTTS:
    """:class:`KovaTTS`'s surface, without the model.

    Records what it was asked for, so a test can check that the demo passed the resolved voice
    and the sampling through rather than only that it produced audio.
    """

    def __init__(
        self,
        voice_names: tuple[str, ...] = (),
        *,
        frames: int = 4,
        transcriber: Any = None,
        fail: Exception | None = None,
        silent: bool = False,
    ) -> None:
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.transcriber = transcriber
        self._voices = tuple(voice_names)
        self._frames = frames
        self._fail = fail
        self._silent = silent
        self.calls: list[dict[str, Any]] = []
        self.cloned: list[dict[str, Any]] = []

    def voices(self) -> list[str]:
        return list(self._voices)

    def voice(self, name: str) -> Voice:
        return Voice(name=name, lora_path=Path(f"/nowhere/{name}"))

    def stream(self, text, voice=None, *, params=None):
        self.calls.append({"text": text, "voice": voice, "params": params})
        if self._fail is not None:
            raise self._fail
        if self._silent:
            yield AudioFrame(np.zeros(0, dtype=np.float32), self.sample_rate, is_final=True)
            return
        for index in range(self._frames):
            yield AudioFrame(sine(0.2 * (index + 1)), self.sample_rate)
        yield AudioFrame(np.zeros(0, dtype=np.float32), self.sample_rate, is_final=True)

    def generate(self, text, voice=None, *, params=None) -> np.ndarray:
        pieces = [frame.samples for frame in self.stream(text, voice, params=params)]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def clone(self, audio, transcript=None, *, name=None, sample_rate=SAMPLE_RATE) -> Voice:
        self.cloned.append(
            {"audio": audio, "transcript": transcript, "name": name, "sample_rate": sample_rate}
        )
        if transcript is None:
            if self.transcriber is None:
                raise ValueError("clone() needs the reference transcript")
            transcript = self.transcriber(str(audio))
        return Voice(name=name or "cloned", ref_codes=(1, 2, 3, 4), ref_text=transcript)

    def save(self, wav: np.ndarray, path, sample_rate: int | None = None) -> Path:
        from kova_tts.audio import save_wav

        return save_wav(path, wav, sample_rate or self.sample_rate)


@pytest.fixture
def reference_wav(tmp_path: Path) -> Path:
    """A three-second recording, synthesized: nothing audio-shaped is ever committed."""
    from kova_tts.audio import save_wav

    return save_wav(tmp_path / "reference.wav", sine(3.0), SAMPLE_RATE)


# ------------------------------------------------------------------------ driving the endpoint


class Stream:
    """One run of the streaming endpoint, taken apart the way the player takes it apart."""

    def __init__(self, status: int, payloads: list[dict[str, Any]]) -> None:
        self.status = status
        self.payloads = payloads

    @property
    def chunks(self) -> list[bytes]:
        """The decoded PCM of each ``chunk`` event, in the order they arrived."""
        return [base64.b64decode(item["audio"]) for item in self.payloads if "audio" in item]

    @property
    def pcm(self) -> bytes:
        """Every chunk concatenated: exactly what the player has when the stream ends."""
        return b"".join(self.chunks)

    @property
    def done(self) -> dict[str, Any] | None:
        return next((item for item in self.payloads if "chunks" in item), None)

    @property
    def error(self) -> dict[str, Any] | None:
        return next((item for item in self.payloads if "error" in item), None)


def speak(client: Any, path: str, **body: Any) -> Stream:
    """POST one synthesis request and collect the events it streams back."""
    payloads: list[dict[str, Any]] = []
    with client.stream("POST", path, json=body) as response:
        if response.status_code != 200:
            response.read()
            return Stream(response.status_code, [response.json()])
        for line in response.iter_lines():
            if line.startswith("data:"):
                payloads.append(json.loads(line[5:].strip()))
    return Stream(response.status_code, payloads)


@pytest.fixture
def client(demo: Any):
    """A client for the whole demo -- page and stream -- against a fake engine.

    Yields ``(client, session, fake)``. Function-scoped: the session owns the lock the endpoint
    reserves, and a test that leaves it held must not reach the next one.
    """
    from fastapi.testclient import TestClient

    def build(fake: Any = None, **options: Any) -> tuple[Any, Any, Any]:
        fake = fake if fake is not None else FakeTTS()
        session = demo.DemoSession(tts=fake, **options)
        return TestClient(demo.build_app(session)), session, fake

    return build


# --------------------------------------------------------------------------------- the demo: UI


def test_build_ui_constructs_without_launching(demo: Any) -> None:
    import gradio as gr

    fake = FakeTTS(("alto", "tenor"))
    ui = demo.build_ui(demo.DemoSession(tts=fake))

    assert isinstance(ui, gr.Blocks)
    # The page opens on the base model; the LoRAs are one source-switch away.
    by_id = {block.elem_id: block for block in ui.blocks.values() if block.elem_id}
    assert [choice[1] for choice in by_id["kova-source"].choices] == [
        demo.SOURCE_BASE,
        demo.SOURCE_CLONE,
        demo.SOURCE_LORA,
    ]
    assert [choice[1] for choice in by_id["kova-voice"].choices] == [demo.BASE_VOICE]
    # Cloning opens on reading the passage, which only the microphone can do; a file comes with
    # freestyle.
    assert [choice[1] for choice in by_id["kova-clone-mode"].choices] == [
        demo.CLONE_READ,
        demo.CLONE_FREE,
    ]
    assert by_id["kova-clone-mode"].value == demo.CLONE_READ
    assert by_id["kova-reference"].sources == ["microphone"]
    # Building the page must not generate anything -- cached examples would do exactly that,
    # and on a real engine it is a minute of synthesis before the first visitor arrives.
    assert not fake.calls


def test_the_page_does_not_use_gradios_streaming_audio(demo: Any) -> None:
    """The one component this page may never grow back.

    ``gr.Audio(streaming=True)`` is served as HLS: every frame is re-encoded to AAC and handed
    over as its own segment, which is lossy and clicks at each join. The player takes the PCM
    over server-sent events instead, so no streaming audio component may reappear here.
    """
    import gradio as gr

    ui = demo.build_ui(demo.DemoSession(tts=FakeTTS()))

    assert not any(isinstance(block, gr.Audio) and block.streaming for block in ui.blocks.values())
    # The player is markup plus a load event; both have to be present or the page is inert.
    html = [block for block in ui.blocks.values() if isinstance(block, gr.HTML)]
    assert any("kova-status" in str(block.value) for block in html)


def test_the_player_is_wired_to_the_endpoint(demo: Any) -> None:
    script = demo.player_js(stream_path="/somewhere/else")

    assert '"/somewhere/else"' in script
    assert "window.kovaDemo" in script
    assert "AudioBufferSourceNode" in script or "createBufferSource" in script
    assert str(demo.MAX_CHARS) in script


def test_build_ui_survives_with_no_engine_at_all(demo: Any, monkeypatch: Any) -> None:
    """A machine with nothing configured still gets a page, and is told what to fix."""
    monkeypatch.setenv("KOVA_DISABLE_DOTENV", "1")
    monkeypatch.setenv("KOVA_MODEL_PATH", "/does/not/exist/kova")
    monkeypatch.delenv("KOVA_LORA_DIR", raising=False)
    from kova_tts import paths

    paths.reset_dotenv_cache()

    session = demo.DemoSession(loader=lambda: pytest.fail("must not load on page build"))
    ui = demo.build_ui(session)

    assert ui is not None
    notices = "\n".join(session.notices())
    assert "kova-tts paths" in notices
    assert "KOVA_LORA_DIR" in notices


def test_examples_are_original_prose(demo: Any) -> None:
    assert len(demo.EXAMPLES) >= 3
    assert all(prompt.strip().endswith((".", "?", "!")) for prompt in demo.EXAMPLES)


def test_the_app_serves_the_page_and_the_stream(demo: Any, client: Any) -> None:
    http, _session, _fake = client(FakeTTS(("alto",)))

    with http:
        page = http.get("/")
        schema = http.get("/openapi.json").json()

    assert page.status_code == 200
    assert "kova-clip" in page.text, "the finished-clip player is part of the page"
    assert demo.STREAM_PATH in schema["paths"], "the protocol the page speaks is documented"


# -------------------------------------------------------------------------- the demo: streaming


def test_the_stream_is_the_samples_the_model_made(demo: Any, client: Any) -> None:
    """The point of the endpoint: what the browser plays is bit-for-bit what the codec decoded.

    Nothing between :meth:`KovaTTS.stream` and the speakers may resample, re-encode or
    otherwise touch the audio, so the concatenated chunks must equal ``generate()`` exactly.
    """
    http, _session, fake = client()

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Hello there, this is a test.")

    assert stream.status == 200
    assert len(stream.chunks) == 4, "each decoded frame reaches the browser on its own"
    assert stream.pcm == to_pcm_bytes(fake.generate("Hello there, this is a test."))
    assert stream.done == {
        "chunks": 4,
        "samples": len(stream.pcm) // 2,
        "duration_seconds": round(len(stream.pcm) / 2 / OUTPUT_SAMPLE_RATE, 3),
        "sample_rate": OUTPUT_SAMPLE_RATE,
    }


def test_generating_three_times_in_a_row_works_every_time(demo: Any, client: Any) -> None:
    """Streaming has to work every time, not only on the first press of a page.

    Nothing about a run may be left behind on the server, so three requests over one connection
    have to produce three identical, complete streams. The browser half of the same guarantee is
    ``clearPlayback()``, which is what stops the *player* carrying state between runs.
    """
    http, _session, _fake = client()

    with http:
        runs = [speak(http, demo.STREAM_PATH, text="Say it again.") for _ in range(3)]

    assert [run.status for run in runs] == [200, 200, 200]
    assert all(run.done is not None and run.error is None for run in runs)
    assert len({run.pcm for run in runs}) == 1, "the same request must give the same audio"
    assert all(len(run.chunks) == 4 for run in runs)


def test_the_voice_reaches_the_engine(demo: Any, client: Any) -> None:
    http, _session, fake = client(FakeTTS(("alto",)))

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Say this.", voice="alto")

    assert stream.status == 200
    assert fake.calls[0]["voice"] == "alto"


def test_the_sampling_controls_reach_the_engine(demo: Any, client: Any) -> None:
    http, _session, fake = client()

    with http:
        speak(
            http,
            demo.STREAM_PATH,
            text="Hello.",
            sampling={"temperature": 0.7, "top_p": 0.85, "top_k": 30, "max_tokens": 512},
        )

    params = fake.calls[0]["params"]
    assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.85, 30)
    assert params.max_tokens == 512
    # Anything the request left out keeps the tuned preset rather than a library default.
    assert params.repetition_penalty == demo.TTS_SAMPLING.repetition_penalty


def test_a_cloned_voice_is_streamable_by_name(demo: Any, client: Any, reference_wav: Path) -> None:
    """Why the demo serves its own endpoint: a clone is an object, not a name on disk."""
    http, session, fake = client()
    session.clone_voice(str(reference_wav), "This is what the clip says.", "mine")

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Now say something new.", voice="mine")

    assert stream.status == 200 and stream.done is not None
    assert isinstance(fake.calls[0]["voice"], Voice)
    assert fake.calls[0]["voice"].name == "mine"
    # And a clone gets the cloning preset, exactly as the picker would have shown.
    assert fake.calls[0]["params"].max_tokens == demo.CLONE_SAMPLING.max_tokens


@pytest.fixture
def preset_dir(tmp_path: Path, reference_wav: Path) -> Path:
    """Two zero-shot presets in metadata.csv, plus rows and clips that must be ignored."""
    import shutil

    folder = tmp_path / "presets"
    folder.mkdir()
    for name in ("voice_01", "voice_02", "orphan"):
        shutil.copy(reference_wav, folder / f"{name}.wav")
    (folder / "metadata.csv").write_text(
        "file_name,text,seconds\n"
        "voice_02.wav,And a second.,3.0\n"
        'voice_01.wav,"The first preset speaks, with a comma.",3.0\n'
        "untranscribed.wav,,3.0\n"  # no text: skipped
        "missing.wav,Its audio is not there.,3.0\n",  # no file: skipped
        encoding="utf-8",
    )
    return folder


def test_presets_are_the_rows_with_audio_and_a_transcript(demo: Any, preset_dir: Path) -> None:
    presets = demo.load_presets(preset_dir)
    # orphan.wav has audio but no row; untranscribed has no text; missing has no audio.
    assert list(presets) == ["voice_01", "voice_02"]
    assert presets["voice_01"][1] == "The first preset speaks, with a comma."
    assert demo.load_presets(None) == {} and demo.load_presets(preset_dir / "missing") == {}


def test_every_source_offers_its_own_voices(
    demo: Any, preset_dir: Path, reference_wav: Path
) -> None:
    session = demo.DemoSession(tts=FakeTTS(("alto",)), zero_shot_dir=preset_dir)
    session.clone_voice(str(reference_wav), "Some words.", "mine")

    assert session.sources() == [
        demo.SOURCE_BASE,
        demo.SOURCE_PRESET,
        demo.SOURCE_CLONE,
        demo.SOURCE_LORA,
    ]
    # Professional cloning stays on the switch with nothing installed, so it never changes shape.
    assert demo.DemoSession(tts=FakeTTS(())).sources() == [
        demo.SOURCE_BASE,
        demo.SOURCE_CLONE,
        demo.SOURCE_LORA,
    ]
    assert session.choices(demo.SOURCE_BASE) == [(demo.BASE_LABEL, demo.BASE_VOICE)]
    assert [v for _, v in session.choices(demo.SOURCE_LORA)] == ["alto"]
    assert [v for _, v in session.choices(demo.SOURCE_PRESET)] == [
        demo.PRESET_PREFIX + "voice_01",
        demo.PRESET_PREFIX + "voice_02",
    ]
    assert [v for _, v in session.choices(demo.SOURCE_CLONE)] == ["mine"]
    for value, source in (
        ("", demo.SOURCE_BASE),
        ("alto", demo.SOURCE_LORA),
        (demo.PRESET_PREFIX + "voice_02", demo.SOURCE_PRESET),
        ("mine", demo.SOURCE_CLONE),
    ):
        assert session.source_of(value) == source
    # The preview shows the preset's own clip and its exact transcript.
    audio, text = session.preset_reference(demo.PRESET_PREFIX + "voice_02")
    assert audio.endswith("voice_02.wav") and text == "And a second."


def test_a_preset_is_cloned_once_then_reused(demo: Any, client: Any, preset_dir: Path) -> None:
    http, _session, fake = client(zero_shot_dir=preset_dir)
    value = demo.PRESET_PREFIX + "voice_01"

    with http:
        first = speak(http, demo.STREAM_PATH, text="Say this.", voice=value)
        second = speak(http, demo.STREAM_PATH, text="And this.", voice=value)

    assert first.status == second.status == 200
    # Encoded from its transcript, not transcribed, and only on first use.
    assert (
        len(fake.cloned) == 1
        and fake.cloned[0]["transcript"] == "The first preset speaks, with a comma."
    )
    assert all(isinstance(call["voice"], Voice) for call in fake.calls)
    assert fake.calls[0]["params"].max_tokens == demo.CLONE_SAMPLING.max_tokens


def test_an_unknown_preset_is_refused(demo: Any, client: Any, preset_dir: Path) -> None:
    http, _session, fake = client(zero_shot_dir=preset_dir)
    with http:
        stream = speak(http, demo.STREAM_PATH, text="Say this.", voice=demo.PRESET_PREFIX + "nope")
    assert stream.status == 422 and not fake.calls


def test_the_bundled_presets_all_have_transcripts(demo: Any) -> None:
    presets = demo.load_presets(demo.DEFAULT_ZERO_SHOT_DIR)
    assert len(presets) == 50
    assert all(path.suffix == ".flac" and text for path, text in presets.values())


def test_preset_follows_the_voice(demo: Any, reference_wav: Path) -> None:
    from kova_tts import CLONE_SAMPLING, TTS_SAMPLING

    session = demo.DemoSession(tts=FakeTTS(("alto",)))
    assert session.preset("alto") == TTS_SAMPLING
    assert session.preset(demo.BASE_VOICE) == TTS_SAMPLING

    session.clone_voice(str(reference_wav), "Some words.", "mine")
    assert session.preset("mine") == CLONE_SAMPLING
    # The two presets differ only in token budget today; the sliders have to reach both.
    assert TTS_SAMPLING.max_tokens != CLONE_SAMPLING.max_tokens


# ------------------------------------------------------------------- the demo: the sad paths


def test_empty_text_is_a_sentence_not_an_exception(demo: Any, client: Any) -> None:
    http, _session, fake = client()

    with http:
        stream = speak(http, demo.STREAM_PATH, text="   ")

    assert stream.status == 422
    assert "empty" in stream.payloads[0]["message"]
    assert not fake.calls


def test_text_over_the_limit_is_refused_politely(demo: Any, client: Any) -> None:
    http, _session, fake = client()

    with http:
        stream = speak(http, demo.STREAM_PATH, text="word " * demo.MAX_CHARS)

    assert stream.status == 422
    assert "characters" in stream.payloads[0]["message"]
    assert not fake.calls


def test_no_voices_installed_still_offers_the_base_voice(demo: Any, client: Any) -> None:
    http, session, _fake = client(FakeTTS(()))

    assert session.choices() == [(demo.BASE_LABEL, demo.BASE_VOICE)]
    assert session.resolve(demo.BASE_VOICE) is None
    assert "No LoRA voices installed" in "\n".join(session.notices())
    with http:
        assert speak(http, demo.STREAM_PATH, text="Hello.").done is not None


def test_a_second_generation_is_told_to_wait(demo: Any, client: Any, monkeypatch: Any) -> None:
    """An overlapping request has to be a sentence with a status code, not a traceback."""
    import session as demo_session  # importable because the demo put its own directory on the path

    monkeypatch.setattr(demo_session, "BUSY_TIMEOUT", 0.05)
    http, session, _fake = client()

    session._lock.acquire()  # stand in for a generation already in flight
    try:
        with http:
            stream = speak(http, demo.STREAM_PATH, text="The second one.")
    finally:
        session._lock.release()

    assert stream.status == 409
    assert stream.payloads[0] == {"error": "busy", "message": demo.BUSY}

    # And the engine is free again the moment the first one lets go.
    with http:
        assert speak(http, demo.STREAM_PATH, text="And now?").done is not None


def test_missing_weights_point_at_the_paths_command(demo: Any, client: Any) -> None:
    from fastapi.testclient import TestClient

    def loader() -> Any:
        raise MissingArtifact("Model directory not found at /nowhere.")

    session = demo.DemoSession(loader=loader)
    with TestClient(demo.build_app(session)) as http:
        stream = speak(http, demo.STREAM_PATH, text="Hello.")

    assert stream.status == 503
    assert "kova-tts paths" in stream.payloads[0]["message"]
    assert "/nowhere" in stream.payloads[0]["message"]


def test_generation_failure_arrives_as_an_error_event(demo: Any, client: Any) -> None:
    """Once the stream is open the status code is gone, so a failure is a terminal event."""
    http, _session, _fake = client(FakeTTS(fail=RuntimeError("CUDA out of memory")))

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Hello.")

    assert stream.status == 200 and stream.done is None
    assert stream.error is not None
    assert "CUDA out of memory" in stream.error["message"]


def test_an_overlapping_engine_is_reported_as_busy(demo: Any, client: Any) -> None:
    """The engine's own reentrancy guard, reached by a server sharing this process."""
    boom = RuntimeError("This generator is already running a request.")
    http, _session, _fake = client(FakeTTS(fail=boom))

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Hello.")

    assert stream.error is not None and stream.error["message"] == demo.BUSY


def test_a_failed_generation_still_releases_the_engine(demo: Any, client: Any) -> None:
    http, session, _fake = client(FakeTTS(fail=RuntimeError("nope")))

    with http:
        speak(http, demo.STREAM_PATH, text="Hello.")

    assert not session._lock.locked(), "a failure must not leave the model reserved"


def test_silence_from_the_model_is_a_stream_with_no_chunks(demo: Any, client: Any) -> None:
    """The player says so; the protocol just reports an honest zero."""
    http, _session, _fake = client(FakeTTS(silent=True))

    with http:
        stream = speak(http, demo.STREAM_PATH, text="Hello.")

    assert stream.chunks == []
    assert stream.done == {
        "chunks": 0,
        "samples": 0,
        "duration_seconds": 0.0,
        "sample_rate": OUTPUT_SAMPLE_RATE,
    }


# ------------------------------------------------------------------------- the demo: cloning


def test_cloning_adds_a_usable_voice(demo: Any, reference_wav: Path) -> None:
    fake = FakeTTS(("alto",))
    session = demo.DemoSession(tts=fake)

    message, name = session.clone_voice(str(reference_wav), "This is what the clip says.", "mine")

    assert name == "mine"
    assert "Cloned **mine**" in message
    assert ("mine (cloned)", "mine") in session.choices()
    assert isinstance(session.resolve("mine"), Voice)


def test_cloned_names_do_not_collide(demo: Any, reference_wav: Path) -> None:
    session = demo.DemoSession(tts=FakeTTS(("mine",)))

    _, first = session.clone_voice(str(reference_wav), "Words.", "mine")
    _, second = session.clone_voice(str(reference_wav), "Words.", "mine")

    assert first == "mine-2" and second == "mine-3"


def test_cloning_without_a_reference_says_so(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS())

    message, name = session.clone_voice(None, "", "")

    assert name is None and "Upload a recording" in message


def test_a_too_short_reference_is_refused(demo: Any, tmp_path: Path) -> None:
    from kova_tts.audio import save_wav

    clip = save_wav(tmp_path / "short.wav", sine(0.3), SAMPLE_RATE)
    session = demo.DemoSession(tts=FakeTTS())

    message, name = session.clone_voice(str(clip), "Too short.", "")

    assert name is None and "0.3 s" in message


def test_a_silent_reference_is_refused(demo: Any, tmp_path: Path) -> None:
    from kova_tts.audio import save_wav

    clip = save_wav(tmp_path / "silent.wav", np.zeros(SAMPLE_RATE * 3, np.float32), SAMPLE_RATE)
    session = demo.DemoSession(tts=FakeTTS())

    message, name = session.clone_voice(str(clip), "Nothing here.", "")

    assert name is None and "silent" in message


def test_cloning_while_the_model_is_speaking_waits_its_turn(demo: Any, reference_wav: Path) -> None:
    """The clone tab and the streaming endpoint share one lock, because they share one model."""
    session = demo.DemoSession(tts=FakeTTS())
    session._lock.acquire()
    try:
        message, name = session.clone_voice(str(reference_wav), "Words.", "mine")
    finally:
        session._lock.release()

    assert name is None and message == demo.BUSY


def test_cloning_without_asr_asks_for_the_transcript(demo: Any, reference_wav: Path) -> None:
    session = demo.DemoSession(tts=FakeTTS(transcriber=None))

    message, name = session.clone_voice(str(reference_wav), "", "")

    assert name is None and "type exactly what" in message


def test_cloning_transcribes_when_it_can(demo: Any, reference_wav: Path) -> None:
    session = demo.DemoSession(tts=FakeTTS(transcriber=lambda path: "Heard by the transcriber."))

    message, name = session.clone_voice(str(reference_wav), "", "auto")

    assert name == "auto"
    assert "transcribed automatically" in message
    assert "Heard by the transcriber." in message


def test_a_missing_asr_dependency_is_explained(demo: Any, reference_wav: Path, monkeypatch) -> None:
    """Parakeet needs librosa from the demo extra; its absence has to read like a suggestion."""

    def unavailable(*_args: Any, **_options: Any) -> Any:
        raise ImportError("ParakeetFeatureExtractor requires the librosa library.")

    monkeypatch.setattr(sys.modules["session"], "load_parakeet", unavailable)
    session = demo.DemoSession(tts=FakeTTS(transcriber=demo.build_transcriber()))

    message, name = session.clone_voice(str(reference_wav), "", "")

    assert name is None
    assert "librosa" in message and "type the transcript" in message


def test_a_recording_is_transcribed_the_moment_it_arrives(demo: Any, reference_wav: Path) -> None:
    heard: list[str] = []

    def transcriber(path: str) -> str:
        heard.append(path)
        return "What the recording says."

    # No engine at all: the clone tab must fill the box without loading the TTS model.
    session = demo.DemoSession(loader=lambda: pytest.fail("must not load"), transcriber=transcriber)

    status, text = session.transcribe_reference(str(reference_wav))

    assert text == "What the recording says." and heard == [str(reference_wav)]
    assert "Transcribed automatically" in status
    assert session.transcribe_reference(None) == ("", None)


def test_a_recording_that_never_arrived_says_so(demo: Any, tmp_path: Path) -> None:
    """The upload failing (e.g. an unwritable temp dir) must not read as 'nothing recorded'."""
    session = demo.DemoSession(transcriber=lambda path: pytest.fail("nothing to transcribe"))

    status, text = session.transcribe_reference(str(tmp_path / "gone.wav"))

    assert text is None and "did not reach the server" in status and "GRADIO_TEMP_DIR" in status


def test_clone_uses_the_page_transcriber_for_an_empty_box(demo: Any, reference_wav: Path) -> None:
    fake = FakeTTS(transcriber=None)
    session = demo.DemoSession(tts=fake, transcriber=lambda path: "Heard on the page.")

    message, name = session.clone_voice(str(reference_wav), "", "mine")

    assert name == "mine" and fake.cloned[0]["transcript"] == "Heard on the page."
    assert "transcribed automatically" in message


def test_uploads_go_to_a_directory_this_user_owns(demo: Any) -> None:
    """A /tmp/gradio created by another user made every upload fail with PermissionError."""
    import os

    temp = Path(os.environ["GRADIO_TEMP_DIR"])
    assert temp != Path("/tmp/gradio")
    temp.mkdir(parents=True, exist_ok=True)
    assert os.access(temp, os.W_OK)


# ------------------------------------------------------------------------------ the node pack


def test_the_pack_exports_what_comfyui_reads(comfy: Any) -> None:
    classes = comfy.NODE_CLASS_MAPPINGS
    names = comfy.NODE_DISPLAY_NAME_MAPPINGS

    assert set(classes) == set(names)
    assert all(isinstance(key, str) and key for key in classes)
    assert all(isinstance(value, str) and value for value in names.values())
    assert {"KovaTTSLoader", "KovaTTSGenerate", "KovaTTSCloneVoice"} == set(classes)


def test_every_node_follows_the_conventions(comfy: Any) -> None:
    for key, node in comfy.NODE_CLASS_MAPPINGS.items():
        spec = node.INPUT_TYPES()
        assert isinstance(spec, dict) and "required" in spec, key
        for section in spec.values():
            for field, declaration in section.items():
                assert isinstance(declaration, tuple), f"{key}.{field}"
                assert isinstance(declaration[0], (str, list)), f"{key}.{field}"
        assert isinstance(node.RETURN_TYPES, tuple) and node.RETURN_TYPES, key
        assert len(node.RETURN_NAMES) == len(node.RETURN_TYPES), key
        assert callable(getattr(node, node.FUNCTION)), key
        assert node.CATEGORY, key


def test_comfyui_is_never_imported(comfy: Any) -> None:
    """The pack has to import on a machine that has kova-tts and no ComfyUI."""
    assert "comfy" not in sys.modules
    assert "folder_paths" not in sys.modules


def test_the_loader_returns_one_engine_and_keeps_it(comfy: Any, monkeypatch: Any) -> None:
    import kova_tts

    built: list[dict[str, Any]] = []

    class Factory:
        @staticmethod
        def from_pretrained(model=None, **options: Any) -> FakeTTS:
            built.append({"model": model, **options})
            return FakeTTS()

    monkeypatch.setattr(kova_tts, "KovaTTS", Factory, raising=False)
    comfy.nodes._ENGINES.clear()
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSLoader"]()

    first = node.load(device="cpu", precision="float32", model="", codec="", lora_dir="")
    second = node.load(device="cpu", precision="float32", model="", codec="", lora_dir="")

    assert len(first) == len(node.RETURN_TYPES) == 1
    assert first[0] is second[0], "the model must survive across executions"
    assert len(built) == 1
    # An empty path widget means "resolve it the usual way", not "look for a file called ''".
    assert built[0]["model"] is None and built[0]["codec"] is None
    assert built[0]["transcriber"] is not None
    comfy.nodes._ENGINES.clear()


def test_generate_returns_the_audio_it_declares(comfy: Any) -> None:
    import torch

    fake = FakeTTS()
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"]()

    result = node.generate(fake, "Speak this sentence.", voice_name="alto")

    assert len(result) == len(node.RETURN_TYPES) == 1
    audio = result[0]
    assert set(audio) == {"waveform", "sample_rate"}
    assert isinstance(audio["waveform"], torch.Tensor)
    assert audio["waveform"].ndim == 3 and audio["waveform"].shape[:2] == (1, 1)
    assert audio["sample_rate"] == OUTPUT_SAMPLE_RATE
    assert fake.calls[0]["voice"] == "alto"


def test_generate_runs_again_every_time_it_is_queued(comfy: Any) -> None:
    """Every generation is a fresh take, so ComfyUI must never answer from its cache."""
    changed = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"].IS_CHANGED(text="Same words.")
    assert changed != changed  # NaN: never equal to what ComfyUI saw last time


def test_generate_prefers_a_connected_voice(comfy: Any) -> None:
    fake = FakeTTS()
    voice = Voice(name="cloned", ref_codes=(5, 6, 7), ref_text="Some words.")
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"]()

    node.generate(fake, "Speak this.", voice=voice, voice_name="ignored")

    assert fake.calls[0]["voice"] is voice


def test_generate_refuses_empty_text(comfy: Any) -> None:
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"]()

    with pytest.raises(ValueError, match="empty"):
        node.generate(FakeTTS(), "   ")


def test_clone_returns_the_voice_it_declares(comfy: Any) -> None:
    fake = FakeTTS()
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSCloneVoice"]()
    audio = comfy.nodes.to_comfy_audio(sine(3.0), SAMPLE_RATE)

    result = node.clone(fake, audio, "mine", "This is what the clip says.")

    assert len(result) == len(node.RETURN_TYPES) == 1
    assert isinstance(result[0], Voice) and result[0].name == "mine"
    # The waveform reaches the engine as numpy, never as a ComfyUI dict or a tensor.
    assert isinstance(fake.cloned[0]["audio"], np.ndarray)


def test_clone_hands_the_engine_the_clip_at_its_own_rate(comfy: Any) -> None:
    """Not at the model's 48 kHz, and not forced to 32 kHz either: the engine decides what the
    encoder gets, and a 16 kHz clip is encoded natively when the encoder allows it."""
    fake = FakeTTS()
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSCloneVoice"]()
    audio = comfy.nodes.to_comfy_audio(sine(3.0, rate=16_000), 16_000)

    node.clone(fake, audio, "mine", "This is what the clip says.")

    assert fake.cloned[0]["sample_rate"] == 16_000
    assert fake.cloned[0]["audio"].size == 3 * 16_000


def test_clone_writes_a_file_when_it_has_to_transcribe(comfy: Any) -> None:
    seen: list[str] = []

    def transcriber(path: str) -> str:
        seen.append(path)
        assert Path(path).is_file(), "the transcriber is handed a real file"
        return "Transcribed."

    fake = FakeTTS(transcriber=transcriber)
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSCloneVoice"]()

    voice = node.clone(fake, comfy.nodes.to_comfy_audio(sine(3.0)), "mine")[0]

    assert voice.ref_text == "Transcribed."
    assert seen and not Path(seen[0]).exists(), "the temporary file is cleaned up"


def test_clone_without_a_transcript_or_asr_says_what_to_do(comfy: Any) -> None:
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSCloneVoice"]()

    with pytest.raises(ValueError, match="transcript"):
        node.clone(FakeTTS(transcriber=None), comfy.nodes.to_comfy_audio(sine(3.0)), "mine")


# -------------------------------------------------------------------- the node pack: AUDIO type


def test_audio_round_trips(comfy: Any) -> None:
    original = sine(1.0)

    restored = comfy.audio.from_comfy_audio(comfy.audio.to_comfy_audio(original, SAMPLE_RATE))

    assert restored.dtype == np.float32
    assert np.allclose(restored, original)


def test_audio_is_downmixed_and_resampled(comfy: Any) -> None:
    import torch

    stereo = torch.stack([torch.ones(16_000), torch.full((16_000,), -0.5)]).reshape(1, 2, -1)

    mono = comfy.audio.from_comfy_audio({"waveform": stereo, "sample_rate": 16_000})

    assert mono.ndim == 1
    assert abs(mono.size - 32_000) <= 2, "one second at 16 kHz is one second at 32 kHz"
    assert np.allclose(np.mean(mono[1000:-1000]), 0.25, atol=1e-3)


def test_audio_batches_take_the_first_item(comfy: Any) -> None:
    import torch

    batch = torch.stack([torch.full((1, 8000), 0.5), torch.full((1, 8000), -0.5)])

    first = comfy.audio.from_comfy_audio({"waveform": batch, "sample_rate": SAMPLE_RATE})

    assert np.allclose(first, 0.5)


def test_a_non_audio_input_says_what_was_expected(comfy: Any) -> None:
    with pytest.raises(ValueError, match="Load Audio"):
        comfy.audio.from_comfy_audio({"samples": []})
