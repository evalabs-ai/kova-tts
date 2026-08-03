"""The demo and the ComfyUI pack, on a CPU with no weights.

Both live in ``apps/``, outside the installed package, because neither is a library: the demo
is a program and the node pack is a directory that gets linked into ``custom_nodes``. They are
loaded here by path, exactly the way their real callers load them -- ``kova-tts demo`` reaches
the demo through its file, and ComfyUI imports the node directory as a package.

Everything runs against :class:`FakeTTS`, which has the surface of
:class:`~kova_tts.engine.tts.KovaTTS` and a sine wave where the model would be. That is enough
to check what actually breaks in these two files: the streaming callback yielding progressively,
the states a first-time user lands in, and the ComfyUI type conversion at the boundary.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kova_codec.constants import SAMPLE_RATE
from kova_tts import AudioFrame, MissingArtifact, Voice

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
    and the seed through rather than only that it produced audio.
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
        self.sample_rate = SAMPLE_RATE
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

    def stream(self, text, voice=None, *, params=None, seed=None):
        self.calls.append({"text": text, "voice": voice, "params": params, "seed": seed})
        if self._fail is not None:
            raise self._fail
        if self._silent:
            yield AudioFrame(np.zeros(0, dtype=np.float32), self.sample_rate, is_final=True)
            return
        for index in range(self._frames):
            yield AudioFrame(sine(0.2 * (index + 1)), self.sample_rate)
        yield AudioFrame(np.zeros(0, dtype=np.float32), self.sample_rate, is_final=True)

    def generate(self, text, voice=None, *, params=None, seed=None) -> np.ndarray:
        pieces = [frame.samples for frame in self.stream(text, voice, params=params, seed=seed)]
        return np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.float32)

    def clone(self, audio, transcript=None, *, name=None) -> Voice:
        self.cloned.append({"audio": audio, "transcript": transcript, "name": name})
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


# --------------------------------------------------------------------------------- the demo: UI


def test_build_ui_constructs_without_launching(demo: Any) -> None:
    import gradio as gr

    fake = FakeTTS(("alto", "tenor"))
    ui = demo.build_ui(demo.DemoSession(tts=fake))

    assert isinstance(ui, gr.Blocks)
    dropdowns = [block for block in ui.blocks.values() if isinstance(block, gr.Dropdown)]
    assert [choice[1] for choice in dropdowns[0].choices] == [demo.BASE_VOICE, "alto", "tenor"]
    # The whole point of the page: an audio output that plays while it is still being written.
    assert any(isinstance(block, gr.Audio) and block.streaming for block in ui.blocks.values())
    # Building the page must not generate anything -- cached examples would do exactly that,
    # and on a real engine it is a minute of synthesis before the first visitor arrives.
    assert not fake.calls


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


# -------------------------------------------------------------------------- the demo: streaming


def test_speak_streams_progressively(demo: Any) -> None:
    fake = FakeTTS(frames=4)
    session = demo.DemoSession(tts=fake)

    steps = list(session.speak("Hello there, this is a test.", demo.BASE_VOICE))

    chunks = [step[0] for step in steps if isinstance(step[0], tuple)]
    assert len(chunks) == 4, "each decoded frame should reach the browser on its own"
    assert all(rate == SAMPLE_RATE and samples.dtype == np.int16 for rate, samples in chunks)
    # Audio arrives before the finished clip does, which is what "streaming" has to mean.
    assert not any(isinstance(step[1], tuple) for step in steps[:-1])

    rate, whole = steps[-1][1]
    assert rate == SAMPLE_RATE
    assert whole.size == sum(samples.size for _, samples in chunks)
    assert "First audio in" in steps[-1][2] and "real time" in steps[-1][2]


def test_speak_passes_the_voice_and_a_concrete_seed(demo: Any) -> None:
    fake = FakeTTS(("alto",))
    session = demo.DemoSession(tts=fake)

    steps = list(session.speak("Say this.", "alto", seed=4321))

    assert fake.calls[0]["voice"] == "alto"
    assert fake.calls[0]["seed"] == 4321
    assert "4321" in steps[-1][2]

    # A negative seed means "draw one", but the drawn seed is still reported, so the result
    # can be reproduced by typing it back in.
    list(session.speak("Again.", "alto", seed=-1))
    assert isinstance(fake.calls[1]["seed"], int) and fake.calls[1]["seed"] >= 0


def test_speak_uses_the_sampling_controls(demo: Any) -> None:
    fake = FakeTTS()
    session = demo.DemoSession(tts=fake)

    list(session.speak("Hello.", demo.BASE_VOICE, 0.7, 0.85, 30, 1.2, 512, 7))

    params = fake.calls[0]["params"]
    assert (params.temperature, params.top_p, params.top_k) == (0.7, 0.85, 30)
    assert (params.repetition_penalty, params.max_tokens) == (1.2, 512)


def test_preset_follows_the_voice(demo: Any) -> None:
    from kova_tts import CLONE_SAMPLING, TTS_SAMPLING

    session = demo.DemoSession(tts=FakeTTS(("alto",)))
    assert session.preset("alto") == TTS_SAMPLING
    assert session.preset(demo.BASE_VOICE) == TTS_SAMPLING

    session.clone_voice(str(_write_reference(session)), "Some words.", "mine")
    assert session.preset("mine") == CLONE_SAMPLING


def _write_reference(session: Any) -> Path:
    import tempfile

    from kova_tts.audio import save_wav

    directory = Path(tempfile.mkdtemp(prefix="kova-test-"))
    return save_wav(directory / "clip.wav", sine(3.0), SAMPLE_RATE)


# ------------------------------------------------------------------- the demo: the sad paths


def test_empty_text_is_a_sentence_not_an_exception(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS())

    steps = list(session.speak("   ", demo.BASE_VOICE))

    assert len(steps) == 1
    assert steps[0][0] is demo.CLEAR and steps[0][1] is demo.CLEAR
    assert "Type something" in steps[0][2]


def test_text_over_the_limit_is_refused_politely(demo: Any) -> None:
    fake = FakeTTS()
    session = demo.DemoSession(tts=fake)

    steps = list(session.speak("word " * demo.MAX_CHARS, demo.BASE_VOICE))

    assert not fake.calls
    assert "characters" in steps[-1][2]


def test_no_voices_installed_still_offers_the_base_voice(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS(()))

    assert session.choices() == [(demo.BASE_LABEL, demo.BASE_VOICE)]
    assert session.resolve(demo.BASE_VOICE) is None
    assert "No LoRA voices installed" in "\n".join(session.notices())
    assert list(session.speak("Hello.", demo.BASE_VOICE))[-1][1] is not None


def test_a_second_generation_is_told_to_wait(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS())

    running = session.speak("The first one.", demo.BASE_VOICE)
    next(running)  # takes the engine
    try:
        steps = list(session.speak("The second one.", demo.BASE_VOICE))
    finally:
        running.close()

    assert len(steps) == 1
    assert steps[0][0] is None and steps[0][1] is None, "a busy page must not clear the player"
    assert steps[0][2] == demo.BUSY

    # Closing the first generator releases the engine again.
    assert list(session.speak("And now?", demo.BASE_VOICE))[-1][1] is not None


def test_missing_weights_point_at_the_paths_command(demo: Any) -> None:
    def loader() -> Any:
        raise MissingArtifact("Model directory not found at /nowhere.")

    session = demo.DemoSession(loader=loader)

    steps = list(session.speak("Hello.", demo.BASE_VOICE))

    assert "kova-tts paths" in steps[-1][2]
    assert "/nowhere" in steps[-1][2]


def test_generation_failure_is_reported_not_raised(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS(fail=RuntimeError("CUDA out of memory")))

    steps = list(session.speak("Hello.", demo.BASE_VOICE))

    assert "CUDA out of memory" in steps[-1][2]


def test_an_overlapping_engine_is_reported_as_busy(demo: Any) -> None:
    """The engine's own reentrancy guard, reached by a server sharing this process."""
    boom = RuntimeError("This generator is already running a request.")
    session = demo.DemoSession(tts=FakeTTS(fail=boom))

    assert list(session.speak("Hello.", demo.BASE_VOICE))[-1][2] == demo.BUSY


def test_silence_from_the_model_is_explained(demo: Any) -> None:
    session = demo.DemoSession(tts=FakeTTS(silent=True))

    steps = list(session.speak("Hello.", demo.BASE_VOICE))

    assert "no audio" in steps[-1][2]


# ------------------------------------------------------------------------- the demo: cloning


def test_cloning_adds_a_usable_voice(demo: Any, reference_wav: Path) -> None:
    fake = FakeTTS(("alto",))
    session = demo.DemoSession(tts=fake)

    message, name = session.clone_voice(str(reference_wav), "This is what the clip says.", "mine")

    assert name == "mine"
    assert "Cloned **mine**" in message
    assert ("mine (cloned)", "mine") in session.choices()

    list(session.speak("Now say something new.", "mine"))
    assert isinstance(fake.calls[0]["voice"], Voice)
    assert fake.calls[0]["voice"].name == "mine"


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


def test_the_missing_data_extra_is_explained(demo: Any, reference_wav: Path, monkeypatch) -> None:
    """The ``data`` extra is optional, so its absence has to read like a suggestion."""
    from kova_tts.data import asr

    def unavailable(**_options: Any) -> Any:
        raise asr.MissingDependency("Transcription needs faster-whisper.")

    monkeypatch.setattr(asr, "load_transcriber", unavailable)
    session = demo.DemoSession(tts=FakeTTS(transcriber=demo.build_transcriber()))

    message, name = session.clone_voice(str(reference_wav), "", "")

    assert name is None
    assert "faster-whisper" in message and "type the transcript" in message


# ------------------------------------------------------------------------- the demo: plumbing


def test_updates_translate_the_sentinels(demo: Any) -> None:
    import gradio as gr

    cleared, untouched, value = demo._updates((demo.CLEAR, None, (SAMPLE_RATE, np.zeros(4))))

    assert cleared is None, "CLEAR empties the component"
    assert isinstance(untouched, dict) and not set(untouched) - {"__type__"}
    assert isinstance(value, tuple)
    assert isinstance(gr.update(), dict)


def test_pcm16_is_clipped_not_wrapped(demo: Any) -> None:
    loud = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)

    converted = demo._pcm16(loud)

    assert converted.dtype == np.int16
    assert converted.tolist() == [-32767, -32767, 0, 32767, 32767]


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

    result = node.generate(fake, "Speak this sentence.", seed=11, voice_name="alto")

    assert len(result) == len(node.RETURN_TYPES) == 1
    audio = result[0]
    assert set(audio) == {"waveform", "sample_rate"}
    assert isinstance(audio["waveform"], torch.Tensor)
    assert audio["waveform"].ndim == 3 and audio["waveform"].shape[:2] == (1, 1)
    assert audio["sample_rate"] == SAMPLE_RATE
    assert fake.calls[0]["voice"] == "alto" and fake.calls[0]["seed"] == 11


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
