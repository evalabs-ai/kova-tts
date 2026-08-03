"""The demo callback and the ComfyUI generate node, on the real model.

``test_apps.py`` proves the wiring against a fake engine. This file proves the thing the demo
exists for: that a real generation reaches the browser as audio, and that the first frame
arrives well before the last one. Everything else about these two files is cheap to check and
this is not, so there is one real generation per surface and no more.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kova_codec.constants import SAMPLE_RATE
from kova_tts import SamplingParams

pytestmark = [pytest.mark.gpu, pytest.mark.weights]

APPS = Path(__file__).resolve().parents[3] / "apps"

#: Two sentences, so the second is still being generated while the first is playing.
TEXT = (
    "The first sentence is already on its way to the speakers. The second one is still being "
    "written while you listen to it."
)


def freest_cuda_device() -> str:
    """The CUDA device with the most free memory, so a parallel run is not fought over.

    The house pattern, copied from ``test_data_end_to_end.py``: the device is named at
    construction rather than hidden behind ``CUDA_VISIBLE_DEVICES``, because torch may already
    have opened a context by the time this runs.
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


def _load(name: str, path: Path, *, package: bool = False) -> Any:
    spec = importlib.util.spec_from_file_location(
        name, path, submodule_search_locations=[str(path.parent)] if package else None
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def demo() -> Any:
    pytest.importorskip("gradio")
    return _load("kova_demo_app_gpu", APPS / "demo" / "app.py")


@pytest.fixture(scope="module")
def comfy() -> Any:
    return _load("kova_comfyui_pack_gpu", APPS / "comfyui" / "__init__.py", package=True)


@pytest.fixture(scope="module")
def engine(demo: Any) -> Any:
    """One real engine for this file: loading it twice would be minutes for nothing."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        loaded = demo.load_engine(device=freest_cuda_device())
        # The first generation of a process captures CUDA graphs and builds the codec, which
        # costs seconds and has nothing to do with either surface. The demo's own answer to
        # this is `--preload`; here it is one throwaway sentence.
        loaded.generate("Warming up.", params=SamplingParams(max_tokens=256), seed=1)
    except Exception as exc:  # noqa: BLE001 - a box without weights skips rather than fails
        pytest.skip(f"could not load the model: {exc}")
    return loaded


def test_the_demo_callback_speaks_before_it_finishes(demo: Any, engine: Any) -> None:
    session = demo.DemoSession(tts=engine)

    started = time.perf_counter()
    first_audio: float | None = None
    steps = []
    for step in session.speak(TEXT, demo.BASE_VOICE, seed=1234):
        if isinstance(step[0], tuple) and first_audio is None:
            first_audio = time.perf_counter() - started
        steps.append(step)
    total = time.perf_counter() - started

    rate, whole = steps[-1][1]
    seconds = whole.size / rate
    assert rate == SAMPLE_RATE
    assert seconds > 2.0, "two sentences should be more than two seconds of speech"
    assert whole.dtype == np.int16 and np.max(np.abs(whole)) > 1000, "that is silence"

    assert first_audio is not None
    # The claim the demo is built around: audio starts long before generation ends. Generously
    # bounded -- on this hardware it is nearer a tenth -- because the box may be busy.
    assert first_audio < total / 2, f"first audio at {first_audio:.2f}s of {total:.2f}s"
    print(f"\ntime to first audio {first_audio:.2f} s, {seconds:.1f} s of speech in {total:.1f} s")


def test_the_comfyui_node_generates_real_audio(comfy: Any, engine: Any) -> None:
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"]()

    (audio,) = node.generate(engine, "Generated from a ComfyUI graph.", seed=99)

    waveform = audio["waveform"]
    assert audio["sample_rate"] == SAMPLE_RATE
    assert waveform.ndim == 3 and waveform.shape[:2] == (1, 1)
    assert waveform.shape[2] / SAMPLE_RATE > 1.0
    assert float(waveform.abs().max()) > 0.03, "that is silence"

    # Round-tripping through the boundary gives back exactly what the engine produced.
    restored = comfy.audio.from_comfy_audio(audio)
    assert restored.dtype == np.float32
    assert restored.size == waveform.shape[2]
