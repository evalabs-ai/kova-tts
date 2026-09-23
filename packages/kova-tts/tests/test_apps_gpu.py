"""The demo's streaming endpoint and the ComfyUI generate node, on the real model.

``test_apps.py`` proves the wiring against a fake engine. This file proves the two things the
demo exists for and only a real generation can show: that audio reaches the browser long before
the utterance is finished, and that what reaches it is the model's own samples -- not a
re-encode of them. Everything else about these two files is cheap to check and this is not, so
there is one real generation per surface, plus the repeat that proves the page is not one-shot.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kova_codec.constants import OUTPUT_SAMPLE_RATE
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


@pytest.fixture(scope="module")
def demo_url(demo: Any, engine: Any) -> Any:
    """The demo served by a real uvicorn, on a free port, for the life of this file.

    Not ``TestClient``: it collects a streaming response before handing it over, so every chunk
    appears to arrive at once and the one measurement this file exists to make -- how long until
    the first frame -- comes out equal to the whole generation. A socket does not lie about that.
    """
    import socket
    import threading

    import uvicorn

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    session = demo.DemoSession(tts=engine)
    server = uvicorn.Server(
        uvicorn.Config(demo.build_app(session), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 30
    while not server.started:
        assert thread.is_alive() and time.time() < deadline, "the demo server never started"
        time.sleep(0.05)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=10)


def snr_db(reference: np.ndarray, other: np.ndarray) -> float:
    """Signal-to-noise ratio of `other` against `reference`, in dB.

    The measure that separates the two failure modes: a re-encode in the path shows up here as
    tens of dB, while run-to-run CUDA noise does not.
    """
    error = other[: reference.size] - reference[: other.size]
    return float(10 * np.log10(np.sum(reference**2) / max(float(np.sum(error**2)), 1e-20)))


def stream_once(url: str, **body: Any) -> dict[str, Any]:
    """One run of the streaming endpoint, timed and reassembled the way the player does it."""
    import httpx

    chunks: list[bytes] = []
    done: dict[str, Any] | None = None
    started = time.perf_counter()
    first_audio: float | None = None
    with httpx.Client(timeout=300) as client, client.stream("POST", url, json=body) as response:
        assert response.status_code == 200, response.read()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            payload = json.loads(line[5:].strip())
            if "audio" in payload:
                if first_audio is None:
                    first_audio = time.perf_counter() - started
                chunks.append(base64.b64decode(payload["audio"]))
            elif "chunks" in payload:
                done = payload
            else:  # pragma: no cover - a real failure should fail the test loudly
                raise AssertionError(f"the stream failed: {payload}")
    return {
        "pcm": b"".join(chunks),
        "chunks": len(chunks),
        "done": done,
        "first_audio": first_audio,
        "elapsed": time.perf_counter() - started,
    }


def test_the_demo_streams_the_models_own_samples(demo: Any, engine: Any, demo_url: str) -> None:
    """What the browser plays is what the codec decoded, and it starts arriving immediately.

    Both halves matter. Gradio's streaming audio component re-encodes every frame to AAC, which
    makes the stream and the finished clip different audio with only one of them clean, so the
    page does not use it. Re-running the same seed through :meth:`KovaTTS.stream` and comparing
    waveforms is the check that nothing like it has crept into the path.

    The comparison has a tolerance, and it is worth being precise about why. The seed fixes the
    sampler, so two runs produce the same tokens and the same number of samples -- but CUDA
    reductions are not bit-reproducible run to run, so the decoded waveform differs by a few
    parts in 100,000 (about -85 dBFS, some 60 dB below the quietest thing anyone can hear on
    this material). A codec in the path does not look like that: an AAC round trip adds 32 ms of
    padding per frame and lands nearer 17 dB SNR, which any tolerance loose enough to be useful
    still catches. ``generate()`` is deliberately not the
    reference either: it decodes the whole code sequence at once rather than window by window.
    """
    run = stream_once(demo_url + demo.STREAM_PATH, text=TEXT, seed=1234)

    audio = np.frombuffer(run["pcm"], dtype="<i2")
    seconds = audio.size / OUTPUT_SAMPLE_RATE
    assert seconds > 2.0, "two sentences should be more than two seconds of speech"
    assert np.max(np.abs(audio)) > 1000, "that is silence"
    assert run["done"]["samples"] == audio.size
    assert run["done"]["sample_rate"] == OUTPUT_SAMPLE_RATE

    # Sample for sample, what the browser got is what the codec decoded.
    again = np.concatenate([frame.samples for frame in engine.stream(TEXT, seed=1234)])
    assert audio.size == again.size, "the same seed must produce the same number of samples"
    assert snr_db(again, audio.astype(np.float32) / 32767.0) > 60.0

    assert run["first_audio"] is not None
    # The claim the demo is built around: audio starts long before generation ends. Bounded
    # generously, at half, because the machine running this may be busy.
    assert run["first_audio"] < run["elapsed"] / 2
    print(
        f"\ntime to first audio {run['first_audio']:.2f} s, "
        f"{seconds:.1f} s of speech in {run['elapsed']:.1f} s"
    )


def test_three_generations_in_a_row_all_stream(demo: Any, demo_url: str) -> None:
    """Streaming has to work every time, not only on the first press.

    One server, three requests, the same seed. Anything left behind by a run -- a reservation
    not released, a generator not closed -- shows up here as a refusal or a short stream.
    """
    runs = [
        stream_once(demo_url + demo.STREAM_PATH, text="Once more, with feeling.", seed=99)
        for _ in range(3)
    ]

    assert all(run["chunks"] > 0 and run["done"] is not None for run in runs)
    waves = [np.frombuffer(run["pcm"], dtype="<i2").astype(np.float32) / 32767.0 for run in runs]
    assert len({wave.size for wave in waves}) == 1, "same seed, same length, every time"
    # Same audio too, to within the run-to-run noise of a GPU decode. A run that streamed only
    # part of its audio, or streamed it twice, is nowhere near this.
    assert all(snr_db(waves[0], wave) > 60.0 for wave in waves[1:])
    print("\ntime to first audio per run: " + ", ".join(f"{r['first_audio']:.2f} s" for r in runs))


def test_the_comfyui_node_generates_real_audio(comfy: Any, engine: Any) -> None:
    node = comfy.NODE_CLASS_MAPPINGS["KovaTTSGenerate"]()

    (audio,) = node.generate(engine, "Generated from a ComfyUI graph.", seed=99)

    waveform = audio["waveform"]
    assert audio["sample_rate"] == OUTPUT_SAMPLE_RATE
    assert waveform.ndim == 3 and waveform.shape[:2] == (1, 1)
    assert waveform.shape[2] / OUTPUT_SAMPLE_RATE > 1.0
    assert float(waveform.abs().max()) > 0.03, "that is silence"

    # Round-tripping through the boundary gives back exactly what the engine produced.
    restored = comfy.audio.from_comfy_audio(audio, OUTPUT_SAMPLE_RATE)
    assert restored.dtype == np.float32
    assert restored.size == waveform.shape[2]
