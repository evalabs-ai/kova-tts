"""Shared fixtures.

``kova_codec`` does not depend on ``kova_tts``, so paths are read straight out of the
environment here rather than through ``kova_tts.paths``. Tests that need weights, a GPU or
real speech are marked ``weights`` / ``gpu`` and skip when the machine cannot supply them;
CI runs ``-m "not gpu and not weights"``.

No audio is committed to this repository — every recording carries licensing and
voice-likeness questions not worth inheriting. Tests that only exercise mechanics use
:func:`synthetic_wav`. The one test that genuinely needs real speech, reconstruction
quality, reads :func:`local_speech_wav` from ``KOVA_TEST_AUDIO`` and skips when unset.
"""

from __future__ import annotations

import math
import os
import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from kova_codec import SAMPLE_RATE

#: Seconds of the source recording to run through the codec. Long enough to cover several
#: streaming windows, short enough that a round trip stays under a second.
CLIP_SECONDS = 3.0


def _load_workspace_dotenv() -> None:
    """Populate ``KOVA_*`` from the workspace ``.env``, so a local run picks up checkpoints.

    Deliberately a five-line parser instead of a dependency: this only ever reads the
    gitignored developer ``.env``, and CI sets ``KOVA_DISABLE_DOTENV``.
    """
    # Falsy spellings count as off, matching kova_tts.paths: "0" means "do not disable".
    disable = os.environ.get("KOVA_DISABLE_DOTENV", "").strip().lower()
    if disable not in ("", "0", "false", "no", "off"):
        return
    for parent in Path(__file__).resolve().parents:
        dotenv = parent / ".env"
        if not dotenv.is_file():
            continue
        for line in dotenv.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        return


_load_workspace_dotenv()


@pytest.fixture(scope="session")
def checkpoint_path() -> Path:
    value = os.environ.get("KOVA_CODEC_PATH", "").strip()
    if not value or not Path(value).is_file():
        pytest.skip("set KOVA_CODEC_PATH to a codec checkpoint to run this test")
    return Path(value)


@pytest.fixture(scope="session")
def wavlm_path() -> str:
    """Local WavLM-large directory, or the Hub repo id if none is configured."""
    value = os.environ.get("KOVA_WAVLM_PATH", "").strip()
    if value and not Path(value).is_dir():
        pytest.skip(f"KOVA_WAVLM_PATH={value} is not a directory")
    return value or "microsoft/wavlm-large"


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device("cuda")


@pytest.fixture(scope="session")
def synthetic_wav() -> torch.Tensor:
    """:data:`CLIP_SECONDS` of deterministic speech-like audio, 32 kHz mono float32.

    A gliding harmonic stack under three fixed formant resonances, gated by a syllable-rate
    envelope with pauses. Not speech, but broadband and voiced enough to drive the encoder
    through a realistic range of codes — which is all any test here needs it to do.
    """
    n = int(CLIP_SECONDS * SAMPLE_RATE)
    t = torch.arange(n, dtype=torch.float64) / SAMPLE_RATE

    f0 = 110.0 + 40.0 * torch.sin(2 * math.pi * 0.7 * t)
    phase = 2 * math.pi * torch.cumsum(f0, dim=0) / SAMPLE_RATE
    wav = torch.zeros(n, dtype=torch.float64)
    for k in range(1, 41):
        # Emphasise harmonics near 700/1200/2600 Hz so the spectrum has formant structure.
        hz = k * 130.0
        gain = sum(math.exp(-(((hz - f) / 250.0) ** 2)) for f in (700.0, 1200.0, 2600.0))
        wav += (gain / k) * torch.sin(k * phase)

    syllables = (0.5 + 0.5 * torch.sin(2 * math.pi * 3.5 * t)) ** 2
    wav *= syllables * (t % 1.0 > 0.15)  # a short pause once a second
    return (0.7 * wav / wav.abs().max()).float().contiguous()


@pytest.fixture(scope="session")
def local_speech_wav() -> torch.Tensor:
    """:data:`CLIP_SECONDS` of real speech from ``KOVA_TEST_AUDIO``, as 32 kHz mono float32.

    Opt-in and local only: reconstruction *quality* cannot be judged on a synthetic signal,
    but no recording ships with this repository.
    """
    value = os.environ.get("KOVA_TEST_AUDIO", "").strip()
    if not value or not Path(value).is_file():
        pytest.skip("set KOVA_TEST_AUDIO to a local speech wav to measure reconstruction")

    with wave.open(value, "rb") as handle:
        if handle.getsampwidth() != 2:
            pytest.skip(f"{value} is not 16-bit PCM")
        channels, rate = handle.getnchannels(), handle.getframerate()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")

    wav = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
    if channels > 1:
        wav = wav.reshape(-1, channels).mean(dim=1)
    if rate != SAMPLE_RATE:
        import torchaudio

        wav = torchaudio.functional.resample(wav, rate, SAMPLE_RATE)

    want = int(CLIP_SECONDS * SAMPLE_RATE)
    if wav.numel() < want:
        pytest.skip(f"{value} is shorter than {CLIP_SECONDS} s")
    # Skip the leading half second, which is usually room tone rather than speech.
    start = min(SAMPLE_RATE // 2, wav.numel() - want)
    return wav[start : start + want].contiguous()


@pytest.fixture(scope="session")
def codec(checkpoint_path: Path, wavlm_path: str, cuda_device: torch.device):
    """A full encode+decode codec on the GPU. Session-scoped: loading it takes ~15 s."""
    from kova_codec import KovaCodec

    return KovaCodec.from_checkpoint(
        checkpoint_path, device=cuda_device, dtype=torch.float32, wavlm=wavlm_path
    )
