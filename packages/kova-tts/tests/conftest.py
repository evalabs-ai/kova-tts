"""Shared test setup for kova-tts."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

from kova_tts import paths


@pytest.fixture(autouse=True)
def restore_environment():
    """Undo environment changes a test leaves behind.

    ``python-dotenv`` writes straight into ``os.environ``, which ``monkeypatch`` cannot roll
    back: without this, a test that loads a throwaway ``.env`` would repoint ``KOVA_MODEL_PATH``
    for every test that runs after it.
    """
    snapshot = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(snapshot)
    paths.reset_dotenv_cache()


@pytest.fixture(scope="session")
def cuda_device():
    """The CUDA device with the most free memory, so a machine running other work is left alone."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    best = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.mem_get_info(i)[0])
    return torch.device("cuda", best)


@pytest.fixture(scope="session")
def local_artifact() -> Callable[[str], str]:
    """``callable(env_var) -> path`` for a locally configured artifact, skipping when unset.

    Never downloads: a test that needs gigabytes of weights should say so rather than fetching
    them behind the runner's back.
    """

    def resolve(env_var: str) -> str:
        paths.load_dotenv()
        value = os.environ.get(env_var, "").strip()
        if not value or not Path(value).exists():
            pytest.skip(f"set {env_var} to a local path to run this test")
        return value

    return resolve
