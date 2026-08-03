"""Shared test setup for kova-tts."""

from __future__ import annotations

import os

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
