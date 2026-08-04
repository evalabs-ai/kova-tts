"""The Gradio demo, as a package: ``python -m apps.demo`` or ``apps.demo.main([...])``.

:mod:`app` composes the rest of the directory and is the only file a caller has to reach for.
Importing it puts its own directory on ``sys.path``, so the modules beside it are imported by
plain name and nothing here imports relatively -- which is what lets ``kova-tts demo`` load
``app.py`` straight from its path, with this repository but not this package importable.
"""

from __future__ import annotations

from .app import DemoSession, build_ui, main

__all__ = ["DemoSession", "build_ui", "main"]
