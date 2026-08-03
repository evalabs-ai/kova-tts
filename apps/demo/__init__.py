"""The Gradio demo, as a package: ``python -m apps.demo`` or ``apps.demo.main([...])``.

Everything lives in :mod:`app`, in one file with no relative imports, so it can equally be
loaded straight from its path by a caller that has this repository but not this package on
``sys.path`` -- which is how ``kova-tts demo`` reaches it.
"""

from __future__ import annotations

from .app import DemoSession, build_ui, main

__all__ = ["DemoSession", "build_ui", "main"]
