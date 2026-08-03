"""Kova TTS nodes for ComfyUI.

ComfyUI imports every directory in ``custom_nodes`` as a package and reads two names from it:
``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS``. That is the entire contract, and
this file is all of it -- see :mod:`nodes` for the nodes and :mod:`audio` for the conversion
between ComfyUI's ``AUDIO`` dict and the waveforms ``kova_tts`` works in.

See ``README.md`` for how to install it. ComfyUI itself is never imported here: the pack has to
import cleanly (and be testable) on a machine that has ``kova-tts`` and no ComfyUI at all.
"""

from __future__ import annotations

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
