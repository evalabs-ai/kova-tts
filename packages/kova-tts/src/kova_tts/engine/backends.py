"""Choosing a decode loop, and building it.

Two backends generate codes, and which one is right is decided by the *artifact* rather than
by taste:

* :class:`~kova_tts.engine.generator.Generator` -- torch, a bf16 checkpoint, CUDA graphs when
  there is a CUDA device. The general answer, and the only one on Linux.
* :class:`~kova_tts.engine.mlx_generator.MLXGenerator` -- MLX, a quantized checkpoint with a
  sliced output head. Apple Silicon only, and it needs a model directory converted for it.

The two artifacts are not interchangeable: torch cannot read MLX's quantized tensors, and the
MLX model's output head is a slice that ``transformers`` would size wrongly. So :func:`resolve`
looks at the model directory and picks the backend that can actually load it, and an explicit
choice is only needed to *refuse* one -- for instance to compare the torch path against MLX on
the same machine.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from kova_tts import paths

log = logging.getLogger(__name__)

#: Accepted values of ``backend=`` and ``--backend``.
BACKENDS = ("auto", "torch", "mlx")

#: Config key the Apple conversion writes; see :mod:`kova_tts.engine.mlx_generator`.
MLX_MARKER = "head_vocab_size"


def is_mlx_artifact(model: str | os.PathLike[str]) -> bool:
    """True if `model` is a local directory converted for the MLX backend.

    A Hub repo id answers ``False``: deciding would mean downloading a config before the
    backend that does the downloading has been chosen, and the reason to keep an unconverted
    model on the Hub is to run it under torch.
    """
    config = Path(os.fspath(model)).expanduser() / "config.json"
    if not config.is_file():
        return False
    try:
        return bool(json.loads(config.read_text(encoding="utf-8")).get(MLX_MARKER))
    except (OSError, ValueError):
        # An unreadable config is the loader's problem to report, with the loader's error
        # message. All this function owes the caller is a backend to try.
        return False


def mlx_available() -> bool:
    """True if MLX can be imported here. False on any machine that is not Apple Silicon."""
    from importlib.util import find_spec

    return find_spec("mlx") is not None


def resolve(model: str | os.PathLike[str] | None = None, requested: str | None = None) -> str:
    """Which backend to load `model` with: ``"torch"`` or ``"mlx"``.

    `requested` is a value from :data:`BACKENDS`; ``None`` and ``"auto"`` both mean "whichever
    can load this artifact".
    """
    requested = (requested or "auto").lower()
    if requested not in BACKENDS:
        raise ValueError(f"backend must be one of {', '.join(BACKENDS)}, got {requested!r}.")

    resolved = paths.model_path(model)
    converted = is_mlx_artifact(resolved)

    if requested == "mlx":
        if not mlx_available():
            raise ImportError(
                "backend='mlx' needs the mlx package, which ships for Apple Silicon only. "
                "Install it with `uv sync --extra mlx`, or leave the backend unset."
            )
        return "mlx"
    if requested == "torch":
        if converted:
            raise ValueError(
                f"{resolved} is a model converted for the MLX backend: its weights are "
                f"MLX-quantized and its output head is a slice of the vocabulary, neither of "
                f"which transformers can load. Point at the bf16 checkpoint to use torch."
            )
        return "torch"

    if converted:
        if not mlx_available():
            raise ImportError(
                f"{resolved} was converted for the MLX backend, but mlx is not installed. "
                f"Run `uv sync --extra mlx` on Apple Silicon, or point KOVA_MODEL_PATH at the "
                f"bf16 checkpoint to run under torch."
            )
        return "mlx"
    return "torch"


def load_generator(
    model: str | os.PathLike[str] | None = None,
    *,
    backend: str | None = None,
    device: Any | None = None,
    dtype: Any | None = None,
    max_cache_len: int | None = None,
    cuda_graph: bool | None = None,
) -> Any:
    """Build the generator for `model`, on the backend :func:`resolve` picks.

    `device`, `dtype` and `cuda_graph` describe the torch loop and have no counterpart in MLX,
    which has one device, its own quantized dtypes, and no graph capture. Passing them
    alongside an MLX model is a mistake worth reporting rather than ignoring, so it warns.
    """
    chosen = resolve(model, backend)
    if chosen == "mlx":
        from kova_tts.engine.mlx_generator import DEFAULT_MAX_CACHE_LEN, MLXGenerator

        ignored = [
            name
            for name, value in (("device", device), ("dtype", dtype), ("cuda_graph", cuda_graph))
            if value is not None
        ]
        if ignored:
            log.warning(
                "Ignoring %s: the MLX backend has one device, its own quantized dtypes, and no "
                "graph capture.",
                ", ".join(ignored),
            )
        return MLXGenerator.from_pretrained(
            model,
            max_cache_len=DEFAULT_MAX_CACHE_LEN if max_cache_len is None else max_cache_len,
        )

    import torch

    from kova_tts.engine.generator import DEFAULT_MAX_CACHE_LEN, Generator

    return Generator.from_pretrained(
        model,
        device=device,
        dtype=torch.bfloat16 if dtype is None else dtype,
        max_cache_len=DEFAULT_MAX_CACHE_LEN if max_cache_len is None else max_cache_len,
        cuda_graph=cuda_graph,
    )
