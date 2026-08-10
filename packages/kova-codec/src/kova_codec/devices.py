"""Which accelerator to run on when nobody said.

One function, in the package both halves of the project depend on, so ``kova-codec`` and
``kova-tts`` cannot drift into disagreeing about where "the default device" is -- a codec on
the CPU behind an LM on the GPU is a working configuration, just a slow one, and nothing
would report it.
"""

from __future__ import annotations

import torch


def default_device() -> torch.device:
    """CUDA if there is any, then Apple's Metal backend, then the CPU.

    CUDA first is not a preference between the two: a machine cannot have both, so the order
    only decides what happens on machines that have neither.

    ``torch.backends.mps.is_built()`` is checked as well as ``is_available()`` because a
    non-Apple-Silicon build raises rather than returning False from ``is_available()`` on some
    torch versions, and this is called from constructors that must not throw.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_built() and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_accelerator(device: torch.device | str) -> bool:
    """True for a device with its own memory and kernels, i.e. anything but the CPU."""
    return torch.device(device).type != "cpu"
