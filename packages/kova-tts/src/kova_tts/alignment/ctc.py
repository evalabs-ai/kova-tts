"""CTC forced alignment: the most likely frame-by-frame path through a fixed label sequence.

A port of ctc-forced-aligner's ``forced_align_impl.cpp`` (itself after torchaudio and
flashlight), with two fixes: the back-pointer arrays are sized for every write the forward pass
makes, and the back-trace stops at frame 0 instead of reading a back-pointer for frame -1.
Compiled with numba rather than shipped as a C++ extension, so installing kova-tts never needs a
compiler; ``nogil`` lets alignment run beside generation without holding the GIL.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True, slots=True)
class Segment:
    """A run of frames on one label: ``[start, end]``, both inclusive."""

    label: int
    start: int
    end: int


def forced_align(log_probs: np.ndarray, targets: np.ndarray, blank: int) -> np.ndarray:
    """The best path through `targets` for emissions `log_probs` (``[T, V]``, float32).

    Returns the label on the path at every frame. Raises ``ValueError`` when there are too few
    frames to fit the targets.
    """
    log_probs = np.ascontiguousarray(log_probs, dtype=np.float32)
    targets = np.ascontiguousarray(targets, dtype=np.int64)
    if blank in targets:
        raise ValueError("targets must not contain the blank label")
    path = _forced_align(log_probs, targets, blank)
    if path[0] < 0:
        raise ValueError("targets are too long for the number of frames")
    return path


def merge_repeats(path: np.ndarray) -> list[Segment]:
    """Runs of equal labels in `path`, one :class:`Segment` per run."""
    if len(path) == 0:
        return []
    change = np.flatnonzero(path[1:] != path[:-1])
    starts = np.concatenate(([0], change + 1))
    ends = np.concatenate((change, [len(path) - 1]))
    return [Segment(int(path[s]), int(s), int(e)) for s, e in zip(starts, ends, strict=True)]


@njit(cache=True, nogil=True)
def _forced_align(log_probs, targets, blank):  # pragma: no cover - compiled
    neg_inf = np.float32(-np.inf)
    T = log_probs.shape[0]
    L = targets.shape[0]
    S = 2 * L + 1
    path = np.zeros(T, dtype=np.int64)

    R = 0
    for i in range(1, L):
        if targets[i] == targets[i - 1]:
            R += 1
    if T < L + R:
        path[0] = -1
        return path

    alphas = np.full(2 * S, neg_inf, dtype=np.float32)
    back_offset = np.zeros(max(T - 1, 1), dtype=np.int64)
    back_seek = np.zeros(max(T - 1, 1), dtype=np.int64)
    # Sized S*T plus margin: the upstream (S+1)*(T-L) is too small and overflows.
    bit0 = np.zeros((S + 1) * T, dtype=np.uint8)
    bit1 = np.zeros((S + 1) * T, dtype=np.uint8)

    start = 0 if T - (L + R) > 0 else 1
    end = 1 if S == 1 else 2
    for i in range(start, end):
        label = blank if i % 2 == 0 else targets[i // 2]
        alphas[i] = log_probs[0, label]

    seek = 0
    for t in range(1, T):
        if T - t <= L + R:
            if start % 2 == 1 and targets[start // 2] != targets[start // 2 + 1]:
                start += 1
            start += 1
        if t <= L + R:
            if end % 2 == 0 and end < 2 * L and targets[end // 2 - 1] != targets[end // 2]:
                end += 1
            end += 1
        startloop = start
        cur = (t % 2) * S
        prev = ((t - 1) % 2) * S
        alphas[cur : cur + S] = neg_inf
        back_seek[t - 1] = seek
        back_offset[t - 1] = start
        if start == 0:
            alphas[cur] = alphas[prev] + log_probs[t, blank]
            startloop += 1
            seek += 1
        for i in range(startloop, end):
            x0 = alphas[prev + i]
            x1 = alphas[prev + i - 1]
            x2 = neg_inf
            label = blank if i % 2 == 0 else targets[i // 2]
            # Skipping a blank is allowed between two different labels only.
            if i % 2 != 0 and i != 1 and targets[i // 2] != targets[i // 2 - 1]:
                x2 = alphas[prev + i - 2]
            if x2 > x1 and x2 > x0:
                result = x2
                bit1[seek + i - startloop] = 1
            elif x1 > x0 and x1 > x2:
                result = x1
                bit0[seek + i - startloop] = 1
            else:
                result = x0
            alphas[cur + i] = result + log_probs[t, label]
        seek += end - startloop

    last = ((T - 1) % 2) * S
    state = S - 1 if alphas[last + S - 1] > alphas[last + S - 2] else S - 2
    for t in range(T - 1, -1, -1):
        path[t] = blank if state % 2 == 0 else targets[state // 2]
        if t == 0:
            break
        idx = back_seek[t - 1] + state - back_offset[t - 1]
        state -= (bit1[idx] << 1) | bit0[idx]
    return path
