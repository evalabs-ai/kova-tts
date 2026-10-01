"""Word timings from codec codes: which frames each word of a transcript was spoken in.

Two modes, both over the same forced alignment (:mod:`.ctc`) of the transcript's letters,
separated by ``<star>`` wildcards that absorb silence:

* :meth:`Aligner.align_full` -- every word, in one pass. For codes known to contain all of
  them: a finished chunk.
* :meth:`Aligner.align` -- only the words it is confident about, while the codes are still
  arriving. It aligns every prefix of the word list (k = 1..N) and keeps the prefix whose words
  fit the audio best: words that are really there each score well, and words the audio has not
  reached yet squeeze the earlier ones and drag the score down. Words ending too close to the
  end of the audio, words stretched over more time than their letters can take, and a weak
  final word are held back for the next call.
"""

from __future__ import annotations

import math
import os
import re
import threading
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from kova_codec.constants import TOKEN_RATE
from kova_tts.alignment.ctc import forced_align, merge_repeats
from kova_tts.alignment.model import BLANK_ID, CHAR_TO_ID, STAR_ID, AlignmentModel

#: Milliseconds per codec frame.
STRIDE_MS = 1000.0 / TOKEN_RATE

#: When a commit advances the buffer, keep this many frames before the last committed word's
#: start. That word's committed start can be off by 100-200 ms; the margin lets the next round's
#: anchored alignment snap it back rather than squeeze it onto the following word's audio.
ADVANCE_BACKOFF_FRAMES = 16  # 200 ms


@dataclass(frozen=True, slots=True)
class CommittedWord:
    """One word placed in the audio. Times are milliseconds from the start of the codes."""

    index: int  # position in the word list that was aligned
    word: str
    start_ms: int
    end_ms: int
    log_p: float  # mean log-probability of the word's frames on the aligned path


@dataclass(slots=True)
class _Step:
    """One prefix's alignment: frame spans and scores per word, and where the anchor landed."""

    starts: list[int]
    ends: list[int]  # exclusive
    scores: list[float]  # NaN for a word given no frames
    anchor_start: int = 0
    anchor_end: int = 0


class CommitConfirmer:
    """Emit a word only once two consecutive :meth:`Aligner.align` rounds agree on it.

    A wrong but confident placement -- a word jumped to a later acoustic match, or smeared over
    its neighbour -- does not survive re-alignment once more audio arrives, while a correct one
    reproduces. Costs about one extra round of latency per word.
    """

    def __init__(self, tolerance_ms: int = 100) -> None:
        self.tolerance_ms = tolerance_ms
        self._pending: dict[int, tuple[int, int]] = {}

    def confirm(self, candidates: list[tuple[int, int, int]]) -> int:
        """`candidates` are ``(word index, start_ms, end_ms)`` in order; returns how many of the
        leading ones were seen in the same place last round."""
        n = 0
        for index, start, end in candidates:
            previous = self._pending.get(index)
            if (
                previous is not None
                and abs(previous[0] - start) <= self.tolerance_ms
                and abs(previous[1] - end) <= self.tolerance_ms
            ):
                n += 1
            else:
                break
        self._pending = {index: (start, end) for index, start, end in candidates[n:]}
        return n


class Aligner:
    """Forced alignment of text against codec codes.

    Thread-safe: the model is behind a lock and everything else is per call.

    Args:
        model: The loaded :class:`~kova_tts.alignment.model.AlignmentModel`.
    """

    def __init__(self, model: AlignmentModel) -> None:
        self.model = model
        self._lock = threading.Lock()

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], device: str | None = None) -> Aligner:
        """Load the model -- in half precision on CUDA -- and warm it up."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = AlignmentModel.from_checkpoint(path, device)
        if str(device).startswith("cuda"):
            model = model.half()
        aligner = cls(model)
        aligner.warmup()
        return aligner

    def warmup(self) -> None:
        """One forward and one alignment, so neither compiles on a real request."""
        log_probs = self._log_probs(torch.zeros(80, dtype=torch.long))
        forced_align(log_probs.numpy(), np.asarray([CHAR_TO_ID["a"]]), BLANK_ID)

    # ------------------------------------------------------------------ the two modes

    def align_full(
        self, words: list[str], codes: torch.Tensor, floor_frame: int = 0
    ) -> list[CommittedWord]:
        """Place every word in `codes`, in one pass, with no confidence filtering.

        `floor_frame` is where the previous emitted word ended, in this buffer's frames; words are
        clamped forward to stay at or after it and monotonic.
        """
        if not words:
            return []
        log_probs = self._log_probs(codes).numpy()
        step = self._align_prefix(words, log_probs, safety_margin=5)
        committed = []
        last_end = max(step.anchor_end, floor_frame)
        for i, word in enumerate(words):
            start = max(step.starts[i], last_end)
            end = max(step.ends[i], start)
            committed.append(
                CommittedWord(i, word, int(start * STRIDE_MS), int(end * STRIDE_MS), step.scores[i])
            )
            last_end = end
        return committed

    def align(
        self,
        words: list[str],
        codes: torch.Tensor,
        *,
        anchor: str | None = None,
        floor_frame: int = 0,
        tail_margin_ms: int = 600,
        min_final_log_p: float = -1.6,
        min_frames: int = 8,
        max_overlap_frames: int = 2,
        word_cost: float = 0.06,
        word_cap: float = 0.3,
    ) -> list[CommittedWord]:
        """The leading words of `words` that `codes` confidently contain, so far.

        Args:
            anchor: The last word already committed, whose audio starts just after the head of
                `codes` (the caller keeps :data:`ADVANCE_BACKOFF_FRAMES` before it). Aligned in
                front of `words` to pin the path, checked, and not returned.
            floor_frame: Where the previous emitted word ended, in this buffer's frames.
            tail_margin_ms: Hold back words ending this close to the end of the codes.
            min_final_log_p: Trim the last committed word while it scores below this.
            min_frames: Commit nothing from a buffer shorter than this.
            max_overlap_frames: How far a word may start before its predecessor's end and be
                clamped; beyond that the path mis-anchored and this round stops there.
            word_cost: Flat cost per word when choosing how many words to commit.
            word_cap: Cap on one word's contribution to that choice.
        """
        if not words or codes.shape[-1] < min_frames:
            return []
        deadline_ms = codes.shape[-1] * STRIDE_MS - tail_margin_ms
        log_probs = self._log_probs(codes).numpy()
        steps = [
            self._align_prefix(words[:k], log_probs, safety_margin=0, anchor=anchor)
            for k in range(1, len(words) + 1)
        ]
        k = self._best_prefix(steps, words, word_cost=word_cost, word_cap=word_cap)
        if k == 0:
            return []
        step = steps[k - 1]
        if anchor:
            # The anchor's real onset is inside the back-off at the buffer head. Placed later,
            # the leading star jumped it over real speech; stretched far longer than the word
            # can last, it smeared over what follows. Either way nothing after it is trusted.
            if step.anchor_start > ADVANCE_BACKOFF_FRAMES + 4:
                return []
            if (step.anchor_end - step.anchor_start) * STRIDE_MS > _stretch_limit_ms(anchor) + 100:
                return []

        committed: list[CommittedWord] = []
        last_end = max(step.anchor_end, floor_frame)
        for i in range(k):
            start, end = step.starts[i], step.ends[i]
            if start < last_end:
                # Word 0 against the floor is last round's emission disagreeing slightly with
                # this one: clamp. A later word overlapping its predecessor means the path
                # mis-anchored: stop.
                if i > 0 and last_end - start > max_overlap_frames:
                    break
                start = last_end
                end = max(end, start)
            end_ms = int(end * STRIDE_MS)
            if end_ms > deadline_ms:
                break
            committed.append(
                CommittedWord(i, words[i], int(start * STRIDE_MS), end_ms, step.scores[i])
            )
            last_end = end
        # A word longer than its letters allow has absorbed the speech after it, so everything
        # from there on sits too late.
        for i, word in enumerate(committed):
            if word.end_ms - word.start_ms > _stretch_limit_ms(word.word):
                committed = committed[:i]
                break
        # The commit boundary must be confident; weak words in the middle are kept.
        while committed and (
            math.isnan(committed[-1].log_p) or committed[-1].log_p < min_final_log_p
        ):
            committed.pop()
        return committed

    # ------------------------------------------------------------------ internals

    def _log_probs(self, codes: torch.Tensor) -> torch.Tensor:
        with self._lock:
            return self.model.log_probs(codes)

    def _align_prefix(
        self,
        words: list[str],
        log_probs: np.ndarray,
        *,
        safety_margin: int,
        anchor: str | None = None,
    ) -> _Step:
        """Align ``[<star>, anchor..., <star>, w1..., <star>, w2..., <star>]`` and read off
        each word's frames."""
        n_frames = log_probs.shape[0]
        targets = _targets(words, anchor)
        empty = _Step([0] * len(words), [0] * len(words), [math.nan] * len(words))
        if len(targets) + safety_margin >= n_frames:
            return empty
        try:
            path = forced_align(log_probs, np.asarray(targets), BLANK_ID)
        except ValueError:
            return empty

        # Frame -> index into the target sequence, NaN on blanks.
        position = np.full(n_frames, np.nan)
        cursor = 0
        for segment in merge_repeats(path):
            if segment.label == BLANK_ID:
                continue
            position[segment.start : segment.end + 1] = cursor
            cursor += 1

        # Target position 0 is the leading star; an anchor's letters follow it.
        n_anchor = len(_letters(anchor)) if anchor else 0
        anchor_start = anchor_end = 0
        if n_anchor:
            frames = np.flatnonzero((position >= 1) & (position <= n_anchor))
            if len(frames):
                anchor_start, anchor_end = int(frames[0]), int(frames[-1] + 1)

        ranges = []
        pos = 1 + n_anchor if n_anchor else 0
        for word in words:
            pos += 1  # the star in front of the word
            ranges.append((pos, pos + len(_letters(word)) - 1))
            pos += len(_letters(word))

        # `position` never decreases over its non-NaN frames, so each word's frames are one
        # contiguous run, found by binary search.
        valid = np.flatnonzero(~np.isnan(position))
        values = position[valid]
        path_log_p = log_probs[np.arange(n_frames), path]
        step = _Step([], [], [], anchor_start, anchor_end)
        for first, last in ranges:
            lo = int(np.searchsorted(values, first, side="left"))
            hi = int(np.searchsorted(values, last, side="right"))
            if hi <= lo:
                step.starts.append(0)
                step.ends.append(0)
                step.scores.append(math.nan)
                continue
            frames = valid[lo:hi]
            step.starts.append(int(frames[0]))
            step.ends.append(int(frames[-1] + 1))
            step.scores.append(float(path_log_p[frames].mean()))
        return step

    @staticmethod
    def _best_prefix(
        steps: list[_Step],
        words: list[str],
        *,
        word_cost: float,
        word_cap: float,
        ref_letters: float = 6.0,
        min_weight: float = 0.4,
        max_weight: float = 1.5,
    ) -> int:
        """How many words to commit: argmax over k of Σ wᵢ·min(exp(log_pᵢ), cap) − cost·k.

        The cap stops one clean word outweighing a squeezed tail ahead of it, the cost makes each
        extra word earn its place, and the length weight wᵢ = clip(letters/6, 0.4, 1.5) gives
        short function words little say. k = 0, waiting for more audio, scores 0.
        """
        weights = np.asarray(
            [np.clip(len(_letters(w)) / ref_letters, min_weight, max_weight) for w in words]
        )
        best_k, best = 0, 0.0
        for k in range(1, len(steps) + 1):
            scores = np.asarray(steps[k - 1].scores, dtype=float)
            contribution = np.clip(np.exp(scores), None, word_cap) * weights[:k]
            score = float(np.nansum(contribution)) - word_cost * k
            if score > best:
                best, best_k = score, k
        return best_k


@lru_cache(maxsize=65536)
def _letters(word: str) -> tuple[str, ...]:
    """The characters of `word` the model has labels for; a ``[tag]`` becomes ``@``."""
    cleaned = word.replace("’", "'").lower()
    cleaned = re.sub(r"\[[^\]]*\]", "@", cleaned)
    cleaned = re.sub(r"[^a-z'@]", "", cleaned)
    return tuple(c for c in cleaned if c in CHAR_TO_ID)


def _targets(words: list[str], anchor: str | None) -> list[int]:
    """Label ids: a leading star, the anchor's letters, then each word behind a star, and a
    trailing star for the silence after the last word."""
    ids = [STAR_ID]
    anchor_letters = _letters(anchor) if anchor else ()
    ids.extend(CHAR_TO_ID[c] for c in anchor_letters)
    for i, word in enumerate(words):
        if i > 0 or anchor_letters:
            ids.append(STAR_ID)
        ids.extend(CHAR_TO_ID[c] for c in _letters(word))
    ids.append(STAR_ID)
    return ids


def _stretch_limit_ms(word: str, base_ms: float = 200.0, per_letter_ms: float = 150.0) -> float:
    """Longest a word can plausibly last. Speech runs 50-120 ms a letter, and the silence between
    words belongs to the stars, so a word longer than this has absorbed something else."""
    return base_ms + per_letter_ms * max(1, len(_letters(word)))
