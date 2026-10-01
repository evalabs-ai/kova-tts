"""Word timings for one chunk while its codes are still being generated.

An :class:`AlignmentStream` aligns one chunk incrementally. Words and codes are fed in as they
become known; a background thread re-aligns the
uncommitted tail every time something arrives and commits the words :meth:`Aligner.align`
and the :class:`CommitConfirmer` are sure of, so their timings are available long before the
chunk ends. :meth:`finish` aligns whatever is left in one pass and completes the list.

Committed words are dropped from the buffer as they go -- it is trimmed to just before the last
committed word, which stays in as the anchor for the next round -- so each round aligns a few
seconds of codes, not the whole chunk.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from kova_tts.alignment.aligner import (
    ADVANCE_BACKOFF_FRAMES,
    STRIDE_MS,
    Aligner,
    CommitConfirmer,
    CommittedWord,
)
from kova_tts.normalization import Word


@dataclass(frozen=True, slots=True)
class TimedWord:
    """A word of the chunk and where it was spoken, in milliseconds from the chunk's start."""

    word: Word
    start_ms: int
    end_ms: int


class AlignmentStream:
    """Incremental alignment of one chunk's words against its codes.

    Args:
        aligner: The shared :class:`~kova_tts.alignment.aligner.Aligner`.
        live: Align in the background as codes arrive. Without it nothing happens until
            :meth:`finish`, which is all a caller that only wants the final timings needs.
    """

    def __init__(self, aligner: Aligner, *, live: bool = True) -> None:
        self.aligner = aligner
        #: Every word committed so far, in order.
        self.words: list[TimedWord] = []
        self._taken = 0
        self._codes: list[int] = []
        self._offset = 0  # frames dropped from the front of `_codes`
        self._pending: list[Word] = []
        self._anchor: str | None = None
        self._floor = 0  # absolute frame where the last emitted word ended
        self._confirmer = CommitConfirmer()
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._finished = False
        self._closed = False
        self._error: BaseException | None = None
        self._worker: threading.Thread | None = None
        if live:
            self._worker = threading.Thread(target=self._run, name="kova-alignment", daemon=True)
            self._worker.start()

    # ------------------------------------------------------------------ the producer's side

    def add_words(self, words: Iterable[Word]) -> None:
        with self._lock:
            self._pending.extend(words)
        self._wake.set()

    def add_codes(self, codes: Iterable[int]) -> None:
        with self._lock:
            self._codes.extend(codes)
        self._wake.set()

    def take(self) -> list[TimedWord]:
        """Words committed since the last call."""
        with self._lock:
            new = self.words[self._taken :]
            self._taken = len(self.words)
        return new

    def finish(self) -> list[TimedWord]:
        """Align the rest, and return every word of the chunk with its timing. Blocks."""
        self._finished = True
        if self._worker is None:
            self._final()
        else:
            self._wake.set()
            self._worker.join()
        if self._error is not None:
            raise self._error
        return list(self.words)

    def close(self) -> None:
        """Abandon the chunk: stop the background thread without aligning the rest."""
        self._closed = True
        self._wake.set()

    # ------------------------------------------------------------------ the background thread

    def _run(self) -> None:
        try:
            while True:
                self._wake.wait()
                self._wake.clear()
                if self._closed:
                    return
                self._incremental()
                if self._finished:
                    self._final()
                    return
        except BaseException as exc:  # noqa: BLE001 - re-raised to the caller by finish()
            self._error = exc

    def _snapshot(self) -> tuple[list[Word], torch.Tensor, int]:
        with self._lock:
            words = list(self._pending)
            codes = torch.tensor(self._codes, dtype=torch.long)
            floor = max(0, self._floor - self._offset)
        return words, codes, floor

    def _incremental(self) -> None:
        words, codes, floor = self._snapshot()
        if not words or codes.numel() == 0:
            return
        committed = self.aligner.align(
            [w.normalized for w in words], codes, anchor=self._anchor, floor_frame=floor
        )
        if not committed:
            return
        # Emit only the prefix that this round and the last one agree on.
        offset_ms = int(self._offset * STRIDE_MS)
        emitted = len(self.words)
        n = self._confirmer.confirm(
            [(emitted + w.index, w.start_ms + offset_ms, w.end_ms + offset_ms) for w in committed]
        )
        self._commit(committed[:n], words)

    def _final(self) -> None:
        """No anchor here: with every code and word present, the unanchored full alignment is the
        most reliable, and the leading star absorbs the anchor word's audio at the buffer head."""
        words, codes, floor = self._snapshot()
        committed = []
        if words and codes.numel() > 0:
            committed = self.aligner.align_full(
                [w.normalized for w in words], codes, floor_frame=floor
            )
        self._commit(committed, words)

    def _commit(self, committed: list[CommittedWord], words: list[Word]) -> None:
        if not committed:
            return
        offset_ms = int(self._offset * STRIDE_MS)
        last = committed[-1]
        with self._lock:
            self.words.extend(
                TimedWord(words[w.index], w.start_ms + offset_ms, w.end_ms + offset_ms)
                for w in committed
            )
            self._pending = self._pending[len(committed) :]
            # Keep the back-off before the last committed word's start, so it stays in the
            # buffer as the next round's anchor.
            advance = max(0, int(last.start_ms // STRIDE_MS) - ADVANCE_BACKOFF_FRAMES)
            advance = min(advance, len(self._codes))
            self._anchor = words[last.index].normalized
            self._floor = self._offset + int(round(last.end_ms / STRIDE_MS))
            del self._codes[:advance]
            self._offset += advance
