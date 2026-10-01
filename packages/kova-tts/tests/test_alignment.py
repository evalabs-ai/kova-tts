"""Word alignment: the CTC kernel, the aligner's two modes, the streaming session, and the
aligned carry. A stand-in model whose codes *are* label ids makes the right answer exact: four
frames per letter and eight of silence between words."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kova_tts.alignment import Aligner, AlignmentStream, TimedWord
from kova_tts.alignment.aligner import STRIDE_MS, CommitConfirmer
from kova_tts.alignment.ctc import Segment, forced_align, merge_repeats
from kova_tts.alignment.model import BLANK_ID, CHAR_TO_ID, STAR_ID, VOCAB
from kova_tts.engine.tts import _aligned_carry
from kova_tts.normalization import Word

LETTER_FRAMES = 4
GAP_FRAMES = 8


class LabelModel:
    """Reads each code as the label spoken in that frame, with 90% confidence."""

    def log_probs(self, codes: torch.Tensor) -> torch.Tensor:
        out = torch.full((codes.numel(), len(VOCAB)), math.log(0.1 / (len(VOCAB) - 1)))
        out[torch.arange(codes.numel()), codes.long()] = math.log(0.9)
        return out


def spoken(words: list[str], tail: int = 60) -> tuple[list[int], list[tuple[int, int]]]:
    """Codes that say `words`, and each word's [start, end) frames."""
    codes = [STAR_ID] * GAP_FRAMES
    spans = []
    for word in words:
        start = len(codes)
        for letter in word:
            codes += [CHAR_TO_ID[letter]] * LETTER_FRAMES
        spans.append((start, len(codes)))
        codes += [STAR_ID] * GAP_FRAMES
    return codes + [STAR_ID] * tail, spans


WORDS = ["the", "kettle", "boiled", "over", "again"]


@pytest.fixture
def aligner() -> Aligner:
    return Aligner(LabelModel())


class TestKernel:
    def test_the_path_follows_unambiguous_emissions(self):
        labels = [BLANK_ID, 1, 1, BLANK_ID, 2, 2, 2, 3]
        log_probs = np.full((len(labels), 5), -10.0, dtype=np.float32)
        log_probs[np.arange(len(labels)), labels] = 0.0
        assert forced_align(log_probs, np.array([1, 2, 3]), BLANK_ID).tolist() == labels

    def test_too_many_targets_for_the_frames_is_refused(self):
        log_probs = np.zeros((2, 5), dtype=np.float32)
        with pytest.raises(ValueError):
            forced_align(log_probs, np.array([1, 2, 3]), BLANK_ID)

    def test_runs_are_merged(self):
        assert merge_repeats(np.array([0, 0, 1, 1, 1, 0])) == [
            Segment(0, 0, 1),
            Segment(1, 2, 4),
            Segment(0, 5, 5),
        ]


class TestAligner:
    def test_full_alignment_places_every_word(self, aligner):
        codes, spans = spoken(WORDS)
        placed = aligner.align_full(WORDS, torch.tensor(codes))
        assert [(w.start_ms, w.end_ms) for w in placed] == [
            (int(s * STRIDE_MS), int(e * STRIDE_MS)) for s, e in spans
        ]

    def test_incremental_alignment_holds_back_words_the_audio_has_not_reached(self, aligner):
        codes, spans = spoken(WORDS, tail=0)
        heard = torch.tensor(codes[: spans[2][1] + 50])  # through "boiled", and a pause
        placed = aligner.align(WORDS, heard)
        assert [w.word for w in placed] == WORDS[:3]


class TestStream:
    def test_finish_times_every_word(self, aligner):
        codes, spans = spoken(WORDS)
        stream = AlignmentStream(aligner, live=False)
        stream.add_words(Word(w, w) for w in WORDS)
        stream.add_codes(codes)
        timed = stream.finish()
        assert [t.word.original for t in timed] == WORDS
        assert [(t.start_ms, t.end_ms) for t in timed] == [
            (int(s * STRIDE_MS), int(e * STRIDE_MS)) for s, e in spans
        ]

    def test_live_alignment_commits_before_the_chunk_ends(self, aligner):
        codes, spans = spoken(WORDS)
        stream = AlignmentStream(aligner, live=False)
        stream.add_words(Word(w, w) for w in WORDS)
        early: list[TimedWord] = []
        for i in range(0, len(codes), 4):  # 50 ms rounds, as the background thread would
            stream.add_codes(codes[i : i + 4])
            stream._incremental()
            early += stream.take()
        assert early, "nothing was committed while codes were still arriving"
        timed = stream.finish()
        assert [t.word.original for t in timed] == WORDS
        assert [(t.start_ms, t.end_ms) for t in timed] == [
            (int(s * STRIDE_MS), int(e * STRIDE_MS)) for s, e in spans
        ]

    def test_the_background_thread_finishes_too(self, aligner):
        codes, _ = spoken(WORDS)
        stream = AlignmentStream(aligner, live=True)
        stream.add_words(Word(w, w) for w in WORDS)
        stream.add_codes(codes)
        assert [t.word.original for t in stream.finish()] == WORDS

    def test_timings_are_reported_against_the_written_word(self, aligner):
        codes, _ = spoken(["fifty", "five"])
        stream = AlignmentStream(aligner, live=False)
        stream.add_words([Word("55", "fifty five")])
        stream.add_codes(codes)
        (timed,) = stream.finish()
        assert timed.word.original == "55"


def test_the_confirmer_needs_two_rounds_that_agree():
    confirmer = CommitConfirmer(tolerance_ms=100)
    assert confirmer.confirm([(0, 0, 300)]) == 0
    assert confirmer.confirm([(0, 50, 320), (1, 400, 700)]) == 1
    assert confirmer.confirm([(1, 900, 1200)]) == 0  # moved too far to trust


class TestAlignedCarry:
    def timed(self, *spans: tuple[str, int, int]) -> list[TimedWord]:
        return [TimedWord(Word(w, w.upper()), s, e) for w, s, e in spans]

    def test_keeps_the_words_in_the_last_five_seconds_and_their_codes(self):
        codes = list(range(800))  # ten seconds
        words = self.timed(("a", 0, 2000), ("b", 2000, 5500), ("c", 6000, 9500))
        carry = _aligned_carry(words, codes, complete=True)
        assert carry.text == "B C"  # spoken forms, as the model read them
        assert carry.codes == tuple(codes[160:])

    def test_a_chunk_shorter_than_the_window_carries_all_its_codes(self):
        codes = list(range(200))
        words = self.timed(("a", 100, 900), ("b", 1000, 2400))
        carry = _aligned_carry(words, codes, complete=True)
        assert carry.text == "A B" and carry.codes == tuple(codes)

    def test_nothing_to_carry(self):
        assert _aligned_carry([], [1, 2], complete=True) is None
