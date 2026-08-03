"""Silence detection, splitting, trimming, and the loudness the codec expects.

The arithmetic is checked against waveforms whose silences are placed by construction, so an
assertion can name the sample a cut belongs at rather than describing the shape of the answer.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_data_synth import join, quiet, speech_like

from kova_codec.constants import SAMPLE_RATE, TARGET_LUFS
from kova_tts.audio import normalize_loudness
from kova_tts.data.segment import (
    SegmentSettings,
    frame_db,
    is_silent,
    silent_spans,
    split_on_silence,
    trim_silence,
)

#: Tolerance on a boundary this module reports, in samples: everything is measured on a 10 ms
#: hop over a 25 ms window, so an answer is exact to within one window.
WINDOW = int(0.025 * SAMPLE_RATE)


def test_frame_db_peaks_at_zero_and_floors_on_silence():
    db = frame_db(join(speech_like(0.5), quiet(0.5)))
    assert db.max() == pytest.approx(0.0)
    assert db[-10:].max() < -40.0


def test_frame_db_is_scale_invariant():
    wav = speech_like(0.5)
    assert frame_db(wav) == pytest.approx(frame_db(wav * 0.01), abs=1e-6)


def test_frame_db_on_digital_silence_is_all_minus_infinity():
    assert np.isneginf(frame_db(np.zeros(SAMPLE_RATE, dtype=np.float32))).all()


def test_is_silent_distinguishes_room_tone_from_nothing():
    assert is_silent(np.zeros(1000, dtype=np.float32))
    assert is_silent(np.array([], dtype=np.float32))
    assert not is_silent(quiet(0.1))


# ------------------------------------------------------------------------------ silent spans


def test_silent_spans_finds_an_interior_gap_where_it_was_put():
    wav = join(speech_like(1.0), quiet(0.8), speech_like(1.0))
    ((start, stop),) = silent_spans(wav, min_silence=0.4)
    assert start == pytest.approx(1.0 * SAMPLE_RATE, abs=WINDOW)
    assert stop == pytest.approx(1.8 * SAMPLE_RATE, abs=WINDOW)


def test_silent_spans_ignores_gaps_shorter_than_the_minimum():
    wav = join(speech_like(1.0), quiet(0.2), speech_like(1.0))
    assert silent_spans(wav, min_silence=0.4) == []
    assert len(silent_spans(wav, min_silence=0.1)) == 1


def test_silent_spans_ignores_the_ends():
    wav = join(quiet(1.0), speech_like(1.0), quiet(1.0))
    assert silent_spans(wav, min_silence=0.4) == []


# ---------------------------------------------------------------------------------- splitting


def test_a_short_recording_is_returned_whole():
    wav = speech_like(2.0)
    pieces = split_on_silence(wav, max_seconds=30.0)
    assert len(pieces) == 1
    assert pieces[0] is not None and pieces[0].size == wav.size


def test_split_cuts_in_the_middle_of_the_silence():
    wav = join(speech_like(3.0), quiet(1.0), speech_like(3.0))
    first, second = split_on_silence(wav, max_seconds=4.0, min_silence=0.5)
    # The cut belongs at the centre of the 1 s gap, 3.5 s in.
    assert first.size == pytest.approx(3.5 * SAMPLE_RATE, abs=WINDOW)
    assert first.size + second.size == wav.size


def test_split_takes_the_last_boundary_that_still_fits():
    # Three 2 s utterances with 0.6 s gaps: a 5 s limit should cut once, after the second gap.
    wav = join(speech_like(2.0), quiet(0.6), speech_like(2.0), quiet(0.6), speech_like(2.0))
    pieces = split_on_silence(wav, max_seconds=5.0, min_silence=0.5)
    assert len(pieces) == 2
    assert pieces[0].size == pytest.approx(4.9 * SAMPLE_RATE, abs=WINDOW)


def test_split_leaves_a_recording_with_no_silence_over_length():
    wav = speech_like(8.0)
    pieces = split_on_silence(wav, max_seconds=3.0)
    assert len(pieces) == 1
    assert pieces[0].size == wav.size


def test_split_overshoots_rather_than_cutting_mid_word():
    # The only boundary is past the limit, so the first piece runs long instead of being cut.
    wav = join(speech_like(6.0), quiet(0.6), speech_like(1.0))
    pieces = split_on_silence(wav, max_seconds=4.0, min_silence=0.5)
    assert len(pieces) == 2
    assert pieces[0].size > 4.0 * SAMPLE_RATE


def test_split_pieces_reassemble_into_the_original():
    wav = join(*[part for _ in range(4) for part in (speech_like(2.0), quiet(0.6))])
    pieces = split_on_silence(wav, max_seconds=3.0, min_silence=0.5)
    assert len(pieces) > 2
    assert np.array_equal(np.concatenate(pieces), wav)


# ----------------------------------------------------------------------------------- trimming


def test_trim_keeps_the_speech_and_the_requested_padding():
    wav = join(quiet(1.0), speech_like(2.0), quiet(1.5))
    trimmed = trim_silence(wav, pad_seconds=0.05)
    expected = (2.0 + 2 * 0.05) * SAMPLE_RATE
    assert trimmed.size == pytest.approx(expected, abs=2 * WINDOW)


def test_trim_without_padding_lands_on_the_speech():
    wav = join(quiet(1.0), speech_like(2.0), quiet(1.5))
    assert trim_silence(wav, pad_seconds=0.0).size == pytest.approx(
        2.0 * SAMPLE_RATE, abs=2 * WINDOW
    )


def test_trim_is_idempotent():
    once = trim_silence(join(quiet(1.0), speech_like(2.0), quiet(1.0)))
    assert np.array_equal(trim_silence(once), once)


def test_trim_leaves_a_clip_with_no_silence_alone():
    wav = speech_like(2.0)
    assert trim_silence(wav).size == wav.size


def test_trim_returns_nothing_for_an_entirely_silent_clip():
    assert trim_silence(np.zeros(SAMPLE_RATE, dtype=np.float32)).size == 0
    assert trim_silence(np.array([], dtype=np.float32)).size == 0


def test_trim_does_not_run_past_the_start():
    # Padding wider than the leading silence must clamp, not wrap round to a negative index.
    wav = join(quiet(0.02), speech_like(1.0))
    assert trim_silence(wav, pad_seconds=1.0).size == wav.size


# ---------------------------------------------------------------------------------- loudness


def test_normalisation_hits_the_level_the_codec_was_trained_on():
    import pyloudnorm as pyln

    normalized = normalize_loudness(speech_like(3.0) * 0.05)
    measured = pyln.Meter(SAMPLE_RATE).integrated_loudness(normalized)
    assert measured == pytest.approx(TARGET_LUFS, abs=0.5)


def test_normalisation_does_not_clip():
    assert np.max(np.abs(normalize_loudness(speech_like(3.0) * 0.999))) <= 1.0


# ---------------------------------------------------------------------------------- settings


def test_settings_reject_a_window_that_drops_everything():
    with pytest.raises(ValueError, match="must exceed min_seconds"):
        SegmentSettings(max_seconds=1.0, min_seconds=2.0).validate()


@pytest.mark.parametrize(
    "settings",
    [
        SegmentSettings(top_db=0.0),
        SegmentSettings(min_silence=0.0),
        SegmentSettings(pad_seconds=-0.1),
    ],
)
def test_settings_reject_impossible_thresholds(settings):
    with pytest.raises(ValueError):
        settings.validate()


def test_default_settings_leave_room_under_the_token_limit():
    from kova_codec.constants import TOKEN_RATE

    assert SegmentSettings().max_seconds * TOKEN_RATE < 4096 * 0.8
