"""The pipeline end to end on a fake codec: what gets written, what gets dropped, and why.

Every test here runs on synthesized audio with a deterministic stand-in for the codec, so the
whole file is CPU-only and needs no weights. The real codec is exercised in
``test_data_end_to_end.py``.

The two assertions that matter most: an emitted line round-trips through
:func:`~kova_tts.prompt.parse_audio_tokens` back to the codes the clip encoded to, and the file
this writes is loaded without complaint by the dataset the trainer actually uses.
"""

from __future__ import annotations

import importlib
import json
import math

import numpy as np
import pytest
from test_data_synth import FakeCodec, join, mini_tokenizer, quiet, speech_like, write_wav

from kova_codec.constants import HOP_LENGTH, SAMPLE_RATE
from kova_tts.data.prepare import main, prepare, read_corpus
from kova_tts.data.report import NO_TRANSCRIPT, SILENT, TOO_LONG, TOO_SHORT, UNREADABLE
from kova_tts.data.segment import SegmentSettings
from kova_tts.prompt import parse_audio_tokens
from kova_tts.tokens import BEGIN_OF_TEXT, SPEECH_END, SPEECH_START

TEXT = "Hello there, this is a line of speech."

#: The module, not the function of the same name that ``kova_tts.data`` re-exports. Tests reach
#: for it to stub out the two things that would otherwise need a GPU: the codec and the ASR model.
prep = importlib.import_module("kova_tts.data.prepare")


def build_corpus(root, count=3, seconds=2.0, *, text=TEXT):
    """`count` recordings with sidecar transcripts, named ``clip_00.wav`` upwards."""
    for index in range(count):
        name = f"clip_{index:02d}"
        write_wav(root / f"{name}.wav", speech_like(seconds, f0=100.0 + 7 * index))
        (root / f"{name}.txt").write_text(f"{text} Number {index}.", encoding="utf-8")
    return root


@pytest.fixture
def recordings(tmp_path):
    return build_corpus(tmp_path / "recordings")


@pytest.fixture
def codec():
    return FakeCodec()


# ------------------------------------------------------------------------------- happy path


def test_writes_one_row_per_recording(recordings, codec):
    report = prepare(recordings, encoder=codec)

    rows = read_corpus(recordings / "train.jsonl")
    assert [row["id"] for row in rows] == ["clip_00.wav#0", "clip_01.wav#0", "clip_02.wav#0"]
    assert report.train_rows == 3
    assert report.emitted == 3
    assert report.skipped == 0


def test_rows_carry_the_clip_they_came_from(recordings, codec):
    prepare(recordings, encoder=codec)
    row = read_corpus(recordings / "train.jsonl")[0]
    assert row["audio"] == "clip_00.wav"
    assert row["segment"] == 0
    assert row["seconds"] == pytest.approx(2.0, abs=0.2)


def test_the_line_is_a_training_example_around_the_transcript(recordings, codec):
    prepare(recordings, encoder=codec)
    text = read_corpus(recordings / "train.jsonl")[0]["text"]
    assert text.startswith(f"{BEGIN_OF_TEXT}<|text_prompt_start|>{TEXT} Number 0.")
    assert SPEECH_START in text
    assert text.endswith(SPEECH_END)


def test_codes_round_trip_and_match_the_clip_duration(recordings, codec):
    prepare(recordings, encoder=codec)
    for row in read_corpus(recordings / "train.jsonl"):
        codes = parse_audio_tokens(row["text"])
        assert len(codes) == math.ceil(row["seconds"] * SAMPLE_RATE / HOP_LENGTH)
        assert all(0 <= code < codec.codebook for code in codes)


def test_each_clip_is_encoded_on_its_own(tmp_path, codec):
    prepare(build_corpus(tmp_path / "recordings", count=5), encoder=codec)
    assert [rows for rows, _ in codec.calls] == [1, 1, 1, 1, 1]


def test_every_clip_reaches_the_codec_at_its_own_length(tmp_path, codec):
    root = tmp_path / "recordings"
    durations = (1.0, 8.0, 1.2, 7.5)
    for index, seconds in enumerate(durations):
        write_wav(root / f"clip_{index}.wav", speech_like(seconds))
        (root / f"clip_{index}.txt").write_text(TEXT, encoding="utf-8")

    prepare(root, encoder=codec)
    widths = sorted(width / SAMPLE_RATE for _, width in codec.calls)
    assert widths == pytest.approx(sorted(durations), abs=0.05)


def test_recordings_are_normalised_before_encoding(tmp_path, codec):
    """A quiet recording and a loud one must reach the codec at the same level."""
    root = tmp_path / "recordings"
    for name, gain in (("quiet", 0.02), ("loud", 0.9)):
        write_wav(root / f"{name}.wav", speech_like(3.0) * gain)
        (root / f"{name}.txt").write_text(TEXT, encoding="utf-8")

    class Peaks(FakeCodec):
        def __init__(self):
            super().__init__()
            self.peaks = []

        def encode(self, wav):
            self.peaks += [float(np.max(np.abs(row))) for row in np.atleast_2d(np.asarray(wav))]
            return super().encode(wav)

    peaks = Peaks()
    prepare(root, encoder=peaks)
    assert peaks.peaks[0] == pytest.approx(peaks.peaks[1], rel=0.05)


# ------------------------------------------------------------- what the trainer makes of it


def test_the_corpus_loads_in_the_trainer_dataset(recordings, codec):
    from kova_tts.finetune.dataset import MaskedCausalDataset

    prepare(recordings, encoder=codec)
    tokenizer = mini_tokenizer(codec.codebook)
    dataset = MaskedCausalDataset(recordings / "train.jsonl", tokenizer)

    assert len(dataset) == 3
    assert dataset.report.skipped == 0
    assert dataset.report.no_speech_end == 0


def test_loss_is_masked_up_to_the_first_audio_token(recordings, codec):
    from kova_tts.finetune.dataset import MaskedCausalDataset

    prepare(recordings, encoder=codec)
    tokenizer = mini_tokenizer(codec.codebook)
    item = MaskedCausalDataset(recordings / "train.jsonl", tokenizer)[0]

    start = int((item["input_ids"] == tokenizer.convert_tokens_to_ids(SPEECH_START)).nonzero()[0])
    assert (item["labels"][: start + 1] == -100).all()
    assert (item["labels"][start + 1 :] != -100).all()
    assert int(item["labels"][-1]) == tokenizer.convert_tokens_to_ids(SPEECH_END)


def test_rows_stay_under_the_trainers_max_length(recordings, codec):
    from kova_tts.finetune.dataset import MaskedCausalDataset

    report = prepare(recordings, encoder=codec)
    dataset = MaskedCausalDataset(recordings / "train.jsonl", mini_tokenizer(codec.codebook))
    # The estimate the pipeline used to decide what to keep must not undercount the truth.
    assert dataset.report.longest <= report.longest_tokens


# ------------------------------------------------------------------------ skips and reasons


def test_a_clip_shorter_than_the_minimum_is_reported(tmp_path, codec):
    root = tmp_path / "recordings"
    build_corpus(root, count=1)
    write_wav(root / "tiny.wav", speech_like(0.2))
    (root / "tiny.txt").write_text(TEXT, encoding="utf-8")

    report = prepare(root, encoder=codec)
    assert report.train_rows == 1
    assert [s.reason for s in report.skips] == [TOO_SHORT]
    assert "under the 0.5 s minimum" in report.skips[0].detail


def test_a_silent_recording_is_reported(tmp_path, codec):
    root = tmp_path / "recordings"
    build_corpus(root, count=1)
    write_wav(root / "silence.wav", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    (root / "silence.txt").write_text(TEXT, encoding="utf-8")

    report = prepare(root, encoder=codec)
    assert [s.reason for s in report.skips] == [SILENT]


def test_an_undecodable_file_is_reported_not_raised(tmp_path, codec):
    root = tmp_path / "recordings"
    build_corpus(root, count=1)
    (root / "broken.wav").write_bytes(b"this is not a wav file")
    (root / "broken.txt").write_text(TEXT, encoding="utf-8")

    report = prepare(root, encoder=codec)
    assert report.train_rows == 1
    assert [s.reason for s in report.skips] == [UNREADABLE]
    assert report.skips[0].detail


def test_a_recording_without_a_transcript_is_reported(tmp_path, codec):
    root = tmp_path / "recordings"
    build_corpus(root, count=1)
    write_wav(root / "orphan.wav", speech_like(2.0))

    report = prepare(root, encoder=codec, transcribe=False)
    assert [s.reason for s in report.skips] == [NO_TRANSCRIPT]
    assert "--no-transcribe" in report.skips[0].detail


def test_a_row_over_the_token_budget_is_dropped_here_not_by_the_trainer(recordings, codec):
    report = prepare(recordings, encoder=codec, max_tokens=100)
    assert report.train_rows == 0
    assert {s.reason for s in report.skips} == {TOO_LONG}
    assert "100-token row limit" in report.skips[0].detail


def test_a_long_recording_with_a_transcript_is_not_split_behind_the_users_back(tmp_path, codec):
    root = tmp_path / "recordings"
    write_wav(root / "long.wav", join(speech_like(4.0), quiet(1.0), speech_like(4.0)))
    (root / "long.txt").write_text(TEXT, encoding="utf-8")

    report = prepare(root, encoder=codec, settings=SegmentSettings(max_seconds=5.0))
    assert report.train_rows == 0
    assert report.skips[0].reason == TOO_LONG
    assert "--transcribe" in report.skips[0].detail


def test_the_summary_names_every_reason(tmp_path, codec):
    root = tmp_path / "recordings"
    build_corpus(root, count=1)
    write_wav(root / "tiny.wav", speech_like(0.2))
    (root / "tiny.txt").write_text(TEXT, encoding="utf-8")
    write_wav(root / "silence.wav", np.zeros(SAMPLE_RATE * 2, dtype=np.float32))
    (root / "silence.txt").write_text(TEXT, encoding="utf-8")

    summary = prepare(root, encoder=codec).summary()
    assert "skipped" in summary
    assert TOO_SHORT in summary and SILENT in summary
    assert "tiny.wav" in summary and "silence.wav" in summary
    assert "1 rows" in summary or "(1 rows)" in summary


def test_the_summary_reports_totals(recordings, codec):
    summary = prepare(recordings, encoder=codec).summary()
    assert "3 audio files" in summary
    assert "3 encoded" in summary
    assert "0:06 total" in summary


# ------------------------------------------------------------------------------- val split


def test_val_split_writes_a_second_file_beside_the_first(tmp_path, codec):
    root = build_corpus(tmp_path / "recordings", count=10)
    report = prepare(root, encoder=codec, val_split=0.2)

    assert report.val_output == root / "val.jsonl"
    assert report.train_rows == 8
    assert report.val_rows == 2
    train = {row["id"] for row in read_corpus(root / "train.jsonl")}
    val = {row["id"] for row in read_corpus(root / "val.jsonl")}
    assert not train & val
    assert len(train | val) == 10


def test_val_split_is_reproducible_under_a_fixed_seed(tmp_path, codec):
    root = build_corpus(tmp_path / "recordings", count=10)

    def held_out(seed, name):
        prepare(root, root / f"{name}.jsonl", encoder=FakeCodec(), val_split=0.3, seed=seed)
        return {row["id"] for row in read_corpus(root / f"{name}.val.jsonl")}

    assert held_out(7, "a") == held_out(7, "b")
    assert held_out(7, "a") != held_out(99, "c")


def test_val_output_can_be_named(tmp_path, codec):
    root = build_corpus(tmp_path / "recordings", count=10)
    elsewhere = tmp_path / "splits" / "held_out.jsonl"
    prepare(root, encoder=codec, val_split=0.2, val_output=elsewhere)
    assert len(read_corpus(elsewhere)) == 2


def test_a_corpus_too_small_to_split_still_writes_its_training_rows(tmp_path, codec):
    root = build_corpus(tmp_path / "recordings", count=1)
    report = prepare(root, encoder=codec, val_split=0.5)
    assert report.train_rows == 1
    assert report.val_rows == 0
    assert any("too small" in note for note in report.warnings())


@pytest.mark.parametrize("value", [-0.1, 1.0, 2.0])
def test_an_impossible_val_split_is_rejected_up_front(recordings, codec, value):
    with pytest.raises(ValueError, match="val_split"):
        prepare(recordings, encoder=codec, val_split=value)


# ------------------------------------------------------------------- rerunning and resuming


def test_rerunning_over_the_same_folder_does_not_duplicate_rows(recordings, codec):
    first = prepare(recordings, encoder=codec)
    before = (recordings / "train.jsonl").read_text(encoding="utf-8")

    second = prepare(recordings, encoder=FakeCodec())
    assert second.train_rows == first.train_rows
    assert (recordings / "train.jsonl").read_text(encoding="utf-8") == before


def test_rerunning_encodes_nothing_that_is_already_done(recordings, codec):
    prepare(recordings, encoder=codec)
    again = FakeCodec()
    report = prepare(recordings, encoder=again)

    assert again.clips_encoded == 0
    assert report.emitted == 0
    assert report.reused == 3


def test_a_new_recording_is_added_to_the_existing_corpus(recordings, codec):
    prepare(recordings, encoder=codec)
    write_wav(recordings / "clip_09.wav", speech_like(2.0))
    (recordings / "clip_09.txt").write_text(TEXT, encoding="utf-8")

    again = FakeCodec()
    report = prepare(recordings, encoder=again)
    assert again.clips_encoded == 1
    assert report.train_rows == 4
    assert [row["id"] for row in read_corpus(recordings / "train.jsonl")][-1] == "clip_09.wav#0"


def test_resuming_keeps_the_held_out_rows_out_of_the_training_file(tmp_path):
    root = build_corpus(tmp_path / "recordings", count=10)
    prepare(root, encoder=FakeCodec(), val_split=0.2)
    write_wav(root / "clip_99.wav", speech_like(2.0))
    (root / "clip_99.txt").write_text(TEXT, encoding="utf-8")

    again = FakeCodec()
    report = prepare(root, encoder=again, val_split=0.2)
    assert again.clips_encoded == 1
    train = {row["id"] for row in read_corpus(root / "train.jsonl")}
    val = {row["id"] for row in read_corpus(root / "val.jsonl")}
    assert not train & val
    assert len(train | val) == 11
    assert report.reused == 10


def test_overwrite_starts_the_corpus_again(recordings, codec):
    prepare(recordings, encoder=codec)
    again = FakeCodec()
    report = prepare(recordings, encoder=again, overwrite=True)
    assert again.clips_encoded == 3
    assert report.reused == 0
    assert report.train_rows == 3


def test_rows_written_by_hand_are_not_thrown_away(recordings, codec):
    corpus = recordings / "train.jsonl"
    corpus.write_text(json.dumps({"text": "someone else's row"}) + "\n", encoding="utf-8")
    prepare(recordings, encoder=codec)
    texts = [row["text"] for row in read_corpus(corpus)]
    assert "someone else's row" in texts
    assert len(texts) == 4


def test_a_corrupt_line_in_the_existing_corpus_does_not_stop_the_run(recordings, codec):
    (recordings / "train.jsonl").write_text("{not json\n", encoding="utf-8")
    report = prepare(recordings, encoder=codec)
    assert report.train_rows == 3


# --------------------------------------------------------------------------------- dry run


def test_a_dry_run_writes_nothing_and_loads_no_codec(recordings):
    report = prepare(recordings, dry_run=True)
    assert not (recordings / "train.jsonl").exists()
    assert report.dry_run
    assert report.train_rows == 3
    assert report.total_seconds > 0
    assert "nothing (--dry-run)" in report.summary()


# ------------------------------------------------------------------------------ ASR fallback


class FakeTranscriber:
    """Returns a fixed line per call, and counts them."""

    def __init__(self, text="Transcribed by the model.") -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, wav, sample_rate=SAMPLE_RATE) -> str:
        self.calls += 1
        return f"{self.text} {self.calls}" if self.text else ""


def test_recordings_without_transcripts_are_transcribed(tmp_path, codec):
    root = tmp_path / "recordings"
    write_wav(root / "a.wav", speech_like(2.0))
    write_wav(root / "b.wav", speech_like(2.0))

    asr = FakeTranscriber()
    report = prepare(root, encoder=codec, transcriber=asr)
    assert asr.calls == 2
    assert report.transcribed == 2
    assert "Transcribed by the model." in read_corpus(root / "train.jsonl")[0]["text"]


def test_supplied_transcripts_are_not_re_transcribed(recordings, codec):
    asr = FakeTranscriber()
    prepare(recordings, encoder=codec, transcriber=asr)
    assert asr.calls == 0


def test_forcing_transcription_replaces_the_supplied_text(recordings, codec):
    asr = FakeTranscriber()
    prepare(recordings, encoder=codec, transcriber=asr, transcribe=True)
    assert asr.calls == 3
    assert TEXT not in read_corpus(recordings / "train.jsonl")[0]["text"]


def test_forcing_transcription_lets_a_long_recording_be_split(tmp_path, codec):
    root = tmp_path / "recordings"
    write_wav(root / "long.wav", join(speech_like(4.0), quiet(1.0), speech_like(4.0)))
    (root / "long.txt").write_text(TEXT, encoding="utf-8")

    report = prepare(
        root,
        encoder=codec,
        transcriber=FakeTranscriber(),
        transcribe=True,
        settings=SegmentSettings(max_seconds=5.0),
    )
    assert report.split == 1
    assert [row["id"] for row in read_corpus(root / "train.jsonl")] == ["long.wav#0", "long.wav#1"]


def test_a_clip_the_model_heard_nothing_in_is_dropped(tmp_path, codec):
    root = tmp_path / "recordings"
    write_wav(root / "a.wav", speech_like(2.0))
    report = prepare(root, encoder=codec, transcriber=FakeTranscriber(text=""))
    assert report.train_rows == 0
    assert report.skips[0].reason == "empty transcript"


def test_a_missing_asr_install_is_explained_per_clip(tmp_path, codec, monkeypatch):
    from kova_tts.data.asr import MissingDependency

    def missing(_options):
        raise MissingDependency("Transcription needs faster-whisper ... kova-tts[data]")

    monkeypatch.setattr(prep, "_build_transcriber", missing)
    root = tmp_path / "recordings"
    write_wav(root / "a.wav", speech_like(2.0))

    report = prepare(root, encoder=codec)
    assert report.train_rows == 0
    assert report.skips[0].reason == NO_TRANSCRIPT
    assert "kova-tts[data]" in report.skips[0].detail


# ------------------------------------------------------------------------- the token estimate


@pytest.mark.parametrize(
    "text",
    ["", "a", "Hello there.", "A much longer line of ordinary English prose, spoken aloud."],
)
def test_the_estimate_never_undercounts_a_byte_level_tokenizer(text):
    from kova_tts.data.report import estimate_tokens

    # A byte-level BPE cannot emit more tokens than the string has UTF-8 bytes, so that plus
    # one token per code plus the five structural tags is the real ceiling.
    ceiling = 100 + len(text.strip().encode("utf-8")) + 5
    assert estimate_tokens(text, 100) >= ceiling


def test_the_estimate_bounds_a_multibyte_transcript():
    from kova_tts.data.report import estimate_tokens

    # Scripts outside the vocabulary's comfort zone tokenize far worse than characters-per-token
    # averages suggest; the bound has to survive them.
    assert estimate_tokens("ありがとうございます", 0) >= len("ありがとうございます")


# --------------------------------------------------------------------------------------- CLI


def test_cli_dry_run(recordings, capsys):
    assert main([str(recordings), "--dry-run"]) == 0
    assert "3 audio files" in capsys.readouterr().out


def test_cli_writes_a_corpus(recordings, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep, "load_codec", lambda **_: FakeCodec())
    out = tmp_path / "data" / "train.jsonl"
    assert main([str(recordings), "-o", str(out), "--val-split", "0.34", "--seed", "1"]) == 0

    assert len(read_corpus(out)) == 2
    assert len(read_corpus(tmp_path / "data" / "val.jsonl")) == 1
    assert str(out) in capsys.readouterr().out


def test_cli_reports_a_bad_source_without_a_traceback(tmp_path, capsys):
    assert main([str(tmp_path / "nowhere"), "--dry-run"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_returns_one_when_nothing_could_be_used(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(prep, "load_codec", lambda **_: FakeCodec())
    root = tmp_path / "recordings"
    write_wav(root / "a.wav", speech_like(2.0))
    assert main([str(root), "--no-transcribe"]) == 1


def test_cli_flags_map_onto_the_settings(recordings, capsys):
    assert main([str(recordings), "--dry-run", "--max-seconds", "1.0", "--min-seconds", "0.1"]) == 0
    assert "too long" in capsys.readouterr().out
