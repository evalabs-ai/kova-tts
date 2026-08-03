"""Finding recordings, and pairing each one with the right transcript.

The layouts covered here are the ones users actually arrive with: sidecar text files, a CSV
whose columns are spelled however the exporting tool spelled them, a TSV, a headerless
pipe-delimited manifest, JSONL, and a plain filename-to-text JSON mapping.
"""

from __future__ import annotations

import json

import pytest
from test_data_synth import write_clip

from kova_tts.data.discover import (
    AUDIO_SUFFIXES,
    DiscoveryError,
    clean_text,
    discover,
    find_audio,
    read_metadata,
)


@pytest.fixture
def recordings(tmp_path):
    """Three short recordings, ``a.wav`` / ``b.wav`` / ``nested/c.wav``."""
    root = tmp_path / "recordings"
    write_clip(root / "a.wav", 1.0)
    write_clip(root / "b.wav", 1.0)
    write_clip(root / "nested" / "c.wav", 1.0)
    return root


# --------------------------------------------------------------------------------- find_audio


def test_find_audio_is_recursive_and_sorted(recordings):
    found = find_audio(recordings)
    assert [p.relative_to(recordings).as_posix() for p in found] == [
        "a.wav",
        "b.wav",
        "nested/c.wav",
    ]


def test_find_audio_can_stay_shallow(recordings):
    found = find_audio(recordings, recursive=False)
    assert [p.name for p in found] == ["a.wav", "b.wav"]


def test_find_audio_ignores_non_audio_and_hidden_files(recordings):
    (recordings / "notes.md").write_text("not audio", encoding="utf-8")
    write_clip(recordings / ".hidden.wav", 1.0)
    write_clip(recordings / ".cache" / "d.wav", 1.0)
    assert [p.name for p in find_audio(recordings)] == ["a.wav", "b.wav", "c.wav"]


def test_find_audio_accepts_a_single_file(recordings):
    assert find_audio(recordings / "a.wav") == [recordings / "a.wav"]


def test_missing_directory_says_what_to_point_at(tmp_path):
    with pytest.raises(DiscoveryError, match="folder holding your recordings"):
        find_audio(tmp_path / "nope")


def test_directory_without_audio_lists_the_extensions(tmp_path):
    (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")
    with pytest.raises(DiscoveryError, match=r"\.flac"):
        discover(tmp_path)


# ----------------------------------------------------------------------------------- sidecars


@pytest.mark.parametrize("suffix", [".txt", ".lab"])
def test_sidecar_transcripts_are_paired(recordings, suffix):
    (recordings / f"a{suffix}").write_text("Hello there.\n", encoding="utf-8")
    found = discover(recordings)
    texts = {clip.path.name: clip.text for clip in found.clips}
    assert texts == {"a.wav": "Hello there.", "b.wav": None, "c.wav": None}
    assert "sidecar" in found.transcript_source


def test_sidecar_newlines_are_collapsed(recordings):
    (recordings / "a.txt").write_text("Hello\nthere,\n  world.\n", encoding="utf-8")
    found = discover(recordings)
    assert found.clips[0].text == "Hello there, world."


def test_clean_text_collapses_all_whitespace():
    assert clean_text("  a\t\nb  c \n") == "a b c"


# ------------------------------------------------------------------------- metadata: spellings


@pytest.mark.parametrize(
    ("header", "row"),
    [
        ("file_name,text", "a.wav,Hello there."),
        ("filename,transcript", "a.wav,Hello there."),
        ("audio,sentence", "a.wav,Hello there."),
        ("path,normalized_text", "a.wav,Hello there."),
        ("audio_filepath,transcription", "a.wav,Hello there."),
        ("Audio File,Text", "a.wav,Hello there."),
        ("id,text,duration", "a.wav,Hello there.,1.0"),
    ],
)
def test_csv_column_spellings(recordings, header, row):
    (recordings / "metadata.csv").write_text(f"{header}\n{row}\n", encoding="utf-8")
    found = discover(recordings)
    assert found.clips[0].text == "Hello there."


def test_metadata_description_names_the_columns(recordings):
    (recordings / "metadata.csv").write_text(
        "filename,transcript\na.wav,Hello there.\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert "'filename' -> 'transcript'" in found.transcript_source
    assert "1 rows" in found.transcript_source


def test_extra_columns_are_ignored(recordings):
    (recordings / "metadata.csv").write_text(
        "speaker,file_name,duration,text\nx,a.wav,1.0,Hello there.\n", encoding="utf-8"
    )
    assert discover(recordings).clips[0].text == "Hello there."


def test_quoted_commas_survive(recordings):
    (recordings / "metadata.csv").write_text(
        'file_name,text\na.wav,"Hello, there."\n', encoding="utf-8"
    )
    assert discover(recordings).clips[0].text == "Hello, there."


def test_byte_order_mark_does_not_break_the_header(recordings):
    (recordings / "metadata.csv").write_text(
        "﻿file_name,text\na.wav,Hello there.\n", encoding="utf-8"
    )
    assert discover(recordings).clips[0].text == "Hello there."


# ------------------------------------------------------------------------- metadata: formats


def test_tsv(recordings):
    (recordings / "metadata.tsv").write_text(
        "file_name\ttext\na.wav\tHello there.\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert found.clips[0].text == "Hello there."
    assert "'\\t' delimited" in found.transcript_source


def test_headerless_pipe_delimited(recordings):
    (recordings / "metadata.csv").write_text(
        "a|Hello there.|hello there\nb|Second one.|second one\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert [c.text for c in found.clips] == ["Hello there.", "Second one.", None]
    assert "no header" in found.transcript_source


def test_jsonl(recordings):
    rows = [
        {"audio_filepath": "a.wav", "text": "Hello there.", "duration": 1.0},
        {"audio_filepath": "nested/c.wav", "text": "Third one."},
    ]
    (recordings / "metadata.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert [c.text for c in found.clips] == ["Hello there.", None, "Third one."]


def test_json_array(tmp_path, recordings):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"file": "a.wav", "caption": "Hello there."}]), encoding="utf-8"
    )
    assert discover(recordings, metadata=manifest).clips[0].text == "Hello there."


def test_json_filename_to_text_mapping(tmp_path, recordings):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"a.wav": "Hello there."}), encoding="utf-8")
    found = discover(recordings, metadata=manifest)
    assert found.clips[0].text == "Hello there."
    assert "mapping" in found.transcript_source


def test_broken_jsonl_names_the_line(recordings):
    (recordings / "metadata.jsonl").write_text('{"file_name": "a.wav"}\n{oops\n', encoding="utf-8")
    with pytest.raises(DiscoveryError, match="line 2"):
        discover(recordings)


def test_unrecognisable_json_fields_say_what_to_rename(recordings):
    (recordings / "metadata.jsonl").write_text('{"clip_ref": "a.wav", "words": "hi"}\n', "utf-8")
    with pytest.raises(DiscoveryError, match="Rename them"):
        discover(recordings)


def test_single_column_table_is_rejected(recordings):
    (recordings / "metadata.csv").write_text("a.wav\nb.wav\n", encoding="utf-8")
    with pytest.raises(DiscoveryError, match="nothing to pair"):
        discover(recordings)


# --------------------------------------------------------------------------- matching by name


@pytest.mark.parametrize(
    "name",
    ["a.wav", "a", "./a.wav", "/somewhere/else/a.wav", "A.WAV"],
)
def test_filenames_match_by_path_basename_and_stem(recordings, name):
    (recordings / "metadata.csv").write_text(f"file_name,text\n{name},Hello there.\n", "utf-8")
    assert discover(recordings).clips[0].text == "Hello there."


def test_relative_paths_match_a_nested_recording(recordings):
    (recordings / "metadata.csv").write_text(
        "file_name,text\nnested/c.wav,Third one.\n", encoding="utf-8"
    )
    assert discover(recordings).clips[2].text == "Third one."


def test_rows_naming_a_missing_file_are_reported(recordings):
    (recordings / "metadata.csv").write_text(
        "file_name,text\na.wav,Hello there.\ngone.wav,Nowhere.\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert found.unmatched_rows == ("gone.wav",)
    assert found.clips[0].text == "Hello there."


def test_a_name_matching_two_recordings_is_dropped_not_guessed(tmp_path):
    root = tmp_path / "recordings"
    write_clip(root / "one" / "a.wav", 1.0)
    write_clip(root / "two" / "a.wav", 1.0)
    (root / "metadata.csv").write_text("file_name,text\na.wav,Ambiguous.\n", encoding="utf-8")

    found = discover(root)
    assert found.ambiguous_names == ("a.wav",)
    assert [clip.text for clip in found.clips] == [None, None]


def test_metadata_wins_over_a_sidecar(recordings):
    (recordings / "a.txt").write_text("From the sidecar.", encoding="utf-8")
    (recordings / "metadata.csv").write_text(
        "file_name,text\na.wav,From the manifest.\n", encoding="utf-8"
    )
    assert discover(recordings).clips[0].text == "From the manifest."


def test_metadata_and_sidecars_can_be_mixed(recordings):
    (recordings / "b.txt").write_text("From the sidecar.", encoding="utf-8")
    (recordings / "metadata.csv").write_text(
        "file_name,text\na.wav,From the manifest.\n", encoding="utf-8"
    )
    found = discover(recordings)
    assert [c.text for c in found.clips] == ["From the manifest.", "From the sidecar.", None]
    assert len(found.with_text) == 2
    assert len(found.without_text) == 1


def test_metadata_is_auto_detected_by_name(recordings):
    (recordings / "transcripts.csv").write_text(
        "file_name,text\na.wav,Hello there.\n", encoding="utf-8"
    )
    assert discover(recordings).clips[0].text == "Hello there."


def test_explicit_metadata_path_outside_the_source_directory(tmp_path, recordings):
    manifest = tmp_path / "elsewhere.csv"
    manifest.write_text("file_name,text\na.wav,Hello there.\n", encoding="utf-8")
    assert discover(recordings, metadata=manifest).clips[0].text == "Hello there."


def test_missing_metadata_file_is_an_error(recordings):
    with pytest.raises(DiscoveryError, match="Metadata file not found"):
        discover(recordings, metadata=recordings / "nope.csv")


def test_read_metadata_returns_rows_and_a_description(tmp_path):
    manifest = tmp_path / "metadata.csv"
    manifest.write_text("file_name,text\na.wav,One.\nb.wav,Two.\n", encoding="utf-8")
    rows, description = read_metadata(manifest)
    assert rows == [("a.wav", "One."), ("b.wav", "Two.")]
    assert description.startswith("metadata.csv:")


def test_every_audio_suffix_is_lowercase_and_dotted():
    assert all(s.startswith(".") and s.islower() for s in AUDIO_SUFFIXES)


def test_no_supported_extension_can_be_committed():
    """Every format this pipeline reads must be one ``.gitignore`` blocks.

    No recording is committed to this repository, and an extension the discovery step accepts
    but ``.gitignore`` has never heard of is exactly how the first one would arrive.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    ignored = (root / ".gitignore").read_text(encoding="utf-8").split()
    missing = [s for s in AUDIO_SUFFIXES if f"*{s}" not in ignored]
    assert not missing, f".gitignore does not block {missing}"
