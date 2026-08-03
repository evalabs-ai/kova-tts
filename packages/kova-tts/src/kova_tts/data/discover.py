"""Finding recordings and pairing each one with its transcript.

Two layouts, because those are the two ways people actually have their data:

* **Sidecars** -- ``clip.wav`` beside ``clip.txt``. Nothing to configure.
* **A metadata file** -- one row per recording naming the file and its text. CSV, TSV and
  JSONL, with whatever column names the exporting tool happened to use.

Column names are the awkward part: the same file gets written with ``file_name``/``text``,
``audio``/``transcript``, ``path``/``sentence``. Rather than demand one spelling this module
accepts the common ones and *reports which pair it matched*, so a mis-read column shows up in
the summary instead of silently emptying the corpus.

Filenames from a metadata file are matched against the discovered audio by relative path, then
basename, then stem, so an absolute path exported on another machine still lines up. A name
that would match two different recordings is dropped as ambiguous: guessing there attaches the
wrong transcript to a clip, which is worse than dropping it and saying so.
"""

from __future__ import annotations

import csv
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path

#: Extensions treated as recordings. Everything here is readable by ``soundfile``, except that
#: mp3/m4a support depends on the local libsndfile -- a file that cannot be decoded is reported
#: as a skip rather than assumed absent.
#:
#: This tuple is kept in step with the audio extensions ``.gitignore`` blocks, and a test
#: enforces that. No recording is ever committed to this repository, and an extension the
#: pipeline accepts but ``.gitignore`` does not know about would be the way one slipped in.
AUDIO_SUFFIXES = (".wav", ".flac", ".ogg", ".opus", ".mp3", ".m4a")

#: Sidecar transcript extensions, tried in this order next to each recording. ``.lab`` is what
#: forced aligners emit.
SIDECAR_SUFFIXES = (".txt", ".lab")

#: Metadata filenames looked for in the source directory when none is given explicitly.
METADATA_NAMES = (
    "metadata.csv",
    "metadata.tsv",
    "metadata.jsonl",
    "metadata.json",
    "transcripts.csv",
    "transcripts.tsv",
    "transcripts.jsonl",
    "transcripts.json",
)

#: Column names that hold the recording's filename, in preference order.
FILENAME_KEYS = (
    "file_name",
    "filename",
    "file",
    "audio_file",
    "audio_filepath",
    "audio_path",
    "audio",
    "wav_path",
    "wav",
    "path",
    "clip",
    "utterance_id",
    "id",
    "name",
)

#: Column names that hold the transcript, in preference order.
TEXT_KEYS = (
    "text",
    "transcript",
    "transcription",
    "sentence",
    "normalized_text",
    "text_normalized",
    "raw_text",
    "utterance",
    "caption",
)

#: Delimiters sniffed in a headerless or unlabelled table.
DELIMITERS = (",", "\t", "|", ";")


class DiscoveryError(ValueError):
    """The source directory or metadata file cannot be used, with the reason in the message."""


@dataclass(frozen=True, slots=True)
class Clip:
    """One recording and the transcript found for it, if any."""

    path: Path

    #: ``None`` means no transcript was found; the caller decides whether to transcribe it.
    text: str | None = None


@dataclass(frozen=True, slots=True)
class Discovery:
    """What a scan of a source directory turned up."""

    root: Path
    clips: tuple[Clip, ...]

    #: Human description of where transcripts came from, e.g.
    #: ``"metadata.csv: 'file_name' -> 'text', ',' delimited (412 rows)"``. Printed verbatim in
    #: the run summary, so a wrong column is visible without re-reading the file.
    transcript_source: str

    #: Metadata rows naming a file that was not found under `root`.
    unmatched_rows: tuple[str, ...] = ()

    #: Metadata names that matched more than one recording and were therefore not used.
    ambiguous_names: tuple[str, ...] = ()

    @property
    def with_text(self) -> tuple[Clip, ...]:
        return tuple(c for c in self.clips if c.text)

    @property
    def without_text(self) -> tuple[Clip, ...]:
        return tuple(c for c in self.clips if not c.text)


# ---------------------------------------------------------------------------------- filesystem


def find_audio(source: str | os.PathLike[str], *, recursive: bool = True) -> list[Path]:
    """Every recording under `source`, sorted by relative path so runs reproduce.

    `source` may also be a single audio file, which is what makes ``prepare one-clip.wav``
    work without a special case anywhere else. Hidden files and directories are ignored: a
    stray ``.DS_Store`` or an editor's ``.#clip.wav`` lock is never training data.
    """
    root = Path(source).expanduser()
    if root.is_file():
        return [root]
    if not root.is_dir():
        raise DiscoveryError(
            f"No such directory: {root}. Point this at the folder holding your recordings."
        )

    suffixes = {s.lower() for s in AUDIO_SUFFIXES}
    walker = root.rglob("*") if recursive else root.glob("*")
    found = [
        path
        for path in walker
        if path.is_file()
        and path.suffix.lower() in suffixes
        and not any(part.startswith(".") for part in path.relative_to(root).parts)
    ]
    return sorted(found, key=lambda p: p.relative_to(root).as_posix())


def read_sidecar(path: Path) -> str | None:
    """Transcript sitting next to `path` as ``.txt``/``.lab``, or ``None``."""
    for suffix in SIDECAR_SUFFIXES:
        sidecar = path.with_suffix(suffix)
        if sidecar.is_file():
            text = clean_text(sidecar.read_text(encoding="utf-8", errors="replace"))
            if text:
                return text
    return None


def find_metadata(root: Path) -> Path | None:
    """The first recognised metadata filename directly inside `root`, or ``None``."""
    for name in METADATA_NAMES:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


# ------------------------------------------------------------------------------ metadata files


def clean_text(text: str) -> str:
    """Collapse a transcript to one line of single-spaced text.

    Line breaks matter here: a sidecar written by a text editor usually ends in a newline, and
    a transcript that reaches :func:`~kova_tts.prompt.training_example` with an embedded
    newline produces a JSONL row whose text does not match anything the model saw in training.
    """
    return " ".join(str(text).split())


def read_metadata(path: str | os.PathLike[str]) -> tuple[list[tuple[str, str]], str]:
    """Read a metadata file into ``[(filename, text), ...]`` plus a description of what it is.

    The description names the columns that were matched and the row count; it exists so the
    run summary can show the user what was read without them opening the file.
    """
    file = Path(path).expanduser()
    if not file.is_file():
        raise DiscoveryError(f"Metadata file not found: {file}")

    suffix = file.suffix.lower()
    if suffix in (".jsonl", ".ndjson", ".json"):
        rows, detail = _read_json(file)
    else:
        rows, detail = _read_table(file)

    if not rows:
        raise DiscoveryError(
            f"{file.name} has no usable rows. Each row needs a filename column (one of "
            f"{', '.join(FILENAME_KEYS[:4])}, ...) and a text column (one of "
            f"{', '.join(TEXT_KEYS[:4])}, ...)."
        )
    return rows, f"{file.name}: {detail} ({len(rows)} rows)"


def _normalise_key(key: str) -> str:
    """Fold a column name to its canonical spelling: lowercase, underscores, no BOM."""
    return key.strip().lstrip("﻿").lower().replace(" ", "_").replace("-", "_")


def _pick_column(header: list[str], candidates: tuple[str, ...]) -> int | None:
    """Index of the first column in `header` matching `candidates`, in candidate order."""
    normalised = [_normalise_key(name) for name in header]
    for wanted in candidates:
        if wanted in normalised:
            return normalised.index(wanted)
    return None


def _sniff_delimiter(sample: str, suffix: str) -> str:
    """Delimiter of a table, from its extension when unambiguous and its content otherwise."""
    if suffix == ".tsv":
        return "\t"
    first = sample.splitlines()[0] if sample.splitlines() else ""
    counts = {d: first.count(d) for d in DELIMITERS}
    best = max(counts, key=lambda d: counts[d])
    return best if counts[best] else ","


def _read_table(file: Path) -> tuple[list[tuple[str, str]], str]:
    """CSV/TSV, with or without a header row."""
    # utf-8-sig so a spreadsheet export's BOM does not become part of the first column name.
    text = file.read_text(encoding="utf-8-sig", errors="replace")
    delimiter = _sniff_delimiter(text, file.suffix.lower())
    records = [row for row in csv.reader(io.StringIO(text), delimiter=delimiter) if row]
    if not records:
        raise DiscoveryError(f"{file} is empty.")

    header = records[0]
    name_col = _pick_column(header, FILENAME_KEYS)
    text_col = _pick_column(header, TEXT_KEYS)
    shown = repr(delimiter)

    if name_col is not None and text_col is not None:
        detail = f"{header[name_col]!r} -> {header[text_col]!r}, {shown} delimited"
        body = records[1:]
    else:
        # No recognisable header, so the file is positional: first field names the recording,
        # second holds the text. This is the shape most hand-written and legacy manifests take,
        # and refusing it would send users off to rename columns for no reason.
        if len(header) < 2:
            raise DiscoveryError(
                f"{file.name} has one column per row, so there is nothing to pair. Give it a "
                f"header row naming a filename column and a text column, e.g. "
                f"'file_name{delimiter}text'."
            )
        name_col, text_col = 0, 1
        detail = f"no header, column 1 -> column 2, {shown} delimited"
        body = records

    rows = [
        (row[name_col].strip(), clean_text(row[text_col]))
        for row in body
        if len(row) > max(name_col, text_col) and row[name_col].strip()
    ]
    return rows, detail


def _read_json(file: Path) -> tuple[list[tuple[str, str]], str]:
    """JSONL (one object per line), a JSON array of objects, or a ``{name: text}`` mapping."""
    raw = file.read_text(encoding="utf-8-sig", errors="replace")
    records: list[object]
    shape: str

    stripped = raw.lstrip()
    if stripped.startswith("{") and file.suffix.lower() == ".json":
        mapping = json.loads(raw)
        if all(isinstance(v, str) for v in mapping.values()):
            rows = [(str(k).strip(), clean_text(v)) for k, v in mapping.items()]
            return rows, "filename -> text mapping"
        records = list(mapping.values())
        shape = "JSON object"
    elif stripped.startswith("["):
        records = json.loads(raw)
        shape = "JSON array"
    else:
        records = []
        shape = "JSONL"
        for number, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise DiscoveryError(f"{file.name} line {number} is not valid JSON: {exc}") from exc

    objects = [r for r in records if isinstance(r, dict)]
    if not objects:
        raise DiscoveryError(
            f"{file.name} holds no JSON objects. Expected one object per line with a filename "
            f"field and a text field."
        )

    keys = list(objects[0])
    name_col = _pick_column(keys, FILENAME_KEYS)
    text_col = _pick_column(keys, TEXT_KEYS)
    if name_col is None or text_col is None:
        raise DiscoveryError(
            f"{file.name} has no recognisable filename/text fields (found: "
            f"{', '.join(keys) or 'nothing'}). Rename them to 'file_name' and 'text'."
        )
    name_key, text_key = keys[name_col], keys[text_col]

    rows = []
    for record in objects:
        name = str(record.get(name_key, "")).strip()
        if name:
            rows.append((name, clean_text(record.get(text_key, ""))))
    return rows, f"{shape}, {name_key!r} -> {text_key!r}"


# ---------------------------------------------------------------------------------- pairing


def _lookup_index(files: list[Path], root: Path) -> dict[str, Path | None]:
    """Every way a metadata row might name each file -> that file.

    ``None`` marks a key that two different recordings both answer to; looking one up returns
    no match rather than an arbitrary one.
    """
    index: dict[str, Path | None] = {}
    for file in files:
        rel = file.relative_to(root).as_posix() if file.is_relative_to(root) else file.name
        keys = {rel, rel.rsplit(".", 1)[0], file.name, file.stem, str(file)}
        for key in keys:
            for variant in (key, key.lower()):
                if variant in index and index[variant] != file:
                    index[variant] = None
                else:
                    index.setdefault(variant, file)
    return index


def _match(name: str, index: dict[str, Path | None]) -> Path | None | str:
    """Resolve a metadata filename. Returns the file, ``None`` if unknown, ``"ambiguous"``."""
    as_path = Path(name)
    candidates = [
        name,
        name.lstrip("./"),
        as_path.as_posix(),
        as_path.name,
        as_path.stem,
    ]
    ambiguous = False
    for candidate in candidates:
        for variant in (candidate, candidate.lower()):
            if variant not in index:
                continue
            hit = index[variant]
            if hit is not None:
                return hit
            ambiguous = True
    return "ambiguous" if ambiguous else None


def discover(
    source: str | os.PathLike[str],
    *,
    metadata: str | os.PathLike[str] | None = None,
    recursive: bool = True,
) -> Discovery:
    """Scan `source` for recordings and attach a transcript to each one.

    Precedence is metadata file, then sidecar: an explicit manifest is a deliberate act, a
    stray ``.txt`` next to a clip may be anything. Clips left without text come back with
    ``text=None`` for the caller to transcribe or skip.
    """
    root = Path(source).expanduser()
    files = find_audio(root, recursive=recursive)
    if root.is_file():
        root = root.parent
    if not files:
        raise DiscoveryError(
            f"No audio files under {root}. Supported extensions: {', '.join(AUDIO_SUFFIXES)}."
        )

    metadata_file = Path(metadata).expanduser() if metadata is not None else find_metadata(root)

    texts: dict[Path, str] = {}
    unmatched: list[str] = []
    ambiguous: list[str] = []
    source_description = "none found"

    if metadata_file is not None:
        rows, source_description = read_metadata(metadata_file)
        index = _lookup_index(files, root)
        for name, text in rows:
            hit = _match(name, index)
            if hit == "ambiguous":
                ambiguous.append(name)
            elif hit is None:
                unmatched.append(name)
            elif text:
                texts[hit] = text  # type: ignore[index]

    sidecars = 0
    for file in files:
        if file in texts:
            continue
        sidecar = read_sidecar(file)
        if sidecar:
            texts[file] = sidecar
            sidecars += 1

    if sidecars:
        note = f"{sidecars} from {'/'.join(SIDECAR_SUFFIXES)} sidecars"
        source_description = f"{source_description}, {note}" if metadata_file else note

    return Discovery(
        root=root,
        clips=tuple(Clip(path=file, text=texts.get(file)) for file in files),
        transcript_source=source_description,
        unmatched_rows=tuple(unmatched),
        ambiguous_names=tuple(ambiguous),
    )
