"""The pipeline: a folder of recordings in, a finetuning corpus out.

    discover -> load, downmix, resample to 32 kHz -> split on silence -> trim -> -23 LUFS
             -> encode -> {"text": ...} JSONL

A recording made at 16 kHz or below is kept at 16 kHz rather than upsampled, and encoded
natively through the encoder's 16 kHz path.

Three things about the order are load-bearing. **Trim before normalise**, because leading
silence drags the integrated loudness of a clip down and the gain applied would then be wrong.
**Normalise before encode**, so every clip in the corpus reaches the codec at the same level
whatever it was recorded at. **Transcribe after segmenting**, because
a recording that gets cut into six clips needs six transcripts, and there is no honest way to
divide one transcript across the cuts -- so a clip that *has* a transcript is never split; it is
reported as too long and the user is told to cut it or pass ``--transcribe``.

Re-running is safe. Rows carry the clip they came from, so a second run over the same folder
reuses what is already in the output file and only encodes what is new. That makes the expensive
half restartable: a run interrupted at clip 3000 of 4000 picks up where it stopped.

The output is written by :func:`kova_tts.prompt.training_example` and nothing else. The JSONL
format is defined in exactly one place, and this module is not it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from kova_codec.constants import SAMPLE_RATE
from kova_tts import paths, prompt
from kova_tts.audio import encoder_input_rate, file_sample_rate, load_audio, normalize_loudness
from kova_tts.data.discover import Clip, DiscoveryError, clean_text, discover
from kova_tts.data.encode import Encoder, code_count, encode_clips, load_codec
from kova_tts.data.report import (
    EMPTY_TRANSCRIPT,
    NO_TRANSCRIPT,
    SILENT,
    TOO_LONG,
    TOO_SHORT,
    UNREADABLE,
    PrepareReport,
    estimate_tokens,
)
from kova_tts.data.segment import SegmentSettings, is_silent, split_on_silence, trim_silence

logger = logging.getLogger(__name__)

#: Default token ceiling, matching ``FinetuneConfig.max_length``. Rows above it are dropped by
#: the trainer, so they are dropped here instead, where the reason can be reported.
DEFAULT_MAX_TOKENS = 4096

#: Default corpus filename inside the source directory.
DEFAULT_OUTPUT_NAME = "train.jsonl"

#: Segmentation defaults, as an instance: a slotted dataclass has no readable class attributes,
#: so this is where argparse and the docs get the numbers from.
DEFAULTS = SegmentSettings()


@dataclass(slots=True)
class _Pending:
    """One trimmed, normalised clip on its way to the encoder."""

    path: Path
    rel: str
    segment: int
    wav: np.ndarray
    text: str | None
    #: The rate `wav` is at: 16 kHz for a source recorded at or below it, 32 kHz otherwise.
    rate: int = SAMPLE_RATE

    @property
    def row_id(self) -> str:
        return f"{self.rel}#{self.segment}"

    @property
    def seconds(self) -> float:
        return self.wav.size / self.rate


# ------------------------------------------------------------------------------- corpus files


def read_corpus(path: Path) -> list[dict[str, Any]]:
    """Read an existing JSONL corpus, ignoring blank and unparseable lines.

    Unparseable lines are logged and dropped rather than raised on: this file is being rewritten
    from what it already holds, and one corrupt line should not cost a completed run.
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("%s line %d is not valid JSON; dropping it", path, number)
            continue
        if isinstance(record, dict) and isinstance(record.get("text"), str):
            rows.append(record)
    return rows


def write_corpus(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL, atomically.

    Via a temporary file because the usual call rewrites a corpus that was just read: a crash
    halfway through a plain overwrite would leave the user with neither the old nor the new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    temporary.write_text(body, encoding="utf-8")
    temporary.replace(path)


def default_val_path(output: Path) -> Path:
    """Where the held-out corpus goes when it is not named explicitly.

    ``train.jsonl`` gets ``val.jsonl`` beside it, which is what a finetuning config expects to
    find; anything else gets ``<name>.val.jsonl`` so it cannot collide with a real corpus.
    """
    if output.stem == "train":
        return output.with_name(f"val{output.suffix}")
    return output.with_name(f"{output.stem}.val{output.suffix}")


# ------------------------------------------------------------------------------------ pipeline


def _segment_clip(
    clip: Clip,
    rel: str,
    *,
    settings: SegmentSettings,
    splittable: bool,
    result: PrepareReport,
) -> list[_Pending]:
    """Load one recording and turn it into trimmed, normalised, length-checked clips."""
    try:
        rate = encoder_input_rate(file_sample_rate(clip.path))
        wav = load_audio(clip.path, rate)
    except Exception as exc:  # noqa: BLE001 - any decode failure is reported, never fatal
        result.add_skip(clip.path, UNREADABLE, _reason(exc))
        return []

    if is_silent(wav):
        result.add_skip(clip.path, SILENT, "no signal above the noise floor")
        return []

    pieces = (
        split_on_silence(
            wav,
            rate,
            max_seconds=settings.max_seconds,
            top_db=settings.top_db,
            min_silence=settings.min_silence,
        )
        if splittable
        else [wav]
    )
    if len(pieces) > 1:
        result.split += 1

    pending: list[_Pending] = []
    for index, piece in enumerate(pieces):
        segment = index if len(pieces) > 1 else 0
        trimmed = trim_silence(
            piece, rate, top_db=settings.top_db, pad_seconds=settings.pad_seconds
        )
        seconds = trimmed.size / rate
        if trimmed.size == 0:
            result.add_skip(clip.path, SILENT, "silent after trimming", segment)
            continue
        if seconds < settings.min_seconds:
            result.add_skip(
                clip.path,
                TOO_SHORT,
                f"{seconds:.2f} s under the {settings.min_seconds:g} s minimum",
                segment,
            )
            continue
        if seconds > settings.max_seconds:
            detail = (
                f"{seconds:.1f} s with no silence to split on"
                if splittable
                else f"{seconds:.1f} s and has a transcript, so it cannot be split; "
                f"cut it yourself or re-run with --transcribe"
            )
            result.add_skip(clip.path, TOO_LONG, detail, segment)
            continue

        pending.append(
            _Pending(
                path=clip.path,
                rel=rel,
                segment=segment,
                wav=normalize_loudness(trimmed, rate),
                text=None if splittable else clip.text,
                rate=rate,
            )
        )
    return pending


def _transcribe_pending(
    pending: list[_Pending],
    *,
    transcriber: Any | None,
    make_transcriber: Any,
    result: PrepareReport,
    progress: bool,
) -> list[_Pending]:
    """Fill in the transcripts that are missing, dropping clips ASR could not read."""
    needed = [item for item in pending if not item.text]
    if not needed:
        return pending

    if transcriber is None:
        from kova_tts.data.asr import MissingDependency

        try:
            transcriber = make_transcriber()
        except MissingDependency as exc:
            for item in needed:
                result.add_skip(item.path, NO_TRANSCRIPT, str(exc), item.segment)
            return [item for item in pending if item.text]

    kept: list[_Pending] = []
    for done, item in enumerate(pending, start=1):
        if item.text:
            kept.append(item)
            continue
        item.text = clean_text(transcriber.transcribe(item.wav, item.rate))
        if not item.text:
            result.add_skip(item.path, EMPTY_TRANSCRIPT, "ASR heard no speech", item.segment)
            continue
        result.transcribed += 1
        kept.append(item)
        if progress:
            _show(f"transcribing {done}/{len(pending)} clips")
    if progress:
        _show("")
    return kept


def prepare(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str] | None = None,
    *,
    metadata: str | os.PathLike[str] | None = None,
    recursive: bool = True,
    settings: SegmentSettings | None = None,
    val_split: float = 0.0,
    val_output: str | os.PathLike[str] | None = None,
    seed: int = 42,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    transcribe: bool | None = None,
    transcriber: Any | None = None,
    asr_options: dict[str, Any] | None = None,
    encoder: Encoder | None = None,
    codec_options: dict[str, Any] | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
    progress: bool = False,
) -> PrepareReport:
    """Turn a folder of recordings into a JSONL finetuning corpus.

    Args:
        source: Directory of recordings, or a single audio file.
        output: JSONL to write. Defaults to ``train.jsonl`` inside `source`.
        metadata: Transcript manifest. Auto-detected in `source` when omitted.
        recursive: Descend into subdirectories.
        settings: Segmentation thresholds; see :class:`~kova_tts.data.segment.SegmentSettings`.
        val_split: Fraction held out into a second file, for the trainer's ``eval_loss``.
        val_output: Where that file goes. Defaults to ``val.jsonl`` beside `output`.
        seed: Seeds the held-out shuffle, and nothing else -- the rest is deterministic.
        max_tokens: Rows estimated longer than this are skipped, matching the trainer's
            ``max_length`` so a row is never silently dropped later instead of loudly now.
        transcribe: ``None`` transcribes only what has no transcript, ``True`` re-transcribes
            everything (and so allows long recordings to be split), ``False`` never transcribes.
        transcriber: A loaded :class:`~kova_tts.data.asr.Transcriber`, if one already exists.
        asr_options: Passed to :func:`~kova_tts.data.asr.load_transcriber` when one is built.
        encoder: A loaded codec. Built lazily from the configured checkpoint when omitted, and
            only if there is something to encode -- so a resumed run with nothing new to do
            never pays for WavLM.
        codec_options: Passed to :func:`~kova_tts.data.encode.load_codec` when one is built.
        overwrite: Ignore an existing corpus at `output` instead of adding to it.
        dry_run: Do everything except loading the codec and writing files.
        progress: Print encode/transcribe progress to stderr.

    Returns:
        A :class:`~kova_tts.data.report.PrepareReport`. Nothing here raises for a clip it
        cannot use; every one of those becomes a skip with a reason.
    """
    settings = settings or SegmentSettings()
    settings.validate()
    if not 0.0 <= val_split < 1.0:
        raise ValueError(f"val_split must be in [0.0, 1.0), got {val_split}.")

    source_path = Path(source).expanduser()
    root = source_path if source_path.is_dir() else source_path.parent
    output_path = Path(output).expanduser() if output is not None else root / DEFAULT_OUTPUT_NAME
    val_path: Path | None = None
    if val_split > 0:
        val_path = (
            Path(val_output).expanduser()
            if val_output is not None
            else default_val_path(output_path)
        )

    found = discover(source_path, metadata=metadata, recursive=recursive)
    result = PrepareReport(
        source=source_path,
        output=output_path,
        val_output=val_path,
        transcript_source=found.transcript_source,
        audio_files=len(found.clips),
        unmatched_rows=len(found.unmatched_rows),
        ambiguous_names=len(found.ambiguous_names),
        max_tokens=max_tokens,
        dry_run=dry_run,
    )

    # ---------------------------------------------------------------- what is already done
    # Both halves of an existing corpus are read back, the held-out one included: its rows are
    # part of what has already been encoded, and leaving them out would re-encode every one of
    # them and write a second copy into the training file.
    carried: list[dict[str, Any]] = []
    sibling = val_path or default_val_path(output_path)
    if not overwrite and not dry_run:
        held_out = read_corpus(sibling)
        carried = read_corpus(output_path) + held_out
        if val_path is None and held_out:
            result.stale_val = sibling
    done_rels = {str(row.get("audio")) for row in carried if row.get("audio")}
    result.reused = len(carried)

    # ------------------------------------------------------------------------- segmenting
    pending: list[_Pending] = []
    for clip in found.clips:
        rel = _relative(clip.path, root)
        if rel in done_rels:
            continue
        if clip.text is None and transcribe is False:
            result.add_skip(
                clip.path,
                NO_TRANSCRIPT,
                "no sidecar and no metadata row; drop --no-transcribe to transcribe it",
            )
            continue
        pending += _segment_clip(
            clip,
            rel,
            settings=settings,
            splittable=transcribe is True or clip.text is None,
            result=result,
        )

    # ------------------------------------------------------------------------ transcribing
    if transcribe is not False:
        pending = _transcribe_pending(
            pending,
            transcriber=transcriber,
            make_transcriber=lambda: _build_transcriber(asr_options),
            result=result,
            progress=progress,
        )

    # --------------------------------------------------------------- the token budget check
    encodable: list[_Pending] = []
    for item in pending:
        tokens = estimate_tokens(item.text or "", code_count(item.wav.size, item.rate))
        if tokens > max_tokens:
            result.add_skip(
                item.path,
                TOO_LONG,
                f"~{tokens} tokens over the {max_tokens}-token row limit",
                item.segment,
            )
            continue
        encodable.append(item)

    # ----------------------------------------------------------------------------- encoding
    if encodable and not dry_run and encoder is None:
        encoder = load_codec(**(codec_options or {}))
    all_codes = (
        encode_clips(
            encoder,
            [item.wav for item in encodable],
            sample_rates=[item.rate for item in encodable],
            on_progress=(lambda done, total: _show(f"encoding {done}/{total} clips"))
            if progress
            else None,
        )
        if encodable and not dry_run
        else [[] for _ in encodable]
    )
    if progress and encodable and not dry_run:
        _show("")

    fresh: list[dict[str, Any]] = []
    for item, codes in zip(encodable, all_codes, strict=True):
        text = item.text or ""
        # A dry run still builds the row, minus the one field that needed the codec, so the
        # summary can report duration and token counts without loading 1.2 GB of WavLM.
        row = {
            "id": item.row_id,
            "audio": item.rel,
            "segment": item.segment,
            "seconds": round(item.seconds, 3),
            "tokens": estimate_tokens(text, code_count(item.wav.size, item.rate)),
        }
        if not dry_run:
            row["tokens"] = estimate_tokens(text, len(codes))
            row["text"] = prompt.training_example(text, codes)
        fresh.append(row)
    result.emitted = len(fresh)

    # ------------------------------------------------------------------------------ writing
    rows = _merge(carried, fresh)
    result.total_seconds = sum(float(row.get("seconds", 0.0)) for row in rows)
    if rows:
        result.longest_seconds = max(float(row.get("seconds", 0.0)) for row in rows)
        result.longest_tokens = max(int(row.get("tokens", 0)) for row in rows)

    if dry_run:
        result.train_rows = len(rows)
        return result

    train_rows, val_rows = _split_rows(rows, val_split=val_split, seed=seed)
    write_corpus(output_path, train_rows)
    result.train_rows = len(train_rows)
    if val_path is not None:
        write_corpus(val_path, val_rows)
        result.val_rows = len(val_rows)
    return result


# ------------------------------------------------------------------------------------ helpers


def _relative(path: Path, root: Path) -> str:
    """Stable identity for a recording: its path relative to the source directory."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _reason(exc: Exception) -> str:
    """One-line rendering of an exception, for a skip's detail."""
    text = clean_text(str(exc)) or type(exc).__name__
    return text if len(text) <= 90 else text[:87] + "..."


def _build_transcriber(asr_options: dict[str, Any] | None):
    from kova_tts.data.asr import load_transcriber

    return load_transcriber(**(asr_options or {}))


def _merge(carried: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Existing rows plus new ones, deduplicated on ``id`` and ordered deterministically.

    Rows without an ``id`` -- a hand-written or third-party corpus -- are kept as they are and
    sorted after the rest, so appending to someone else's file never drops their work.
    """
    merged: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for row in [*carried, *fresh]:
        key = row.get("id")
        if isinstance(key, str) and key:
            merged[key] = row
        else:
            anonymous.append(row)
    return [merged[key] for key in sorted(merged)] + anonymous


def _split_rows(
    rows: list[dict[str, Any]], *, val_split: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deterministic train/validation split of an already-ordered row list.

    The shuffle runs over positions in the sorted list, so the same corpus and seed always
    produce the same split whatever order the clips were encoded in.
    """
    if val_split <= 0 or len(rows) < 2:
        return rows, []
    size = min(max(1, int(len(rows) * val_split)), len(rows) - 1)
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train, val = sorted(indices[:-size]), sorted(indices[-size:])
    return [rows[i] for i in train], [rows[i] for i in val]


def _show(message: str) -> None:
    """Overwrite the current stderr line. Empty message ends it."""
    if message:
        print(f"\r{message:<60}", end="", file=sys.stderr, flush=True)
    else:
        print(f"\r{'':<60}\r", end="", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------------------- CLI


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach this command's flags to `parser`, so the top-level CLI can host it too."""
    parser.add_argument("source", type=Path, help="directory of recordings, or one audio file")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help=f"output JSONL (default: SOURCE/{DEFAULT_OUTPUT_NAME})",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="transcript manifest (CSV/TSV/JSONL); auto-detected in SOURCE when omitted",
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="do not descend into subdirectories"
    )

    group = parser.add_argument_group("segmentation")
    group.add_argument("--max-seconds", type=float, default=DEFAULTS.max_seconds)
    group.add_argument("--min-seconds", type=float, default=DEFAULTS.min_seconds)
    group.add_argument(
        "--silence-db",
        type=float,
        default=DEFAULTS.top_db,
        help="silence threshold, dB below the loudest frame",
    )
    group.add_argument("--min-silence", type=float, default=DEFAULTS.min_silence)
    group.add_argument(
        "--pad", type=float, default=DEFAULTS.pad_seconds, help="silence kept at each end"
    )
    group.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="row length limit, matching the trainer's max_length",
    )

    group = parser.add_argument_group("transcription")
    transcribe = group.add_mutually_exclusive_group()
    transcribe.add_argument(
        "--transcribe",
        action="store_true",
        help="transcribe every clip with ASR, replacing any supplied transcript; "
        "this is also what allows long recordings to be split",
    )
    transcribe.add_argument(
        "--no-transcribe",
        action="store_true",
        help="never transcribe; recordings without a transcript are skipped",
    )
    group.add_argument("--asr-model", default=None, help="faster-whisper model (default: small)")
    group.add_argument("--asr-language", default=None, help="ISO code; detected per clip if unset")
    group.add_argument("--asr-device", default=None, choices=("cuda", "cpu"))
    group.add_argument("--asr-compute-type", default=None, help="e.g. float16, int8")

    group = parser.add_argument_group("output")
    group.add_argument("--val-split", type=float, default=0.0, help="fraction held out (0-1)")
    group.add_argument("--val-out", type=Path, default=None, help="held-out JSONL")
    group.add_argument("--seed", type=int, default=42, help="seeds the held-out shuffle")
    group.add_argument(
        "--overwrite", action="store_true", help="ignore an existing corpus instead of adding to it"
    )
    group.add_argument(
        "--dry-run", action="store_true", help="report what would happen without encoding anything"
    )

    group = parser.add_argument_group("codec")
    group.add_argument("--codec", type=Path, default=None, help="codec checkpoint")
    group.add_argument("--wavlm", type=Path, default=None, help="WavLM-large directory or repo id")
    group.add_argument("--device", default=None, help="torch device for the codec")

    parser.add_argument("-v", "--verbose", action="store_true", help="log each step")
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kova-tts prepare-data",
        description="Turn a folder of recordings into a JSONL corpus for LoRA finetuning.",
    )
    return add_arguments(parser)


def run(args: argparse.Namespace) -> int:
    """Execute a parsed command. Split out so the top-level CLI can call it directly."""
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    transcribe = True if args.transcribe else (False if args.no_transcribe else None)
    asr_options = {
        key: value
        for key, value in (
            ("model", args.asr_model),
            ("language", args.asr_language),
            ("device", args.asr_device),
            ("compute_type", args.asr_compute_type),
        )
        if value is not None
    }

    result = prepare(
        args.source,
        args.out,
        metadata=args.metadata,
        recursive=not args.no_recursive,
        settings=SegmentSettings(
            max_seconds=args.max_seconds,
            min_seconds=args.min_seconds,
            top_db=args.silence_db,
            min_silence=args.min_silence,
            pad_seconds=args.pad,
        ),
        val_split=args.val_split,
        val_output=args.val_out,
        seed=args.seed,
        max_tokens=args.max_tokens,
        transcribe=transcribe,
        asr_options=asr_options,
        codec_options={"checkpoint": args.codec, "wavlm": args.wavlm, "device": args.device},
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        progress=sys.stderr.isatty(),
    )
    print(result.summary())
    return 0 if result.rows or result.dry_run else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (DiscoveryError, ValueError, paths.MissingArtifact) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
