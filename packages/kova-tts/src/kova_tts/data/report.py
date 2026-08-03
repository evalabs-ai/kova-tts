"""What a preparation run did, and why anything it dropped was dropped.

The whole point of this module is that a corpus which quietly halved must be explainable
without re-running anything. Every clip that does not reach the JSONL leaves a :class:`Skip`
behind carrying the file, the reason and enough detail to act on it -- "too long (41.2 s, no
silence to split on)" tells a user what to fix, "skipped 214 clips" does not.

:meth:`PrepareReport.summary` is what the CLI prints. It is a fixed-label block rather than a
log stream because the useful question after a ten-minute run is "what have I got", not "what
happened at 14:02:11".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Skip reasons. Constants rather than an enum because they are printed verbatim, grouped by
# equality, and occasionally matched on in a test -- an enum would only add a `.value`.
UNREADABLE = "unreadable"
NO_TRANSCRIPT = "no transcript"
EMPTY_TRANSCRIPT = "empty transcript"
SILENT = "silent"
TOO_SHORT = "too short"
TOO_LONG = "too long"

#: Fraction of ``max_tokens`` above which the summary warns. A corpus whose longest rows sit
#: just under the limit will start losing rows the moment anything about it changes.
_CROWDED = 0.8

#: Example filenames shown per skip reason before collapsing into a count.
_EXAMPLES = 3


@dataclass(frozen=True, slots=True)
class Skip:
    """One clip that did not make it into the corpus."""

    path: Path
    reason: str

    #: The numbers behind the reason, e.g. ``"41.2 s, no silence to split on"``.
    detail: str = ""

    #: Index within the recording when it was split, otherwise ``None``.
    segment: int | None = None

    def describe(self) -> str:
        # The segment index is only worth showing when the recording was actually cut up:
        # "clip.wav#0" for a file that produced one clip is noise in a summary.
        name = f"{self.path.name}#{self.segment}" if self.segment else self.path.name
        return f"{name} ({self.detail})" if self.detail else name


@dataclass(slots=True)
class PrepareReport:
    """Counts, durations and skips from one run of :func:`~kova_tts.data.prepare.prepare`."""

    source: Path
    output: Path

    #: Where transcripts came from, as :class:`~kova_tts.data.discover.Discovery` described it.
    transcript_source: str = "none found"

    audio_files: int = 0

    #: Clips encoded during this run, and clips carried over from a previous run's output.
    emitted: int = 0
    reused: int = 0

    #: Clips whose transcript came from the ASR model rather than the user.
    transcribed: int = 0

    #: Recordings that were cut into more than one clip.
    split: int = 0

    total_seconds: float = 0.0
    longest_seconds: float = 0.0
    longest_tokens: int = 0
    max_tokens: int = 4096

    train_rows: int = 0
    val_output: Path | None = None
    val_rows: int = 0

    #: Metadata rows naming a file that was not found, and names matching several files.
    unmatched_rows: int = 0
    ambiguous_names: int = 0

    #: True when the run stopped before encoding and wrote nothing.
    dry_run: bool = False

    #: A held-out file left over from a run with a different ``--val-split``. Its rows were
    #: read back so nothing is re-encoded, but they now also live in the training file.
    stale_val: Path | None = None

    skips: list[Skip] = field(default_factory=list)

    # ------------------------------------------------------------------------------ counting

    def add_skip(self, path: Path, reason: str, detail: str = "", segment: int | None = None):
        """Record a dropped clip. Returns the :class:`Skip` so callers can log it too."""
        skip = Skip(path=path, reason=reason, detail=detail, segment=segment)
        self.skips.append(skip)
        return skip

    @property
    def skipped(self) -> int:
        return len(self.skips)

    @property
    def rows(self) -> int:
        """Rows written across the training and validation files."""
        return self.train_rows + self.val_rows

    def reasons(self) -> list[tuple[str, int]]:
        """Skip reasons with their counts, most common first."""
        return Counter(skip.reason for skip in self.skips).most_common()

    def warnings(self) -> list[str]:
        """Things that are not errors but will cost the user rows or quality if ignored."""
        notes: list[str] = []
        if self.longest_tokens > self.max_tokens * _CROWDED:
            notes.append(
                f"longest row is {self.longest_tokens} tokens, "
                f"{self.longest_tokens / self.max_tokens:.0%} of the {self.max_tokens}-token "
                f"limit; lower --max-seconds to leave headroom"
            )
        if self.unmatched_rows:
            notes.append(
                f"{self.unmatched_rows} metadata rows name a file that is not in the source "
                f"directory"
            )
        if self.ambiguous_names:
            notes.append(
                f"{self.ambiguous_names} metadata names match more than one recording and were "
                f"ignored; use paths relative to the source directory to disambiguate"
            )
        if self.rows and self.val_rows == 0 and self.val_output is not None:
            notes.append("val split rounded to zero rows; the corpus is too small to hold any out")
        if self.stale_val is not None:
            notes.append(
                f"{self.stale_val.name} is left over from a run with --val-split; its rows are "
                f"now in the training file too, so delete it or pass --val-split again"
            )
        return notes

    # ----------------------------------------------------------------------------- rendering

    def summary(self) -> str:
        """The block the CLI prints when a run finishes."""
        lines = [
            _row("source", f"{self.source} ({self.audio_files} audio files)"),
            _row("transcripts", self.transcript_source),
        ]
        if self.transcribed:
            lines.append(_row("transcribed", f"{self.transcribed} clips by ASR"))

        made = f"{self.emitted} encoded"
        if self.reused:
            made += f", {self.reused} reused from the existing corpus"
        if self.split:
            made += f", from {self.split} recordings that were split"
        lines.append(_row("clips", made))

        if self.rows:
            longest = f"{self.longest_seconds:.1f} s / {self.longest_tokens} tokens"
            lines.append(
                _row("duration", f"{format_duration(self.total_seconds)} total, longest {longest}")
            )

        if self.dry_run:
            lines.append(_row("written", "nothing (--dry-run)"))
        else:
            written = f"{self.output} ({self.train_rows} rows)"
            if self.val_output is not None:
                written += f", {self.val_output} ({self.val_rows} rows)"
            lines.append(_row("written", written))

        if self.skips:
            lines.append(_row("skipped", f"{self.skipped} clips"))
            lines.extend(self._skip_lines())
        for note in self.warnings():
            lines.append(_row("warning", note))
        return "\n".join(lines)

    def _skip_lines(self) -> list[str]:
        """One indented line per skip reason, with a few examples each."""
        by_reason: dict[str, list[Skip]] = {}
        for skip in self.skips:
            by_reason.setdefault(skip.reason, []).append(skip)

        lines = []
        for reason, count in self.reasons():
            examples = by_reason[reason]
            shown = ", ".join(skip.describe() for skip in examples[:_EXAMPLES])
            if count > _EXAMPLES:
                shown += f", +{count - _EXAMPLES} more"
            lines.append(f"{'':<13}{count:>5}  {reason:<16} {shown}")
        return lines


def _row(label: str, value: str) -> str:
    return f"{label:<13}{value}"


def format_duration(seconds: float) -> str:
    """``h:mm:ss`` for a corpus, ``m:ss`` for anything under an hour."""
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def estimate_tokens(text: str, code_count: int) -> int:
    """Upper bound on the tokenized length of one training row, without a tokenizer.

    The audio codes are exact -- one token each -- and dominate at 80 per second. The text is
    counted in UTF-8 bytes, which is a genuine bound rather than an average: the tokenizer is
    byte-level BPE, so no token covers less than one byte. It runs about 4x high on English
    prose and much closer on scripts that do not fit the vocabulary well.

    Deliberately loose in that direction. Under-counting costs a row the trainer silently
    drops for exceeding ``max_length``; over-counting costs a clip split slightly earlier than
    it strictly had to be. At the default 30-second limit the audio is 2400 tokens and the text
    a few hundred, so the slack never comes close to mattering.
    """
    structural = 5  # BOS, the two text-prompt tags, speech_start, speech_end
    return code_count + max(1, len(text.strip().encode("utf-8"))) + structural
