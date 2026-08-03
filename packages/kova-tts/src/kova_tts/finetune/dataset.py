"""The finetuning corpus: JSONL in, masked causal-LM batches out.

One JSON object per line, one field that matters::

    {"text": "<|begin_of_text|><|text_prompt_start|>...<|speech_start|>...<|speech_end|>"}

Build that string with :func:`kova_tts.prompt.training_example` -- the exact byte layout is what
the checkpoint was trained on. Any other field on the line is ignored, so a corpus that carries
provenance (source clip, duration, speaker) needs no stripping first.

Two things to know:

**Lines are tokenized here**, with ``add_special_tokens=False``, rather than stored as ids. A
corpus is therefore a readable text file that can be diffed, grepped and hand-edited, at a
load-time cost a per-voice corpus never notices.

**Loss starts after ``<|speech_start|>``.** Every label up to and including that tag is
``-100``; supervision runs from the first audio token through ``<|speech_end|>`` inclusive. The
model is being taught to speak the given text in a voice, not to reproduce the prompt or to
guess where a prompt ends.

Bad lines are skipped and counted rather than raised on. A corpus is usually machine-generated
from an ASR pipeline, and losing a training run at line 9,000 because one clip tokenized long is
worse than dropping the clip and saying so -- see :class:`LoadReport`, which train.py logs.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from kova_tts.finetune.loss import WEIGHT_KEY, EndingWeight, ramp_weights
from kova_tts.tokens import SPEECH_END, SPEECH_START

logger = logging.getLogger(__name__)

#: Rows are tokenized in chunks this size, so a huge corpus never materialises as one giant
#: list of Python strings *and* one giant list of id lists at the same time.
_TOKENIZE_CHUNK = 256


class DatasetError(ValueError):
    """The corpus could not be used at all -- missing file, or every line unusable."""


@dataclass(frozen=True, slots=True)
class LoadReport:
    """What :class:`MaskedCausalDataset` made of a JSONL file.

    The counters exist so that dropped lines are visible: a corpus that lost 30% of its lines to
    ``too_long`` is a max_length problem, not a training problem, and the difference should not
    take a wasted GPU-hour to notice.
    """

    path: Path
    lines: int
    kept: int
    unparseable: int = 0
    no_text_field: int = 0
    no_speech_start: int = 0
    too_long: int = 0
    no_speech_end: int = 0

    #: Longest line seen, in tokens, *including* lines dropped for exceeding max_length -- so
    #: this is the number to compare max_length against when tuning it.
    longest: int = 0

    @property
    def skipped(self) -> int:
        return self.unparseable + self.no_text_field + self.no_speech_start + self.too_long

    def summary(self) -> str:
        """One-line human summary, with the reason breakdown only when something was dropped."""
        parts = [f"{self.path.name}: {self.kept}/{self.lines} examples", f"longest {self.longest}"]
        reasons = [
            ("unparseable JSON", self.unparseable),
            ("no 'text' field", self.no_text_field),
            (f"no {SPEECH_START}", self.no_speech_start),
            ("over max_length", self.too_long),
        ]
        dropped = [f"{count} {label}" for label, count in reasons if count]
        if dropped:
            parts.append("skipped " + ", ".join(dropped))
        if self.no_speech_end:
            parts.append(f"{self.no_speech_end} without {SPEECH_END} (they teach no ending)")
        return "; ".join(parts)


@dataclass(frozen=True, slots=True)
class _Row:
    """One usable example, tokenized once at load time."""

    ids: np.ndarray
    speech_start: int
    speech_end: int | None


class MaskedCausalDataset(Dataset):
    """JSONL corpus as masked causal-LM examples, tokenized eagerly at construction.

    Eager tokenization buys the validation in :class:`LoadReport`: a corpus is fully checked --
    and any unusable line reported -- before a single weight is loaded onto the GPU.

    When `ending_weight` is active each item carries an extra ``ending_w`` channel; see
    :mod:`kova_tts.finetune.loss`. Validation views are built without it so that ``eval_loss``
    stays a plain cross entropy.
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        ending_weight: EndingWeight | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_length = int(max_length)
        self.ending_weight = ending_weight if ending_weight and ending_weight.ramps else None

        self.speech_start_id = _require_token(tokenizer, SPEECH_START)
        self.speech_end_id = _require_token(tokenizer, SPEECH_END)

        self._rows, self.report = self._load(tokenizer)
        if not self._rows:
            raise DatasetError(
                f"No usable training examples in {self.path}. {self.report.summary()}. "
                f"Each line must be a JSON object with a 'text' field containing "
                f"{SPEECH_START}; build it with kova_tts.prompt.training_example()."
            )

    # ------------------------------------------------------------------------------ loading

    def _load(self, tokenizer: Any) -> tuple[list[_Row], LoadReport]:
        if not self.path.is_file():
            raise DatasetError(
                f"Training corpus not found at {self.path}. Point 'dataset' in your config at a "
                f"JSONL file, one {{'text': ...}} object per line."
            )

        texts: list[str] = []
        lines = unparseable = no_text = 0
        with self.path.open(encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                lines += 1
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError:
                    unparseable += 1
                    continue
                text = record.get("text") if isinstance(record, dict) else None
                if not isinstance(text, str) or not text:
                    no_text += 1
                    continue
                texts.append(text)

        rows: list[_Row] = []
        no_speech_start = too_long = no_speech_end = 0
        longest = 0
        for chunk in _chunks(texts, _TOKENIZE_CHUNK):
            for encoded in tokenizer(chunk, add_special_tokens=False)["input_ids"]:
                ids = np.asarray(encoded, dtype=np.int64)
                longest = max(longest, ids.size)
                if ids.size > self.max_length:
                    too_long += 1
                    continue
                starts = np.flatnonzero(ids == self.speech_start_id)
                if starts.size == 0:
                    no_speech_start += 1
                    continue
                ends = np.flatnonzero(ids == self.speech_end_id)
                if ends.size == 0:
                    no_speech_end += 1
                rows.append(
                    _Row(
                        ids=ids,
                        speech_start=int(starts[0]),
                        speech_end=int(ends[0]) if ends.size else None,
                    )
                )

        report = LoadReport(
            path=self.path,
            lines=lines,
            kept=len(rows),
            unparseable=unparseable,
            no_text_field=no_text,
            no_speech_start=no_speech_start,
            too_long=too_long,
            no_speech_end=no_speech_end,
            longest=longest,
        )
        if report.skipped:
            logger.warning("%s", report.summary())
        else:
            logger.info("%s", report.summary())
        return rows, report

    # ------------------------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self._rows[index]
        input_ids = torch.from_numpy(row.ids)
        labels = input_ids.clone()
        # Inclusive of <|speech_start|> itself: the prompt is context, not a target.
        labels[: row.speech_start + 1] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
        }
        if self.ending_weight is not None:
            item[WEIGHT_KEY] = ramp_weights(labels, row.speech_end, self.ending_weight)
        return item

    @property
    def longest(self) -> int:
        """Token count of the longest *kept* example."""
        return max((row.ids.size for row in self._rows), default=0)

    def view(
        self,
        indices: Sequence[int],
        *,
        ending_weight: EndingWeight | None,
    ) -> MaskedCausalDataset:
        """A subset sharing this dataset's tokenized rows, optionally reweighted.

        Used to split train/validation without tokenizing the file twice, and to hand the
        validation half an unweighted view of the same rows.
        """
        clone = object.__new__(type(self))
        clone.path = self.path
        clone.max_length = self.max_length
        clone.ending_weight = ending_weight if ending_weight and ending_weight.ramps else None
        clone.speech_start_id = self.speech_start_id
        clone.speech_end_id = self.speech_end_id
        clone._rows = [self._rows[i] for i in indices]
        clone.report = self.report
        return clone


def _require_token(tokenizer: Any, token: str) -> int:
    """Token id for a structural tag, with an error that names the likely cause."""
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise DatasetError(
            f"Tokenizer has no {token} token, so loss masking is undefined. The base model must "
            f"be a Kova TTS checkpoint -- check 'model' in your config (or KOVA_MODEL_PATH)."
        )
    return int(token_id)


def _chunks(items: list[str], size: int) -> Iterator[list[str]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def build_datasets(
    *,
    train_path: str | Path,
    tokenizer: Any,
    max_length: int = 4096,
    ending_weight: EndingWeight | None = None,
    val_path: str | Path | None = None,
    val_split: float = 0.0,
    seed: int = 42,
) -> tuple[MaskedCausalDataset, MaskedCausalDataset | None]:
    """Load the training corpus and a validation set, from a file or by splitting the train set.

    An explicit `val_path` wins over `val_split`. Either way the validation half is built
    *without* the ending weights, so ``eval_loss`` remains a plain cross entropy and stays
    comparable across runs that tune the ending recipe.
    """
    train = MaskedCausalDataset(
        train_path, tokenizer, max_length=max_length, ending_weight=ending_weight
    )

    if val_path is not None:
        val = MaskedCausalDataset(val_path, tokenizer, max_length=max_length)
        return train, val

    if val_split <= 0:
        return train, None

    val_size = max(1, int(len(train) * val_split))
    if val_size >= len(train):
        raise DatasetError(
            f"val_split={val_split} would leave no training examples: the corpus has "
            f"{len(train)} usable lines. Lower val_split or add data."
        )

    indices = list(range(len(train)))
    random.Random(seed).shuffle(indices)
    train_indices, val_indices = indices[:-val_size], indices[-val_size:]
    return (
        train.view(train_indices, ending_weight=ending_weight),
        train.view(val_indices, ending_weight=None),
    )
