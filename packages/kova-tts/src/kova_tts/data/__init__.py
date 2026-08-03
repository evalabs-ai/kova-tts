"""Dataset preparation: a folder of recordings in, a finetuning corpus out.

The step between "here are my recordings" and :mod:`kova_tts.finetune`. Point it at your own
audio -- no dataset ships with this repository, and none is downloaded::

    kova-tts prepare-data recordings/ --val-split 0.05

    from kova_tts.data import prepare
    report = prepare("recordings/", "data/voice/train.jsonl", val_split=0.05)
    print(report.summary())

Two input shapes are understood, and they can be mixed in one folder:

* **Audio with transcripts** -- either ``clip.wav`` beside ``clip.txt``, or a metadata file
  (CSV, TSV, JSONL) mapping filename to text under whatever column names it happens to use.
* **Audio alone** -- transcribed with faster-whisper. That is the only thing the ``data`` extra
  is needed for; with transcripts in hand, nothing here imports it.

The pipeline is::

    discover -> load, downmix, resample to 32 kHz -> split on silence -> trim -> -23 LUFS
             -> encode -> {"text": ...} JSONL

Everything it drops is counted and explained (see :class:`~kova_tts.data.report.PrepareReport`),
runs are deterministic given a ``--seed``, and re-running over the same folder adds to the corpus
rather than duplicating it.
"""

from __future__ import annotations

from kova_tts.data.discover import (
    AUDIO_SUFFIXES,
    Clip,
    Discovery,
    DiscoveryError,
    discover,
    find_audio,
    read_metadata,
)
from kova_tts.data.encode import Encoder, encode_clips, load_codec
from kova_tts.data.prepare import add_arguments, build_parser, main, prepare, read_corpus
from kova_tts.data.report import PrepareReport, Skip
from kova_tts.data.segment import SegmentSettings, split_on_silence, trim_silence

__all__ = [
    "AUDIO_SUFFIXES",
    "Clip",
    "Discovery",
    "DiscoveryError",
    "Encoder",
    "PrepareReport",
    "SegmentSettings",
    "Skip",
    "add_arguments",
    "build_parser",
    "discover",
    "encode_clips",
    "find_audio",
    "load_codec",
    "main",
    "prepare",
    "read_corpus",
    "read_metadata",
    "split_on_silence",
    "trim_silence",
]
