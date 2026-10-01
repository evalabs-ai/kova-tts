"""Word timestamps: when each word of the text is spoken in the generated audio.

A small model reads the codec codes the LM produced (:mod:`.model`), a CTC forced alignment
places the transcript's letters on them (:mod:`.ctc`), and :mod:`.aligner` turns that into word
timings -- for a finished chunk in one pass, or incrementally while it is still being
generated (:mod:`.stream`).

The timings also decide what one chunk hands to the next: :class:`~kova_tts.engine.tts.KovaTTS`
carries the last few seconds of words and their codes into the next prompt, which is what lets
it work in larger chunks than it can without an aligner.

>>> from kova_tts.alignment import load_aligner
>>> aligner = load_aligner()  # alignment.pt, resolved like every other checkpoint
"""

from __future__ import annotations

import os

from kova_tts.alignment.aligner import Aligner, CommittedWord
from kova_tts.alignment.stream import AlignmentStream, TimedWord


def load_aligner(
    path: str | os.PathLike[str] | None = None, *, device: str | None = None
) -> Aligner:
    """The aligner, from `path`, ``KOVA_ALIGNMENT_PATH``, or the Hugging Face Hub."""
    from kova_tts import paths

    return Aligner.from_checkpoint(paths.alignment_path(path), device=device)


__all__ = ["Aligner", "AlignmentStream", "CommittedWord", "TimedWord", "load_aligner"]
