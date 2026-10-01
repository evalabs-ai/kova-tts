"""Text preparation: sentence chunks, their spoken form, and a map between the two.

Split into chunks, then normalize each one and map its words::

    >>> from kova_tts.normalization import preprocess
    >>> [s.normalized for s in preprocess("It costs $12.50. Call me at 3pm.", 300)]
    ['It costs twelve dollars fifty cents. Call me at three PM.']

Each :class:`Sentence` keeps its original text, the text the model is given, and the
word-by-word map between them, which is how a word timestamp measured on the spoken words is
reported against the word the caller wrote.

Normalization needs pynini, the ``normalize`` extra; without it the text is spoken as written
(:func:`~kova_tts.normalization.normalizer.available` says which). Splitting always works.
"""

from __future__ import annotations

from dataclasses import dataclass

from kova_tts.normalization.normalizer import available, normalize_text
from kova_tts.normalization.split import sentences, split_and_merge
from kova_tts.normalization.words import Word, identity, map_words


@dataclass(frozen=True, slots=True)
class Sentence:
    """One chunk of text: as written, as it will be spoken, and word by word."""

    original: str
    normalized: str
    words: tuple[Word, ...]


def preprocess_sentence(text: str, normalize: bool = True) -> Sentence:
    """`text` as one :class:`Sentence`, normalized unless `normalize` is false."""
    if not normalize:
        return Sentence(text, text, tuple(identity(text)))
    normalized = normalize_text(text)
    return Sentence(text, normalized, tuple(map_words(text, normalized)))


def preprocess(text: str, max_chars: int, normalize: bool = True) -> list[Sentence]:
    """`text` split into chunks of at most `max_chars`, each prepared for the model."""
    return [preprocess_sentence(chunk, normalize) for chunk in split_and_merge(text, max_chars)]


__all__ = [
    "Sentence",
    "Word",
    "available",
    "normalize_text",
    "preprocess",
    "preprocess_sentence",
    "sentences",
    "split_and_merge",
]
