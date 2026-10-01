"""Splitting text into sentences, and packing sentences into chunks of a bounded length.

Sentence boundaries come from pysbd, which knows that "Dr. Smith" and "U.S. law" do not end a
sentence. The split is lossless -- the pieces join back into the input exactly, whitespace
included -- which is what lets callers keep character offsets into the text they were given.
"""

from __future__ import annotations

import re

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=False)


def sentences(text: str) -> list[str]:
    """`text` cut at its sentence boundaries."""
    return list(_segmenter.segment(text))


def stable_sentence_end(text: str) -> int:
    """Offset just past the first sentence of streamed `text` that is surely complete, or 0.

    ``!`` and ``?`` end a sentence. ``.`` and ``…`` only once more text follows them, since the
    next characters might still turn "3." into "3.5" or "Dr." into a name. A sentence inside an
    unclosed ``[tag`` is not complete either.
    """
    if not any(mark in text for mark in ".!?…"):
        return 0
    content = text.lstrip()
    offset = len(text) - len(content)
    segments = sentences(content)
    if "".join(segments) != content:
        return 0  # pysbd lost characters; its offsets cannot be trusted
    for segment in segments:
        offset += len(segment)
        tail = segment.rstrip().rstrip("\"”’'")
        followed = bool(text[offset:].strip())
        ended = tail.endswith(("!", "?")) or (followed and tail.endswith((".", "…")))
        if ended and tags_closed(text[:offset]):
            return offset
    return 0


def last_sentence_end(text: str) -> int:
    """Offset just past the last sentence of `text` that ends in terminal punctuation, or 0."""
    end = position = 0
    for segment in sentences(text):
        position += len(segment)
        if segment.rstrip().endswith((".", "!", "?", "…")):
            end = position
    return end


def tags_closed(text: str) -> bool:
    """True unless `text` ends inside a ``[tag]``."""
    return text.rfind("[") <= text.rfind("]")


def split_and_merge(text: str, max_length: int, *, preserve_first: bool = False) -> list[str]:
    """Sentences packed greedily into chunks of at most `max_length` characters.

    A sentence longer than that on its own is broken at punctuation, then at whitespace, then
    anywhere. `preserve_first` keeps the first sentence as a chunk of its own, so a streaming
    caller can start speaking before the rest has been packed.
    """
    pieces: list[str] = []
    for sentence in sentences(text):
        if len(sentence) > max_length:
            pieces.extend(_split_too_long(sentence, max_length))
        else:
            pieces.append(sentence)
    if preserve_first and pieces:
        return pieces[:1] + _merge(pieces[1:], max_length)
    return _merge(pieces, max_length)


def _split_too_long(sentence: str, max_length: int) -> list[str]:
    """Break one over-long sentence at the least damaging places available."""
    phrases = _merge(_split_keeping(sentence, r"[,;:!?]+"), max_length)
    words: list[str] = []
    for phrase in phrases:
        if len(phrase) > max_length:
            words.extend(_merge(_split_keeping(phrase, r"\s+"), max_length))
        else:
            words.append(phrase)
    pieces: list[str] = []
    for group in words:
        if len(group) > max_length:
            pieces.extend(_merge(list(group), max_length))
        else:
            pieces.append(group)
    return _merge(pieces, max_length)


def _split_keeping(text: str, delimiter: str) -> list[str]:
    """Split on `delimiter`, keeping each delimiter on the piece before it."""
    parts = re.split(f"({delimiter})", text)
    pieces = []
    for i in range(0, len(parts), 2):
        piece = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
        if piece:
            pieces.append(piece)
    return pieces


def _merge(pieces: list[str], max_length: int) -> list[str]:
    """Join consecutive pieces while the result stays within `max_length`."""
    merged: list[str] = []
    current = ""
    for piece in pieces:
        if len(current) + len(piece) <= max_length:
            current += piece
        else:
            if current:
                merged.append(current)
            current = piece
    if current:
        merged.append(current)
    return merged
