"""Which spoken words each written word became.

Normalization rewrites a sentence as a whole, so "Pay $55 now" comes back as "Pay fifty five
dollars now" with nothing saying that "$55" is now three words. Word timestamps need that map:
the aligner times the spoken words the model actually said, and the caller wants times for the
words it actually sent. :func:`map_words` rebuilds it by normalizing each written word on its
own and lining the result up against the sentence's normalization with Needleman-Wunsch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from kova_tts.normalization.normalizer import normalize_text

TOKEN_PATTERN = re.compile(r"\[[^\[\]]*\]|\S+")

#: Characters a written word may consist of and still be its own spoken form.
_SPOKEN_AS_WRITTEN = set("-—'.,!?:;\"’")


@dataclass(frozen=True, slots=True)
class Word:
    """One written word and what it is spoken as. Either side may be several words."""

    original: str
    normalized: str


def tokens(text: str) -> list[str]:
    """Whitespace-separated words, keeping a ``[tag]`` whole even if it contains spaces."""
    if text == "":
        return [""]
    return TOKEN_PATTERN.findall(text)


def identity(text: str) -> list[Word]:
    """The map for text that is spoken as written."""
    return [Word(word, word) for word in tokens(text)]


def map_words(original: str, normalized: str) -> list[Word]:
    """Pair each written word of `original` with its spoken form in `normalized`."""
    if original.strip() == normalized.strip():
        return identity(normalized)
    return _align_arrays(_expand(original), tokens(normalized))


@lru_cache(maxsize=2048)
def _expand_word(word: str) -> tuple[str, ...]:
    # Cached: it is a pure function of one word, and the same words come up constantly.
    return tuple(tokens(normalize_text(word)))


def _expand(original: str) -> list[tuple[str, list[str]]]:
    """Each written word with the spoken words it becomes on its own."""
    mapping = []
    for word in tokens(original):
        if word.startswith("[") and word.endswith("]"):
            mapping.append((word, [word]))
        elif all(char.isalpha() or char in _SPOKEN_AS_WRITTEN for char in word):
            mapping.append((word, [word]))
        elif len(word) <= 128:
            mapping.append((word, list(_expand_word(word))))
        else:
            mapping.append((word, tokens(normalize_text(word))))
    return mapping


def _align_arrays(expanded: list[tuple[str, list[str]]], normalized: list[str]) -> list[Word]:
    """Group the sentence's spoken words back under the written words they came from.

    ``[("55", ["fifty", "five"])]`` lined up against ``["fifty", "five"]`` gives
    ``[Word("55", "fifty five")]``.
    """
    flattened = [spoken for _, words in expanded for spoken in words]
    alignment = _needleman_wunsch(flattened, normalized)

    # Which occurrence of which written word each flattened position belongs to.
    owner: list[tuple[str, int]] = []
    seen: dict[str, int] = {}
    for word, words in expanded:
        seen[word] = seen.get(word, -1) + 1
        owner.extend((word, seen[word]) for _ in words)

    result: list[tuple[str, str]] = []
    current: tuple[str, int] | None = None
    parts: list[str] = []
    position = 0
    for aligned_original, aligned_normalized in alignment:
        if aligned_original != "" and position < len(owner):
            expected = owner[position]
            position += 1
        else:
            expected = (aligned_original, 0)
        if current is not None and current != expected:
            if parts:
                result.append((current[0], " ".join(parts)))
            current = expected
            parts = [aligned_normalized] if aligned_normalized else []
        else:
            if current is None:
                current = expected
            if aligned_normalized:
                parts.append(aligned_normalized)
    if current is not None:
        result.append((current[0], " ".join(parts)))
    return [Word(original, normalized) for original, normalized in result]


def _similarity(a: str, b: str, match: int, mismatch: int) -> float:
    if a.lower() == b.lower():
        return match
    from Levenshtein import ratio

    similarity = ratio(a, b)
    return similarity * match if similarity > 0.5 else mismatch


def _needleman_wunsch(
    original: list[str],
    normalized: list[str],
    match: int = 2,
    mismatch: int = -1,
    gap: int = -1,
) -> list[tuple[str, str]]:
    """Global alignment of two word sequences; a gap is ``""``.

    Spoken words with no written partner are merged into the next word that has one, so every
    pair but a trailing run has a written side.
    """
    m, n = len(original), len(normalized)
    score = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        score[i][0] = score[i - 1][0] + gap
    for j in range(1, n + 1):
        score[0][j] = score[0][j - 1] + gap
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            diagonal = score[i - 1][j - 1] + _similarity(
                original[i - 1], normalized[j - 1], match, mismatch
            )
            score[i][j] = max(diagonal, score[i - 1][j] + gap, score[i][j - 1] + gap)

    pairs: list[tuple[str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            diagonal = score[i - 1][j - 1] + _similarity(
                original[i - 1], normalized[j - 1], match, mismatch
            )
            if score[i][j] == diagonal:
                pairs.append((original[i - 1], normalized[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and score[i][j] == score[i - 1][j] + gap:
            pairs.append((original[i - 1], ""))
            i -= 1
        elif j > 0 and score[i][j] == score[i][j - 1] + gap:
            pairs.append(("", normalized[j - 1]))
            j -= 1
        else:
            break
    pairs.reverse()

    merged: list[tuple[str, str]] = []
    i = 0
    while i < len(pairs):
        if pairs[i][0] != "":
            merged.append(pairs[i])
            i += 1
            continue
        pending = [pairs[i][1]]
        j = i + 1
        while j < len(pairs) and pairs[j][0] == "":
            pending.append(pairs[j][1])
            j += 1
        if j < len(pairs):
            merged.append((pairs[j][0], " ".join([*pending, pairs[j][1]])))
            i = j + 1
        else:
            merged.extend(("", word) for word in pending)
            i = j
    return merged
