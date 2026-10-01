"""Written form to spoken form: ``"$12.50"`` becomes ``"twelve dollars fifty cents"``.

The model reads what it is given, so "Dr. Smith paid $40 on 3/14" spoken without normalization
comes out as letters and guesses. This runs the English NeMo grammars in :mod:`.nemo`, with two
fixes around them:

* ``[tags]`` such as ``[laughs]`` pass through untouched. NeMo would read the word inside.
* A minus sign in front of a number or a price is spoken as "minus". NeMo reads a lone dash
  before a number as a dash.

pynini, which runs the grammars, is the ``normalize`` extra. Without it :func:`normalize_text`
returns its input unchanged and says so once in the log.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

#: The compiled grammars, tracked in the package. See ``nemo/README.md`` to rebuild them.
CACHE_DIR = Path(__file__).parent / "nemo" / "cache"

TAG_PATTERN = re.compile(r"\[[^\[\]]*\]")

# An ASCII dash can delimit a price range, so only rewrite it as a unary money sign at a token
# boundary. U+2212 is explicitly a mathematical minus; before a number it also disambiguates
# subtraction from a dash/range.
NUMERIC_SIGN_PATTERN = re.compile(
    r"(?<![\w.+$€£−-])(?:[-−]\s*(?P<currency_after>[$€£])\s*"
    r"|(?P<currency_before>[$€£])\s*[-−]\s*)(?=[0-9]|\.[0-9])"
    r"|(?<![0-9][eE])−\s*(?P<unicode_currency>[$€£])?\s*(?=[0-9]|\.[0-9])"
)

_normalizer: Any = None
_unavailable = False
_lock = threading.Lock()


def available() -> bool:
    """True when pynini is installed, so :func:`normalize_text` really normalizes."""
    try:
        import pynini  # noqa: F401
    except ImportError:
        return False
    return True


def normalize_text(text: str) -> str:
    """`text` in its spoken form, or unchanged when pynini is not installed."""
    normalizer = _get_normalizer()
    if normalizer is None:
        return text
    text, tags = _protect_tags(text)
    text = NUMERIC_SIGN_PATTERN.sub(_spoken_numeric_sign, text)
    text = re.sub(r"\s+", " ", text).strip()
    # One call at a time: the server normalizes from more than one thread, and the grammars are
    # one shared object.
    with _lock:
        normalized = normalizer.normalize(text, punct_pre_process=True, punct_post_process=True)
    normalized = _restore_tags(normalized, tags)
    # Trailing quotes are dropped.
    return normalized.rstrip("\"'”")


def _get_normalizer() -> Any:
    """The NeMo normalizer, loaded once. Loading the grammars takes a few seconds."""
    global _normalizer, _unavailable
    if _normalizer is not None or _unavailable:
        return _normalizer
    with _lock:
        if _normalizer is None and not _unavailable:
            if not available():
                _unavailable = True
                log.warning(
                    "Text normalization is off: pynini is not installed, so numbers, dates and "
                    "symbols are spoken as written. Install the `normalize` extra to turn it on "
                    "(see docs/installation.md)."
                )
                return None
            from kova_tts.normalization.nemo import Normalizer

            _normalizer = Normalizer(
                input_case="cased",
                lang="en",
                cache_dir=str(CACHE_DIR),
                overwrite_cache=False,
                post_process=True,
            )
    return _normalizer


def _spoken_numeric_sign(match: re.Match[str]) -> str:
    currency = (
        match.group("currency_after")
        or match.group("currency_before")
        or match.group("unicode_currency")
        or ""
    )
    return " minus " + currency


def _tag_placeholder(index: int) -> str:
    """A deterministic, purely alphabetic placeholder, which NeMo leaves intact."""
    letters = []
    while True:
        letters.append(chr(ord("A") + index % 26))
        index = index // 26 - 1
        if index < 0:
            break
    return f"KOVATAGPLACEHOLDER{''.join(reversed(letters))}TOKEN"


def _protect_tags(text: str) -> tuple[str, dict[str, str]]:
    tags: dict[str, str] = {}
    next_index = 0

    def replace_tag(match: re.Match[str]) -> str:
        nonlocal next_index
        placeholder = _tag_placeholder(next_index)
        next_index += 1
        while placeholder in text or placeholder in tags:
            placeholder = _tag_placeholder(next_index)
            next_index += 1
        tags[placeholder] = match.group(0)
        return placeholder

    return TAG_PATTERN.sub(replace_tag, text), tags


def _restore_tags(text: str, tags: dict[str, str]) -> str:
    for placeholder, tag in tags.items():
        text = text.replace(placeholder, tag)
    return text
