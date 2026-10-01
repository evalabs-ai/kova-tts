"""Explicit spoken forms and literal identifier spelling, compiled into the FAR.

Capitalization alone cannot distinguish an acronym from an ordinary word. Only
known pronunciations and dotted initialisms are expanded; other words survive.
Mixed alphanumeric identifiers are spelled individually, preserving every digit.
"""
import pynini
from pynini.lib import pynutil

from ..graph_utils import NEMO_ALPHA, NEMO_DIGIT, NEMO_UPPER, NEMO_SIGMA
from ..utils import get_abs_path, load_labels


def pronunciations(case_sensitive=True):
    rows = load_labels(get_abs_path('data/lexical/pronunciations.tsv'))
    forms = list(rows)
    for original, spoken in rows:
        if original[-1].isalnum():
            for suffix in ("'s", "’s"):
                forms.append((original + suffix, spoken + suffix))
            # The apostrophe makes the final individual letter plural, rather
            # than introducing an extra spoken S (GPU's -> G P U's).
            if original.isalpha():
                plural = "'s" if spoken[-1].isupper() else 's'
                forms.append((original + 's', spoken + plural))
    if not case_sensitive:
        forms += [(original.lower(), spoken) for original, spoken in forms]
    graph = pynini.string_map(forms)
    dotted = NEMO_UPPER + pynini.closure(
        pynutil.delete('.') + pynutil.insert(' ') + NEMO_UPPER, 1)
    dotted += pynini.closure(pynutil.delete('.'), 0, 1)
    return (graph | dotted).optimize()


def identifiers(mixed=True):
    digit_words = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    rows = [(str(i), word) for i, word in enumerate(digit_words)]
    rows += [(chr(i), chr(i).upper()) for i in range(ord('a'), ord('z') + 1)]
    rows += [(chr(i), chr(i)) for i in range(ord('A'), ord('Z') + 1)]
    rows += [('-', 'dash'), ('_', 'underscore')]
    symbol = pynini.string_map(rows)
    spelling = symbol + pynini.closure(pynutil.insert(' ') + symbol)
    separator = pynini.union('-', '_')
    atom = NEMO_ALPHA | NEMO_DIGIT
    # Both letters and digits are required; numeric-first IDs are valid too.
    # Units, ordinals and scientific notation keep their earlier classes.
    shape = atom + pynini.closure(atom | (separator + atom))
    if mixed:
        shape @= NEMO_SIGMA + NEMO_ALPHA + NEMO_SIGMA
        shape @= NEMO_SIGMA + NEMO_DIGIT + NEMO_SIGMA
    return (shape @ spelling).optimize()


def with_emphasis(graph):
    stars = pynini.closure(pynini.accep('*'))
    return (stars + graph + stars).optimize()


def versions(cardinal):
    part = (NEMO_DIGIT ** (1, 4)) @ cardinal.graph
    return (pynini.cross(pynini.union('v', 'V'), 'version ') + part
            + pynini.closure(pynini.cross('.', ' point ') + part, 2)).optimize()


def explicit_codes():
    # An explicit code/ID cue disambiguates strings that otherwise look like
    # scientific notation, a date or a unit. It also permits letter-only IDs.
    cues = pynini.string_map([(word, word) for word in
        ('code ', 'Code ', 'code: ', 'Code: ', 'code is ', 'Code is ',
         'serial number ', 'Serial number ')] +
        [(word, 'I D' + word[2:]) for word in ('ID ', 'ID: ', 'ID is ')])
    return (cues + with_emphasis(identifiers(mixed=False))).optimize()
