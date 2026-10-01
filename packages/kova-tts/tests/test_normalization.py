"""Text preparation: sentence splitting, the readiness rule for streamed text, normalization and
the word map. The normalization cases need pynini and are skipped without it."""

from __future__ import annotations

import pytest

from kova_tts.normalization import Word, preprocess, preprocess_sentence, split_and_merge
from kova_tts.normalization.split import last_sentence_end, stable_sentence_end
from kova_tts.normalization.words import identity, map_words

needs_pynini = pytest.mark.skipif(
    not __import__("kova_tts.normalization").normalization.available(),
    reason="normalization needs the normalize extra (pynini)",
)


class TestSplitting:
    def test_abbreviations_do_not_end_a_sentence(self):
        assert split_and_merge("Dr. Smith went to the U.S. office. Then home.", 20) == [
            "Dr. Smith went to ",
            "the U.S. office. ",
            "Then home.",
        ]

    def test_short_sentences_are_packed_up_to_the_limit(self):
        assert split_and_merge("One. Two. Three.", 300) == ["One. Two. Three."]

    def test_nothing_is_lost(self):
        text = "First, a clause; then another: and a very long tail without any stops at all"
        assert "".join(split_and_merge(text, 16)) == text
        assert all(len(piece) <= 16 for piece in split_and_merge(text, 16))


class TestReadiness:
    """When streamed text holds a sentence that can no longer change."""

    def test_a_question_or_exclamation_is_complete_at_once(self):
        assert stable_sentence_end("Is it ready?") == len("Is it ready?")
        assert stable_sentence_end("Yes!") == len("Yes!")

    def test_a_period_waits_for_what_follows(self):
        assert stable_sentence_end("It costs 3.") == 0
        assert stable_sentence_end("It costs 3. More") == len("It costs 3. ")

    def test_only_the_first_sentence_is_returned(self):
        assert stable_sentence_end("One! Two! Three") == len("One! ")

    def test_a_sentence_end_inside_an_open_tag_does_not_count(self):
        assert stable_sentence_end("Oh [wow! and") == 0

    def test_the_last_terminated_sentence(self):
        assert last_sentence_end("One. Two. Three") == len("One. Two. ")
        assert last_sentence_end("no stop at all") == 0


class TestWordMap:
    def test_unnormalized_text_maps_to_itself(self):
        assert preprocess_sentence("Hello there.", normalize=False).words == (
            Word("Hello", "Hello"),
            Word("there.", "there."),
        )

    def test_a_tag_is_one_word(self):
        assert [w.original for w in identity("Well [long pause] then")] == [
            "Well",
            "[long pause]",
            "then",
        ]

    def test_a_written_word_can_become_several_spoken_ones(self, monkeypatch):
        import kova_tts.normalization.words as words

        monkeypatch.setattr(words, "normalize_text", lambda w: {"55": "fifty five"}.get(w, w))
        words._expand_word.cache_clear()
        try:
            assert map_words("Pay 55 now", "Pay fifty five now") == [
                Word("Pay", "Pay"),
                Word("55", "fifty five"),
                Word("now", "now"),
            ]
        finally:
            words._expand_word.cache_clear()


@needs_pynini
class TestNormalization:
    @pytest.mark.parametrize(
        ("written", "spoken"),
        [
            (
                "It costs $12.50. Call me at 3pm.",
                "It costs twelve dollars fifty cents. Call me at three PM.",
            ),
            (
                "Dr. Smith paid -$40 on 3/14/2024 [laughs] for 2.5kg of U.S. beef.",
                "doctor Smith paid minus forty dollars on march fourteenth twenty twenty four "
                "[laughs] for two point five kilograms of U S beef.",
            ),
            (
                "The temperature dropped to −5 degrees; that's 23°F at 10:30 a.m.",
                "The temperature dropped to minus five degrees; that's twenty three degrees "
                "Fahrenheit at ten thirty AM",
            ),
        ],
    )
    def test_written_text_is_spoken_as_words(self, written, spoken):
        assert [s.normalized for s in preprocess(written, 300)] == [spoken]

    def test_every_spoken_word_belongs_to_a_written_one(self):
        (sentence,) = preprocess("I have 55 apples and $20.", 300)
        assert [(w.original, w.normalized) for w in sentence.words] == [
            ("I", "I"),
            ("have", "have"),
            ("55", "fifty five"),
            ("apples", "apples"),
            ("and", "and"),
            ("$20.", "twenty dollars."),
        ]
