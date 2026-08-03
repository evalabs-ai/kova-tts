"""Prompt strings: exact bytes, because these are what the checkpoint was trained on."""

from __future__ import annotations

import pytest

from kova_tts import prompt


class TestAudioTokens:
    def test_formats_each_code_as_a_tag(self):
        assert prompt.format_audio_tokens([0, 42, 8191]) == "<|s_0|><|s_42|><|s_8191|>"

    def test_empty_sequence_is_the_empty_string(self):
        assert prompt.format_audio_tokens([]) == ""

    def test_parses_codes_back_out(self):
        assert prompt.parse_audio_tokens("<|s_7|><|s_0|>") == [7, 0]

    def test_ignores_surrounding_text_and_other_tags(self):
        text = "hello <|s_1|> world <|speech_end|><|s_2|>"
        assert prompt.parse_audio_tokens(text) == [1, 2]

    def test_round_trip(self):
        codes = [0, 1, 2, 129, 4095, 8191]
        assert prompt.parse_audio_tokens(prompt.format_audio_tokens(codes)) == codes

    def test_no_audio_tokens_yields_nothing(self):
        assert prompt.parse_audio_tokens("just some text") == []


class TestTtsPrompt:
    def test_exact_string(self):
        assert (
            prompt.tts_prompt("Hello there.")
            == "<|text_prompt_start|>Hello there.<|text_prompt_end|><|speech_start|>"
        )

    def test_text_is_stripped(self):
        assert prompt.tts_prompt("  Hello there.\n") == prompt.tts_prompt("Hello there.")

    def test_no_bos_because_the_engine_adds_it(self):
        assert not prompt.tts_prompt("Hi").startswith("<|begin_of_text|>")


class TestClonePrompt:
    def test_exact_string(self):
        assert prompt.clone_prompt("Ref text.", "Target text.", [0, 5]) == (
            "<|begin_of_text|><|text_prompt_start|>Ref text. Target text."
            "<|text_prompt_end|><|speech_start|><|s_0|><|s_5|>"
        )

    def test_reference_transcript_comes_first_separated_by_one_space(self):
        built = prompt.clone_prompt("  Ref.  ", "\tTarget.\n", [])
        assert "<|text_prompt_start|>Ref. Target.<|text_prompt_end|>" in built

    def test_includes_bos_exactly_once(self):
        assert prompt.clone_prompt("a", "b", [1]).count("<|begin_of_text|>") == 1

    def test_ends_with_the_reference_codes_so_the_model_continues(self):
        built = prompt.clone_prompt("a", "b", [1, 2, 3])
        assert built.endswith("<|s_1|><|s_2|><|s_3|>")
        assert "<|speech_end|>" not in built

    def test_reference_codes_survive_a_round_trip(self):
        codes = [11, 22, 33]
        assert prompt.parse_audio_tokens(prompt.clone_prompt("a", "b", codes)) == codes


class TestTrainingExample:
    def test_exact_string(self):
        assert prompt.training_example("Hi.", [3]) == (
            "<|begin_of_text|><|text_prompt_start|>Hi.<|text_prompt_end|>"
            "<|speech_start|><|s_3|><|speech_end|>"
        )

    def test_is_the_tts_prompt_with_bos_and_a_terminated_continuation(self):
        example = prompt.training_example("Hi.", [3])
        assert example.startswith("<|begin_of_text|>" + prompt.tts_prompt("Hi."))
        assert example.endswith("<|speech_end|>")

    @pytest.mark.parametrize("text", ["Hi.", " Hi.\n"])
    def test_text_is_stripped(self, text):
        assert prompt.training_example(text, []) == prompt.training_example("Hi.", [])
