"""Sampling, against hand-built logits with answers worked out by hand."""

from __future__ import annotations

import math

import pytest
import torch

from kova_tts.engine import sampling


def logits(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class TestRepetitionPenalty:
    def test_positive_logits_are_divided(self):
        out = sampling.apply_repetition_penalty(logits(2.0, 4.0), torch.tensor([1]), 2.0)
        assert out.tolist() == [2.0, 2.0]

    def test_negative_logits_are_multiplied(self):
        # Dividing a negative logit would raise it, which is the opposite of a penalty.
        out = sampling.apply_repetition_penalty(logits(-2.0, -4.0), torch.tensor([1]), 2.0)
        assert out.tolist() == [-2.0, -8.0]

    def test_unseen_tokens_are_untouched(self):
        out = sampling.apply_repetition_penalty(logits(3.0, 3.0), torch.tensor([0]), 3.0)
        assert out.tolist() == [1.0, 3.0]

    def test_a_repeated_token_is_penalised_once(self):
        once = sampling.apply_repetition_penalty(logits(4.0, 1.0), torch.tensor([0]), 2.0)
        twice = sampling.apply_repetition_penalty(logits(4.0, 1.0), torch.tensor([0, 0, 0]), 2.0)
        assert once.tolist() == twice.tolist()

    def test_a_bool_mask_and_an_index_list_agree(self):
        values = logits(1.0, -1.0, 3.0, 0.5)
        mask = torch.tensor([True, False, True, False])
        by_index = sampling.apply_repetition_penalty(values, torch.tensor([0, 2]), 1.4)
        by_mask = sampling.apply_repetition_penalty(values, mask, 1.4)
        assert torch.equal(by_index, by_mask)

    def test_a_penalty_of_one_changes_nothing(self):
        values = logits(1.0, -1.0)
        assert torch.equal(
            sampling.apply_repetition_penalty(values, torch.tensor([0]), 1.0), values
        )

    def test_an_empty_history_changes_nothing(self):
        values = logits(1.0, -1.0)
        empty = torch.zeros(0, dtype=torch.long)
        assert torch.equal(sampling.apply_repetition_penalty(values, empty, 2.0), values)

    def test_a_mismatched_mask_is_rejected(self):
        with pytest.raises(ValueError, match="must match the logits"):
            sampling.apply_repetition_penalty(logits(1.0, 2.0), torch.tensor([True]), 2.0)


class TestTopK:
    def test_keeps_exactly_k(self):
        out = sampling.apply_top_k(logits(1.0, 5.0, 3.0, 2.0), 2)
        assert out.tolist() == [-math.inf, 5.0, 3.0, -math.inf]

    def test_zero_disables_it(self):
        values = logits(1.0, 5.0, 3.0)
        assert torch.equal(sampling.apply_top_k(values, 0), values)

    def test_k_beyond_the_vocabulary_disables_it(self):
        values = logits(1.0, 5.0, 3.0)
        assert torch.equal(sampling.apply_top_k(values, 99), values)

    def test_ties_on_the_threshold_are_all_kept(self):
        out = sampling.apply_top_k(logits(4.0, 4.0, 1.0), 1)
        assert out.tolist() == [4.0, 4.0, -math.inf]


class TestTopP:
    def test_keeps_the_smallest_set_reaching_p(self):
        # softmax over these is exactly [0.6, 0.3, 0.1] after log; build it that way instead.
        values = torch.log(torch.tensor([0.6, 0.3, 0.1]))
        out = sampling.apply_top_p(values, 0.9)
        assert out[2] == -math.inf
        assert out[0] != -math.inf and out[1] != -math.inf

    def test_the_most_likely_token_always_survives(self):
        values = torch.log(torch.tensor([0.95, 0.04, 0.01]))
        out = sampling.apply_top_p(values, 0.5)
        assert out[0] != -math.inf
        assert out[1] == -math.inf and out[2] == -math.inf

    def test_p_of_one_changes_nothing(self):
        values = logits(1.0, 2.0, 3.0)
        assert torch.equal(sampling.apply_top_p(values, 1.0), values)

    def test_masking_is_scattered_back_onto_the_original_order(self):
        values = torch.log(torch.tensor([0.1, 0.6, 0.3]))
        out = sampling.apply_top_p(values, 0.9)
        assert out[0] == -math.inf
        assert out[1] != -math.inf and out[2] != -math.inf

    def test_rejects_p_outside_the_range(self):
        with pytest.raises(ValueError, match="top_p must be"):
            sampling.apply_top_p(logits(1.0, 2.0), 0.0)


class TestTemperature:
    def test_divides(self):
        assert sampling.apply_temperature(logits(2.0, 4.0), 2.0).tolist() == [1.0, 2.0]

    def test_one_is_a_no_op(self):
        values = logits(2.0, 4.0)
        assert torch.equal(sampling.apply_temperature(values, 1.0), values)

    def test_zero_is_left_to_sample(self):
        values = logits(2.0, 4.0)
        assert torch.equal(sampling.apply_temperature(values, 0.0), values)


class TestSample:
    def test_temperature_zero_is_argmax(self):
        values = logits(1.0, 9.0, 3.0)
        for _ in range(5):
            assert int(sampling.sample(values, temperature=0.0)) == 1

    def test_temperature_zero_argmax_is_taken_after_the_penalty(self):
        # The whole point of penalising before going greedy: an unpenalised argmax loop on this
        # model repeats one code forever.
        values = logits(4.0, 3.9)
        seen = torch.tensor([True, False])
        assert int(sampling.sample(values, temperature=0.0)) == 0
        assert (
            int(sampling.sample(values, temperature=0.0, previous=seen, repetition_penalty=2.0))
            == 1
        )

    def test_the_same_seed_gives_the_same_draw(self):
        values = torch.randn(64, generator=torch.Generator().manual_seed(0))
        first = [
            int(
                sampling.sample(values, temperature=1.0, generator=torch.Generator().manual_seed(7))
            )
            for _ in range(3)
        ]
        assert len(set(first)) == 1

    def test_different_seeds_eventually_differ(self):
        values = torch.zeros(1000)  # uniform: any two seeds almost surely disagree
        draws = {
            int(
                sampling.sample(values, temperature=1.0, generator=torch.Generator().manual_seed(s))
            )
            for s in range(8)
        }
        assert len(draws) > 1

    def test_masked_tokens_are_never_drawn(self):
        values = logits(10.0, 9.9, 0.0, 0.0)
        gen = torch.Generator().manual_seed(3)
        drawn = {
            int(sampling.sample(values, temperature=1.0, top_k=2, generator=gen)) for _ in range(50)
        }
        assert drawn <= {0, 1}

    def test_top_p_and_top_k_compose_in_the_documented_order(self):
        values = torch.log(torch.tensor([0.4, 0.3, 0.2, 0.1]))
        expected = sampling.apply_top_p(
            sampling.apply_top_k(sampling.apply_temperature(values, 0.5), 3), 0.9
        )
        gen_a = torch.Generator().manual_seed(11)
        gen_b = torch.Generator().manual_seed(11)
        drawn = int(sampling.sample(values, temperature=0.5, top_k=3, top_p=0.9, generator=gen_a))
        wanted = int(torch.multinomial(torch.softmax(expected, -1), 1, generator=gen_b)[0])
        assert drawn == wanted

    def test_rejects_a_batch_of_logits(self):
        with pytest.raises(ValueError, match="batch-1"):
            sampling.sample(torch.zeros(2, 8))
