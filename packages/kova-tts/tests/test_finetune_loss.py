"""The ending-weight recipe, checked against numbers computed by hand.

This is where an off-by-one hides: the weight channel lives in label space and is shifted with
the labels, so a weight that lands one position early would upweight the wrong audio token and
degrade a run without ever failing. Every assertion here therefore names exact positions and
exact values rather than checking shapes.
"""

from __future__ import annotations

import math

import pytest
import torch

from kova_tts.finetune.loss import (
    WEIGHT_KEY,
    EndingWeight,
    EndingWeightTrainer,
    ramp_weights,
    weighted_lm_loss,
)

SPEECH_END_ID = 136450


def plain_labels(length: int, *, masked_through: int, speech_end_at: int | None = None):
    """Labels shaped like a real example: masked prompt, then audio, then EOS."""
    labels = torch.arange(1, length + 1, dtype=torch.long)
    labels[: masked_through + 1] = -100
    if speech_end_at is not None:
        labels[speech_end_at] = SPEECH_END_ID
    return labels


class TestRampWeights:
    def test_ramp_covers_exactly_the_positions_before_speech_end(self):
        labels = plain_labels(20, masked_through=3, speech_end_at=15)
        weights = ramp_weights(labels, 15, EndingWeight(ramp_tokens=4, ramp_max=4.5))

        assert torch.equal(weights[:11], torch.zeros(11))
        assert weights[11:15].tolist() == pytest.approx([1.0, 2.1666667, 3.3333333, 4.5])
        # <|speech_end|> itself is weighted by label id in the loss, not by the ramp.
        assert weights[15:].tolist() == [0.0] * 5

    def test_default_recipe_is_ten_positions_ramping_to_four_and_a_half(self):
        labels = plain_labels(60, masked_through=5, speech_end_at=50)
        weights = ramp_weights(labels, 50, EndingWeight())

        ramp = weights[40:50]
        assert len(ramp) == 10
        assert ramp[0].item() == pytest.approx(1.0)
        assert ramp[-1].item() == pytest.approx(4.5)
        # Linear, so consecutive gaps are all equal.
        gaps = (ramp[1:] - ramp[:-1]).tolist()
        assert gaps == pytest.approx([3.5 / 9] * 9)
        assert weights[:40].sum().item() == 0.0
        assert weights[50:].sum().item() == 0.0

    def test_ramp_never_reaches_into_the_masked_prompt(self):
        # <|speech_start|> at index 8 means labels 0..8 are -100; only 9 and 10 are real.
        labels = plain_labels(20, masked_through=8, speech_end_at=11)
        weights = ramp_weights(labels, 11, EndingWeight(ramp_tokens=10, ramp_max=4.5))

        assert weights[:9].sum().item() == 0.0
        assert weights[9:11].tolist() == pytest.approx([1.0, 4.5])

    def test_ramp_is_clamped_at_the_start_of_the_sequence(self):
        labels = torch.arange(1, 6, dtype=torch.long)
        weights = ramp_weights(labels, 4, EndingWeight(ramp_tokens=10, ramp_max=3.0))
        assert weights[:4].tolist() == pytest.approx([1.0, 1.6666667, 2.3333333, 3.0])

    def test_no_weights_when_speech_end_is_absent(self):
        labels = plain_labels(20, masked_through=3)
        assert ramp_weights(labels, None, EndingWeight()).sum().item() == 0.0

    def test_no_weights_when_the_ending_itself_is_masked(self):
        labels = plain_labels(20, masked_through=17, speech_end_at=15)
        assert ramp_weights(labels, 15, EndingWeight()).sum().item() == 0.0

    @pytest.mark.parametrize(
        "ending",
        [
            EndingWeight(enabled=False),
            EndingWeight(ramp_tokens=0),
            EndingWeight(ramp_max=1.0),
        ],
    )
    def test_inactive_recipes_produce_no_ramp(self, ending):
        labels = plain_labels(20, masked_through=3, speech_end_at=15)
        assert ramp_weights(labels, 15, ending).sum().item() == 0.0


class TestEndingWeightValidation:
    def test_negative_ramp_tokens_is_rejected(self):
        with pytest.raises(ValueError, match="ramp_tokens must be >= 0"):
            EndingWeight(ramp_tokens=-1).validate()

    def test_ramp_max_below_one_would_downweight_the_ending(self):
        with pytest.raises(ValueError, match="opposite of the intent"):
            EndingWeight(ramp_max=0.5).validate()

    def test_eos_weight_below_one_is_rejected(self):
        with pytest.raises(ValueError, match="eos_token_weight must be >= 1.0"):
            EndingWeight(eos_token_weight=0.0).validate()

    def test_defaults_are_the_documented_recipe(self):
        ending = EndingWeight()
        assert (ending.enabled, ending.ramp_tokens, ending.ramp_max, ending.eos_token_weight) == (
            True,
            10,
            4.5,
            3.0,
        )
        ending.validate()

    def test_activity_flags(self):
        assert EndingWeight().active
        assert not EndingWeight(enabled=False).active
        # ramp_tokens=0 keeps the flat EOS upweight and nothing else: still active, no ramp.
        flat = EndingWeight(ramp_tokens=0)
        assert flat.active and not flat.ramps and flat.weights_eos


def manual_loss(logits, labels, weights_by_position):
    """Reference implementation, written the long way round with an explicit Python loop."""
    total, denominator = 0.0, 0.0
    for batch in range(labels.shape[0]):
        for pos in range(1, labels.shape[1]):
            target = int(labels[batch, pos])
            if target == -100:
                continue
            row = logits[batch, pos - 1].double()
            nll = -(row[target] - torch.logsumexp(row, dim=0)).item()
            weight = weights_by_position[batch][pos]
            total += nll * weight
            denominator += weight
    return total / denominator


class TestWeightedLmLoss:
    @pytest.fixture
    def batch(self):
        torch.manual_seed(0)
        vocab = 12
        logits = torch.randn(2, 6, vocab)
        labels = torch.tensor(
            [
                [-100, -100, 3, 4, 5, 9],
                [-100, 2, 7, 8, 9, -100],
            ],
            dtype=torch.long,
        )
        return logits, labels, vocab

    def test_unweighted_matches_a_hand_computed_mean(self, batch):
        logits, labels, _ = batch
        expected = manual_loss(logits, labels, [[1.0] * 6, [1.0] * 6])
        actual = weighted_lm_loss(logits, labels, None, speech_end_id=9, eos_token_weight=1.0)
        assert actual.item() == pytest.approx(expected, rel=1e-6)

    def test_weights_are_applied_at_the_shifted_positions(self, batch):
        logits, labels, _ = batch
        ending_w = torch.zeros(2, 6)
        ending_w[0, 3] = 2.0
        ending_w[0, 4] = 4.0
        ending_w[1, 2] = 3.0

        by_position = [[1.0] * 6, [1.0] * 6]
        by_position[0][3] = 2.0
        by_position[0][4] = 4.0
        by_position[1][2] = 3.0

        expected = manual_loss(logits, labels, by_position)
        actual = weighted_lm_loss(logits, labels, ending_w, speech_end_id=-1, eos_token_weight=1.0)
        assert actual.item() == pytest.approx(expected, rel=1e-6)

    def test_a_weight_one_position_off_gives_a_different_answer(self, batch):
        """Guards the alignment itself: if the shift were wrong this test would pass silently."""
        logits, labels, _ = batch
        right = torch.zeros(2, 6)
        right[0, 4] = 4.0
        shifted = torch.zeros(2, 6)
        shifted[0, 3] = 4.0

        a = weighted_lm_loss(logits, labels, right, speech_end_id=-1)
        b = weighted_lm_loss(logits, labels, shifted, speech_end_id=-1)
        assert not math.isclose(a.item(), b.item(), rel_tol=1e-4)

    def test_eos_weight_hits_every_speech_end_label(self, batch):
        logits, labels, _ = batch
        by_position = [[1.0] * 6, [1.0] * 6]
        by_position[0][5] = 3.0  # label 9 at (0, 5)
        by_position[1][4] = 3.0  # label 9 at (1, 4)

        expected = manual_loss(logits, labels, by_position)
        actual = weighted_lm_loss(logits, labels, None, speech_end_id=9, eos_token_weight=3.0)
        assert actual.item() == pytest.approx(expected, rel=1e-6)

    def test_eos_weight_overrides_the_ramp_at_the_eos_position(self, batch):
        logits, labels, _ = batch
        ending_w = torch.zeros(2, 6)
        ending_w[0, 5] = 4.5  # a ramp value landing on an EOS label

        by_position = [[1.0] * 6, [1.0] * 6]
        by_position[0][5] = 3.0
        by_position[1][4] = 3.0

        expected = manual_loss(logits, labels, by_position)
        actual = weighted_lm_loss(logits, labels, ending_w, speech_end_id=9, eos_token_weight=3.0)
        assert actual.item() == pytest.approx(expected, rel=1e-6)

    def test_zero_weights_mean_one_not_zero(self, batch):
        logits, labels, _ = batch
        zeros = torch.zeros(2, 6)
        with_channel = weighted_lm_loss(logits, labels, zeros, speech_end_id=-1)
        without = weighted_lm_loss(logits, labels, None, speech_end_id=-1)
        assert with_channel.item() == pytest.approx(without.item(), rel=1e-7)

    def test_num_items_in_batch_normalises_like_the_stock_trainer(self, batch):
        logits, labels, _ = batch
        supervised = int((labels[..., 1:] != -100).sum())
        summed = manual_loss(logits, labels, [[1.0] * 6, [1.0] * 6]) * supervised

        actual = weighted_lm_loss(
            logits, labels, None, speech_end_id=-1, num_items_in_batch=supervised
        )
        assert actual.item() == pytest.approx(summed / supervised, rel=1e-6)

    def test_masked_positions_contribute_nothing(self, batch):
        logits, labels, vocab = batch
        # Rewriting the logits under a masked label must not move the loss at all.
        before = weighted_lm_loss(logits, labels, None, speech_end_id=-1)
        poisoned = logits.clone()
        poisoned[1, 4] = torch.full((vocab,), 50.0)  # predicts label (1, 5), which is -100
        after = weighted_lm_loss(poisoned, labels, None, speech_end_id=-1)
        assert before.item() == pytest.approx(after.item(), rel=1e-7)


class _StubOutput:
    def __init__(self, logits):
        self.logits = logits


class _StubModel(torch.nn.Module):
    """Returns fixed logits, so compute_loss is tested without a real checkpoint."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("_logits", logits)
        self.seen: dict | None = None

    def forward(self, **inputs):
        self.seen = inputs
        return _StubOutput(self._logits)


def make_trainer(ending: EndingWeight) -> EndingWeightTrainer:
    """An EndingWeightTrainer with only the attributes compute_loss touches.

    Constructing a real Trainer needs a model, TrainingArguments and an accelerator; bypassing
    __init__ keeps this a unit test of the loss override rather than of transformers.
    """
    trainer = object.__new__(EndingWeightTrainer)
    trainer.speech_end_id = 9
    trainer.ending = ending
    return trainer


class TestComputeLoss:
    @pytest.fixture
    def inputs(self):
        torch.manual_seed(1)
        logits = torch.randn(2, 6, 12)
        labels = torch.tensor(
            [
                [-100, -100, 3, 4, 5, 9],
                [-100, 2, 7, 8, 9, -100],
            ],
            dtype=torch.long,
        )
        input_ids = labels.clamp(min=0)
        return logits, labels, input_ids

    def test_matches_the_hand_computed_weighted_loss(self, inputs):
        logits, labels, input_ids = inputs
        ending_w = torch.zeros(2, 6)
        ending_w[0, 3] = 2.0
        ending_w[0, 4] = 4.0

        by_position = [[1.0] * 6, [1.0] * 6]
        by_position[0][3] = 2.0
        by_position[0][4] = 4.0
        by_position[0][5] = 3.0  # EOS label
        by_position[1][4] = 3.0  # EOS label
        expected = manual_loss(logits, labels, by_position)

        trainer = make_trainer(EndingWeight())
        loss = trainer.compute_loss(
            _StubModel(logits),
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
                WEIGHT_KEY: ending_w,
            },
        )
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_labels_and_weights_never_reach_the_model(self, inputs):
        logits, labels, input_ids = inputs
        model = _StubModel(logits)
        original = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": labels,
            WEIGHT_KEY: torch.zeros(2, 6),
        }
        make_trainer(EndingWeight()).compute_loss(model, original)

        assert set(model.seen) == {"input_ids", "attention_mask"}
        # compute_loss must not mutate the batch it was handed: Trainer reuses it.
        assert set(original) == {"input_ids", "attention_mask", "labels", WEIGHT_KEY}

    def test_disabled_recipe_reproduces_the_unweighted_loss_exactly(self, inputs):
        logits, labels, input_ids = inputs
        unweighted = manual_loss(logits, labels, [[1.0] * 6, [1.0] * 6])

        trainer = make_trainer(EndingWeight(enabled=False))
        loss = trainer.compute_loss(
            _StubModel(logits),
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
        )
        assert loss.item() == pytest.approx(unweighted, rel=1e-7)

    def test_a_batch_without_the_weight_channel_still_gets_the_eos_upweight(self, inputs):
        logits, labels, input_ids = inputs
        by_position = [[1.0] * 6, [1.0] * 6]
        by_position[0][5] = 3.0
        by_position[1][4] = 3.0
        expected = manual_loss(logits, labels, by_position)

        loss = make_trainer(EndingWeight()).compute_loss(
            _StubModel(logits),
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
        )
        assert loss.item() == pytest.approx(expected, rel=1e-6)

    def test_return_outputs_hands_back_the_model_output(self, inputs):
        logits, labels, input_ids = inputs
        loss, outputs = make_trainer(EndingWeight()).compute_loss(
            _StubModel(logits),
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
            return_outputs=True,
        )
        assert torch.equal(outputs.logits, logits)
        assert loss.ndim == 0
