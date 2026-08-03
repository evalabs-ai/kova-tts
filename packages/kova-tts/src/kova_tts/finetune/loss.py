"""Ending-weighted cross entropy: the recipe that teaches the model to stop.

A LoRA trained with plain cross entropy learns a voice quickly but keeps *ending* the
utterance badly -- it trails off, repeats the last syllable, or runs past the transcript
before finally emitting ``<|speech_end|>``. The cause is arithmetic: an ending is a handful of
token positions out of thousands, so the gradient signal that decides "stop here" is drowned
out by the signal that decides "keep talking in this voice".

So this module reweights rather than resamples: the per-position cross entropy on the last few
real positions before ``<|speech_end|>`` is multiplied by a linear ramp, and the
``<|speech_end|>`` label itself by a constant. The weights land on the tokens that make up the
ending *pattern* -- weighting silence instead would teach the model that any quiet stretch means
"stop", and it would truncate mid-sentence. On by default; see :class:`EndingWeight`.

Alignment is the thing to get right. A causal LM predicts token ``i`` from position ``i-1``,
so ``labels`` are shifted left by one against ``logits`` before the loss. The weight channel
lives in *label* space -- ``ending_w[i]`` scales the cost of predicting ``labels[i]`` -- and is
therefore shifted with the labels, not with the logits. An off-by-one here would quietly weight
the wrong tokens and degrade a run without ever failing a test, which is why
:func:`weighted_lm_loss` is exercised directly against hand-computed numbers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformers import DataCollatorForSeq2Seq, Trainer

#: Name of the per-position weight channel carried alongside ``input_ids``/``labels``.
WEIGHT_KEY = "ending_w"


@dataclass(frozen=True, slots=True)
class EndingWeight:
    """The ending-weight recipe.

    ``ramp_tokens`` positions immediately before ``<|speech_end|>`` get a cross-entropy
    multiplier rising linearly from 1.0 to ``ramp_max``; the ``<|speech_end|>`` label itself
    gets ``eos_token_weight``. Everything else keeps the default weight of 1.

    Setting ``ramp_tokens: 0`` leaves only the flat EOS upweight, and ``enabled: false`` turns
    the whole recipe off.
    """

    #: Master switch. When false the trainer uses stock, exactly unweighted cross entropy.
    enabled: bool = True

    #: How many real token positions before ``<|speech_end|>`` the ramp covers.
    ramp_tokens: int = 10

    #: Multiplier at the last ramped position. The ramp runs 1.0 -> this value inclusive.
    ramp_max: float = 4.5

    #: Multiplier applied wherever the label is ``<|speech_end|>``.
    eos_token_weight: float = 3.0

    def validate(self) -> None:
        """Raise :class:`ValueError` on a setting that would silently do nothing or misbehave."""
        if self.ramp_tokens < 0:
            raise ValueError(
                f"ending_weight.ramp_tokens must be >= 0, got {self.ramp_tokens}. "
                f"Use 0 to disable the ramp and keep only the EOS upweight."
            )
        if self.ramp_max < 1.0:
            raise ValueError(
                f"ending_weight.ramp_max must be >= 1.0, got {self.ramp_max}. "
                f"Values below 1 would *down*weight the ending, the opposite of the intent."
            )
        if self.eos_token_weight < 1.0:
            raise ValueError(
                f"ending_weight.eos_token_weight must be >= 1.0, got {self.eos_token_weight}. "
                f"Use 1.0 to leave <|speech_end|> at its natural weight."
            )

    @property
    def ramps(self) -> bool:
        """True when the position ramp actually changes anything."""
        return self.enabled and self.ramp_tokens > 0 and self.ramp_max > 1.0

    @property
    def weights_eos(self) -> bool:
        """True when the flat ``<|speech_end|>`` upweight actually changes anything."""
        return self.enabled and self.eos_token_weight > 1.0

    @property
    def active(self) -> bool:
        """True when this recipe changes the loss at all."""
        return self.ramps or self.weights_eos


def ramp_weights(
    labels: torch.Tensor,
    speech_end_index: int | None,
    ending: EndingWeight,
) -> torch.Tensor:
    """Per-position cross-entropy multipliers for one example, in label space.

    Returns a float32 tensor as long as `labels`, zero everywhere except the ramp -- zero means
    "no opinion", which :func:`weighted_lm_loss` reads as the default weight of 1. Encoding the
    default as 0 rather than 1 is what lets the collator right-pad the channel with zeros.

    The ramp stops early at any ``-100`` label, so an example whose ending is masked (or whose
    prompt reaches to within ``ramp_tokens`` of the end) is weighted only over supervised
    positions.
    """
    weights = torch.zeros(labels.numel(), dtype=torch.float32)
    if not ending.ramps or speech_end_index is None:
        return weights

    end = int(speech_end_index)
    if not (0 <= end < labels.numel()) or labels[end].item() == -100:
        # <|speech_end|> was never supervised: nothing here teaches an ending, so weighting the
        # preceding tokens would just amplify mid-utterance audio.
        return weights

    start = max(0, end - ending.ramp_tokens)
    while start < end and labels[start].item() == -100:
        start += 1
    if start < end:
        weights[start:end] = torch.linspace(1.0, ending.ramp_max, end - start)
    return weights


def weighted_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ending_w: torch.Tensor | None = None,
    *,
    speech_end_id: int,
    eos_token_weight: float = 1.0,
    num_items_in_batch: torch.Tensor | int | None = None,
) -> torch.Tensor:
    """Causal-LM cross entropy with a per-position weight channel.

    `logits` is ``(batch, seq, vocab)``, `labels` and `ending_w` are ``(batch, seq)`` in label
    space; all three are shifted here exactly as a stock causal-LM head shifts them. Positions
    where ``ending_w`` is 0 -- and every position when it is ``None`` -- weigh 1.

    Normalisation follows the stock Trainer: divide by ``num_items_in_batch`` when the Trainer
    supplies it, so gradient accumulation scales identically to an unweighted run, and by the
    total weight otherwise. Logits are upcast to float32 first, as the stock loss does, because
    a bf16 softmax over a 136k vocabulary loses real precision.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    flat_logits = shift_logits.view(-1, shift_logits.size(-1)).float()
    flat_labels = shift_labels.view(-1).to(flat_logits.device)

    # reduction="none" with the default ignore_index leaves masked positions at exactly 0.
    nll = F.cross_entropy(flat_logits, flat_labels, reduction="none", ignore_index=-100)

    if ending_w is None:
        weights = torch.ones_like(nll)
    else:
        flat_w = ending_w[..., 1:].contiguous().reshape(-1).to(nll.device, nll.dtype)
        weights = torch.where(flat_w > 0, flat_w, torch.ones_like(nll))
    if eos_token_weight > 1.0:
        weights = torch.where(
            flat_labels == speech_end_id,
            torch.full_like(weights, float(eos_token_weight)),
            weights,
        )

    valid = flat_labels != -100
    total = (nll * weights)[valid].sum()
    if num_items_in_batch is not None:
        return total / num_items_in_batch
    denominator = weights[valid].sum()
    return total / denominator.clamp(min=1.0)


class EndingWeightCollator(DataCollatorForSeq2Seq):
    """Pads the weight channel alongside ``input_ids`` and ``labels``.

    ``DataCollatorForSeq2Seq`` knows nothing about extra per-position channels and would try to
    tokenizer-pad it, so the channel is lifted out before delegating and right-padded with zeros
    (= weight 1) to the batch width the parent chose, ``pad_to_multiple_of`` included. Items
    without the channel -- validation rows, which are scored with plain cross entropy -- pad to
    all zeros and are therefore unweighted.
    """

    def __call__(self, features: Sequence[dict[str, Any]], return_tensors: str | None = None):
        weights = [f.get(WEIGHT_KEY) for f in features]
        stripped = [{k: v for k, v in f.items() if k != WEIGHT_KEY} for f in features]
        batch = super().__call__(stripped, return_tensors)

        width = batch["input_ids"].shape[1]
        padded = torch.zeros(len(weights), width, dtype=torch.float32)
        for row, weight in enumerate(weights):
            if weight is not None:
                padded[row, : weight.numel()] = weight
        batch[WEIGHT_KEY] = padded
        return batch


class EndingWeightTrainer(Trainer):
    """``Trainer`` whose loss honours the weight channel and the EOS upweight.

    One asymmetry to know about when reading ``eval_loss``: the EOS upweight is keyed on the
    label id, so it applies to evaluation batches too, while the ramp does not -- validation
    examples are built without the weight channel. ``eval_loss`` is therefore comparable across
    runs that share an ``eos_token_weight``, but not across runs that change it.
    """

    def __init__(self, *args: Any, speech_end_id: int, ending: EndingWeight, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.speech_end_id = int(speech_end_id)
        self.ending = ending

    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        forward_inputs = dict(inputs)
        labels = forward_inputs.pop("labels")
        ending_w = forward_inputs.pop(WEIGHT_KEY, None)

        outputs = model(**forward_inputs)
        loss = weighted_lm_loss(
            outputs.logits,
            labels,
            ending_w,
            speech_end_id=self.speech_end_id,
            eos_token_weight=self.ending.eos_token_weight if self.ending.enabled else 1.0,
            num_items_in_batch=num_items_in_batch,
        )
        return (loss, outputs) if return_outputs else loss
