"""Turning one row of logits into one token.

Batch size is always 1 here, so everything operates on a **1-D tensor of logits** rather than
on a padded batch. That keeps the functions small enough to test against hand-built tensors
with known answers, which is the only way to be sure a sampler is right.

The order of operations is not a matter of taste -- it is the order the checkpoint's presets
(:data:`~kova_tts.engine.types.TTS_SAMPLING`, :data:`~kova_tts.engine.types.CLONE_SAMPLING`)
were tuned under, and they only mean what they are supposed to mean under it:

1. repetition penalty, over every token already seen (prompt *and* output),
2. temperature,
3. top-k,
4. top-p,
5. a draw from the surviving distribution.

Penalising before dividing by the temperature matters: the penalty is multiplicative on the
logit, so applying it after a temperature of 0.9 would scale it by 1/0.9 as well.

``temperature == 0`` short-circuits to argmax *after* the repetition penalty, which is the only
form of greedy decoding worth having -- an unpenalised argmax loop on this model walks straight
into a repeating loop of the same code.
"""

from __future__ import annotations

import torch

#: Logit value used to mask a token out. ``-inf`` is exact under softmax and survives division
#: by a temperature, which a large finite negative number does not.
MASKED = float("-inf")


def _check_row(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() != 1:
        raise ValueError(
            f"Expected a single row of logits with shape [vocab], got {tuple(logits.shape)}. "
            f"This engine is batch-1; index the row you want before sampling."
        )
    return logits


def apply_repetition_penalty(
    logits: torch.Tensor,
    previous: torch.Tensor,
    penalty: float,
) -> torch.Tensor:
    """Divide the logits of already-seen tokens by `penalty` (multiply, if negative).

    The CTRL formulation: a positive logit is divided and a negative one multiplied, so the
    penalty always pushes a seen token *down* regardless of sign. Applying it to a negative
    logit by division would raise it instead.

    `previous` is either a 1-D tensor of token indices (duplicates are harmless -- the penalty
    is applied once per distinct token, not once per occurrence) or a bool mask the same length
    as `logits`. The generator keeps a mask, since it updates it once per step; callers holding
    a list of tokens can pass that instead.
    """
    _check_row(logits)
    if penalty == 1.0 or previous.numel() == 0:
        return logits
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {penalty}.")

    if previous.dtype == torch.bool:
        if previous.shape != logits.shape:
            raise ValueError(
                f"A bool `previous` mask must match the logits, got {tuple(previous.shape)} "
                f"vs {tuple(logits.shape)}."
            )
        seen = previous
    else:
        seen = torch.zeros_like(logits, dtype=torch.bool)
        seen[previous.to(torch.long)] = True

    penalised = torch.where(logits > 0, logits / penalty, logits * penalty)
    return torch.where(seen, penalised, logits)


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Divide by `temperature`. A temperature of 0 is returned unchanged -- see :func:`sample`,
    which reads it as a request for argmax rather than for a division by zero."""
    _check_row(logits)
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")
    if temperature in (0.0, 1.0):
        return logits
    return logits / temperature


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask out everything below the `top_k`-th largest logit. ``top_k <= 0`` disables it.

    Ties on the threshold are all kept, so this can leave more than `top_k` candidates alive.
    That matches every other implementation, and only happens on exactly equal logits.
    """
    _check_row(logits)
    if top_k <= 0 or top_k >= logits.numel():
        return logits
    threshold = torch.topk(logits, top_k).values[-1]
    return logits.masked_fill(logits < threshold, MASKED)


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Keep the smallest set of tokens whose probabilities sum to at least `top_p`.

    The most likely token always survives, even when it alone exceeds `top_p`, so this can
    never mask out the whole distribution.
    """
    _check_row(logits)
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if top_p == 1.0:
        return logits

    ordered, order = torch.sort(logits, descending=True)
    probabilities = torch.softmax(ordered, dim=-1)
    # Cumulative mass *excluding* each token: a token is dropped only once the tokens strictly
    # more likely than it already cover top_p. The first entry is 0, which is what keeps the
    # most likely token alive.
    covered = probabilities.cumsum(dim=-1) - probabilities
    drop = covered >= top_p
    # Scatter the decision back onto the original token positions; `order` is a permutation,
    # so every position is written exactly once.
    return logits.masked_fill(torch.zeros_like(drop).scatter(0, order, drop), MASKED)


def sample(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    previous: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample one token index from a single row of logits.

    Returns a 0-dim long tensor on the logits' device, so the caller decides when to pay for
    the device-to-host copy.
    """
    _check_row(logits)
    scores = logits.float()
    if previous is not None:
        scores = apply_repetition_penalty(scores, previous, repetition_penalty)
    if temperature == 0:
        return scores.argmax()
    scores = apply_temperature(scores, temperature)
    scores = apply_top_k(scores, top_k)
    scores = apply_top_p(scores, top_p)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.multinomial(probabilities, 1)[0]
