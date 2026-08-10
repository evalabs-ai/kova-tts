"""Turning one row of logits into one token, inside the MLX graph.

A transcription of :mod:`kova_tts.engine.sampling` into MLX, operation for operation and in
the same order -- repetition penalty, temperature, top-k, top-p, draw -- because that order is
what the checkpoint's presets were tuned under and they only mean what they are supposed to
mean under it. ``tests/test_mlx_sampling.py`` asserts the two agree on the deterministic
stages; the draw itself cannot agree, since torch and MLX have different generators.

**Why this exists rather than reusing the torch sampler.** Sampling on the host means
evaluating the logits, copying them out of the unified buffer, and only then knowing which
token to feed back -- the GPU idles through all of it, once per token. Kept inside the graph,
the sampled token is just another node, the next step can be queued before the host has looked
at this one, and the whole decode pipelines. Measured on a base M1 with the 4-bit g128
artifact: **74.9 tok/s host-side against 81.6 in-graph**, which is the difference between
0.94x and 1.02x real time.

Every knob is a Python value rather than an array, so each ``if`` here is resolved while the
graph is built and a disabled stage costs nothing at all.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

#: Logit value used to mask a token out. ``-inf`` is exact under softmax and survives division
#: by a temperature, which a large finite negative number does not.
MASKED = float("-inf")


def apply_repetition_penalty(logits: mx.array, previous: mx.array, penalty: float) -> mx.array:
    """Divide the logits of already-seen tokens by `penalty` (multiply, if negative).

    The CTRL formulation: a positive logit is divided and a negative one multiplied, so the
    penalty always pushes a seen token *down* regardless of sign.

    `previous` is a boolean array the same length as `logits` -- the generator keeps a mask
    rather than a list, since it updates it once per step and the update has to stay in the
    graph.
    """
    if penalty == 1.0:
        return logits
    if penalty <= 0:
        raise ValueError(f"repetition_penalty must be > 0, got {penalty}.")
    return mx.where(previous, mx.where(logits > 0, logits / penalty, logits * penalty), logits)


def apply_temperature(logits: mx.array, temperature: float) -> mx.array:
    """Divide by `temperature`. A temperature of 0 is returned unchanged -- see :func:`sample`,
    which reads it as a request for argmax rather than for a division by zero."""
    if temperature < 0:
        raise ValueError(f"temperature must be >= 0, got {temperature}.")
    if temperature in (0.0, 1.0):
        return logits
    return logits / temperature


def apply_top_k(logits: mx.array, top_k: int) -> mx.array:
    """Mask out everything below the `top_k`-th largest logit. ``top_k <= 0`` disables it.

    Ties on the threshold are all kept, which matches the torch sampler and every other
    implementation, and only happens on exactly equal logits.
    """
    if top_k <= 0 or top_k >= logits.size:
        return logits
    threshold = mx.sort(logits)[-top_k]
    return mx.where(logits < threshold, MASKED, logits)


def apply_top_p(logits: mx.array, top_p: float) -> mx.array:
    """Keep the smallest set of tokens whose probabilities sum to at least `top_p`.

    The most likely token always survives, even when it alone exceeds `top_p`, so this can
    never mask out the whole distribution.
    """
    if not 0 < top_p <= 1:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}.")
    if top_p == 1.0:
        return logits

    # Descending order, by sorting the negated scores: MLX sorts ascending only. A masked
    # -inf logit negates to +inf and lands at the end, which is where it belongs.
    order = mx.argsort(-logits)
    probabilities = mx.softmax(logits[order])
    # Cumulative mass *excluding* each token: a token is dropped only once the tokens strictly
    # more likely than it already cover top_p. The first entry is 0, which is what keeps the
    # most likely token alive.
    covered = mx.cumsum(probabilities) - probabilities
    drop = covered >= top_p
    # Back to the original positions. `order` is a permutation, so argsort of it is its
    # inverse: a gather, where the torch version scatters. Same result, and MLX gathers are
    # the cheaper of the two.
    return mx.where(drop[mx.argsort(order)], MASKED, logits)


def sample(
    logits: mx.array,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    repetition_penalty: float = 1.0,
    previous: mx.array | None = None,
    key: Any | None = None,
) -> mx.array:
    """Sample one token index from a single row of logits.

    Returns a scalar array, unevaluated: the caller decides when to pay for the trip to the
    host, and the point of this module is that it can queue the next decode step first.

    `key` is an MLX PRNG key. ``None`` draws from the global generator.
    """
    if logits.ndim != 1:
        raise ValueError(
            f"Expected a single row of logits with shape [vocab], got {logits.shape}. "
            f"This engine is batch-1; index the row you want before sampling."
        )
    scores = logits.astype(mx.float32)
    if previous is not None:
        scores = apply_repetition_penalty(scores, previous, repetition_penalty)
    if temperature == 0:
        return mx.argmax(scores)
    scores = apply_temperature(scores, temperature)
    scores = apply_top_k(scores, top_k)
    scores = apply_top_p(scores, top_p)
    return mx.random.categorical(scores, key=key)
