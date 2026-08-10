"""The MLX sampler against the torch one it was transcribed from.

The draw itself cannot agree -- two different generators -- so what is checked is everything
before it: the same logits go in, the same *scores* come out, stage by stage and composed. A
divergence here is a checkpoint sampled under different parameters than its presets name,
which is inaudible in the code and very audible in the output.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from kova_tts.engine import sampling

mx = pytest.importorskip("mlx.core", reason="the MLX sampler needs Apple Silicon")

from kova_tts.engine import mlx_sampling  # noqa: E402  (after the platform guard)

RNG = np.random.default_rng(20240805)


def pair(values: np.ndarray) -> tuple[torch.Tensor, mx.array]:
    """The same logits as a torch tensor and an MLX array."""
    return torch.from_numpy(values.astype(np.float32)), mx.array(values.astype(np.float32))


def same(torch_out: torch.Tensor, mlx_out: mx.array, *, tol: float = 1e-5) -> bool:
    """Equal to within float32 noise, with -inf compared as -inf rather than subtracted."""
    a = torch_out.numpy()
    b = np.array(mlx_out)
    masked = np.isneginf(a)
    return bool(
        np.array_equal(masked, np.isneginf(b))
        and np.allclose(a[~masked], b[~masked], atol=tol, rtol=tol)
    )


@pytest.fixture
def logits():
    return RNG.normal(size=512).astype(np.float32) * 4.0


@pytest.fixture
def seen():
    return RNG.random(512) < 0.3


class TestStages:
    @pytest.mark.parametrize("penalty", [1.0, 1.1, 2.0, 0.5])
    def test_repetition_penalty(self, logits, seen, penalty):
        t, m = pair(logits)
        assert same(
            sampling.apply_repetition_penalty(t, torch.from_numpy(seen), penalty),
            mlx_sampling.apply_repetition_penalty(m, mx.array(seen), penalty),
        )

    @pytest.mark.parametrize("temperature", [0.5, 1.0, 1.1, 3.0])
    def test_temperature(self, logits, temperature):
        t, m = pair(logits)
        assert same(
            sampling.apply_temperature(t, temperature),
            mlx_sampling.apply_temperature(m, temperature),
        )

    @pytest.mark.parametrize("top_k", [0, 1, 75, 512, 9000])
    def test_top_k(self, logits, top_k):
        t, m = pair(logits)
        assert same(sampling.apply_top_k(t, top_k), mlx_sampling.apply_top_k(m, top_k))

    @pytest.mark.parametrize("top_p", [0.1, 0.9, 0.99, 1.0])
    def test_top_p(self, logits, top_p):
        t, m = pair(logits)
        assert same(sampling.apply_top_p(t, top_p), mlx_sampling.apply_top_p(m, top_p))

    def test_top_p_keeps_the_most_likely_token_alone_past_the_threshold(self):
        # One token holds more than top_p of the mass; masking everything would leave the
        # sampler nothing to draw.
        values = np.array([10.0, 0.0, -1.0, -2.0], dtype=np.float32)
        t, m = pair(values)
        assert same(sampling.apply_top_p(t, 0.5), mlx_sampling.apply_top_p(m, 0.5))
        assert not np.isneginf(np.array(mlx_sampling.apply_top_p(m, 0.5))[0])


class TestComposed:
    def test_the_tts_preset_produces_the_same_scores(self, logits, seen):
        """The full chain at the shipped preset, which is the only combination that ships."""
        t, m = pair(logits)
        expected = sampling.apply_top_p(
            sampling.apply_top_k(
                sampling.apply_temperature(
                    sampling.apply_repetition_penalty(t, torch.from_numpy(seen), 1.1), 1.1
                ),
                75,
            ),
            0.9,
        )
        actual = mlx_sampling.apply_top_p(
            mlx_sampling.apply_top_k(
                mlx_sampling.apply_temperature(
                    mlx_sampling.apply_repetition_penalty(m, mx.array(seen), 1.1), 1.1
                ),
                75,
            ),
            0.9,
        )
        assert same(expected, actual)

    def test_greedy_picks_the_same_token(self, logits, seen):
        """``temperature=0`` is argmax *after* the penalty, on both backends."""
        t, m = pair(logits)
        expected = sampling.sample(
            t, temperature=0.0, repetition_penalty=1.1, previous=torch.from_numpy(seen)
        )
        actual = mlx_sampling.sample(
            m, temperature=0.0, repetition_penalty=1.1, previous=mx.array(seen)
        )
        assert int(expected) == int(actual)


class TestDraw:
    def test_a_seed_makes_the_draw_repeatable(self, logits):
        _, m = pair(logits)
        first = mlx_sampling.sample(m, temperature=1.1, top_k=75, key=mx.random.key(7))
        second = mlx_sampling.sample(m, temperature=1.1, top_k=75, key=mx.random.key(7))
        assert int(first) == int(second)

    def test_only_surviving_tokens_are_ever_drawn(self, logits):
        _, m = pair(logits)
        survivors = {
            index
            for index, value in enumerate(np.array(mlx_sampling.apply_top_k(m, 5)))
            if not np.isneginf(value)
        }
        drawn = {
            int(mlx_sampling.sample(m, temperature=1.0, top_k=5, key=mx.random.key(seed)))
            for seed in range(64)
        }
        assert drawn <= survivors

    def test_a_row_of_logits_is_required(self):
        with pytest.raises(ValueError, match="batch-1"):
            mlx_sampling.sample(mx.zeros((2, 4)))
