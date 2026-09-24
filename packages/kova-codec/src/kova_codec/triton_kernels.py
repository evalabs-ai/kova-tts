"""Triton kernels for the decode path on CUDA and ROCm GPUs.

The counterpart of :mod:`kova_codec.mps_kernels`: the same fused anti-aliased activation,
written in Triton, which compiles for whatever GPU it runs on rather than for one architecture.
Every entry point has a pure-torch fallback at its call site, and nothing here is imported
unless a CUDA tensor reaches that call site -- a CPU or Metal process never loads Triton.

The anti-aliased activation is a 2x upsampling filter, SnakeBeta, and a 2x decimating filter.
In torch that is a replicate pad, two depthwise convolutions, an interleave, the nonlinearity,
another pad and a strided convolution -- about eight kernels, each reading and writing a tensor
twice the width of the layer, 43 times per decode. Here it is two kernels:

* :func:`_up_snake_kernel` reads ``x`` once and writes the upsampled, activated signal as two
  phase arrays (even and odd output samples), so both its stores are contiguous;
* :func:`_down_kernel` reads those back and writes the decimated output.

Two rather than one because the decimator is a stride-2, 12-tap stencil over the upsampled
signal: fused, every output would recompute six upsampled samples and their ``sin``.

Deliberately plain, unlike the batch-throughput variant this is derived from: the accurate
``tl.sin`` rather than a polynomial (which moved samples by up to 0.035), and a fixed launch
configuration per length rather than autotuning, which benchmarks every new length before
running it -- the per-shape stall :func:`kova_tts.engine.decoder.cudnn` exists to avoid.
"""

from __future__ import annotations

import functools
import os

import torch

try:  # Triton ships with CUDA builds of torch on Linux; anywhere else the torch path is used.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the platform's torch build
    triton = None

#: Set to ``0`` to force the pure-torch path, for comparison or to rule the kernel out.
ENV_TRITON = "KOVA_CODEC_TRITON"


@functools.lru_cache(maxsize=1)
def available() -> bool:
    """True when Triton is importable and a GPU is visible to launch on."""
    return triton is not None and torch.cuda.is_available()


def enabled() -> bool:
    """:func:`available`, unless ``KOVA_CODEC_TRITON=0`` switches the kernel off.

    Read per call, not cached, so a test or a benchmark can turn the kernel off and on.
    """
    return os.environ.get(ENV_TRITON, "1").strip() != "0" and available()


def _launch_config(length: int) -> tuple[int, int]:
    """``(BLOCK, num_warps)`` for a row of `length` samples.

    Short rows get small blocks: a streaming window is a few dozen frames at the top of the
    decoder, and a wide block would leave most of its lanes masked off. Long rows cap at 1024,
    which keeps a whole block's worth of loads in flight without spilling.
    """
    block = min(1024, max(64, triton.next_power_of_2(length)))
    return block, (2 if block <= 128 else 4 if block <= 512 else 8)


if triton is not None:

    @triton.jit
    def _clipped(ptr, row, index, last):
        """``ptr[row + clamp(index, 0, last)]`` as float32: replicate padding, without a pad."""
        return tl.load(ptr + row + tl.minimum(tl.maximum(index, 0), last)).to(tl.float32)

    @triton.jit
    def _up_snake_kernel(
        x_ptr,
        even_ptr,
        odd_ptr,
        alpha_ptr,
        beta_inv_ptr,
        up_ptr,
        channels,
        length,
        BLOCK: tl.constexpr,
    ):
        """Upsample 2x through the 12-tap filter's two polyphase halves, then SnakeBeta.

        With ``w`` the 12 taps and ``x`` replicate-padded, output samples ``2i`` and ``2i + 1``
        of the upsampler are::

            even[i] = 2 * (x[i-3] w11 + x[i-2] w9 + x[i-1] w7 + x[i] w5 + x[i+1] w3 + x[i+2] w1)
            odd[i]  = 2 * (x[i-2] w10 + x[i-1] w8 + x[i] w6 + x[i+1] w4 + x[i+2] w2 + x[i+3] w0)

        which is ``UpSample1d`` -- replicate pad, polyphase convolution, interleave and crop --
        written out per sample.
        """
        row = tl.program_id(0)
        channel = row % channels
        base = row * length
        i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        last = length - 1

        alpha = tl.load(alpha_ptr + channel)
        beta_inv = tl.load(beta_inv_ptr + channel)
        xm3 = _clipped(x_ptr, base, i - 3, last)
        xm2 = _clipped(x_ptr, base, i - 2, last)
        xm1 = _clipped(x_ptr, base, i - 1, last)
        x0 = _clipped(x_ptr, base, i, last)
        xp1 = _clipped(x_ptr, base, i + 1, last)
        xp2 = _clipped(x_ptr, base, i + 2, last)
        xp3 = _clipped(x_ptr, base, i + 3, last)

        even = 2.0 * (
            xm3 * tl.load(up_ptr + 11)
            + xm2 * tl.load(up_ptr + 9)
            + xm1 * tl.load(up_ptr + 7)
            + x0 * tl.load(up_ptr + 5)
            + xp1 * tl.load(up_ptr + 3)
            + xp2 * tl.load(up_ptr + 1)
        )
        odd = 2.0 * (
            xm2 * tl.load(up_ptr + 10)
            + xm1 * tl.load(up_ptr + 8)
            + x0 * tl.load(up_ptr + 6)
            + xp1 * tl.load(up_ptr + 4)
            + xp2 * tl.load(up_ptr + 2)
            + xp3 * tl.load(up_ptr + 0)
        )
        # SnakeBeta: x + sin(alpha * x)^2 / beta, with exp() and the reciprocal done on the host.
        s_even = tl.sin(even * alpha)
        s_odd = tl.sin(odd * alpha)
        mask = i < length
        tl.store(even_ptr + base + i, even + beta_inv * s_even * s_even, mask=mask)
        tl.store(odd_ptr + base + i, odd + beta_inv * s_odd * s_odd, mask=mask)

    @triton.jit
    def _down_kernel(even_ptr, odd_ptr, out_ptr, down_ptr, length, BLOCK: tl.constexpr):
        """Low-pass and decimate 2x: ``out[t] = sum_k w[k] * up[clamp(2t + k - 5, 0, 2T - 1)]``.

        ``DownSample1d`` -- replicate pad (5, 6) then a stride-2 convolution -- read out of the
        two phase arrays: upsampled sample ``j`` is ``even[j // 2]`` or ``odd[j // 2]`` by its
        parity. Clamping ``j`` itself, not its half, is what makes the edges match exactly.
        """
        base = tl.program_id(0) * length
        t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        mask = t < length
        top = 2 * length - 1
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for k in tl.static_range(12):
            j = tl.minimum(tl.maximum(2 * t + k - 5, 0), top)
            e = tl.load(even_ptr + base + j // 2, mask=mask, other=0.0)
            o = tl.load(odd_ptr + base + j // 2, mask=mask, other=0.0)
            acc += tl.load(down_ptr + k) * tl.where((j & 1) == 0, e, o)
        tl.store(out_ptr + base + t, acc, mask=mask)


def fused_activation1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta_inv: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
) -> torch.Tensor:
    """Upsample, SnakeBeta and downsample ``[B, C, T]`` -- the same contract as the Metal kernel.

    Args:
        x: ``[B, C, T]`` on a CUDA device, float16, bfloat16 or float32.
        alpha: ``exp(SnakeBeta.alpha)``, float32 ``[C]``.
        beta_inv: ``1 / (exp(SnakeBeta.beta) + eps)``, float32 ``[C]``.
        up_filter: The upsampler's 12 taps, float32 ``[12]``.
        down_filter: The decimator's 12 taps, float32 ``[12]``.

    The upsampled signal is kept in float32 between the two kernels whatever ``x`` is, so the
    only rounding to ``x``'s dtype is the final store -- one fewer than the torch path makes.
    """
    b, c, t = x.shape
    x = x.contiguous()
    even = torch.empty((b, c, t), dtype=torch.float32, device=x.device)
    odd = torch.empty_like(even)
    out = torch.empty_like(x)
    block, warps = _launch_config(t)
    grid = (b * c, triton.cdiv(t, block))
    # Triton compiles for, and launches on, torch's *current* device. With more than one GPU the
    # codec's need not be it, and a kernel built for another architecture will not load.
    with torch.cuda.device(x.device):
        _up_snake_kernel[grid](
            x, even, odd, alpha, beta_inv, up_filter, c, t, BLOCK=block, num_warps=warps
        )
        _down_kernel[grid](even, odd, out, down_filter, t, BLOCK=block, num_warps=warps)
    return out
