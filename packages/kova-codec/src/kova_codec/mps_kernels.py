"""Hand-written Metal kernels for the decode path, used only on Apple's Metal backend.

Everything here is an optimisation of a shape the pure-torch modules already implement, and
every entry point has a ``fused_*`` name and a matching pure-torch fallback at its call site.
Nothing in this module is reachable from CUDA or the CPU: callers dispatch on
``x.device.type == "mps"``, and :func:`activation1d_supported` additionally refuses any
configuration the kernel was not derived for.

Why hand-written kernels at all: the codec's generator is a long chain of *small* operations
over *large* tensors, and on Metal that chain is dominated by memory traffic rather than
arithmetic. One second of audio is 200k samples at the output rate, and the anti-aliased
activation alone reads and writes that tensor about ten times over. Fusing the whole
upsample -> nonlinearity -> downsample sandwich into a single pass is worth **11.4x** on that
module, which was half of decode time.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from kova_codec.vq.alias_free_torch.act import Activation1d

#: ``(shader suffix, threads per group, outputs per thread)`` for the fused activation.
#:
#: Both numbers were swept against the decoder's own shapes. Four outputs per thread beats one
#: by ~1.3x: the threadgroup's shared array of upsampled samples is written once and read twelve
#: times per output, so amortising the barrier over four outputs is most of the win.
#:
#: :data:`_WIDE` is used wherever the row is long enough to fill it. :data:`_NARROW` exists
#: because a streaming window is only a few dozen frames at the top of the decoder, and a wide
#: tile would leave most of its threads with no output to write.
_NARROW = ("64", 64, 1)
_WIDE = ("128", 128, 4)

#: The fused activation kernel is derived for this exact resampler geometry (see the module
#: docstring of ``alias_free_torch.resample``). Anything else falls back to torch.
_RATIO = 2
_KERNEL_SIZE = 12

_ACTIVATION_SHADER = """
#include <metal_stdlib>
using namespace metal;

// One fused pass of UpSample1d -> SnakeBeta -> DownSample1d over a [B, C, T] contiguous tensor.
//
// Writing U for the 2x upsampled signal and X for the (replicate-padded) input, the three
// stages compose to
//
//     out[n] = sum_{k=0..11} d[k] * snake(U[clamp(2n + k - 5, 0, 2T-1)])
//     U[j]   = 2 * sum_{s=0..5} X[clamp((15 + j) / 2 + s - 10, 0, T-1)] * f[((15+j) & 1) + 2*(5-s)]
//     snake(u) = u + sin(u * alpha)^2 / beta
//
// where f is the 12-tap Kaiser-sinc upsampling filter split into its two polyphase components
// and d is the matching 12-tap decimation filter. The clamps are the `replicate` padding the
// torch modules apply, hoisted to the point of use.
//
// A naive thread-per-output would evaluate 12 upsampled samples (and 12 sines) per output, six
// times more than exist. Instead each threadgroup evaluates every upsampled sample its outputs
// need exactly once into threadgroup memory, and only then decimates -- so the sine count
// matches the 2x oversampling rate rather than the filter width.
//
// The input is read straight from device memory rather than staged in threadgroup memory:
// neighbouring threads read overlapping 11-sample windows, which is what the cache is for, and
// not staging removes a barrier. Measured faster at every shape in the decoder.
template<typename T_, uint TG, uint PER>
inline void act1d_tile(device T_* out, device const T_* inp,
                       device const float* alpha, device const float* beta_inv,
                       device const float* upf, device const float* dnf,
                       uint T, uint ntiles, uint C,
                       threadgroup T_* su, uint tid, uint tg) {
    const uint OUTS = TG * PER;         // outputs this threadgroup produces
    const uint NU   = 2 * OUTS + 12;    // upsampled samples they need between them

    uint tile = tg % ntiles;
    uint row  = tg / ntiles;            // row = b * C + c
    uint n0   = tile * OUTS;
    device const T_* x = inp + (ulong)row * T;

    float a = alpha[row % C];
    float bi = beta_inv[row % C];
    for (uint q = tid; q < NU; q += TG) {
        int j = clamp(int(2 * n0 + q) - 5, 0, int(2 * T) - 1);
        uint jj = uint(j) + 15;
        uint m = jj >> 1;
        uint p = jj & 1u;
        float u = 0.0f;
        for (uint s = 0; s < 6; ++s) {
            u = fma(float(x[clamp(int(m + s) - 10, 0, int(T) - 1)]), upf[p + 2 * (5 - s)], u);
        }
        u *= 2.0f;
        float sn = sin(u * a);
        // Stored at the tensor's own precision: the decimation below is a weighted mean whose
        // taps sum to one, so rounding here cannot accumulate past what the output already is.
        su[q] = T_(fma(sn * sn, bi, u));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint b = 0; b < PER; ++b) {
        uint local = tid + b * TG;
        uint n = n0 + local;
        if (n >= T) return;
        float acc = 0.0f;
        for (uint k = 0; k < 12; ++k) acc = fma(dnf[k], float(su[2 * local + k]), acc);
        out[(ulong)row * T + n] = T_(acc);
    }
}

#define ACT1D_KERNEL(NAME, T_, TG, PER)                                     \\
kernel void NAME(device T_* out             [[buffer(0)]],                  \\
                 device const T_* inp       [[buffer(1)]],                  \\
                 device const float* alpha  [[buffer(2)]],                  \\
                 device const float* binv   [[buffer(3)]],                  \\
                 device const float* upf    [[buffer(4)]],                  \\
                 device const float* dnf    [[buffer(5)]],                  \\
                 constant uint& T           [[buffer(6)]],                  \\
                 constant uint& ntiles      [[buffer(7)]],                  \\
                 constant uint& C           [[buffer(8)]],                  \\
                 uint tid [[thread_position_in_threadgroup]],               \\
                 uint tg  [[threadgroup_position_in_grid]]) {               \\
    threadgroup T_ su[2 * TG * PER + 12];                                   \\
    act1d_tile<T_, TG, PER>(out, inp, alpha, binv, upf, dnf, T, ntiles, C, su, tid, tg); \\
}

ACT1D_KERNEL(act1d_h64,  half,   64, 1)
ACT1D_KERNEL(act1d_h128, half,  128, 4)
ACT1D_KERNEL(act1d_f64,  float,  64, 1)
ACT1D_KERNEL(act1d_f128, float, 128, 4)
"""


@functools.lru_cache(maxsize=1)
def _library() -> Any:
    return torch.mps.compile_shader(_ACTIVATION_SHADER)


def available() -> bool:
    """True when this build of torch can compile Metal shaders at runtime.

    ``torch.mps.compile_shader`` arrived in torch 2.7. Older builds fall back to the pure-torch
    modules rather than failing, so the codec still runs -- just slower.
    """
    if not hasattr(torch.mps, "compile_shader"):
        return False
    try:
        _library()
    except Exception:  # a driver or toolchain that will not build the shader
        return False
    return True


def activation1d_supported(module: Activation1d) -> bool:
    """True when `module` has the exact geometry :func:`fused_activation1d` was derived for."""
    from kova_codec.vq.activations import SnakeBeta

    return (
        isinstance(module.act, SnakeBeta)
        and module.up_ratio == _RATIO
        and module.down_ratio == _RATIO
        and module.upsample.kernel_size == _KERNEL_SIZE
        and module.downsample.kernel_size == _KERNEL_SIZE
        and module.downsample.lowpass.padding
        and module.downsample.lowpass.padding_mode == "replicate"
    )


def fused_activation1d(
    x: torch.Tensor,
    alpha: torch.Tensor,
    beta_inv: torch.Tensor,
    up_filter: torch.Tensor,
    down_filter: torch.Tensor,
) -> torch.Tensor:
    """Upsample, SnakeBeta and downsample ``[B, C, T]`` in one pass.

    Args:
        x: Contiguous ``[B, C, T]`` on MPS, float16 or float32.
        alpha: ``exp(SnakeBeta.alpha)``, float32 ``[C]``.
        beta_inv: ``1 / (exp(SnakeBeta.beta) + eps)``, float32 ``[C]``.
        up_filter: The upsampler's 12 taps, float32 ``[12]``.
        down_filter: The decimator's 12 taps, float32 ``[12]``.
    """
    b, c, t = x.shape
    x = x.contiguous()
    out = torch.empty_like(x)
    # A tile wider than the row would leave most of its threads with no output to write.
    name, group, per = _NARROW if t <= _WIDE[1] * _WIDE[2] else _WIDE
    outs = group * per
    ntiles = (t + outs - 1) // outs
    dtype = "h" if x.dtype == torch.float16 else "f"
    kernel = getattr(_library(), f"act1d_{dtype}{name}")
    kernel(
        out,
        x,
        alpha,
        beta_inv,
        up_filter,
        down_filter,
        t,
        ntiles,
        c,
        threads=b * c * ntiles * group,
        group_size=group,
    )
    return out
