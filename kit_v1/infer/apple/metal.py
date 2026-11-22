"""Metal operations for the FP32 KiT inference plan."""

from functools import cache
from operator import index
from pathlib import Path

import torch


@cache
def _compiled_ops(width):
    return MetalOps(width)


def metal_ops(width=1024):
    return _compiled_ops(index(width))


@cache
def kernels_available(width=1024):
    """Check the compiled pipelines before selecting the optimized sampler."""
    try:
        if not torch.backends.mps.is_available():
            return False
        metal_ops(width)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _supports_reduction(kernel):
    return kernel.thread_execution_width == 32 and kernel.max_threads_per_threadgroup >= 128


class MetalOps:
    def __init__(self, width=1024):
        self.width = index(width)
        if self.width <= 0 or self.width % 32:
            raise ValueError("The fused Metal RMSNorm needs a positive width divisible by 32")
        try:
            source = Path(__file__).with_name("kernels.metal").read_text()
            library = torch.mps.compile_shader(f"#define KIT_MODEL_WIDTH {self.width}u\n" + source)
            self._qk = library.kit_qknorm_rope128
            self._gate_norm = library.gated_norm
            self._silu = library.packed_f32
            if not (_supports_reduction(self._qk) and _supports_reduction(self._gate_norm)):
                raise RuntimeError("KiT's Metal reductions require 32-lane SIMD groups")
            if self._silu.max_threads_per_threadgroup < 256:
                raise RuntimeError("KiT's Metal SwiGLU requires a 256-thread group")
        except (AttributeError, OSError) as exc:
            raise RuntimeError("KiT's FP32 Metal kernels are unavailable") from exc

        # Only this kernel is optional. Full QKV projection has the same model semantics.
        self.tail_kernel = None
        try:
            kernel = library.norm_rope_one
            if _supports_reduction(kernel):
                self.tail_kernel = kernel
        except (AttributeError, OSError, RuntimeError):
            pass

    def silu(self, packed):
        width = packed.shape[-1] // 2
        out = packed.new_empty((*packed.shape[:-1], width))
        self._silu(packed, out, width, threads=out.numel(), group_size=256)
        return out

    def gate_norm(self, x, delta, gate, norm, prefix, gate_offset, norm_offset):
        if x.shape[-1] != self.width:
            raise ValueError("Residual width differs from the compiled Metal RMSNorm")
        residual = torch.empty_like(x)
        hidden = torch.empty_like(x)
        self._gate_norm(
            x,
            delta,
            *gate,
            *norm,
            residual,
            hidden,
            x.shape[1],
            prefix,
            gate_offset,
            norm_offset,
            threads=x.shape[0] * x.shape[1] * 32,
            group_size=128,
        )
        return residual, hidden

    def qk(self, qkv, attention):
        batch, sequence, _ = qkv.shape
        heads = attention.n_heads
        rope = attention.rope
        rope._ensure(sequence, qkv.device)
        q = qkv.new_empty((batch, heads, sequence, 128))
        k = torch.empty_like(q)
        self._qk(
            qkv,
            attention.q_norm.weight,
            attention.k_norm.weight,
            rope.cos_cached,
            rope.sin_cached,
            q,
            k,
            sequence,
            heads,
            threads=batch * sequence * heads * 32,
            group_size=128,
        )
        return q, k

    def _rotate_one(self, projection, weight, rope, heads, projections, offset):
        batch, sequence, _ = projection.shape
        rope._ensure(sequence + offset, projection.device)
        out = projection.new_empty((batch, heads, sequence, 128))
        self.tail_kernel(
            projection,
            weight,
            rope.cos_cached,
            rope.sin_cached,
            out,
            sequence,
            heads,
            projections,
            offset,
            threads=batch * sequence * heads * 32,
            group_size=128,
        )
        return out

    def target_qkv(self, x, q_weight, kv_weight, attention, prefix):
        batch, seq, dim = x.shape
        heads = attention.n_heads
        q = (x[:, prefix:].reshape(-1, dim) @ q_weight).view(batch, seq - prefix, dim)
        kv = (x.reshape(-1, dim) @ kv_weight).view(batch, seq, 2 * dim)
        q = self._rotate_one(q, attention.q_norm.weight, attention.rope, heads, 1, prefix)
        k = self._rotate_one(kv, attention.k_norm.weight, attention.rope, heads, 2, 0)
        v = kv.view(batch, seq, 2, heads, dim // heads)[:, :, 1].transpose(1, 2)
        return q, k, v
