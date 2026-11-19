"""Optional FP32 inference operators for KiT-L on Apple Silicon."""

import ctypes
import hashlib
import math
import os
import platform
import subprocess
import tempfile
from pathlib import Path

import torch

_SOURCE = Path(__file__).with_name("cpu_kernels.cpp")
_FLAGS = (
    "-std=c++17",
    "-O3",
    "-fblocks",
    "-ffp-contract=off",
    "-arch",
    "arm64",
    "-dynamiclib",
)


def _bind(path):
    lib = ctypes.CDLL(str(path))
    ptr, size = ctypes.c_void_p, ctypes.c_size_t
    for name, types in (
        ("kit_qk_rope", [ptr] * 7 + [size] * 3 + [ctypes.c_float]),
        ("kit_gate_norm", [ptr] * 8 + [size] * 5),
        ("kit_swiglu", [ptr] * 2 + [size] * 2),
    ):
        fn = getattr(lib, name)
        fn.argtypes = types
        fn.restype = None
    return lib


def _load_library(directory):
    import fcntl

    digest = hashlib.sha256(_SOURCE.read_bytes())
    digest.update(repr((_FLAGS, platform.mac_ver()[0], platform.machine())).encode())
    key = digest.hexdigest()[:24]
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    st = directory.stat()
    if st.st_uid != os.getuid() or st.st_mode & 0o022:
        raise OSError("Native library cache must be owned and writable only by the current user")
    dest = directory / f"{key}.dylib"
    with (directory / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.is_file():
            try:
                return _bind(dest)
            except (OSError, AttributeError):
                pass  # A damaged cached binary can be rebuilt from the packaged source.
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            output = Path(temp) / "kernels.dylib"
            subprocess.run(
                [
                    "/usr/bin/clang++",
                    *_FLAGS,
                    str(_SOURCE),
                    "-framework",
                    "Accelerate",
                    "-o",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=45,
            )
            lib = _bind(output)
            os.replace(output, dest)
            return lib


def _pointers(*tensors):
    if not all(
        t.device.type == "cpu"
        and t.layout == torch.strided
        and t.dtype == torch.float32
        and t.is_contiguous()
        and not t.is_neg()
        and t.numel() > 0
        and not (torch.is_grad_enabled() and t.requires_grad)
        for t in tensors
    ):
        raise ValueError("Native inference requires nonempty contiguous CPU float32 tensors")
    return [t.data_ptr() for t in tensors]


class _CPUOps:
    def __init__(self, lib):
        self._lib = lib

    def qk(self, qkv, attention):
        batch, sequence, width = qkv.shape
        heads, dim = attention.n_heads, attention.head_dim
        qnorm, knorm = attention.q_norm, attention.k_norm
        if (
            dim != 128
            or heads <= 0
            or width != 3 * heads * dim
            or qnorm is None
            or knorm is None
            or qnorm.weight is None
            or knorm.weight is None
            or qnorm.weight.shape != (dim,)
            or knorm.weight.shape != (dim,)
            or qnorm.eps != knorm.eps
            or not math.isfinite(qnorm.eps)
            or qnorm.eps <= 0
        ):
            raise ValueError(
                "Native Q/K requires 128-dimensional heads and matching affine RMSNorm"
            )
        attention.rope._ensure(sequence, qkv.device)
        cos, sin = attention.rope.cos_cached, attention.rope.sin_cached
        if cos.shape != sin.shape or cos.ndim != 2 or cos.size(0) < sequence or cos.size(1) != dim:
            raise ValueError("RoPE tables have an incompatible shape")
        pointers = _pointers(qkv, qnorm.weight, knorm.weight, cos, sin)
        q = qkv.new_empty((batch, sequence, heads, dim))
        k = torch.empty_like(q)
        self._lib.kit_qk_rope(
            *pointers, q.data_ptr(), k.data_ptr(), batch, sequence, heads, qnorm.eps
        )
        return q.transpose(1, 2), k.transpose(1, 2)

    def gate_norm(self, x, delta, gate, norm, prefix, gate_offset, norm_offset):
        batch, sequence, width = x.shape
        gc, gt = gate
        nc, nt = norm
        if (
            width != 1024
            or delta.shape != x.shape
            or not 0 <= prefix <= sequence
            or gate_offset not in (2, 5)
            or norm_offset not in (0, 3)
            or any(v.shape != (6 * width,) for v in (gc, gt, nc, nt))
        ):
            raise ValueError("Unexpected KiT-L residual or modulation shape or offset")
        pointers = _pointers(x, delta, gc, gt, nc, nt)
        residual, hidden = torch.empty_like(x), torch.empty_like(x)
        self._lib.kit_gate_norm(
            *pointers,
            residual.data_ptr(),
            hidden.data_ptr(),
            batch * sequence,
            sequence,
            prefix,
            gate_offset,
            norm_offset,
        )
        return residual, hidden

    def silu(self, packed):
        batch, sequence, size = packed.shape
        if size % 8:
            raise ValueError("Packed SwiGLU requires a half-width divisible by four")
        pointers = _pointers(packed)
        width = size // 2
        out = packed.new_empty((batch, sequence, width))
        self._lib.kit_swiglu(*pointers, out.data_ptr(), batch * sequence, width)
        return out


def load_cpu_ops():
    """Return the optional native backend, or None when compilation or loading is unavailable."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    try:
        directory = Path.home() / "Library" / "Caches" / "KiT" / "cpu-kernels"
        return _CPUOps(_load_library(directory))
    except (OSError, AttributeError, RuntimeError, subprocess.SubprocessError):
        return None
