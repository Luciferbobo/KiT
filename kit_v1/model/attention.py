"""Bidirectional multi-head attention with QK-Norm and RoPE.

Backends (ModelConfig.attn_backend):
- "flash2": flash_attn varlen, needs CUDA sm80+ and flash-attn installed
- "sdpa": F.scaled_dot_product_attention
- "math": plain softmax attention
- "auto": flash2 if available, else sdpa
"""
from __future__ import annotations

import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kit_v1.model.norms import RMSNorm
from kit_v1.model.rope import RotaryEmbedding

# warn only once about the flash2 -> sdpa dtype fallback
_FLASH2_DTYPE_WARNED = False


def _flash_attn_available() -> bool:
    """Whether flash_attn can be imported."""
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(requested: str) -> str:
    """Resolve the backend name. "auto" picks flash2 when usable, else sdpa."""
    if requested == "auto":
        if (_flash_attn_available() and torch.cuda.is_available()
                and torch.cuda.get_device_capability() >= (8, 0)):
            return "flash2"
        return "sdpa"
    if requested == "flash2":
        if not _flash_attn_available():
            raise RuntimeError(
                "attn_backend='flash2' but flash_attn is not installed; "
                "install flash-attn>=2 or use attn_backend='auto'/'sdpa'.")
        if not torch.cuda.is_available():
            raise RuntimeError(
                "attn_backend='flash2' requires a CUDA GPU (cuda is not available); "
                "use attn_backend='auto'/'sdpa'/'math' instead.")
        if torch.cuda.get_device_capability() < (8, 0):
            raise RuntimeError(
                "attn_backend='flash2' requires sm80+ (A100/H100); "
                f"current capability={torch.cuda.get_device_capability()}; "
                "on V100 etc. use attn_backend='auto'/'sdpa'.")
        return "flash2"
    if requested in ("sdpa", "math"):
        return requested
    raise ValueError(f"unknown attn_backend: {requested!r} (choose auto/flash2/sdpa/math)")


def _dense_attn_mask(pad_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """pad_mask [B,S] bool (True=valid) -> attn_mask [B,1,S,S] bool (True=attend).

    The diagonal is always allowed so padded query rows don't become all -inf (NaN).
    """
    mask = pad_mask[:, None, None, :]                          # [B,1,1,S]
    eye = torch.eye(seq_len, dtype=torch.bool, device=pad_mask.device)
    return mask | eye[None, None, :, :]                        # [B,1,S,S]


class MultiheadAttention(nn.Module):
    """qkv projection -> QK-Norm -> RoPE -> attention -> output projection."""

    def __init__(self, d: int, n_heads: int, qk_norm: bool = True,
                 rope_theta: float = 10000.0, backend: str = "auto") -> None:
        super().__init__()
        if d % n_heads != 0:
            raise ValueError(f"d={d} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        # QK-Norm: RMSNorm over head_dim, weights shared across heads
        self.q_norm: Optional[RMSNorm] = RMSNorm(self.head_dim, affine=True) if qk_norm else None
        self.k_norm: Optional[RMSNorm] = RMSNorm(self.head_dim, affine=True) if qk_norm else None
        self.rope = RotaryEmbedding(self.head_dim, theta=rope_theta)
        self.backend = resolve_backend(backend)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B,S,d]; pad_mask [B,S] bool (True=valid, None=all valid); positions [S] long."""
        B, S, d = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)                            # [B,S,H,Dh]
        q = q.transpose(1, 2)                                  # [B,H,S,Dh]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if positions is None:
            positions = torch.arange(S, device=x.device)
        q, k = self.rope.apply_rope(q, k, positions)

        if self.backend == "flash2":
            out = self._attn_flash2(q, k, v, pad_mask)
        elif self.backend == "sdpa":
            out = self._attn_sdpa(q, k, v, pad_mask)
        else:
            out = self._attn_math(q, k, v, pad_mask)

        out = out.transpose(1, 2).reshape(B, S, d)             # merge heads
        return self.out_proj(out)

    def _attn_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        attn_mask = None
        if pad_mask is not None:
            attn_mask = _dense_attn_mask(pad_mask, q.shape[2])
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    def _attn_math(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Plain softmax(QK^T / sqrt(d)) @ V."""
        scale = self.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B,H,S,S]
        if pad_mask is not None:
            attn_mask = _dense_attn_mask(pad_mask, q.shape[2])
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        return torch.matmul(attn, v)

    def _attn_flash2(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     pad_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """flash_attn varlen path: unpad valid tokens, attend, repad."""
        # flash-attn only supports fp16/bf16; fall back to sdpa otherwise
        if q.dtype not in (torch.float16, torch.bfloat16):
            global _FLASH2_DTYPE_WARNED
            if not _FLASH2_DTYPE_WARNED:
                _FLASH2_DTYPE_WARNED = True
                warnings.warn(
                    "flash2 only supports fp16/bf16, current dtype="
                    f"{q.dtype}; falling back to sdpa this time (similar warnings will not repeat).",
                    RuntimeWarning, stacklevel=3)
            return self._attn_sdpa(q, k, v, pad_mask)
        try:
            from flash_attn import flash_attn_varlen_func
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "flash2 backend requires flash-attn>=2, import failed: "
                f"{e!r}; use the sdpa/math backend instead.") from e

        B, H, S, Dh = q.shape
        if pad_mask is None:
            pad_mask = torch.ones(B, S, dtype=torch.bool, device=q.device)
        # [B,H,S,Dh] -> unpadded [total, H, Dh]
        q_ = q.transpose(1, 2).reshape(B * S, H, Dh)
        k_ = k.transpose(1, 2).reshape(B * S, H, Dh)
        v_ = v.transpose(1, 2).reshape(B * S, H, Dh)
        flat_mask = pad_mask.reshape(B * S)
        indices = flat_mask.nonzero(as_tuple=False).squeeze(-1)  # valid token indices
        q_u, k_u, v_u = q_[indices], k_[indices], v_[indices]
        seqlens = pad_mask.sum(dim=1, dtype=torch.int32)          # [B]
        cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
        max_seqlen = int(seqlens.max().item())
        out_u = flash_attn_varlen_func(
            q_u, k_u, v_u,
            cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen, max_seqlen_k=max_seqlen,
            causal=False,  # bidirectional
        )
        # repad with zeros
        out = torch.zeros(B * S, H, Dh, dtype=out_u.dtype, device=out_u.device)
        out[indices] = out_u
        return out.view(B, S, H, Dh).transpose(1, 2)              # [B,H,S,Dh]
