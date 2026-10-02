"""1D rotary position embedding (LLaMA-style rotate_half, fp32 cos/sin cache)."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """Applies RoPE to q/k. positions are token indices in the assembled sequence."""

    def __init__(self, head_dim: int, theta: float = 10000.0,
                 max_positions: int = 4096) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even, got {head_dim}")
        self.head_dim = head_dim
        self.theta = theta
        cos, sin = self._build(max_positions)
        # non-persistent: not saved in state_dict
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (self.theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim))
        pos = torch.arange(n, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)                     # [n, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)                # [n, d]
        return emb.cos(), emb.sin()

    def _ensure(self, max_pos: int, device: torch.device) -> None:
        if max_pos > self.cos_cached.shape[0]:
            cos, sin = self._build(max_pos)
            self.cos_cached = cos.to(device)
            self.sin_cached = sin.to(device)

    def apply_rope(self, q: torch.Tensor, k: torch.Tensor,
                   positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """q/k [B,H,S,Dh]; positions [S] long."""
        self._ensure(int(positions.max().item()) + 1 if positions.numel() else 0,
                     q.device)
        cos = self.cos_cached.to(q.device)[positions]          # [S, Dh] fp32
        sin = self.sin_cached.to(q.device)[positions]
        cos = cos[None, None, :, :]                            # [1,1,S,Dh]
        sin = sin[None, None, :, :]
        qf, kf = q.float(), k.float()
        q_out = qf * cos + _rotate_half(qf) * sin
        k_out = kf * cos + _rotate_half(kf) * sin
        return q_out.to(q.dtype), k_out.to(k.dtype)

    forward = apply_rope
