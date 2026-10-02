"""Shared AdaLN trunk. Called twice (tgt with real t, ctx with t=0) with shared weights."""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedAdaLN(nn.Module):
    """h = silu(W1(t_emb + c_base)); returns c_mod = W2(h) and head_mod = W_head(h)."""

    def __init__(self, d: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, d)
        self.w2 = nn.Linear(d, 6 * d)        # shift1, scale1, gate1, shift2, scale2, gate2
        self.w_head = nn.Linear(d, 2 * d)    # final shift, scale
        # zero init so blocks start as identity
        nn.init.zeros_(self.w2.weight)
        nn.init.zeros_(self.w2.bias)
        nn.init.zeros_(self.w_head.weight)
        nn.init.zeros_(self.w_head.bias)

    def forward(self, t_emb_vec: torch.Tensor,
                c_base: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """t_emb_vec [B,d], c_base [B,d] -> (c_mod [B,6d], head_mod [B,2d])."""
        h = F.silu(self.w1(t_emb_vec + c_base))
        return self.w2(h), self.w_head(h)
