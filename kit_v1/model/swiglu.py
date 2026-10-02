"""SwiGLU feed-forward."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """w2(silu(w1 x) * w3 x), all Linear layers without bias."""

    def __init__(self, d: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)   # gate
        self.w3 = nn.Linear(d, hidden, bias=False)   # value
        self.w2 = nn.Linear(hidden, d, bias=False)   # output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
