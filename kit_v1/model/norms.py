"""RMSNorm."""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """RMSNorm computed in fp32, cast back to the input dtype.

    affine=False where AdaLN supplies scale/shift; affine=True for QK-Norm.
    """

    def __init__(self, dim: int, affine: bool = True, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        if self.affine:
            out = out * self.weight.float()
        return out.to(dtype)
