"""Exponential moving average of model parameters (fp32 shadow copy).

The inference loader (:func:`kit_v1.infer.loader.load_model_and_config`) reads
``ckpt["ema"]["shadow"]`` as a flat ``{param_name: tensor}`` mapping, which is
exactly what :meth:`EMA.state_dict` returns.
"""
from __future__ import annotations

from typing import Dict

import torch


class EMA:
    """Decay ``shadow = decay * shadow + (1 - decay) * param`` over float params."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {}
        for name, p in model.state_dict().items():
            if torch.is_floating_point(p):
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        src = model.state_dict()
        for name, p in src.items():
            if name not in self.shadow:
                continue
            s = self.shadow[name]
            s.mul_(self.decay).add_(p.detach().float(), alpha=1.0 - self.decay)

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, shadow: Dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().float().clone() for k, v in shadow.items()}


__all__ = ["EMA"]
