"""Checkpoint save / load.

The on-disk layout is exactly what :func:`kit_v1.infer.loader.load_model_and_config`
expects::

    {
      "model":    {name: tensor},          # raw KlineDiT state_dict
      "ema":      {"shadow": {name: tensor}},  # fp32 EMA parameters
      "config":   {...},                   # Config.to_dict()
      "step":     int,
      "optimizer": {...},                  # only for resume
    }
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch

from kit_v1.config import Config


def save_checkpoint(path: Union[str, Path], model: torch.nn.Module, ema,
                    cfg: Config, step: int,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> None:
    """Atomically write a checkpoint (temp file + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "ema": {"shadow": {k: v.detach().cpu() for k, v in ema.state_dict().items()}},
        "config": cfg.to_dict(),
        "step": int(step),
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    tmp = path.with_name(path.name + ".tmp")
    torch.save(ckpt, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Union[str, Path], model: torch.nn.Module,
                    ema=None, optimizer: Optional[torch.optim.Optimizer] = None,
                    map_location: str = "cpu", strict: bool = True) -> int:
    """Load model/ema/optimizer weights; returns the saved step (0 if absent)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    if ema is not None and "ema" in ckpt:
        shadow = ckpt["ema"].get("shadow", ckpt["ema"]) if isinstance(ckpt["ema"], dict) \
            else ckpt["ema"]
        ema.load_state_dict(shadow)
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return int(ckpt.get("step", 0))


__all__ = ["save_checkpoint", "load_checkpoint"]
