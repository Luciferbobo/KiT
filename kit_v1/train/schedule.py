"""Learning-rate schedule: linear warmup, constant plateau, cosine decay to min_lr.

Optionally a two-stage schedule: once ``step >= lr_drop_step`` the whole curve is
scaled by ``lr_drop_ratio``.  All knobs live in :class:`kit_v1.config.TrainConfig`.
"""
from __future__ import annotations

import math

from kit_v1.config import TrainConfig


def lr_at(step: int, cfg: TrainConfig) -> float:
    """Learning rate at optimizer ``step`` (0-indexed)."""
    warmup = max(1, int(cfg.warmup_steps))
    total = max(1, int(cfg.total_steps))
    if step < warmup:
        base = cfg.lr * float(step + 1) / float(warmup)
    else:
        decay_start = max(warmup, int(total * (1.0 - float(cfg.decay_frac))))
        if step < decay_start or decay_start >= total:
            base = cfg.lr
        else:
            p = float(step - decay_start) / float(total - decay_start)
            p = min(1.0, max(0.0, p))
            base = cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1.0 + math.cos(math.pi * p))
    if cfg.lr_drop_step > 0 and step >= cfg.lr_drop_step:
        base *= float(cfg.lr_drop_ratio)
    return base


__all__ = ["lr_at"]
