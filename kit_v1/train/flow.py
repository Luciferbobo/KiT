"""Rectified-flow / flow-matching objective and condition dropout.

Matches the convention of :mod:`kit_v1.infer.sampler` and the paper
(§3.1, §3.5, Eq. 1, 2, 8):

    z_t = (1 - t) * x + t * eps,     t ~ sigmoid(N(0, 1)),   eps ~ N(0, I)
    L   = (1/Lh) * sum_i || v_theta(z_t, t)_i - (eps_i - x_i) ||^2

evaluated **only over the ``Lh`` target positions** (context tokens attend but
do not contribute to the loss).  The history part of ``z`` stays clean
(bit-identical), only the target part is noised.

Training also applies the paper's three stochastic condition-dropout schemes
(§3.5 / App. D), which are exactly the inference-time CFG branches:

* instrument + sector embeddings -> shared ``[UNK]`` (slot 0)   (`inst_dropout`)
* all identity signals -> ``[NULL]`` (slot 0)                  (`allcond_dropout`)
* the whole history context zeroed                            (`hist_dropout`)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

# identity (window-constant) embeddings live in cond as [B] long tensors
_IDENTITY_KEYS = ("market", "sector", "inst", "scale")
# per-bar calendar features, zeroed over the context in the history-drop branch
_CALENDAR_KEYS = ("clock", "sess", "dow", "event", "month", "yday")


def sample_logit_normal_t(bs: int, device, generator: Optional[torch.Generator] = None,
                          mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    """``t = sigmoid(N(mean, std))`` in (0, 1), shape [bs]."""
    u = torch.randn(bs, device=device, generator=generator) * std + mean
    return torch.sigmoid(u)


def apply_condition_dropout(
    x_ctx: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    Lh: int,
    inst_dropout: float = 0.0,
    allcond_dropout: float = 0.0,
    hist_dropout: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply the three CFG dropouts per sample (returns modified copies)."""
    if inst_dropout <= 0.0 and allcond_dropout <= 0.0 and hist_dropout <= 0.0:
        return x_ctx, cond
    B = x_ctx.shape[0]
    Lc = cond["clock"].shape[1] - int(Lh)
    cond = dict(cond)

    def _mask(rate: float) -> torch.Tensor:
        return torch.rand(B, device=x_ctx.device, generator=generator) < rate

    if allcond_dropout > 0.0:
        m = _mask(allcond_dropout)
        for k in _IDENTITY_KEYS:
            v = cond[k].clone()
            v[m] = 0                                  # slot 0 = [NULL]
            cond[k] = v
    if inst_dropout > 0.0:
        m = _mask(inst_dropout)
        for k in ("inst", "sector"):
            v = cond[k].clone()
            v[m] = 0                                  # slot 0 = [UNK]
            cond[k] = v
    if hist_dropout > 0.0:
        m = _mask(hist_dropout)
        x_ctx = x_ctx.clone()
        x_ctx[m] = 0
        for k in _CALENDAR_KEYS:
            v = cond[k].clone()
            v[m, :Lc] = 0
            cond[k] = v
        pm = cond["pad_mask"].clone()
        pm[m, :Lc] = False
        cond["pad_mask"] = pm
    return x_ctx, cond


def flow_matching_loss(
    model,
    x_ctx: torch.Tensor,
    x_tgt: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    Lh: int,
    inst_dropout: float = 0.0,
    allcond_dropout: float = 0.0,
    hist_dropout: float = 0.0,
    generator: Optional[torch.Generator] = None,
) -> Dict[str, torch.Tensor]:
    """Flow-matching MSE over the target bars (Eq. 8), with CFG condition dropout."""
    x_ctx, cond = apply_condition_dropout(
        x_ctx, cond, Lh, inst_dropout, allcond_dropout, hist_dropout, generator)
    B = x_tgt.shape[0]
    t = sample_logit_normal_t(B, x_tgt.device, generator).to(x_tgt.dtype)
    eps = torch.randn(x_tgt.shape, device=x_tgt.device, dtype=x_tgt.dtype,
                      generator=generator)
    t_b = t.view(B, 1, 1)
    z_tgt = (1.0 - t_b) * x_tgt + t_b * eps
    v_target = eps - x_tgt
    z = torch.cat([x_ctx, z_tgt], dim=1)
    v_pred = model(z, t, cond, Lh)
    # (1/Lh) * sum_i ||v_i - (eps_i - x_i)||^2, averaged over the batch.
    loss = ((v_pred - v_target) ** 2).sum(dim=-1).mean()
    return {"loss": loss, "v_pred": v_pred, "v_target": v_target, "t": t}


__all__ = ["sample_logit_normal_t", "apply_condition_dropout", "flow_matching_loss"]
