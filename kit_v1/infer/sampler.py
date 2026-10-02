"""Rectified Flow sampler (Euler / Heun).

Convention: z_t = (1 - t)*x + t*eps, t=1 is pure noise, dz/dt = v.
Sampling integrates t from 1 to 0 with z <- z - dt*v. The context part stays clean;
only the target part is updated.
"""
from __future__ import annotations

import contextlib
from typing import Dict, Optional

import torch

# cond fields expanded along K
_COND_KEYS = ("clock", "sess", "dow", "event", "month", "yday",
              "market", "sector", "inst", "scale", "pad_mask")
# identity ids zeroed in the unconditional CFG branch (embedding slot 0 = null)
_CFG_NULL_KEYS = ("market", "sector", "inst", "scale")
# per-bar conditions zeroed over the context in the no-history branch
_HIST_NULL_KEYS = ("clock", "sess", "dow", "event", "month", "yday")


def make_nohist_cond(
    cond: Dict[str, torch.Tensor], Lc: int
) -> Dict[str, torch.Tensor]:
    """Build the "no history" condition: the first Lc positions get zeroed inputs and pad_mask=False.

    Identity ids are kept. A missing pad_mask is treated as all valid. The caller must also zero the context of z.
    """
    out = dict(cond)
    for k in _HIST_NULL_KEYS:
        v = cond[k].clone()
        v[:, :Lc] = 0
        out[k] = v
    pm = cond.get("pad_mask")
    if pm is None:
        ref = cond["clock"]
        pm = torch.ones(ref.shape[0], ref.shape[1], dtype=torch.bool,
                        device=ref.device)
    else:
        pm = pm.clone()
    pm[:, :Lc] = False
    out["pad_mask"] = pm
    return out


def _chunked_forward(
    model: torch.nn.Module,
    z_full: torch.Tensor,
    t_vec: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    Lh: int,
    chunk_size: int,
) -> torch.Tensor:
    """Forward in chunks along the batch dim to limit memory."""
    n = z_full.shape[0]
    if n <= chunk_size:
        return model(z_full, t_vec, cond, Lh)
    outs = []
    for s in range(0, n, chunk_size):
        e = min(s + chunk_size, n)
        sub = {k: v[s:e] for k, v in cond.items()}
        outs.append(model(z_full[s:e], t_vec[s:e], sub, Lh))
    return torch.cat(outs, dim=0)


# precision -> autocast dtype
_INFER_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


def _autocast(precision: str, device: torch.device):
    """autocast for bf16/fp16 on CUDA; a no-op for fp32 or CPU."""
    if precision not in _INFER_DTYPES:
        raise ValueError(f"unknown precision: {precision!r} (choose fp32/fp16/bf16)")
    dt = _INFER_DTYPES[precision]
    if dt is None or device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dt)


@torch.no_grad()
def flow_sample(
    model: torch.nn.Module,
    x_ctx: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    Lh: int,
    steps: int = 32,
    solver: str = "euler",
    n_paths: int = 64,
    cfg_w: float = 1.0,
    hist_cfg_w: float = 1.0,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    chunk_size: int = 256,
    precision: str = "fp32",
) -> torch.Tensor:
    """Sample n_paths future paths per window, in normalized feature space.

    Args:
        model: KlineDiT, forward(z, t, cond, Lh) -> velocity [B',Lh,F].
        x_ctx: [B,Lc,F] clean history.
        cond: window conditions with L=Lc+Lh. clock [B,L,16], sess [B,L,2], yday [B,L,8] (float32);
            dow/event/month [B,L] and market/sector/inst/scale [B] (int64);
            pad_mask [B,L] bool (optional).
        steps: number of integration steps, uniform from t=1 to 0.
        solver: "euler" or "heun" (the last step is always Euler).
        cfg_w: identity guidance weight; 1.0 disables it.
        hist_cfg_w: history guidance weight; 1.0 disables it. Guided velocity is
            v_full + (cfg_w-1)*(v_full-v_noid) + (hist_cfg_w-1)*(v_full-v_nohist).
        generator: controls the initial noise (one torch.randn call of shape (B*K, Lh, F)).
        device: defaults to the model's device.
        chunk_size: max sequences per forward pass.
        precision: "fp32", "bf16" or "fp16". Only the model forward runs in low precision.

    Returns:
        paths [B,K,Lh,F].
    """
    if solver not in ("euler", "heun"):
        raise ValueError(f"solver must be euler/heun, got {solver!r}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    B, Lc, F = x_ctx.shape
    K = int(n_paths)
    BK = B * K
    L = Lc + Lh
    if cond["clock"].shape[1] != L:
        raise ValueError(
            f"cond sequence length {cond['clock'].shape[1]} does not match Lc+Lh={L}")

    # expand to a B*K batch
    x_ctx_rep = x_ctx.to(device).repeat_interleave(K, dim=0)        # [BK,Lc,F]
    cond_rep: Dict[str, torch.Tensor] = {}
    for key in _COND_KEYS:
        val = cond.get(key)
        if val is None:
            if key == "pad_mask":
                continue  # optional
            raise KeyError(f"cond is missing required field {key!r}")
        cond_rep[key] = val.to(device).repeat_interleave(K, dim=0)

    use_id_cfg = float(cfg_w) != 1.0
    use_hist_cfg = float(hist_cfg_w) != 1.0
    if use_hist_cfg and "pad_mask" not in cond_rep:
        # the no-history branch needs a pad_mask
        cond_rep["pad_mask"] = torch.ones(BK, L, dtype=torch.bool, device=device)
    # branches: [full] (+ noid) (+ nohist), run in one batched forward
    branch_conds = [cond_rep]
    if use_id_cfg:
        # identity ids zeroed, calendar kept
        branch_conds.append({
            k: (torch.zeros_like(v) if k in _CFG_NULL_KEYS else v)
            for k, v in cond_rep.items()
        })
    if use_hist_cfg:
        # context cleared, identity kept
        branch_conds.append(make_nohist_cond(cond_rep, Lc))
        x_ctx_zero = torch.zeros_like(x_ctx_rep)
    n_br = len(branch_conds)
    if n_br > 1:
        cond_multi = {k: torch.cat([bc[k] for bc in branch_conds], dim=0)
                      for k in branch_conds[0]}

    def velocity(z_tgt: torch.Tensor, t_val: float) -> torch.Tensor:
        """Velocity of the target part; returned in fp32."""
        z_full = torch.cat([x_ctx_rep, z_tgt], dim=1)               # [BK,L,F]
        t_vec = torch.full((BK,), float(t_val),
                           device=device, dtype=x_ctx_rep.dtype)
        with _autocast(precision, device):
            if n_br == 1:
                return _chunked_forward(model, z_full, t_vec, cond_rep, Lh,
                                        chunk_size).float()
            zs = [z_full] * (2 if use_id_cfg else 1)   # full and noid share z
            if use_hist_cfg:
                zs.append(torch.cat([x_ctx_zero, z_tgt], dim=1))  # zeroed ctx
            v_all = _chunked_forward(
                model, torch.cat(zs, dim=0), t_vec.repeat(n_br),
                cond_multi, Lh, chunk_size)
            parts = v_all.chunk(n_br, dim=0)
            v_full = parts[0]
            v = v_full
            i = 1
            if use_id_cfg:
                v = v + (cfg_w - 1.0) * (v_full - parts[i])
                i += 1
            if use_hist_cfg:
                v = v + (hist_cfg_w - 1.0) * (v_full - parts[i])
        return v.float()

    # initial noise
    gen_device = generator.device if generator is not None else torch.device("cpu")
    z_tgt = torch.randn(BK, Lh, F, generator=generator,
                        device=gen_device, dtype=x_ctx_rep.dtype).to(device)

    # t grid from 1 to 0
    grid = torch.linspace(1.0, 0.0, steps + 1)
    for i in range(steps):
        t_cur = float(grid[i])
        t_next = float(grid[i + 1])
        dt = t_cur - t_next                                          # > 0
        v1 = velocity(z_tgt, t_cur)
        if solver == "euler" or i == steps - 1:
            z_tgt = z_tgt - dt * v1
        else:
            # Heun
            z_e = z_tgt - dt * v1
            v2 = velocity(z_e, t_next)
            z_tgt = z_tgt - dt * 0.5 * (v1 + v2)

    return z_tgt.view(B, K, Lh, F)
