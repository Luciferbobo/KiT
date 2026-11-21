"""Flow sampler for the cached-context KiT inference prototype."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from kit_v1.model.cached_dit import CachedKlineDiT


_COND_KEYS = (
    "clock",
    "sess",
    "dow",
    "event",
    "month",
    "yday",
    "market",
    "sector",
    "inst",
    "scale",
    "pad_mask",
)


@torch.no_grad()
def flow_sample_cached(
    model,
    x_ctx: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    Lh: int,
    steps: int = 32,
    n_paths: int = 32,
    generator: Optional[torch.Generator] = None,
    device: Optional[torch.device] = None,
    precision: str = "bf16",
    progress: Optional[Callable[[int, int], None]] = None,
    chunk_size: int = 0,
) -> torch.Tensor:
    """Sample paths while encoding the fixed context only once."""
    batch = x_ctx.shape[0]
    if chunk_size > 0 and batch > chunk_size:
        outputs = []
        for start in range(0, batch, chunk_size):
            end = min(batch, start + chunk_size)
            outputs.append(
                flow_sample_cached(
                    model,
                    x_ctx[start:end],
                    {key: value[start:end] for key, value in cond.items()},
                    Lh,
                    steps=steps,
                    n_paths=n_paths,
                    generator=generator,
                    device=device,
                    precision=precision,
                    progress=None,
                    chunk_size=0,
                )
            )
        return torch.cat(outputs, dim=0)

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    model.eval()
    cached = CachedKlineDiT(model)
    batch, context_len, _ = x_ctx.shape
    path_count = int(n_paths)
    batch_paths = batch * path_count

    autocast_dtype = {
        "fp32": None,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[precision]

    with torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
        enabled=autocast_dtype is not None and device.type == "cuda",
    ):
        cond = {key: value.to(device) for key, value in cond.items()}
        context_cache, context_valid = cached.prepare_context(
            x_ctx.to(device), cond, path_repeat=path_count
        )
        target_cond = {
            key: value.to(device).repeat_interleave(path_count, dim=0)
            for key, value in cond.items()
            if key in _COND_KEYS
        }
        noise = torch.randn(
            batch_paths,
            Lh,
            model.cfg.in_features,
            generator=generator,
            device=generator.device if generator is not None else torch.device("cpu"),
            dtype=x_ctx.dtype,
        ).to(device)

        grid = torch.linspace(1.0, 0.0, steps + 1)
        for index in range(steps):
            t_current = float(grid[index])
            t_next = float(grid[index + 1])
            delta = t_current - t_next
            t_vec = torch.full(
                (batch_paths,),
                t_current,
                device=device,
                dtype=x_ctx.dtype,
            )
            velocity = cached.forward_target(
                noise,
                t_vec,
                target_cond,
                context_cache,
                context_valid,
                context_len,
            )
            noise = noise - delta * velocity.float()
            if progress is not None:
                progress(index + 1, steps)

    return noise.view(batch, path_count, Lh, model.cfg.in_features)
