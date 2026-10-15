"""Distributed setup and model wrapping (none / DDP / FSDP).

Launch with ``torchrun --nproc_per_node=N train.py ...``; the process-group
environment variables are read from the standard ``RANK`` / ``WORLD_SIZE`` /
``LOCAL_RANK`` set by torchrun.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from kit_v1.config import TrainConfig


def distributed_info() -> Tuple[int, int, int]:
    """Return ``(rank, world_size, local_rank)`` from the environment."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return (int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]),
                int(os.environ.get("LOCAL_RANK", 0)))
    return 0, 1, 0


def init_distributed(rank: int, world_size: int) -> None:
    """Initialise the default process group when running multi-process."""
    if world_size <= 1:
        return
    if dist.is_available() and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def is_main(rank: int) -> bool:
    return rank == 0


def pick_device(local_rank: int) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def wrap_model(model: torch.nn.Module, cfg: TrainConfig, local_rank: int,
               world_size: int) -> torch.nn.Module:
    """Wrap ``model`` according to ``TrainConfig.parallel``."""
    mode = (cfg.parallel or "none").lower()
    if mode == "none" or world_size <= 1:
        return model
    if mode == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        device_ids = [local_rank] if torch.cuda.is_available() else None
        return DDP(model, device_ids=device_ids, find_unused_parameters=False)
    if mode == "fsdp":
        from torch.distributed.fsdp import (FullyShardedDataParallel as FSDP,
                                             ShardingStrategy)
        strat = (ShardingStrategy.SHARD_GRAD_OP if cfg.fsdp_sharding == "grad_op"
                 else ShardingStrategy.FULL_SHARD)
        return FSDP(model, sharding_strategy=strat,
                    device_id=local_rank if torch.cuda.is_available() else None,
                    use_orig_params=True)
    raise ValueError(f"unknown train.parallel {cfg.parallel!r} (none/ddp/fsdp)")


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module of a DDP/FSDP wrapper (or the model itself)."""
    m: Optional[torch.nn.Module] = model
    seen = set()
    while m is not None and hasattr(m, "module") and id(m) not in seen:
        seen.add(id(m))
        m = m.module
    return m if m is not None else model


__all__ = ["distributed_info", "init_distributed", "is_main", "pick_device",
           "wrap_model", "unwrap"]
