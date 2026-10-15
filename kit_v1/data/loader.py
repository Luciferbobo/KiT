"""DataLoader construction and the numpy -> torch collate function."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader

from kit_v1.config import Config
from kit_v1.data.dataset import KlineWindowDataset
from kit_v1.data.sampler import ScaleBatchSampler

_FLOAT_KEYS = ("clock", "sess", "yday")
_INT_KEYS = ("dow", "event", "month", "market", "sector", "inst", "scale")


def collate_windows(batch) -> Dict[str, object]:
    """Stack a list of dataset items into a batch of tensors.

    Returns ``{"x_ctx": [B,Lc,5], "x_tgt": [B,Lh,5], "cond": {...}, "code": [...]}``
    with ``cond`` holding ``clock [B,L,16]``, ``sess [B,L,2]``, ``yday [B,L,8]``
    (float32), ``dow/event/month`` and ``market/sector/inst/scale`` (int64) and
    ``pad_mask [B,L]`` (bool).
    """
    x_ctx = torch.from_numpy(np.stack([b["x_ctx"] for b in batch])).float()
    x_tgt = torch.from_numpy(np.stack([b["x_tgt"] for b in batch])).float()
    cond: Dict[str, torch.Tensor] = {}
    for k in _FLOAT_KEYS:
        cond[k] = torch.from_numpy(np.stack([b["cond"][k] for b in batch])).float()
    for k in _INT_KEYS:
        cond[k] = torch.stack(
            [torch.as_tensor(b["cond"][k]) for b in batch]).long()
    cond["pad_mask"] = torch.from_numpy(
        np.stack([b["cond"]["pad_mask"] for b in batch])).bool()
    return {"x_ctx": x_ctx, "x_tgt": x_tgt, "cond": cond,
            "code": [b["code"] for b in batch],
            "market": [b["market"] for b in batch],
            "scale": [b["scale"] for b in batch]}


def build_dataloader(
    cfg: Config,
    split: str = "train",
    data_root: Optional[Union[str, "os.PathLike"]] = None,
    rank: int = 0,
    world_size: int = 1,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
) -> Tuple[KlineWindowDataset, DataLoader]:
    """Build the dataset and a scale-grouped ``DataLoader`` for a split.

    ``pin_memory`` defaults to False; the training loop enables it only when the
    model is actually on CUDA (pinning starts a CUDA thread even for CPU runs,
    which can hang process teardown on some Windows/WDDM systems).
    """
    ds = KlineWindowDataset(cfg, split=split, data_root=data_root)
    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 1)
    pm = bool(pin_memory) if pin_memory is not None else False

    if split == "train":
        sampler = ScaleBatchSampler(
            ds, batch_size=int(cfg.train.micro_batch),
            temperature_tau=float(cfg.data.temperature_tau),
            market_weights=cfg.data.market_weights_map(),
            shuffle=True, seed=cfg.train.seed, rank=rank, world_size=world_size,
            drop_last=True)
    else:
        sampler = ScaleBatchSampler(
            ds, batch_size=int(cfg.train.micro_batch), temperature_tau=1.0,
            shuffle=False, rank=rank, world_size=world_size, drop_last=False)

    loader = DataLoader(
        ds, batch_sampler=sampler, collate_fn=collate_windows,
        num_workers=int(num_workers), pin_memory=pm,
        persistent_workers=int(num_workers) > 0)
    # expose the sampler so the loop can call set_epoch()
    loader._kit_sampler = sampler  # type: ignore[attr-defined]
    return ds, loader


def set_loader_epoch(loader: DataLoader, epoch: int) -> None:
    """Advance the sampler epoch (shuffling) for the next pass over the data."""
    sampler = getattr(loader, "_kit_sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


__all__ = ["collate_windows", "build_dataloader", "set_loader_epoch"]
