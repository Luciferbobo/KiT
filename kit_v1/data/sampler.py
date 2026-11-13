"""Scale-grouped batch sampler.

Batches must be homogeneous in ``(Lc, Lh)`` because every sample is padded to
its own scale window, so a batch never mixes timescales.  This sampler groups
the windows by ``(market, timescale)`` and emits batches that come from a single
group.

Sampling policy (``DataConfig.temperature_tau``):

* ``tau == 1.0`` -> "epoch traversal": every window is used once per epoch,
  batches are produced round-robin over the groups (a shuffled epoch).
* ``tau != 1.0`` -> weighted draw of groups with replacement; a group weight is
  the market weight (or ``#windows ** tau`` when ``market_weights`` is empty)
  spread over the group's windows, then indices are drawn uniformly inside the
  group.  ``tau < 1`` flattens the market distribution, ``tau > 1`` sharpens it.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Sampler

from kit_v1.data.dataset import KlineWindowDataset


class ScaleBatchSampler(Sampler):
    """Yield lists of indices that all share one ``(market, timescale)`` group."""

    def __init__(self, dataset: KlineWindowDataset, batch_size: int,
                 temperature_tau: float = 1.0,
                 market_weights: Optional[Dict[str, float]] = None,
                 shuffle: bool = True, seed: int = 0,
                 rank: int = 0, world_size: int = 1,
                 drop_last: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = max(1, int(batch_size))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = max(1, int(world_size))
        self.drop_last = bool(drop_last)
        self.epoch = 0

        groups: Dict[Tuple[str, str], List[int]] = {}
        for i, s in enumerate(dataset.samples):
            groups.setdefault((s.market, s.scale), []).append(i)
        self.groups = groups
        self.keys = sorted(groups.keys())

        n_market: Dict[str, int] = {}
        for (market, _), idx in groups.items():
            n_market[market] = n_market.get(market, 0) + len(idx)
        if market_weights:
            mw = {m: float(market_weights.get(m, 0.0)) for m in n_market}
        else:
            tau = float(temperature_tau)
            mw = {m: float(c) ** tau for m, c in n_market.items()}
        if sum(mw.values()) <= 0:
            mw = {m: 1.0 for m in n_market}
        total_mw = sum(mw.values())
        self.group_weight = {
            (market, scale): (mw[market] / total_mw) * (len(idx) / n_market[market])
            for (market, scale), idx in groups.items()
        }
        self.epoch_traversal = (float(temperature_tau) == 1.0)
        self.weight_tensor = torch.tensor([self.group_weight[k] for k in self.keys],
                                          dtype=torch.double)
        self._n_batches_global = self._count_batches()

    def _count_batches(self) -> int:
        if self.epoch_traversal:
            total = 0
            for idx in self.groups.values():
                n = len(idx)
                if self.drop_last:
                    total += n // self.batch_size
                else:
                    total += math.ceil(n / self.batch_size)
            return total
        n_samples = len(self.dataset)
        return max(1, n_samples // self.batch_size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_traversal(self, rng: np.random.Generator):
        """Yield one epoch of scale-homogeneous batches (round-robin over groups)."""
        pools = {}
        for k in self.keys:
            idx = list(self.groups[k])
            if self.shuffle:
                rng.shuffle(idx)
            pools[k] = idx
        active = sorted(self.keys, key=lambda k: -len(pools[k]))
        while active:
            for k in list(active):
                pool = pools[k]
                if len(pool) >= self.batch_size:
                    yield pool[:self.batch_size]
                    pools[k] = pool[self.batch_size:]
                elif pool:                       # remainder smaller than one batch
                    if not self.drop_last:
                        yield pool
                    pools[k] = []
                if not pools[k]:
                    active.remove(k)

    def _iter_weighted(self, rng: np.random.Generator, n_batches: int):
        """Yield ``n_batches`` batches drawn with replacement by group weight."""
        for _ in range(n_batches):
            g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
            k = self.keys[int(torch.multinomial(self.weight_tensor, 1,
                                                generator=g).item())]
            pool = self.groups[k]
            picks = rng.integers(0, len(pool), size=self.batch_size)
            yield [pool[p] for p in picks]

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        if self.epoch_traversal:
            src = self._iter_traversal(rng if self.shuffle
                                       else np.random.default_rng(0))
        else:
            src = self._iter_weighted(rng, self._count_batches())
        for j, batch in enumerate(src):
            if self.world_size == 1 or (j % self.world_size) == self.rank:
                yield batch

    def __len__(self) -> int:
        if self.world_size > 1:
            return math.ceil(self._n_batches_global / self.world_size)
        return self._n_batches_global


__all__ = ["ScaleBatchSampler"]
