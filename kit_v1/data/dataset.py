"""Map-style training/validation dataset of (history + horizon) K-line windows.

A window is anchored at ``anchor``: the bar index of the first *target* bar.
The model reads the ``Lc`` bars before the anchor (padding at the front when
fewer than ``Lc`` are available) and predicts the ``Lh`` bars from the anchor
on.  This mirrors exactly what :func:`kit_v1.infer.window.prepare_inference_window`
builds, so a training sample and an inference sample are the same object.

The dataset is intentionally torch-free: it returns numpy arrays and the collate
function in :mod:`kit_v1.data.loader` turns them into tensors.  Signals are
produced with the same feature/normalisation/calendar code as inference.

Data layout (see ``data/README.md``)::

    <data_root>/<market>/<timescale>/<code>.parquet
    <data_root>/instruments.json     (code -> {name, market, sector, vocab_index, ...})
    <data_root>/stats.json           (<market>|<timescale>|<feature> -> scale)
    <data_root>/trade_dates.csv      (optional A-share style trading calendar)
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from kit_v1.config import Config
from kit_v1.data.calendar import (
    MARKET_IDS,
    TIMESCALE_IDS,
    attach_window_calendar,
    bars_per_day,
    load_trading_days,
)
from kit_v1.data.features import FEATURE_NAMES, compute_features
from kit_v1.data.normalize import load_stats, normalize, stats_key


def _end_of_day(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """A bare date becomes the last nanosecond of that day (inclusive cutoff)."""
    t = pd.Timestamp(ts)
    if t == t.normalize():
        return t + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return t


@dataclass(frozen=True)
class WindowSample:
    """One training window: instrument + scale + anchor index in the feature array."""

    market: str
    scale: str
    code: str
    anchor: int          # index of the first target bar
    Lc: int
    Lh: int


class KlineWindowDataset:
    """Builds the list of valid windows for a split and returns them on demand."""

    def __init__(self, cfg: Config, split: str = "train",
                 data_root: Optional[Union[str, Path]] = None) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        self.cfg = cfg
        self.d = cfg.data
        self.split = split
        self.data_root = Path(data_root if data_root is not None
                              else self.d.data_root)
        if not self.data_root.is_dir():
            raise FileNotFoundError(
                f"data_root does not exist: {self.data_root} "
                "(download the demo data or run scripts/prepare_data.py)")

        inst_path = self.data_root / "instruments.json"
        if not inst_path.is_file():
            raise FileNotFoundError(f"missing {inst_path}")
        self.instruments: Dict[str, dict] = json.loads(
            inst_path.read_text(encoding="utf-8"))

        stats_path = Path(self.d.stats_path) if self.d.stats_path \
            else self.data_root / "stats.json"
        self.stats = load_stats(stats_path)
        self.trading_days = load_trading_days(self.data_root / "trade_dates.csv")

        self._dt_cache: "OrderedDict[Tuple[str, str, str], np.ndarray]" = OrderedDict()
        self._feat_cache: "OrderedDict[Tuple[str, str, str], np.ndarray]" = OrderedDict()
        self._cache_cap = int(self.d.cache_max_inst) if int(self.d.cache_max_inst) > 0 else None

        self.markets: List[str] = []
        self.samples: List[WindowSample] = []
        self._build_index()

    # ------------------------------------------------------------------ index
    def _selected_scales(self) -> Optional[List[str]]:
        if self.split == "train":
            return self.d.timescales_list()
        return self.d.val_timescales_list()

    def _discover(self) -> List[Tuple[str, str]]:
        """(market, scale) pairs that have a directory with parquet files."""
        selected = self._selected_scales()
        out: List[Tuple[str, str]] = []
        for market_dir in sorted(p for p in self.data_root.iterdir() if p.is_dir()):
            market = market_dir.name
            if market not in MARKET_IDS:
                continue
            for ts_dir in sorted(p for p in market_dir.iterdir() if p.is_dir()):
                scale = ts_dir.name
                if scale not in TIMESCALE_IDS:
                    continue
                if selected is not None and scale not in selected:
                    continue
                if any(ts_dir.glob("*.parquet")):
                    out.append((market, scale))
        return out

    def _read_datetimes(self, path: Path) -> np.ndarray:
        key = (path.parent.parent.name, path.parent.name, path.stem)
        cached = self._dt_cache.get(key)
        if cached is not None:
            self._dt_cache.move_to_end(key)
            return cached
        try:
            s = pd.read_parquet(path, columns=["datetime"])["datetime"]
        except Exception:
            from kit_v1.data.io import canon_ohlcv_columns
            s = canon_ohlcv_columns(pd.read_parquet(path))["datetime"]
        arr = pd.DatetimeIndex(s).astype("datetime64[ns]").astype(np.int64)
        self._dt_cache[key] = arr
        self._trim(self._dt_cache)
        return arr

    def _trim(self, cache: "OrderedDict") -> None:
        if self._cache_cap is None:
            return
        while len(cache) > self._cache_cap:
            cache.popitem(last=False)

    def _anchor_range(self, dt: np.ndarray, Lc: int, Lh: int, trunc_min: int
                      ) -> Tuple[int, int]:
        """Inclusive [lo, hi] range of valid anchor indices for the split."""
        n = len(dt)
        lo = max(1, min(trunc_min, Lc))
        if self.split == "train":
            nt = int(np.searchsorted(dt, _end_of_day(self.d.train_end).value,
                                     side="right"))
            hi = nt - Lh
        elif self.split == "val":
            hi = int(np.searchsorted(dt, _end_of_day(self.d.val_end).value,
                                     side="right")) - Lh
            lo = max(lo, int(np.searchsorted(dt, pd.Timestamp(self.d.val_start).value,
                                             side="left")))
        else:  # test
            lo = max(lo, int(np.searchsorted(dt, pd.Timestamp(self.d.test_start).value,
                                             side="left")))
            hi = n - Lh
        hi = min(hi, n - Lh)
        if hi < lo:
            return 1, 0  # empty
        return lo, hi

    def _build_index(self) -> None:
        d = self.d
        pairs = self._discover()
        self.markets = sorted({m for m, _ in pairs})
        for market, scale in pairs:
            Lc, Lh = d.window_for(scale)
            trunc_min = d.trunc_min_for(scale)
            stride = d.stride_for(scale)
            jitter = float(d.stride_jitter) if stride > 1 else 0.0
            scale_dir = self.data_root / market / scale
            for path in sorted(scale_dir.glob("*.parquet")):
                code = path.stem
                dt = self._read_datetimes(path)
                if len(dt) < Lh + 1:
                    continue
                lo, hi = self._anchor_range(dt, Lc, Lh, trunc_min)
                if hi < lo:
                    continue
                if stride > 1:
                    anchors = list(range(lo, hi + 1, stride))
                    if jitter > 0 and anchors:
                        rng = np.random.default_rng(abs(hash((market, scale, code))) % (2**32))
                        maxj = int(round(stride * jitter))
                        if maxj > 0:
                            anchors = [int(np.clip(a + int(rng.integers(-maxj, maxj + 1)),
                                                   lo, hi)) for a in anchors]
                else:
                    anchors = list(range(lo, hi + 1))
                cap = int(d.per_inst_cap)
                if cap > 0 and len(anchors) > cap:
                    idx = np.linspace(0, len(anchors) - 1, cap).round().astype(int)
                    anchors = [anchors[i] for i in sorted(set(idx.tolist()))]
                for a in anchors:
                    self.samples.append(WindowSample(market, scale, code, int(a), Lc, Lh))
        if not self.samples:
            raise RuntimeError(
                f"no {self.split} windows built from {self.data_root} "
                f"(check data_root, timescales and train/val date ranges)")

    # --------------------------------------------------------------- features
    def _features(self, market: str, scale: str, code: str, path: Path) -> np.ndarray:
        key = (market, scale, code)
        cached = self._feat_cache.get(key)
        if cached is not None:
            self._feat_cache.move_to_end(key)
            return cached
        from kit_v1.data.io import read_ohlcv
        df = read_ohlcv(path)
        hl_bars = float(self.d.vol_ema_half_life_days) * bars_per_day(market, scale)
        feats = compute_features(df["open"].to_numpy(), df["high"].to_numpy(),
                                 df["low"].to_numpy(), df["close"].to_numpy(),
                                 df["volume"].to_numpy(), half_life_bars=hl_bars)
        cols = [normalize(feats[name], float(self.stats[stats_key(market, scale, name)]),
                          c=self.d.soft_clip_c) for name in FEATURE_NAMES]
        arr = np.stack(cols, axis=1).astype(np.float32)
        self._feat_cache[key] = arr
        self._trim(self._feat_cache)
        return arr

    # ------------------------------------------------------------------ item
    def __len__(self) -> int:
        return len(self.samples)

    def sample_at(self, i: int) -> WindowSample:
        return self.samples[i]

    def _hist_len(self, s: WindowSample, rng: np.random.Generator) -> int:
        full = min(s.Lc, s.anchor)                       # bars actually available
        if full <= 0:
            return 0
        if self.split == "train" and self.d.trunc_prob > 0.0 \
                and rng.random() < float(self.d.trunc_prob):
            lo = min(s.Lc, max(1, self.d.trunc_min_for(s.scale)), full)
            if lo < full:
                return int(rng.integers(lo, full + 1))   # inclusive of full
        return full

    def __getitem__(self, i: int) -> Dict[str, object]:
        s = self.samples[i]
        scale_dir = self.data_root / s.market / s.scale
        feats = self._features(s.market, s.scale, s.code, scale_dir / f"{s.code}.parquet")
        rng = np.random.default_rng()
        hist_len = self._hist_len(s, rng)
        hist_len = max(1, min(hist_len, s.Lc, s.anchor))
        pad = s.Lc - hist_len
        ctx_start = s.anchor - hist_len

        seg = feats[ctx_start:s.anchor + s.Lh]           # [hist_len + Lh, 5]
        x_ctx = np.zeros((s.Lc, len(FEATURE_NAMES)), dtype=np.float32)
        x_ctx[pad:] = seg[:hist_len]
        x_tgt = np.ascontiguousarray(seg[hist_len:s.Lh + hist_len], dtype=np.float32)

        dt_all = self._read_datetimes(scale_dir / f"{s.code}.parquet")
        ts_seg = pd.DatetimeIndex(dt_all[ctx_start:s.anchor + s.Lh])
        prev_ts = pd.to_datetime(int(dt_all[ctx_start - 1]), unit="ns") \
            if ctx_start > 0 else None
        cal = attach_window_calendar(ts_seg, s.market, s.scale, n_ctx=hist_len,
                                     prev_ts=prev_ts, trading_days=self.trading_days)

        L = s.Lc + s.Lh
        clock = np.zeros((L, cal["clock"].shape[1]), dtype=np.float32)
        sess = np.zeros((L, 2), dtype=np.float32)
        yday = np.zeros((L, cal["yday"].shape[1]), dtype=np.float32)
        dow = np.zeros(L, dtype=np.int64)
        month = np.zeros(L, dtype=np.int64)
        event = np.zeros(L, dtype=np.int64)
        clock[pad:] = cal["clock"]
        sess[pad:] = cal["sess"]
        yday[pad:] = cal["yday"]
        dow[pad:] = cal["dow"]
        month[pad:] = cal["month"]
        event[pad:] = cal["event"]
        pad_mask = np.ones(L, dtype=bool)
        pad_mask[:pad] = False

        inst = self.instruments.get(s.code, {})
        cond = {
            "clock": clock, "sess": sess, "yday": yday,
            "dow": dow, "event": event, "month": month, "pad_mask": pad_mask,
            "market": np.int64(MARKET_IDS[s.market]),
            "sector": np.int64(int(inst.get("sector", 0))),
            "inst": np.int64(int(inst.get("vocab_index", 0))),
            "scale": np.int64(TIMESCALE_IDS[s.scale]),
        }
        return {"x_ctx": x_ctx, "x_tgt": x_tgt, "cond": cond, "code": s.code,
                "market": s.market, "scale": s.scale}


__all__ = ["KlineWindowDataset", "WindowSample"]
