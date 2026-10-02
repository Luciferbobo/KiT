"""Per-bucket robust normalization with tanh soft clipping.

- Buckets are (market, timescale, feature): x' = x / (1.4826 * MAD), fitted only
  on data up to train_end (inclusive). The mean is not subtracted.
- Soft clip: x'' = c * tanh(x' / c), c = 6.0.
- stats.json keys: "<market>|<timescale>|<feature>" -> scale.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd

from kit_v1.data.calendar import bars_per_day
from kit_v1.data.features import FEATURE_NAMES, compute_features

DEFAULT_SOFT_CLIP_C: float = 6.0
_MAD_CONST: float = 1.4826


def stats_key(market: str, timescale: str, feature: str) -> str:
    """Build a stats.json key."""
    return f"{market}|{timescale}|{feature}"


def fit_mad_scale(x: np.ndarray) -> float:
    """Robust scale 1.4826 * MAD; falls back to max(std, 1e-8) if it is below 1e-8."""
    arr = np.asarray(x, dtype=np.float64)
    med = float(np.median(arr))
    scale = _MAD_CONST * float(np.median(np.abs(arr - med)))
    if scale < 1e-8:
        scale = max(float(np.std(arr)), 1e-8)
    return float(scale)


def _end_of_day(ts: Union[str, pd.Timestamp]) -> pd.Timestamp:
    """If ts is a bare date, extend it to the last nanosecond of that day."""
    t = pd.Timestamp(ts)
    if t == t.normalize():
        t = t + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return t


def fit_stats(
    data_root: Union[str, Path],
    train_end: Union[str, pd.Timestamp],
    half_life_days: float = 5.0,
) -> Dict[str, float]:
    """Fit MAD scales per (market, timescale, feature) over a data tree.

    Layout: <data_root>/<market>/<timescale>/<instrument_id>.parquet with columns
    datetime (tz-naive), open, high, low, close, volume. Only rows with
    datetime <= train_end (whole day) are used.

    Args:
        half_life_days: volume EMA half-life in trading days; converted to bars
            with bars_per_day(market, timescale).
    """
    root = Path(data_root)
    cutoff = _end_of_day(train_end)
    buckets: Dict[str, list] = {}
    for market_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        market = market_dir.name
        for ts_dir in sorted(p for p in market_dir.iterdir() if p.is_dir()):
            timescale = ts_dir.name
            hl_bars = half_life_days * bars_per_day(market, timescale)
            for pq in sorted(ts_dir.glob("*.parquet")):
                try:
                    # push the datetime filter down to pyarrow to cut IO
                    df = pd.read_parquet(
                        pq, columns=["datetime", "open", "high", "low",
                                     "close", "volume"],
                        filters=[("datetime", "<=", cutoff)])
                except Exception:
                    # fallback: read everything
                    df = pd.read_parquet(pq)
                df = df[pd.DatetimeIndex(df["datetime"]) <= cutoff]
                if len(df) < 2:
                    continue
                feats = compute_features(
                    df["open"].to_numpy(),
                    df["high"].to_numpy(),
                    df["low"].to_numpy(),
                    df["close"].to_numpy(),
                    df["volume"].to_numpy(),
                    half_life_bars=hl_bars,
                )
                for name in FEATURE_NAMES:
                    buckets.setdefault(stats_key(market, timescale, name), []).append(
                        feats[name]
                    )
    stats: Dict[str, float] = {}
    for key, chunks in buckets.items():
        stats[key] = fit_mad_scale(np.concatenate(chunks))
    return stats


def save_stats(stats: Dict[str, float], path: Union[str, Path]) -> None:
    """Write stats to a JSON file."""
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, sort_keys=True, ensure_ascii=False)


def load_stats(path: Union[str, Path]) -> Dict[str, float]:
    """Read stats from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return {str(k): float(v) for k, v in json.load(f).items()}


def normalize(
    x: np.ndarray,
    scale: float,
    c: float = DEFAULT_SOFT_CLIP_C,
) -> np.ndarray:
    """y = c * tanh((x / scale) / c), clipped to the open interval (-c, c)."""
    xp = np.asarray(x, dtype=np.float64) / float(scale)
    y = c * np.tanh(xp / c)
    bound = c * (1.0 - 1e-9)
    return np.clip(y, -bound, bound)


def denormalize(
    y: np.ndarray,
    scale: float,
    c: float = DEFAULT_SOFT_CLIP_C,
) -> np.ndarray:
    """Inverse of normalize; y/c is clamped to +/-(1-1e-6) to keep atanh finite."""
    r = np.asarray(y, dtype=np.float64) / c
    r = np.clip(r, -(1.0 - 1e-6), 1.0 - 1e-6)
    return np.arctanh(r) * c * float(scale)
