"""Aggregate OHLCV bars from a finer timescale to a coarser one (session aware).

The aggregation follows the same bar convention as :mod:`kit_v1.data.calendar`:
a bar's timestamp is its *close* time and, within a trading day, a coarse bar
aggregates every fine bar whose close minute falls in ``(prev_close, close]``
where the closes come from the market's session grid.  This guarantees that the
generated series line up exactly with ``bars_per_session`` / ``gen_calendar``
used by inference, so training and inference see identical grids.

``aggregate_bars`` is a pure function on a DataFrame; ``aggregate_tree`` walks a
directory of per-instrument parquet files and writes the multi-timescale tree
consumed by the training/eval pipeline.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from kit_v1.data.calendar import (
    SESSION_TEMPLATES,
    TIMESCALE_MINUTES,
    _daily_grid_minutes,
    _owning_days,
)
from kit_v1.data.io import list_parquet, read_ohlcv

_DEFAULT_AGG_TIMESCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")


def aggregate_bars(df: pd.DataFrame, market: str, timescale: str) -> pd.DataFrame:
    """Aggregate a finer OHLCV frame to ``timescale`` for ``market``.

    Args:
        df: frame with canonical columns (see :mod:`kit_v1.data.io`), sorted or
            not; the row order is ignored and bars are placed on the session grid.
        market: market key, e.g. ``"ashare"``.
        timescale: target timescale key, e.g. ``"5m"``.

    Returns a frame with ``datetime, open, high, low, close, volume``.
    """
    if timescale not in TIMESCALE_MINUTES:
        raise ValueError(f"unknown timescale {timescale!r}")
    if market not in SESSION_TEMPLATES:
        raise ValueError(f"unknown market {market!r}")

    idx = pd.DatetimeIndex(df["datetime"])
    if len(idx) == 0:
        return df.iloc[0:0].copy()

    grid = np.asarray(sorted(_daily_grid_minutes(market, timescale)), dtype=np.float64)
    own = _owning_days(idx, market)
    # force nanoseconds (a DatetimeIndex may be stored at us/ms resolution)
    day_ns = own.normalize().astype("datetime64[ns]").astype(np.int64)
    minutes = (idx.hour * 60 + idx.minute).to_numpy().astype(np.float64)
    if SESSION_TEMPLATES[market][-1][1] == 1440:  # 24/7: a 00:00 close is 1440
        minutes[minutes == 0.0] = 1440.0

    # smallest grid close that is >= this bar's close minute
    bucket = np.searchsorted(grid, minutes, side="left").astype(np.int64)
    valid = bucket < len(grid)

    sub = pd.DataFrame({
        "_day": day_ns,
        "_bucket": bucket,
        "open": df["open"].to_numpy(dtype=np.float64),
        "high": df["high"].to_numpy(dtype=np.float64),
        "low": df["low"].to_numpy(dtype=np.float64),
        "close": df["close"].to_numpy(dtype=np.float64),
        "volume": df["volume"].to_numpy(dtype=np.float64),
    })
    sub = sub[np.asarray(valid, dtype=bool)]
    if sub.empty:
        return df.iloc[0:0].copy()

    g = sub.groupby(["_day", "_bucket"], sort=True)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
        "volume": g["volume"].sum(),
    })
    day_part = out.index.get_level_values("_day").to_numpy().astype(np.int64)
    bucket_part = out.index.get_level_values("_bucket").to_numpy().astype(np.int64)
    # day_part holds epoch nanoseconds; be explicit about the unit (pandas >= 2
    # defaults bare integers to seconds, which would overflow here).
    out.insert(0, "datetime",
               pd.to_datetime(day_part, unit="ns")
               + pd.to_timedelta(grid[bucket_part] * 60.0, unit="s"))
    return out.reset_index(drop=True)


def _find_source_files(raw_root: Path, market: str, source_timescale: str):
    """Locate ``<code>.parquet`` source files in the common raw layouts."""
    for cand in (raw_root,
                 raw_root / market / source_timescale,
                 raw_root / source_timescale,
                 raw_root / market):
        files = list_parquet(cand)
        if files:
            return files
    return []


def aggregate_tree(
    raw_root: Union[str, Path],
    out_root: Union[str, Path],
    market: str,
    target_timescales: Sequence[str] = _DEFAULT_AGG_TIMESCALES,
    source_timescale: str = "1m",
    workers: int = 8,
    overwrite: bool = False,
) -> List[str]:
    """Aggregate a raw fine-scale tree into ``<out_root>/<market>/<scale>/<code>.parquet``.

    Returns the sorted list of instrument codes that were written.
    """
    raw_root, out_root = Path(raw_root), Path(out_root)
    if source_timescale not in TIMESCALE_MINUTES:
        raise ValueError(f"unknown source_timescale {source_timescale!r}")
    src_minutes = TIMESCALE_MINUTES[source_timescale]
    target_timescales = tuple(target_timescales)
    bad = [t for t in target_timescales if t not in TIMESCALE_MINUTES]
    if bad:
        raise ValueError(f"unknown target timescales: {bad}")
    for t in target_timescales:
        if TIMESCALE_MINUTES[t] < src_minutes:
            raise ValueError(
                f"target {t} is finer than the source {source_timescale}; "
                "aggregation can only coarsen")

    files = _find_source_files(raw_root, market, source_timescale)
    if not files:
        raise FileNotFoundError(
            f"no {source_timescale} parquet files found under {raw_root}")

    def _one(path: Path) -> str:
        code = path.stem
        df = read_ohlcv(path)
        for ts in target_timescales:
            dst = out_root / market / ts / f"{code}.parquet"
            if dst.is_file() and not overwrite:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            agg = df if ts == source_timescale else aggregate_bars(df, market, ts)
            tmp = dst.with_suffix(".parquet.tmp")
            agg.to_parquet(tmp, index=False)
            tmp.replace(dst)
        return code

    codes = sorted({p.stem for p in files})
    workers = max(1, int(workers))
    if workers == 1:
        for p in files:
            _one(p)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(_one, files))
    return codes


__all__ = ["aggregate_bars", "aggregate_tree"]
