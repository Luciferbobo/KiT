"""Build model inputs from raw K-line history (parquet or DataFrame)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
import torch

from kit_v1.data.calendar import (MARKET_IDS, TIMESCALE_IDS, TimestampLike,
                                     TradingDays, attach_window_calendar,
                                     bars_per_day, gen_future_timestamps)
from kit_v1.data.features import (FEATURE_NAMES, compute_features,
                                     compute_volume_ema, ema_alpha)
from kit_v1.data.normalize import (DEFAULT_SOFT_CLIP_C, normalize,
                                      stats_key)


def prepare_inference_window(
    source: Union[str, Path, pd.DataFrame],
    market: str,
    timescale: str,
    Lc: int,
    Lh: int,
    stats: Mapping[str, float],
    soft_clip_c: float = DEFAULT_SOFT_CLIP_C,
    half_life_days: float = 5.0,
    end_ts: Optional[TimestampLike] = None,
    sector_id: int = 0,
    inst_id: int = 0,
    trading_days: TradingDays = None,
) -> Dict[str, Any]:
    """Build inputs for a single window (batch size 1).

    Args:
        source: parquet path or DataFrame with columns [datetime, open, high, low, close, volume].
        end_ts: last timestamp to use (inclusive); defaults to the last bar.
        sector_id / inst_id: embedding ids; 0 means unknown.
        trading_days: optional trading-day table; without it only weekends are skipped.

    If fewer than Lc bars are available, the front is padded (zeros, pad_mask=False).

    Returns a dict:
        x_ctx           float32 [1,Lc,5] normalized features
        cond            condition dict for the model, L=Lc+Lh
        prev_close      last context close
        prev_ema        volume EMA state at the first future bar
        half_life_bars  volume EMA half-life in bars
        ctx_timestamps  timestamps of the valid context bars
        future_timestamps  timestamps of the Lh future bars
        n_pad           number of padded bars (Lc - valid context length)
    """
    if isinstance(source, (str, Path)):
        df = pd.read_parquet(source)
    else:
        df = source
    ts_all = pd.DatetimeIndex(df["datetime"])
    if end_ts is not None:
        keep = ts_all <= pd.Timestamp(end_ts)
        df = df.loc[keep]
        ts_all = ts_all[keep]
    n_total = len(df)
    if n_total < 2:
        raise ValueError(f"history up to end_ts has only {n_total} bars, cannot build an inference window")

    o = df["open"].to_numpy(dtype=np.float64)
    h = df["high"].to_numpy(dtype=np.float64)
    l = df["low"].to_numpy(dtype=np.float64)
    c = df["close"].to_numpy(dtype=np.float64)
    v = df["volume"].to_numpy(dtype=np.float64)

    # features use all available history so the volume EMA is warmed up
    half_life_bars = float(half_life_days) * bars_per_day(market, timescale)
    feats = compute_features(o, h, l, c, v, half_life_bars=half_life_bars)
    cols = []
    for name in FEATURE_NAMES:
        scale = float(stats[stats_key(market, timescale, name)])
        cols.append(normalize(feats[name], scale, c=soft_clip_c))
    x_all = np.stack(cols, axis=1).astype(np.float32)                # [N,5]

    # last n_eff bars form the context
    n_eff = min(int(Lc), n_total)
    n_pad = int(Lc) - n_eff
    ctx_ts = ts_all[n_total - n_eff:]
    x_ctx = np.zeros((int(Lc), len(FEATURE_NAMES)), dtype=np.float32)
    x_ctx[n_pad:] = x_all[n_total - n_eff:]

    # future timestamps follow the exchange calendar
    fut_ts = gen_future_timestamps(ctx_ts[-1], int(Lh), market, timescale,
                                   trading_days=trading_days)
    window_ts = ctx_ts.append(fut_ts)                                # n_eff + Lh
    # bar just before the context, if any
    prev_ts = ts_all[n_total - n_eff - 1] if n_total > n_eff else None
    cal = attach_window_calendar(window_ts, market, timescale,
                                 n_ctx=n_eff, prev_ts=prev_ts,
                                 trading_days=trading_days)

    # padded positions keep zero calendar features
    L = int(Lc) + int(Lh)
    clock = np.zeros((L, cal["clock"].shape[1]), dtype=np.float32)
    sess = np.zeros((L, 2), dtype=np.float32)
    dow = np.zeros(L, dtype=np.int64)
    month = np.zeros(L, dtype=np.int64)
    yday = np.zeros((L, cal["yday"].shape[1]), dtype=np.float32)
    event = np.zeros(L, dtype=np.int64)
    pad_mask = np.ones(L, dtype=bool)
    clock[n_pad:] = cal["clock"]
    sess[n_pad:] = cal["sess"]
    dow[n_pad:] = cal["dow"]
    month[n_pad:] = cal["month"]
    yday[n_pad:] = cal["yday"]
    event[n_pad:] = cal["event"]
    pad_mask[:n_pad] = False

    # EMA at the first future bar: fold in the last historical volume
    ema = compute_volume_ema(v, half_life_bars)
    alpha = ema_alpha(half_life_bars)
    prev_ema = float((1.0 - alpha) * ema[-1] + alpha * v[-1])

    cond: Dict[str, torch.Tensor] = {
        "clock": torch.from_numpy(clock).unsqueeze(0),               # f32 [1,L,16]
        "sess": torch.from_numpy(sess).unsqueeze(0),                 # f32 [1,L,2]
        "dow": torch.from_numpy(dow).unsqueeze(0),                   # i64 [1,L]
        "event": torch.from_numpy(event).unsqueeze(0),               # i64 [1,L]
        "month": torch.from_numpy(month).unsqueeze(0),               # i64 [1,L]
        "yday": torch.from_numpy(yday).unsqueeze(0),                 # f32 [1,L,8]
        "market": torch.tensor([MARKET_IDS[market]], dtype=torch.int64),
        "sector": torch.tensor([int(sector_id)], dtype=torch.int64),
        "inst": torch.tensor([int(inst_id)], dtype=torch.int64),
        "scale": torch.tensor([TIMESCALE_IDS[timescale]], dtype=torch.int64),
        "pad_mask": torch.from_numpy(pad_mask).unsqueeze(0),         # bool [1,L]
    }
    return {
        "x_ctx": torch.from_numpy(x_ctx).unsqueeze(0),               # f32 [1,Lc,5]
        "cond": cond,
        "prev_close": float(c[-1]),
        "prev_ema": prev_ema,
        "half_life_bars": half_life_bars,
        "ctx_timestamps": ctx_ts,
        "future_timestamps": fut_ts,
        "n_pad": n_pad,
    }
