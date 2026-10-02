"""Post-processing: denormalize, rebuild OHLCV, limit-band projection, path summaries (numpy float64)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Union

import numpy as np
import torch

from kit_v1.data.features import FEATURE_NAMES, invert_features
from kit_v1.data.normalize import DEFAULT_SOFT_CLIP_C, denormalize, stats_key

# quantile bands
QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)
_Q_KEYS = ("q05", "q25", "q50", "q75", "q95")

ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Tensor / array -> float64 ndarray."""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)


def denormalize_paths(
    paths: ArrayLike,
    market: str,
    timescale: str,
    stats: Mapping[str, float],
    c: float = DEFAULT_SOFT_CLIP_C,
) -> np.ndarray:
    """Denormalize per feature using stats. paths: [..., 5] in FEATURE_NAMES order; same shape returned."""
    arr = _to_numpy(paths)
    if arr.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"paths last dim must be {len(FEATURE_NAMES)}, got {arr.shape}")
    out = np.empty_like(arr)
    for j, name in enumerate(FEATURE_NAMES):
        scale = float(stats[stats_key(market, timescale, name)])
        out[..., j] = denormalize(arr[..., j], scale, c=c)
    return out


def paths_to_ohlcv(
    feats: ArrayLike,
    prev_close: float,
    prev_ema: float,
    half_life_bars: float,
) -> Dict[str, np.ndarray]:
    """Convert denormalized feature paths [K,Lh,5] to OHLCV paths.

    Each path is rebuilt independently from prev_close / prev_ema, with r_up and r_dn clamped to >= 0.
    Returns {"open","high","low","close","volume"} -> [K,Lh].
    """
    arr = _to_numpy(feats)
    if arr.ndim != 3 or arr.shape[-1] != len(FEATURE_NAMES):
        raise ValueError(f"feats must be [K,Lh,5], got {arr.shape}")
    K, Lh, _ = arr.shape
    out = {k: np.empty((K, Lh), dtype=np.float64)
           for k in ("open", "high", "low", "close", "volume")}
    for k in range(K):
        p = arr[k]
        ohlcv = invert_features(
            r_gap=p[:, 0],
            r_body=p[:, 1],
            r_up=np.maximum(p[:, 2], 0.0),   # keeps H >= max(O,C)
            r_dn=np.maximum(p[:, 3], 0.0),   # keeps L <= min(O,C)
            v=p[:, 4],
            prev_close=float(prev_close),
            prev_ema=float(prev_ema),
            half_life_bars=float(half_life_bars),
        )
        for name in out:
            out[name][k] = ohlcv[name]
    return out


def apply_limit_projection(
    ohlcv: Mapping[str, np.ndarray],
    prev_close: float,
    limit_rate: Optional[float],
) -> Dict[str, np.ndarray]:
    """Clamp O/H/L/C bar by bar to [pc*(1-rate), pc*(1+rate)], where pc is the previous projected close.

    H/L are then made consistent with O/C. Returns copies unchanged if limit_rate is None.
    """
    o = _to_numpy(ohlcv["open"]).copy()
    h = _to_numpy(ohlcv["high"]).copy()
    l = _to_numpy(ohlcv["low"]).copy()
    c = _to_numpy(ohlcv["close"]).copy()
    v = _to_numpy(ohlcv["volume"]).copy()
    if limit_rate is None:
        return {"open": o, "high": h, "low": l, "close": c, "volume": v}
    rate = float(limit_rate)
    if o.ndim == 1:  # single path [Lh] is allowed
        o, h, l, c = o[None], h[None], l[None], c[None]
        v = v[None]
        squeeze = True
    else:
        squeeze = False
    K, Lh = o.shape
    pc = np.full(K, float(prev_close), dtype=np.float64)
    for t in range(Lh):
        lo, hi = pc * (1.0 - rate), pc * (1.0 + rate)
        o[:, t] = np.clip(o[:, t], lo, hi)
        c[:, t] = np.clip(c[:, t], lo, hi)
        h[:, t] = np.clip(h[:, t], lo, hi)
        l[:, t] = np.clip(l[:, t], lo, hi)
        h[:, t] = np.maximum(h[:, t], np.maximum(o[:, t], c[:, t]))
        l[:, t] = np.minimum(l[:, t], np.minimum(o[:, t], c[:, t]))
        pc = c[:, t]
    if squeeze:
        o, h, l, c, v = o[0], h[0], l[0], c[0], v[0]
    return {"open": o, "high": h, "low": l, "close": c, "volume": v}


def _band(x: np.ndarray, axis: int = 0) -> Dict[str, np.ndarray]:
    """Quantile bands along the sample axis."""
    qs = np.quantile(x, QUANTILES, axis=axis)
    return {k: qs[i] for i, k in enumerate(_Q_KEYS)}


def _dist(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """Samples plus their quantiles."""
    d: Dict[str, np.ndarray] = {"samples": samples}
    d.update(_band(samples, axis=0))
    return d


def summarize_paths(
    paths: Union[ArrayLike, Mapping[str, np.ndarray]],
    prev_close: Optional[float] = None,
) -> Dict[str, object]:
    """Summarize a set of paths.

    paths is either denormalized features [K,Lh,5] or an OHLCV dict of [K,Lh] arrays.
    For an OHLCV dict, the first return uses prev_close, or ln(C_0/O_0) if it is None.

    Returns:
        close_return_bands  per-bar quantile bands of close log return, each [Lh]
        channel_bands       {channel: per-bar quantile bands}
        terminal_return     cumulative log return over Lh (samples + quantiles)
        max_drawdown        per-path max drawdown (samples + quantiles)
        realized_vol        per-path std of per-bar log returns (samples + quantiles)
    """
    if isinstance(paths, Mapping):
        close = _to_numpy(paths["close"])                            # [K,Lh]
        r = np.empty_like(close)
        if prev_close is not None:
            r[:, 0] = np.log(close[:, 0] / float(prev_close))
        else:
            r[:, 0] = np.log(close[:, 0] / _to_numpy(paths["open"])[:, 0])
        if close.shape[1] > 1:
            r[:, 1:] = np.diff(np.log(close), axis=1)
        channel_bands = {name: _band(_to_numpy(paths[name]), axis=0)
                         for name in ("open", "high", "low", "close", "volume")}
    else:
        arr = _to_numpy(paths)                                       # [K,Lh,5]
        if arr.ndim != 3 or arr.shape[-1] != len(FEATURE_NAMES):
            raise ValueError(f"paths must be [K,Lh,5] or an ohlcv dict, got {arr.shape}")
        r = arr[:, :, 0] + arr[:, :, 1]  # close-to-close log return
        channel_bands = {name: _band(arr[:, :, j], axis=0)
                         for j, name in enumerate(FEATURE_NAMES)}

    terminal = r.sum(axis=1)                                         # [K]
    # max drawdown of the cumulative log return, starting from 0
    cum = np.cumsum(r, axis=1)                                       # [K,Lh]
    run_max = np.maximum.accumulate(np.maximum(cum, 0.0), axis=1)
    max_dd = (run_max - cum).max(axis=1)                             # [K]
    # 0 when Lh == 1
    rv = r.std(axis=1, ddof=1) if r.shape[1] > 1 else np.zeros(r.shape[0])

    return {
        "close_return_bands": _band(r, axis=0),
        "channel_bands": channel_bands,
        "terminal_return": _dist(terminal),
        "max_drawdown": _dist(max_dd),
        "realized_vol": _dist(rv),
    }
