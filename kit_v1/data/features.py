"""Candle features and their inverse (float64, numpy).

    r_gap  = ln(O_t / C_{t-1})             open gap (0 for the first bar)
    r_body = ln(C_t / O_t)                 body
    r_up   = ln(H_t / max(O_t, C_t)) >= 0  upper shadow
    r_dn   = ln(min(O_t, C_t) / L_t) >= 0  lower shadow
    v_t    = ln((V_t + 1) / (EMA_t + 1))   log volume relative to its EMA

EMA_t uses only V_0..V_{t-1} (EMA_0 = V_0, so v_0 = 0):
    EMA_{t+1} = (1 - alpha) * EMA_t + alpha * V_t,  alpha = 1 - 2^(-1/half_life_bars)
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

# Feature order; also the feature part of the stats.json keys.
FEATURE_NAMES: Tuple[str, ...] = ("r_gap", "r_body", "r_up", "r_dn", "v")


def ema_alpha(half_life_bars: float) -> float:
    """EMA smoothing factor from a half-life in bars."""
    if half_life_bars <= 0:
        raise ValueError(f"half_life_bars 必须为正数，收到 {half_life_bars}")
    return float(1.0 - 2.0 ** (-1.0 / half_life_bars))


def compute_volume_ema(volume: np.ndarray, half_life_bars: float) -> np.ndarray:
    """Volume EMA where EMA[t] depends only on V[0..t-1]; EMA[0] = V[0]."""
    v = np.asarray(volume, dtype=np.float64)
    if v.ndim != 1:
        raise ValueError("volume 必须是一维数组")
    n = v.shape[0]
    alpha = ema_alpha(half_life_bars)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    import pandas as pd
    y = pd.Series(v).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    ema = np.empty(n, dtype=np.float64)
    ema[0] = v[0]
    ema[1:] = y[:-1]
    return ema


def compute_features(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    half_life_bars: float,
) -> Dict[str, np.ndarray]:
    """OHLCV -> dict of the 5 features (float64 arrays).

    Args:
        open_/high/low/close/volume: 1-D arrays of equal length (adjusted prices).
        half_life_bars: volume EMA half-life in bars.
    """
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    vol = np.asarray(volume, dtype=np.float64)
    n = o.shape[0]
    if not (h.shape[0] == l.shape[0] == c.shape[0] == vol.shape[0] == n):
        raise ValueError("OHLCV 各数组长度必须一致")

    r_gap = np.zeros(n, dtype=np.float64)
    if n > 1:
        r_gap[1:] = np.log(o[1:] / c[:-1])
    r_body = np.log(c / o)
    body_hi = np.maximum(o, c)
    body_lo = np.minimum(o, c)
    r_up = np.log(h / body_hi)   # H >= max(O,C) => r_up >= 0
    r_dn = np.log(body_lo / l)   # L <= min(O,C) => r_dn >= 0

    ema = compute_volume_ema(vol, half_life_bars)
    v_feat = np.log((vol + 1.0) / (ema + 1.0))

    return {"r_gap": r_gap, "r_body": r_body, "r_up": r_up, "r_dn": r_dn, "v": v_feat}


def invert_features(
    r_gap: np.ndarray,
    r_body: np.ndarray,
    r_up: np.ndarray,
    r_dn: np.ndarray,
    v: np.ndarray,
    prev_close: float,
    prev_ema: float,
    half_life_bars: float,
) -> Dict[str, np.ndarray]:
    """Rebuild OHLCV from features by chaining from the previous bar.

    Args:
        prev_close: close before the first bar. For a round trip with
            compute_features, pass the first open.
        prev_ema: volume EMA at the first bar. For a round trip, pass the first volume.
        half_life_bars: EMA half-life, same as used for encoding.

    Volume is clamped to >= 0.
    """
    r_gap = np.asarray(r_gap, dtype=np.float64)
    r_body = np.asarray(r_body, dtype=np.float64)
    r_up = np.asarray(r_up, dtype=np.float64)
    r_dn = np.asarray(r_dn, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    n = r_gap.shape[0]

    alpha = ema_alpha(half_life_bars)
    o = np.empty(n, dtype=np.float64)
    h = np.empty(n, dtype=np.float64)
    l = np.empty(n, dtype=np.float64)
    c = np.empty(n, dtype=np.float64)
    vol = np.empty(n, dtype=np.float64)

    c_prev = float(prev_close)
    ema = float(prev_ema)
    for t in range(n):
        o[t] = c_prev * np.exp(r_gap[t])
        c[t] = o[t] * np.exp(r_body[t])
        h[t] = max(o[t], c[t]) * np.exp(r_up[t])
        l[t] = min(o[t], c[t]) * np.exp(-r_dn[t])
        vol[t] = max((ema + 1.0) * np.exp(v[t]) - 1.0, 0.0)
        ema = (1.0 - alpha) * ema + alpha * vol[t]
        c_prev = c[t]

    return {"open": o, "high": h, "low": l, "close": c, "volume": vol}
