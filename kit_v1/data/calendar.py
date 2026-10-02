"""Trading calendar, per-bar calendar features, and event tagging.

Conventions:
- Slot 0 of every ID table is reserved for [UNK]/[NULL].
- An intraday bar's datetime is its close time and lies on the grid implied by
  SESSION_TEMPLATES; a 1d bar's datetime is the market close of that day.
- For 24/7 markets (e.g. crypto) a bar closing at 00:00 belongs to the previous
  calendar day: dow/month/yday use ts - 1ns, while clock uses the real close time.
- An optional trading_days table (ascending datetime64[D], see load_trading_days)
  makes holidays count as closed days like weekends. Without it, or beyond its
  last date, only Saturdays and Sundays are skipped.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

TimestampLike = Union[str, pd.Timestamp, np.datetime64]
TradingDays = Optional[np.ndarray]   # ascending datetime64[D]; None = skip weekends only

# ID tables (slot 0 reserved for [UNK]/[NULL])
MARKET_IDS: Dict[str, int] = {"ashare": 1, "hk": 2, "us": 3, "futures": 4, "crypto": 5}
TIMESCALE_IDS: Dict[str, int] = {
    "1m": 1, "5m": 2, "15m": 3, "30m": 4, "1h": 5, "2h": 6, "1d": 7,
}
TIMESCALE_MINUTES: Dict[str, int] = {
    "1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60, "2h": 120, "1d": 1440,
}

# Scales with exactly one bar per trading day, stamped at the market close.
DAILY_SINGLE_BAR_SCALES: Tuple[str, ...] = ("1d",)

# Trading sessions per market as (open_min, close_min), local minutes of the day.
SESSION_TEMPLATES: Dict[str, List[Tuple[int, int]]] = {
    "ashare": [(570, 690), (780, 900)],   # 9:30-11:30, 13:00-15:00
    "hk": [(570, 720), (780, 960)],       # 9:30-12:00, 13:00-16:00
    "us": [(570, 960)],                   # 9:30-16:00
    "futures": [(540, 900)],              # 9:00-15:00
    "crypto": [(0, 1440)],                # 24/7
}


def _session_close_minutes(s: int, e: int, step: int) -> List[int]:
    """Close minutes of the bars in session (s, e); the last bar may be a short stub."""
    mins: List[int] = []
    m = s + step
    while m < e:
        mins.append(m)
        m += step
    mins.append(e)
    return mins


def _market_session_grids(market: str, timescale: str) -> List[List[int]]:
    """Per-session close-minute grids (1d returns [[market close minute]])."""
    if timescale in DAILY_SINGLE_BAR_SCALES:
        return [[SESSION_TEMPLATES[market][-1][1]]]
    step = TIMESCALE_MINUTES[timescale]
    return [_session_close_minutes(s, e, step)
            for s, e in SESSION_TEMPLATES[market]]


def bars_per_session(market: str, timescale: str) -> List[int]:
    """Number of bars in each session, e.g. US 1h => [7], 2h => [4]; 1d => [1]."""
    if timescale in DAILY_SINGLE_BAR_SCALES:
        return [1]
    return [len(g) for g in _market_session_grids(market, timescale)]


def bars_per_day(market: str, timescale: str) -> int:
    """Total bars per trading day (ashare 1h => 4, US 1h => 7, any 1d => 1)."""
    return int(sum(bars_per_session(market, timescale)))


def _close_minute(market: str) -> int:
    """Market close minute of the day (1440 maps to 0)."""
    return SESSION_TEMPLATES[market][-1][1] % 1440


def _daily_grid_minutes(market: str, timescale: str) -> List[int]:
    """Close minutes of all bars in a day (1440 means next-day 00:00)."""
    out: List[int] = []
    for grid in _market_session_grids(market, timescale):
        out.extend(grid)
    return out


def _owning_days(idx: pd.DatetimeIndex, market: str) -> pd.DatetimeIndex:
    """Owning day of each bar: 24/7 markets shift 00:00 closes back by 1ns."""
    if SESSION_TEMPLATES[market][-1][1] != 1440:
        return idx
    return idx - pd.Timedelta(1, "ns")


def gen_calendar(
    timestamps: Union[Sequence[TimestampLike], pd.DatetimeIndex, np.ndarray],
    market: str,
    timescale: str,
) -> Dict[str, np.ndarray]:
    """Per-bar calendar features.

    Returns:
        clock [N,16] float32: phi = 2*pi*minute_of_day/1440, [sin(k*phi), cos(k*phi)] for k=1..8
        sess  [N,2]  float32: [bars_since_open, bars_to_close] / bars_per_session, in [0,1]
        dow   [N]    int64: day of week (0 = Monday)
        month [N]    int64: month (0 = January)
        yday  [N,8]  float32: [sin(k*phi_y), cos(k*phi_y)] for k=1..4,
              phi_y = 2*pi*(dayofyear-1)/year_length
    For 1d scales sess is fixed at 0.5/0.5 and clock uses the market close minute.
    """
    idx = pd.DatetimeIndex(timestamps)
    n = len(idx)
    own = _owning_days(idx, market)
    dow = own.dayofweek.to_numpy().astype(np.int64)
    month = (own.month.to_numpy() - 1).astype(np.int64)
    # Phase uses the real year length; sin/cos are computed on unique phases only.
    year_len = np.where(np.asarray(own.is_leap_year), 366.0, 365.0)
    phi_y = 2.0 * np.pi * (own.dayofyear.to_numpy().astype(np.float64) - 1.0) / year_len
    ky = np.arange(1, 5, dtype=np.float64)
    uq_y, inv_y = np.unique(phi_y, return_inverse=True)
    ang_y = uq_y[:, None] * ky[None, :]
    yday_u = np.empty((len(uq_y), 8), dtype=np.float32)
    yday_u[:, 0::2] = np.sin(ang_y)
    yday_u[:, 1::2] = np.cos(ang_y)
    yday = yday_u[inv_y]

    if timescale in DAILY_SINGLE_BAR_SCALES:
        minutes = np.full(n, _close_minute(market), dtype=np.float64)
        sess = np.full((n, 2), 0.5, dtype=np.float32)
    else:
        minutes = (idx.hour * 60 + idx.minute).to_numpy().astype(np.float64)
        grids = _market_session_grids(market, timescale)
        sessions = SESSION_TEMPLATES[market]
        # a 00:00 close on a 24/7 market maps back to 1440
        m = minutes.copy()
        if sessions[-1][1] == 1440:
            m[m == 0.0] = 1440.0
        sess = np.zeros((n, 2), dtype=np.float32)
        placed = np.zeros(n, dtype=bool)
        # locate each bar by exact match on the session grid
        for grid in grids:
            b = len(grid)
            garr = np.asarray(grid, dtype=np.float64)
            pos = np.searchsorted(garr, m)
            in_range = (~placed) & (pos < b)
            hit = in_range.copy()
            if in_range.any():
                hit[in_range] = garr[pos[in_range]] == m[in_range]
            if hit.any():
                j = pos[hit]
                sess[hit, 0] = j / b
                sess[hit, 1] = (b - 1 - j) / b
                placed |= hit
        if not placed.all():
            # off-grid timestamps: snap to the nearest session edge
            for i in np.nonzero(~placed)[0]:
                mi = m[i]
                dists = [min(abs(mi - s), abs(mi - e)) for s, e in sessions]
                kk = int(np.argmin(dists))
                b = len(grids[kk])
                j = 0 if abs(mi - sessions[kk][0]) <= abs(mi - sessions[kk][1]) else b - 1
                sess[i, 0] = j / b
                sess[i, 1] = (b - 1 - j) / b

    phi = 2.0 * np.pi * (minutes % 1440.0) / 1440.0
    k = np.arange(1, 9, dtype=np.float64)
    uq_c, inv_c = np.unique(phi, return_inverse=True)
    ang = uq_c[:, None] * k[None, :]
    clock_u = np.empty((len(uq_c), 16), dtype=np.float32)
    clock_u[:, 0::2] = np.sin(ang)
    clock_u[:, 1::2] = np.cos(ang)
    clock = clock_u[inv_c]

    return {"clock": clock, "sess": sess, "dow": dow, "month": month, "yday": yday}


def load_trading_days(path: Union[str, Path]) -> TradingDays:
    """Read a trading-day CSV (first column YYYYMMDD) into ascending datetime64[D].

    Returns None if the file is missing, or (with a warning) if it cannot be parsed.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        col = pd.read_csv(p, encoding="utf-8-sig").iloc[:, 0].astype(str)
        days = pd.to_datetime(col, format="%Y%m%d", errors="coerce").dropna()
        arr = np.sort(days.to_numpy().astype("datetime64[D]"))
        return arr if len(arr) else None
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"交易日表解析失败（{p}）：{e}；回退只跳周末口径",
                      RuntimeWarning, stacklevel=2)
        return None


def load_market_trading_days(data_root: Union[str, Path],
                             market: str) -> TradingDays:
    """Load the trading-day table for a market.

    Tries <data_root>/<market>/trade_dates.csv, then <data_root>/trade_dates.csv,
    else returns None.
    """
    root = Path(data_root)
    td = load_trading_days(root / str(market) / "trade_dates.csv")
    if td is None:
        td = load_trading_days(root / "trade_dates.csv")
    return td


def _is_trading_day(day: pd.Timestamp, trading_days: TradingDays) -> bool:
    """Whether day is a trading day (weekday check if no table or beyond its end)."""
    if trading_days is None:
        return day.dayofweek < 5
    d64 = np.datetime64(day.date())
    if d64 > trading_days[-1]:
        return day.dayofweek < 5
    i = int(np.searchsorted(trading_days, d64))
    return i < len(trading_days) and trading_days[i] == d64


def gen_future_timestamps(
    last_ts: TimestampLike,
    n: int,
    market: str,
    timescale: str,
    trading_days: TradingDays = None,
) -> pd.DatetimeIndex:
    """Close timestamps of the next n bars after last_ts.

    Holidays are skipped if trading_days is given; otherwise only Saturdays and
    Sundays (so 24/7 markets are treated as weekday-only here).
    """
    ts = pd.Timestamp(last_ts)
    grid = _daily_grid_minutes(market, timescale)
    out: List[pd.Timestamp] = []
    day = ts.normalize()
    while len(out) < n:
        if _is_trading_day(day, trading_days):
            for m in grid:
                cand = day + pd.Timedelta(minutes=m)
                if cand > ts:
                    out.append(cand)
                    if len(out) == n:
                        break
        day += pd.Timedelta(days=1)
    return pd.DatetimeIndex(out)


def _expected_next_ns(idx: pd.DatetimeIndex, market: str, timescale: str,
                      trading_days: TradingDays = None) -> np.ndarray:
    """Vectorized close time (epoch ns) of the bar after each timestamp.

    Matches gen_future_timestamps(ts, 1, ...)[0].
    """
    grid_sec = np.asarray(_daily_grid_minutes(market, timescale), dtype=np.int64) * 60
    day_ns = idx.normalize().asi8
    sec = (idx.asi8 - day_ns) // 1_000_000_000
    dow = idx.dayofweek.to_numpy()
    # first grid point strictly after the current time
    pos = np.searchsorted(grid_sec, sec, side="right")
    has_more = pos < len(grid_sec)
    # weekend-only fallback: Fri +3, Sat +2, Sun +1, otherwise +1
    shift = np.where(dow >= 5, 7 - dow, np.where(dow == 4, 3, 1)).astype(np.int64)
    fallback_day = day_ns + shift * 86_400_000_000_000
    if trading_days is None:
        same_day = (dow < 5) & has_more
        exp_day = np.where(same_day, day_ns, fallback_day)
    else:
        td_ns = trading_days.astype("datetime64[ns]").astype(np.int64)
        beyond = day_ns > td_ns[-1]
        li = np.searchsorted(td_ns, day_ns, side="left")
        is_td = (li < len(td_ns)) & (td_ns[np.minimum(li, len(td_ns) - 1)] == day_ns)
        same_day = np.where(beyond, dow < 5, is_td) & has_more
        # next trading day strictly after today; fall back past the table end
        ri = np.searchsorted(td_ns, day_ns, side="right")
        next_td = td_ns[np.minimum(ri, len(td_ns) - 1)]
        exhausted = ri >= len(td_ns)
        next_day = np.where(beyond | exhausted, fallback_day, next_td)
        exp_day = np.where(same_day, day_ns, next_day)
    exp_sec = np.where(same_day, grid_sec[np.minimum(pos, len(grid_sec) - 1)], grid_sec[0])
    return exp_day + exp_sec * 1_000_000_000


def detect_events(
    timestamps: Union[Sequence[TimestampLike], pd.DatetimeIndex, np.ndarray],
    market: str,
    timescale: str,
    prev_ts: Optional[TimestampLike] = None,
    gap_multiplier: float = 3.0,
    trading_days: TradingDays = None,
) -> np.ndarray:
    """Tag each bar with an event code (int64 [N]).

    - 2: first bar of the series (when prev_ts is given, the first bar is compared
      against prev_ts like any other bar).
    - 1: gap to the previous bar exceeds gap_multiplier times the expected gap
      from the calendar (overnight, lunch, weekends and holidays are not gaps).
    - 0: otherwise.
    """
    idx = pd.DatetimeIndex(timestamps)
    n = len(idx)
    ev = np.zeros(n, dtype=np.int64)
    if n == 0:
        return ev
    if prev_ts is None:
        ev[0] = 2
        start = 1
        prev_idx = idx[:-1]
    else:
        start = 0
        prev_idx = pd.DatetimeIndex([pd.Timestamp(prev_ts)]).append(idx[:-1])
    if n > start:
        expected_ns = _expected_next_ns(prev_idx, market, timescale, trading_days)
        expected_gap = (expected_ns - prev_idx.asi8) / 1e9
        actual_gap = (idx[start:].asi8 - prev_idx.asi8) / 1e9
        ev[start:] = np.where(actual_gap > gap_multiplier * expected_gap, 1, ev[start:])
    return ev


def attach_window_calendar(
    timestamps: Union[Sequence[TimestampLike], pd.DatetimeIndex, np.ndarray],
    market: str,
    timescale: str,
    n_ctx: int,
    events: Optional[np.ndarray] = None,
    prev_ts: Optional[TimestampLike] = None,
    trading_days: TradingDays = None,
) -> Dict[str, np.ndarray]:
    """Calendar features and events for a (ctx + tgt) window.

    Args:
        n_ctx: length of the context part; event is forced to 0 after it.
        events: optional precomputed events; computed from timestamps if None.
        prev_ts: timestamp before the window (only used when events is None).
        trading_days: optional trading-day table (only used when events is None).
    """
    cal = gen_calendar(timestamps, market, timescale)
    if events is None:
        event = detect_events(timestamps, market, timescale, prev_ts=prev_ts,
                              trading_days=trading_days)
    else:
        event = np.asarray(events, dtype=np.int64).copy()
        if event.shape[0] != len(cal["dow"]):
            raise ValueError("events 长度必须与 timestamps 一致")
    event = event.astype(np.int64, copy=True)
    event[n_ctx:] = 0
    cal["event"] = event
    return cal
