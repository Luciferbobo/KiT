# Evaluating KiT: magnitude, risk and direction

KiT samples **K future paths** for every anchor. What you can read from them depends on the statistic
you compute. This note lists the choices that most often change evaluation results. The reference
implementation of every metric below is [`eval/cal_metrics.py`](../eval/cal_metrics.py).

## 1. Use the model's fixed windows

Each timescale has a fixed history length `Lc` and horizon `Lh`. They are part of the trained model, not
tunable inputs; a different window gives the model an input it was not trained on.

| Timescale | 1m | 5m | 15m | 30m | 1h | 2h | 1d |
|---|---|---|---|---|---|---|---|
| History `Lc` | 1200 | 960 | 800 | 640 | 400 | 360 | 250 |
| Horizon `Lh` | 120 | 96 | 64 | 40 | 20 | 20 | 20 |

For a horizon shorter than `Lh` (e.g. the next 12 bars), forecast with the full window and read the first
bars of each path. For daily questions use the `1d` timescale instead of rolling an intraday model across
the session.

## 2. Use an out-of-sample checkpoint

| Checkpoint | Training data | Out-of-sample from |
|---|---|---|
| `step_market_284M.pt` | 2016-01-01 to 2025-12-31 | 2026-01-01 |
| `step_market_284M_train260410.pt` | 2016-01-01 to 2026-04-10 | 2026-04-11 |

Evaluating `step_market_284M_train260410.pt` on January–April 2026 measures in-sample fit.

## 3. Magnitude: take the spread of the paths, not the size of their mean

Let `r[k, h]` be the log return of path `k` at bar `h` (relative to the previous bar; `P[k, -1]` is the
anchor close).

| Statistic | Formula | Use it? |
|---|---|---|
| Within-path volatility (official `vol`) | `mean_k( std_h(r[k, :], ddof=1) )` | Yes: volatility over the window |
| Expected absolute move | `mean_k( | sum_h r[k, :H] | )` | Yes: size of the move over the next `H` bars |
| Dispersion across paths | `std_k( sum_h r[k, :H] )` | Yes: uncertainty of the `H`-bar return |
| Size of the mean path | `| mean_k( sum_h r[k, :H] ) |` | **No** |

Averaging paths before taking the absolute value lets up-paths and down-paths cancel. The result is
systematically smaller than any single path's move, and the shrinkage depends on K (fewer paths cancel
less). It measures how confident the model is about **direction**, not how large the move will be.

On the live demo (100 A-shares, `1d`, January–November 2026, about 21,000 stock-days, next-day
horizon):

| Predicted magnitude | Mean predicted | Mean realized | Realized / predicted |
|---|---|---|---|
| `| mean_k r[k, 0] |` | 1.02% | 2.49% | 2.44× |
| `mean_k | r[k, 0] |` | 2.20% | 2.49% | 1.13× |
| `std_k( r[k, 0] )` | 2.55% | 2.49% | 0.98× |

Use enough paths for magnitude work: with K = 8 the per-anchor estimate is noisy; we use K ≥ 32.

```python
import numpy as np

def path_statistics(close_paths, anchor_close, horizon):
    """close_paths: [K, Lh] predicted closes; anchor_close: last real close before the forecast."""
    log_prices = np.log(np.concatenate([np.full((len(close_paths), 1), anchor_close), close_paths], axis=1))
    r = np.diff(log_prices, axis=1)              # per-bar log returns, [K, Lh]
    window = r[:, :horizon].sum(axis=1)          # H-bar return of each path, [K]
    return {
        "volatility": r.std(axis=1, ddof=1).mean(),     # official vol, full window
        "expected_abs_move": np.abs(window).mean(),
        "dispersion": window.std(ddof=1),
        "mean_return": window.mean(),                   # direction signal; not a magnitude
    }
```

## 4. Rank across stocks at each anchor

The paper's metrics are **cross-sectional**: at each anchor, correlate predictions with outcomes across
stocks (Spearman for RankIC), then average over anchors. A correlation over time for a single
instrument answers a different question (volatility timing for that stock) and has far fewer
independent observations; with about 180 daily anchors, quintile means rest on about 36 points each and
the middle buckets are within noise.

Always compare with a simple baseline. For risk, the natural one is trailing realized volatility. On the
live demo, the daily cross-sectional Spearman with realized volatility is:

| Horizon | KiT predicted volatility | Trailing 20-day volatility | Days KiT ranks better |
|---|---|---|---|
| Next day (absolute return) | 0.34 | 0.30 | 67% |
| Next 5 days | 0.54 | 0.47 | 79% |
| Next 20 days | 0.63 | 0.53 | 89% |

The live numbers and charts are on the overview page of the
[demo](https://trading.etfnav.org/?lang=en) ("KiT risk forecasts").

## 5. Evaluate direction separately

`mean_return` is a ranking signal. Evaluate it as a cross-sectional RankIC or a ranked portfolio over
the forecast window (see `eval/backtest.py`), not as the hit rate of one series. Next-bar hit rates near
50% on a single stock are expected and say little about the ranking signal. If you report hit rates,
include the naive baselines of the same sample (always up, always down).

Magnitude and direction answer different questions and can disagree: a stock can be ranked high-risk
with no clear direction. For position sizing, volatility filters and risk budgets, use the magnitude
statistics in section 3.

## 6. Calibrating the level

Ranking metrics ignore the level of the forecast. If a horizon needs calibrated levels (e.g. MAE in
percent, or volatility targeting), fit a monotone map on a held-out window and apply it to later data:
a single scale factor `realized / predicted`, or isotonic regression. A monotone map leaves the ranking,
and therefore RankIC and the quintiles, unchanged. Use one of the statistics in section 3 first; most
of an apparent under-forecast disappears once the mean-path statistic is replaced.
