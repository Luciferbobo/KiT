"""Cross-sectional stock selection backtest on the results of infer_for_eval.py.

For every scale and anchor, rank the stocks by the mean predicted window return, hold the top N
with equal weight until the window ends, then chain the windows. A-share rules are applied:
  * buy at the anchor close; skipped if the stock closes limit-up or is in its first 5 listing days
  * sell at the last bar of the window; if it is limit-down there, the position is carried over
  * T+1 holds automatically (buy at the close, sell in a later window)
  * costs: commission 2.5bp + transfer fee 0.1bp per side, stamp tax 5bp on sell, slippage per side
Stocks that stay in the top N are kept (no trading). For 1m (half-day windows) everything is sold
at the window end and bought again at the next close.

Usage:
  python eval/backtest.py [--top 10] [--slip 10] [--scales 1m,5m,...]

Output (results/backtest_results/):
  backtest_top<N>.png   equity curves for 1m..2h vs. 100-stock mean / median and CSI500
  summary_top<N>.csv    one row per scale
CSI500 daily closes are downloaded once with akshare and cached in data/csi500_daily.csv.
"""
import argparse
import glob
import os
import re
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "infer_for_eval" / "raw"
DAILY = ROOT / "data" / "ashare" / "1d"
INDEX_CACHE = ROOT / "data" / "csi500_daily.csv"
OUT = ROOT / "results" / "backtest_results"
SCALES = ["1m", "5m", "15m", "30m", "1h", "2h", "1d"]
PLOT_SCALES = SCALES[:6]
EPS = 0.005          # within 0.5% of the limit counts as sealed

C = dict(top="#2a78d6", mean="#eb6834", median="#1baf7a", idx="#e87ba4",
         ink="#0b0b0b", ink2="#52514e", grid="#e6e5e1", bg="#fcfcfb")


def cost_rates(slip_bp):
    comm, transfer, stamp = 2.5, 0.1, 5.0
    return (comm + transfer + slip_bp) * 1e-4, (comm + transfer + stamp + slip_bp) * 1e-4


def limit_rate(inst):
    return 0.20 if inst.split(".")[0].startswith(("688", "689", "300", "301")) else 0.10


class Daily:
    """Daily closes of the universe, used for limit-up / limit-down checks."""

    def __init__(self, insts):
        self.d = {}
        for it in insts:
            df = pd.read_parquet(DAILY / f"{it}.parquet", columns=["datetime", "close"])
            self.d[it] = (df["datetime"].dt.normalize().values, df["close"].to_numpy(float))

    def prev_close(self, inst, date):
        dates, close = self.d[inst]
        i = np.searchsorted(dates, np.datetime64(date), side="left")
        return close[i - 1] if i > 0 else np.nan

    def buy_blocked(self, inst, date):
        dates, close = self.d[inst]
        i = np.searchsorted(dates, np.datetime64(date), side="left")
        if i >= len(dates) or dates[i] != np.datetime64(date) or i < 5:
            return True
        return close[i] / close[i - 1] - 1 >= limit_rate(inst) - EPS

    def sell_blocked(self, inst, end_ts, last_close):
        pc = self.prev_close(inst, end_ts.normalize())
        if not np.isfinite(pc) or not np.isfinite(last_close):
            return False
        return last_close / pc - 1 <= -limit_rate(inst) + EPS


def load_windows(scale):
    rows = []
    for f in sorted(glob.glob(str(RAW / scale / "*.npz"))):
        z = np.load(f, allow_pickle=True)
        act = np.exp(z["act_bar_ret"].astype(np.float64).sum(-1)) - 1
        sig = z["pred_bar_ret"].astype(np.float64).sum(-1).mean(1)
        ok = np.isfinite(act) & np.isfinite(sig)
        rows.append(dict(
            anchor=pd.Timestamp(re.search(r"_(\d{8})_", f).group(1)),
            end=pd.Timestamp(z["fut_ts"].max()),
            insts=[str(s) for s in z["inst"][ok]], sig=sig[ok], act=act[ok],
            last_close=z["act_ohlcv"][ok, -1, 3].astype(np.float64)))
    for r in rows:
        r["pos"] = {n: i for i, n in enumerate(r["insts"])}
    return rows


def simulate(scale, rows, daily, top, slip_bp, tradable=True, costs=True):
    """Returns the equity curve after each window, the final equity after liquidation, and stats."""
    buy_rate, sell_rate = cost_rates(slip_bp) if costs else (0.0, 0.0)
    keep_overlap = scale != "1m"
    cash, hold = 1.0, {}          # hold: inst -> position value
    curve, stats = [], []
    prev = None
    for r in rows:
        order = list(np.array(r["insts"])[np.argsort(-r["sig"])])
        cand = [n for n in order if not (tradable and daily.buy_blocked(n, r["anchor"].normalize()))]
        target = cand[:top]
        top_limit_up = np.mean([daily.buy_blocked(n, r["anchor"].normalize()) for n in order[:top]])
        stuck, traded = 0, 0.0
        for n in list(hold):                      # sell names that are no longer wanted
            if keep_overlap and n in target:
                continue
            blocked = (tradable and prev is not None and n in prev["pos"] and
                       daily.sell_blocked(n, prev["end"], prev["last_close"][prev["pos"][n]]))
            if blocked:
                stuck += 1
                continue
            cash += hold[n] * (1 - sell_rate)
            traded += hold[n]
            del hold[n]
        new = [n for n in target if n not in hold]    # buy the missing names with the available cash
        if new:
            equity = cash + sum(hold.values())
            per = min(equity / top, cash / (1 + buy_rate) / len(new))
            for n in new:
                hold[n] = per
                cash -= per * (1 + buy_rate)
                traded += per
        for n in list(hold):                      # window return
            hold[n] *= 1 + (r["act"][r["pos"][n]] if n in r["pos"] else 0.0)
        curve.append(cash + sum(hold.values()))
        stats.append(dict(turnover=traded, stuck=stuck, top_limit_up=top_limit_up))
        prev = r
    final = curve[-1] - sum(hold.values()) * sell_rate
    return np.array(curve), final, pd.DataFrame(stats)


def universe_chain(rows, fn):
    eq, out = 1.0, []
    for r in rows:
        eq *= 1 + fn(r["act"])
        out.append(eq)
    return np.array(out)


def max_dd(curve):
    c = np.concatenate([[1.0], curve])
    return (c / np.maximum.accumulate(c) - 1).min()


def load_csi500():
    if not INDEX_CACHE.exists():
        try:
            import akshare as ak
            ak.stock_zh_index_daily(symbol="sh000905")[["date", "close"]].to_csv(INDEX_CACHE, index=False)
        except Exception as e:
            print(f"CSI500 not available ({e!r}); it will be left out. Install akshare or provide {INDEX_CACHE}")
            return None
    return pd.read_csv(INDEX_CACHE, parse_dates=["date"]).set_index("date")["close"]


def style(ax):
    ax.set_facecolor(C["bg"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.tick_params(colors=C["ink2"], labelsize=8, length=0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10, help="number of stocks held")
    ap.add_argument("--slip", type=float, default=10.0, help="slippage per side, bp")
    ap.add_argument("--scales", default=",".join(SCALES))
    args = ap.parse_args()

    scales = [s for s in args.scales.split(",") if s in SCALES and glob.glob(str(RAW / s / "*.npz"))]
    if not scales:
        raise SystemExit(f"No inference results found in {RAW}. Run eval/infer_for_eval.py first.")
    idx = load_csi500()
    data = {}
    daily = None
    for s in scales:
        rows = load_windows(s)
        daily = daily or Daily(rows[0]["insts"])
        net, net_final, st = simulate(s, rows, daily, args.top, args.slip)
        gross, _, _ = simulate(s, rows, daily, args.top, args.slip, costs=False)
        _, naive_final, _ = simulate(s, rows, daily, args.top, args.slip, tradable=False)
        data[s] = dict(rows=rows, net=net, net_final=net_final, gross=gross, st=st, naive=naive_final,
                       mean=universe_chain(rows, np.mean), median=universe_chain(rows, np.median))
        print(f"{s}: {len(rows)} windows, net {net_final - 1:+.1%}")

    def idx_ret(rows):
        if idx is None:
            return np.nan
        return (idx[idx.index <= rows[-1]["end"].normalize()].iloc[-1] /
                idx[idx.index <= rows[0]["anchor"]].iloc[-1] - 1)

    t_start = min(d["rows"][0]["anchor"] for d in data.values())
    t_end = max(d["rows"][-1]["end"] for d in data.values())
    if idx is not None:
        base = idx[idx.index <= t_start].iloc[-1]
        idx_plot = idx[(idx.index >= t_start) & (idx.index <= t_end.normalize())]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), facecolor=C["bg"])
    for ax in axes.ravel():
        ax.axis("off")
    summary = []
    for s in scales:
        d = data[s]
        rows = d["rows"]
        if s in PLOT_SCALES:
            ax = axes.ravel()[PLOT_SCALES.index(s)]
            ax.axis("on")
            style(ax)
            x = np.concatenate([[rows[0]["anchor"].to_datetime64()], [r["end"].to_datetime64() for r in rows]])
            one = lambda c: np.concatenate([[1.0], c]) - 1
            if idx is not None:
                ax.plot(idx_plot.index + pd.Timedelta(hours=15), idx_plot / base - 1, color=C["idx"], lw=1.4,
                        label="CSI500")
            ax.plot(x, one(d["mean"]), color=C["mean"], lw=1.4, label="100-stock mean")
            ax.plot(x, one(d["median"]), color=C["median"], lw=1.4, label="100-stock median")
            ax.plot(x, one(d["gross"]), color=C["top"], lw=1.6, ls="--", label=f"KiT top{args.top}, no costs")
            ax.plot(x, one(d["net"]), color=C["top"], lw=2.2, label=f"KiT top{args.top}, net of costs")
            ax.axhline(0, color=C["ink2"], lw=0.6)
            ax.set_title(f"{s}   net {d['net_final'] - 1:+.1%}   (CSI500 {idx_ret(rows):+.1%})",
                         loc="left", fontsize=11, color=C["ink"])
        st = d["st"]
        summary.append(dict(
            scale=s, windows=len(rows), start=rows[0]["anchor"].date(), end=rows[-1]["end"].date(),
            net=d["net_final"] - 1, no_costs=d["gross"][-1] - 1, no_limit_filter=d["naive"] - 1,
            mean_100=d["mean"][-1] - 1, median_100=d["median"][-1] - 1, csi500=idx_ret(rows),
            max_drawdown=max_dd(d["net"]), two_way_turnover_per_window=st["turnover"].mean(),
            top_limit_up_share=st["top_limit_up"].mean(), stuck_per_window=st["stuck"].mean()))

    OUT.mkdir(parents=True, exist_ok=True)
    h, l = next(ax for ax in axes.ravel() if ax.lines).get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, fontsize=11, labelcolor=C["ink"])
    fig.suptitle(f"KiT top {args.top} of 100 A-shares, equal weight, limit-up/down and A-share costs applied "
                 f"(slippage {args.slip:g}bp/side)", x=0.01, ha="left", fontsize=14, color=C["ink"])
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(OUT / f"backtest_top{args.top}.png", dpi=130, facecolor=C["bg"])
    pd.DataFrame(summary).to_csv(OUT / f"summary_top{args.top}.csv", index=False, float_format="%.4f")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
