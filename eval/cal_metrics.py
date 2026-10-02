#!/usr/bin/env python
"""Compute three cross-sectional metrics (return, volatility, price path) from infer_for_eval.py results.

Needs only numpy / pandas. Metrics are computed per anchor over all stocks, then averaged over anchors:
  return  pred = sum of per-bar log returns over Lh, averaged over K paths; actual = same sum on real bars
  vol     pred = std (ddof=1) of per-bar log returns, averaged over K paths; actual = same on real bars
  price   pred = mean over K paths of cumulative log return vs. anchor close [N,Lh]; actual = real cumulative
          log return [N,Lh]. Correlation is taken at each future bar, then averaged over bars.
          (Raw prices would be dominated by price level, hence cumulative log returns.)
Each is reported as IC (Pearson) and RankIC (Spearman), plus std across anchors and ICIR = mean/std.

Output (default <results/infer_for_eval>/metrics/):
  metrics_summary.csv     one row per scale (anchor average) + ALL row (equal weight per scale)
  metrics_by_anchor.csv   one row per anchor
  metrics.json            both tables + per-bar price IC/RankIC (averaged over anchors per scale)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
METRICS = ("ret", "vol", "price")
FNAME = re.compile(r"^(?P<scale>\w+?)_(?P<date>\d{8})_k(?P<k>\d+)_s(?P<s>\d+)\.npz$")


def _rank(a: np.ndarray) -> np.ndarray:
    """Average rank (ties averaged), so Spearman = Pearson of ranks."""
    return pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)


def pearson(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3 or not (np.isfinite(a).all() and np.isfinite(b).all()):
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return float("nan")
    return pearson(_rank(a), _rank(b))


def cs_corr_per_bar(pred: np.ndarray, act: np.ndarray, fn) -> np.ndarray:
    """pred/act [N,Lh]: cross-sectional correlation at each bar -> [Lh]."""
    return np.array([fn(pred[:, h], act[:, h]) for h in range(pred.shape[1])])


def nanmean(x) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def anchor_metrics(z) -> dict:
    """The three metrics for one anchor; z is the npz content."""
    bar = z["pred_bar_ret"].astype(np.float64)               # [N,K,Lh]
    act_bar = z["act_bar_ret"].astype(np.float64)            # [N,Lh]

    pred_ret = bar.sum(axis=-1).mean(axis=1)                 # [N]
    act_ret = act_bar.sum(axis=-1)
    pred_vol = bar.std(axis=-1, ddof=1).mean(axis=1)
    act_vol = act_bar.std(axis=-1, ddof=1)
    pred_cum = bar.cumsum(axis=-1).mean(axis=1)              # [N,Lh]
    act_cum = act_bar.cumsum(axis=-1)

    price_ic = cs_corr_per_bar(pred_cum, act_cum, pearson)
    price_rk = cs_corr_per_bar(pred_cum, act_cum, spearman)
    return {
        "n": int(bar.shape[0]), "Lh": int(bar.shape[-1]),
        "ret_ic": pearson(pred_ret, act_ret), "ret_rankic": spearman(pred_ret, act_ret),
        "vol_ic": pearson(pred_vol, act_vol), "vol_rankic": spearman(pred_vol, act_vol),
        "price_ic": nanmean(price_ic), "price_rankic": nanmean(price_rk),
        "_price_bar_ic": price_ic, "_price_bar_rankic": price_rk,
    }


def find_files(raw: Path, k: int, steps: int):
    files = []
    for p in sorted(raw.rglob("*.npz")):
        m = FNAME.match(p.name)
        if m and ".tmp" not in p.name:
            files.append((m["scale"], m["date"], int(m["k"]), int(m["s"]), p))
    if not files:
        raise SystemExit(f"{raw}  has no inference result npz files; run infer_for_eval.py first")
    cfgs = sorted({(f[2], f[3]) for f in files})
    if k > 0 or steps > 0:
        files = [f for f in files if (k <= 0 or f[2] == k) and (steps <= 0 or f[3] == steps)]
    elif len(cfgs) > 1:
        raise SystemExit(f"directory mixes several (K,steps): {cfgs}; choose with --k / --steps")
    if not files:
        raise SystemExit(f"no results match K={k} steps={steps}; available: {cfgs}")
    return files


def summarize(df: pd.DataFrame) -> dict:
    row = {"n_anchors": len(df), "n_pairs": int(df["n"].sum())}
    for m in METRICS:
        for kind in ("ic", "rankic"):
            col = f"{m}_{kind}"
            v = df[col].to_numpy(dtype=np.float64)
            v = v[np.isfinite(v)]
            mean = float(v.mean()) if v.size else float("nan")
            std = float(v.std(ddof=1)) if v.size > 1 else float("nan")
            row[col] = mean
            row[f"{col}_std"] = std
            row[f"{col}_icir"] = mean / std if std and np.isfinite(std) and std > 0 else float("nan")
    return row


def print_table(summary: pd.DataFrame) -> None:
    cols = ["scale", "n_anchors", "n_pairs"] + [f"{m}_{k}" for m in METRICS for k in ("ic", "rankic")]
    names = {"ret_ic": "retIC", "ret_rankic": "retRank", "vol_ic": "volIC", "vol_rankic": "volRank",
             "price_ic": "priceIC", "price_rankic": "priceRank"}
    hdr = f"{'scale':5} {'anch':>4} {'pairs':>6} " + " ".join(f"{names[c]:>9}" for c in cols[3:])
    print("\n" + hdr)
    print("-" * len(hdr))
    for _, r in summary.iterrows():
        print(f"{r['scale']:5} {int(r['n_anchors']):4d} {int(r['n_pairs']):6d} "
              + " ".join(f"{r[c]:+9.4f}" for c in cols[3:]))


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="KiT inference results -> IC and RankIC of return/volatility/price series")
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results" / "infer_for_eval",
                    help="output directory of infer_for_eval.py (contains raw/)")
    ap.add_argument("--out", type=Path, default=None, help="metrics output directory, default <results/infer_for_eval>/metrics")
    ap.add_argument("--scales", default=",".join(SCALES))
    ap.add_argument("--k", type=int, default=0, help="use only results with this K, 0=no limit (may be omitted if the directory has only one)")
    ap.add_argument("--steps", type=int, default=0, help="use only results with this number of Euler steps, 0=no limit")
    args = ap.parse_args()

    out = args.out or (args.results / "metrics")
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]
    files = [f for f in find_files(args.results / "raw", args.k, args.steps) if f[0] in scales]
    if not files:
        raise SystemExit(f"no results for {scales}")

    rows, bar_curves = [], {}
    for scale, date, k, steps, path in files:
        with np.load(path) as z:
            m = anchor_metrics(z)
        anchor = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        bar_curves.setdefault(scale, []).append((m.pop("_price_bar_ic"), m.pop("_price_bar_rankic")))
        rows.append({"scale": scale, "anchor": anchor, **m})

    by_anchor = pd.DataFrame(rows)
    by_anchor["_o"] = by_anchor["scale"].map({s: i for i, s in enumerate(SCALES)})
    by_anchor = by_anchor.sort_values(["_o", "anchor"]).drop(columns="_o").reset_index(drop=True)

    srows = []
    for scale in [s for s in SCALES if s in set(by_anchor["scale"])]:
        srows.append({"scale": scale, **summarize(by_anchor[by_anchor["scale"] == scale])})
    summary = pd.DataFrame(srows)
    # ALL: equal weight per scale (average anchors within a scale first)
    allrow = {"scale": "ALL", "n_anchors": int(summary["n_anchors"].sum()),
              "n_pairs": int(summary["n_pairs"].sum())}
    for c in summary.columns:
        if c not in ("scale", "n_anchors", "n_pairs"):
            allrow[c] = nanmean(summary[c])
    summary = pd.concat([summary, pd.DataFrame([allrow])], ignore_index=True)

    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "metrics_summary.csv", index=False, float_format="%.6f")
    by_anchor.to_csv(out / "metrics_by_anchor.csv", index=False, float_format="%.6f")
    curves = {}
    for scale, lst in bar_curves.items():
        curves[scale] = {
            "price_ic_by_bar": np.nanmean(np.stack([a for a, _ in lst]), axis=0).tolist(),
            "price_rankic_by_bar": np.nanmean(np.stack([b for _, b in lst]), axis=0).tolist(),
        }
    (out / "metrics.json").write_text(json.dumps({
        "summary": json.loads(summary.to_json(orient="records")),
        "by_anchor": json.loads(by_anchor.to_json(orient="records")),
        "price_by_bar": curves,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print_table(summary)
    print(f"\nwrote {out / 'metrics_summary.csv'}")
    print(f"wrote {out / 'metrics_by_anchor.csv'}")
    print(f"wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    main()
