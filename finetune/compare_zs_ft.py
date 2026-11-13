#!/usr/bin/env python
"""Zero-shot (pretrained) vs fine-tuned KiT: metrics, quintile charts and tables.

Runs the same cross-sectional evaluation on both checkpoints over the custom
data tree and writes an analyst-friendly report:

    <out>/metrics_by_scale.csv   IC / RankIC (mean +- SE, paired delta + t) by scale
    <out>/anchors.csv            per-anchor metrics for both models
    <out>/quintiles.csv          mean realised return per predicted-return quintile
    <out>/summary.json           everything, machine readable (for an LLM)
    <out>/summary.md             markdown tables
    <out>/quintiles.png          quintile bar chart (ZS vs FT)
    <out>/rankic_by_scale.png    RankIC bars by scale (ZS vs FT)
    <out>/ic_over_time.png       per-anchor return RankIC over time
    <out>/equity_curve.png       cumulative long-short (Q_top - Q_bottom) for --curve-scale

Correctness notes:
    * Cross-sections are formed at the **exact target-bar timestamp** (not the
      day), so only windows that start at the same instant are compared.
    * Both models share the **same initial noise** per cross-section (a generator
      seeded from (market, scale, timestamp)), so ZS vs FT reflect the model, not
      sampling luck.
    * Deltas are paired over anchors and reported with an SE and t-statistic.

A bar's close-to-close log return is `r_gap + r_body` (denormalised), so no price
levels are needed: terminal return is their horizon sum, realised volatility the
per-bar sample std. Metric definitions follow `eval/cal_metrics.py`.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: F401,E402  (load before pandas)
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.data.dataset import KlineWindowDataset  # noqa: E402
from kit_v1.data.normalize import load_stats  # noqa: E402
from kit_v1.infer import denormalize_paths, flow_sample, load_model_and_config  # noqa: E402

SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
METRIC_KEYS = ("ret_ic", "ret_rankic", "vol_ic", "vol_rankic", "price_ic", "price_rankic")
_COLORS = {"ZS": "#9aa0a6", "FT": "#1a73e8"}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def _clean(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[np.isfinite(x)]


def _mean_se(x):
    x = _clean(x)
    if x.size == 0:
        return float("nan"), float("nan")
    se = float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")
    return float(x.mean()), se


def _se(x):
    x = _clean(x)
    return float(x.std(ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def _nanmean(x) -> float:
    x = _clean(x)
    return float(x.mean()) if x.size else float("nan")


def _rank(a) -> np.ndarray:
    return pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average").to_numpy()


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


def _section_seed(market: str, scale: str, ts) -> int:
    key = f"{market}|{scale}|{pd.Timestamp(ts).isoformat()}".encode("utf-8")
    return int.from_bytes(hashlib.md5(key).digest()[:8], "little") % (2 ** 63)


# --------------------------------------------------------------------------- #
# data selection (exact target timestamp)
# --------------------------------------------------------------------------- #
def build_cross_sections(ds: KlineWindowDataset, scales, n_anchors: int,
                         min_size: int, max_inst: int):
    by = defaultdict(lambda: defaultdict(list))
    for i, s in enumerate(ds.samples):
        if scales and s.scale not in scales:
            continue
        dt = ds._read_datetimes(ds.data_root / s.market / s.scale / f"{s.code}.parquet")
        ts = pd.Timestamp(int(dt[s.anchor]), unit="ns")
        by[(s.market, s.scale)][ts].append((i, s.market, s.code))
    sections = []
    for (market, scale), per_ts in sorted(by.items()):
        times = [t for t in sorted(per_ts) if len(per_ts[t]) >= min_size]
        if not times:
            continue
        if n_anchors > 0 and len(times) > n_anchors:
            idx = np.linspace(0, len(times) - 1, n_anchors).round().astype(int)
            times = [times[j] for j in sorted(set(idx.tolist()))]
        for t in times:
            items = per_ts[t]
            if max_inst > 0 and len(items) > max_inst:
                step = max(1, len(items) // max_inst)
                items = items[::step][:max_inst]
            sections.append(((market, scale, t), items))
    return sections


# --------------------------------------------------------------------------- #
# evaluation -> records (batched, shared noise)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, ds, sections, stats, device, steps, n_paths, precision):
    """One batched sampling call per cross-section; noise seeded from the section."""
    records = []
    for (market, scale, ts), items in sections:
        items_obj = [ds[i] for i, _, _ in items]
        Lh = items_obj[0]["x_tgt"].shape[0]
        x_ctx = torch.from_numpy(np.stack([o["x_ctx"] for o in items_obj])).float()
        cond = {k: torch.stack([torch.as_tensor(o["cond"][k]) for o in items_obj])
                for k in items_obj[0]["cond"]}
        gen = torch.Generator().manual_seed(_section_seed(market, scale, ts))
        paths = flow_sample(model, x_ctx, cond, Lh=Lh, steps=steps, n_paths=n_paths,
                            precision=precision, device=device, generator=gen)
        pred = denormalize_paths(paths.float().cpu().numpy(), market, scale, stats)
        rp = pred[..., 0] + pred[..., 1]                            # [N,K,Lh]
        act = denormalize_paths(np.stack([o["x_tgt"] for o in items_obj]),
                                market, scale, stats)               # [N,Lh,5]
        ra = act[..., 0] + act[..., 1]                              # [N,Lh]
        n = rp.shape[0]
        if n < 3:
            continue
        records.append({
            "market": market, "scale": scale, "ts": ts,
            "date": str(pd.Timestamp(ts).date()), "n": n,
            "pred_ret": rp.sum(axis=-1).mean(axis=-1),
            "act_ret": ra.sum(axis=-1),
            "pred_vol": (rp.std(axis=-1, ddof=1).mean(axis=-1) if Lh > 1
                         else np.zeros(n)),
            "act_vol": ra.std(axis=-1, ddof=1) if Lh > 1 else np.zeros(n),
            "pred_cum": rp.cumsum(axis=-1).mean(axis=1),            # [N,Lh]
            "act_cum": ra.cumsum(axis=-1),
        })
    return records


def anchor_metrics(r):
    h = r["pred_cum"].shape[1]
    return {
        "ret_ic": pearson(r["pred_ret"], r["act_ret"]),
        "ret_rankic": spearman(r["pred_ret"], r["act_ret"]),
        "vol_ic": pearson(r["pred_vol"], r["act_vol"]),
        "vol_rankic": spearman(r["pred_vol"], r["act_vol"]),
        "price_ic": _nanmean([pearson(r["pred_cum"][:, i], r["act_cum"][:, i])
                              for i in range(h)]),
        "price_rankic": _nanmean([spearman(r["pred_cum"][:, i], r["act_cum"][:, i])
                                  for i in range(h)]),
    }


def per_scale_rows(records):
    m = defaultdict(lambda: defaultdict(list))
    for r in records:
        for k, v in anchor_metrics(r).items():
            m[r["scale"]][k].append(v)
    return {s: {k: np.asarray(v) for k, v in d.items()} for s, d in m.items()}


def _pool(maps, scales, key):
    arrs = [maps[s][key] for s in scales if key in maps.get(s, {})]
    return np.concatenate(arrs) if arrs else np.array([])


def summarize(zs_map, ft_map=None):
    """Per-scale mean/SE and paired ZS->FT deltas for every metric."""
    scales = sorted(zs_map)
    out = {}
    for s in scales:
        row = {}
        for mk in METRIC_KEYS:
            za = zs_map[s].get(mk, np.array([]))
            zm, zse = _mean_se(za)
            cell = {"zs_mean": zm, "zs_se": zse, "n": int(np.isfinite(za).sum())}
            if ft_map and s in ft_map and mk in ft_map[s]:
                fa = ft_map[s][mk]
                fm, fse = _mean_se(fa)
                k = min(len(za), len(fa))
                d = _clean(fa[:k] - za[:k])
                dse = _se(d)
                cell.update({"ft_mean": fm, "ft_se": fse,
                             "delta_mean": _nanmean(d), "delta_se": dse,
                             "delta_t": _nanmean(d) / dse if dse and dse > 0 else float("nan")})
            row[mk] = cell
        out[s] = row
    if scales:
        allrow = {}
        for mk in METRIC_KEYS:
            za = _pool(zs_map, scales, mk)
            zm, zse = _mean_se(za)
            cell = {"zs_mean": zm, "zs_se": zse, "n": int(np.isfinite(za).sum())}
            if ft_map:
                fa = _pool(ft_map, scales, mk)
                fm, fse = _mean_se(fa)
                ds = []
                for s in scales:
                    if s in ft_map and mk in ft_map[s] and mk in zs_map[s]:
                        k = min(len(zs_map[s][mk]), len(ft_map[s][mk]))
                        ds.append(ft_map[s][mk][:k] - zs_map[s][mk][:k])
                d = _clean(np.concatenate(ds)) if ds else np.array([])
                dse = _se(d)
                cell.update({"ft_mean": fm, "ft_se": fse,
                             "delta_mean": _nanmean(d), "delta_se": dse,
                             "delta_t": _nanmean(d) / dse if dse and dse > 0 else float("nan")})
            allrow[mk] = cell
        out["ALL"] = allrow
    return out


# --------------------------------------------------------------------------- #
# quintiles + per-scale equity curve
# --------------------------------------------------------------------------- #
def quintile_table(records, q=5, scale=None):
    acc = defaultdict(list)
    curve = defaultdict(list)
    for r in records:
        if scale and r["scale"] != scale:
            continue
        n = r["n"]
        if n < q:
            continue
        ranks = np.empty(n)
        ranks[np.argsort(r["pred_ret"])] = np.arange(n)
        b = np.clip((ranks / n * q).astype(int), 0, q - 1)
        ar = r["act_ret"]
        for k in range(q):
            m = b == k
            if m.any():
                acc[k].append(ar[m].mean())
        curve[r["date"]].append(ar[b == q - 1].mean() - ar[b == 0].mean())
    quint = {k: _nanmean(v) for k, v in sorted(acc.items())}
    long_short = _nanmean([np.mean(v) for v in curve.values()])
    dates = sorted(curve)
    equity = np.cumsum([float(np.mean(curve[d])) for d in dates]).tolist() if dates else []
    return quint, long_short, dates, equity


def _pick_curve_scale(records, requested: str) -> str:
    if requested:
        return requested
    counts = defaultdict(int)
    for r in records:
        counts[r["scale"]] += 1
    return max(counts, key=counts.get) if counts else ""


# --------------------------------------------------------------------------- #
# charts
# --------------------------------------------------------------------------- #
def make_charts(summary, zs_recs, ft_recs, out_dir: Path, q: int, curve_scale: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [s for s in SCALES if s in summary]
    qz, lsz, _, _ = quintile_table(zs_recs, q)
    qf, lsf, _, _ = (quintile_table(ft_recs, q) if ft_recs else ({}, float("nan"), [], []))
    w = 0.38

    # 1. quintiles
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ks = sorted(set(qz) | set(qf))
    x = np.arange(len(ks))
    ax.bar(x - w / 2, [qz.get(k, np.nan) for k in ks], w, label="ZS", color=_COLORS["ZS"])
    if ft_recs:
        ax.bar(x + w / 2, [qf.get(k, np.nan) for k in ks], w, label="FT", color=_COLORS["FT"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{k+1}" for k in ks])
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylabel("mean realised return")
    ax.set_title(f"Return quintiles (Q1..Q{q}); long-short ZS={lsz:.4f}"
                 + (f" FT={lsf:.4f}" if ft_recs else ""))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "quintiles.png", dpi=150)
    plt.close(fig)

    # 2. RankIC by scale (ret + vol)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric, title in zip(axes, ("ret", "vol"), ("Return RankIC", "Volatility RankIC")):
        mk = f"{metric}_rankic"
        x = np.arange(len(scales))
        ax.bar(x - w / 2, [summary[s][mk]["zs_mean"] for s in scales], w,
               label="ZS", color=_COLORS["ZS"],
               yerr=[summary[s][mk]["zs_se"] for s in scales], capsize=2)
        if ft_recs:
            ax.bar(x + w / 2, [summary[s][mk].get("ft_mean", np.nan) for s in scales], w,
                   label="FT", color=_COLORS["FT"],
                   yerr=[summary[s][mk].get("ft_se", np.nan) for s in scales], capsize=2)
        ax.set_xticks(x)
        ax.set_xticklabels(scales)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "rankic_by_scale.png", dpi=150)
    plt.close(fig)

    # 3. per-anchor Return RankIC over time
    fig, ax = plt.subplots(figsize=(9, 4))
    for name, recs in (("ZS", zs_recs), ("FT", ft_recs)):
        if not recs:
            continue
        pts = sorted(((r["ts"], anchor_metrics(r)["ret_rankic"]) for r in recs),
                     key=lambda t: t[0])
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.scatter(xs, ys, s=14, alpha=0.5, color=_COLORS[name])
        roll = pd.Series(ys).rolling(max(1, min(10, len(ys))), min_periods=1).mean()
        ax.plot(xs, roll, color=_COLORS[name], label=f"{name} (rolling mean)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title("Return RankIC per anchor")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "ic_over_time.png", dpi=150)
    plt.close(fig)

    # 4. long-short equity curve for one scale (avoids mixing return scales)
    cs = _pick_curve_scale(zs_recs, curve_scale)
    _, lsz_c, dates, eqz = quintile_table(zs_recs, q, scale=cs)
    if dates:
        _, lsf_c, _, eqf = (quintile_table(ft_recs, q, scale=cs) if ft_recs
                            else (float("nan"), float("nan"), [], []))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(pd.to_datetime(dates), eqz, label="ZS", color=_COLORS["ZS"])
        if ft_recs:
            ax.plot(pd.to_datetime(dates), eqf, label="FT", color=_COLORS["FT"])
        ax.set_title(f"Cumulative long-short (Q_top - Q_bottom), scale={cs}")
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(out_dir / "equity_curve.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #
# report tables
# --------------------------------------------------------------------------- #
def _fmt(v, nd=4):
    return f"{v:.{nd}f}" if np.isfinite(v) else ""


def write_report(summary, zs_recs, ft_recs, zs_map, ft_map, out_dir: Path, q: int,
                 meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [s for s in SCALES if s in summary] + (["ALL"] if "ALL" in summary else [])
    has_ft = bool(ft_recs)

    # metrics_by_scale.csv
    rows = []
    for s in scales:
        row = {"scale": s, "n_anchors": summary[s][METRIC_KEYS[0]]["n"]}
        for mk in METRIC_KEYS:
            c = summary[s][mk]
            row[f"{mk}_zs"] = c["zs_mean"]
            row[f"{mk}_zs_se"] = c["zs_se"]
            if has_ft and "ft_mean" in c:
                row[f"{mk}_ft"] = c["ft_mean"]
                row[f"{mk}_ft_se"] = c["ft_se"]
                row[f"{mk}_delta"] = c["delta_mean"]
                row[f"{mk}_delta_se"] = c["delta_se"]
                row[f"{mk}_delta_t"] = c["delta_t"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "metrics_by_scale.csv", index=False)

    # anchors.csv
    arows = []
    for name, recs in (("ZS", zs_recs), ("FT", ft_recs)):
        for r in recs:
            arows.append({"model": name, "scale": r["scale"], "timestamp": str(r["ts"]),
                          "date": r["date"], "n": r["n"], **anchor_metrics(r)})
    pd.DataFrame(arows).to_csv(out_dir / "anchors.csv", index=False)

    # quintiles.csv
    qz, lsz, _, _ = quintile_table(zs_recs, q)
    qf, lsf, _, _ = quintile_table(ft_recs, q) if has_ft else ({}, float("nan"), [], [])
    qrows = [{"quintile": k, "zs": qz.get(k, np.nan), "ft": qf.get(k, np.nan)}
             for k in sorted(set(qz) | set(qf))]
    pd.DataFrame(qrows).to_csv(out_dir / "quintiles.csv", index=False)

    # summary.json
    payload = {"meta": meta, "scales": {s: summary[s] for s in summary},
               "quintiles": {"zs": qz, "ft": qf},
               "long_short": {"zs": lsz, "ft": lsf}}
    (out_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # summary.md
    L = ["# KiT zero-shot vs fine-tuned", "",
         f"- data: `{meta['data_root']}`  split: `{meta['split']}`",
         f"- anchors scored: {meta['n_anchors']}  paths: {meta['n_paths']}  steps: {meta['steps']}",
         f"- ZS: `{meta['ckpt_zs']}`", f"- FT: `{meta['ckpt_ft']}`",
         "- metrics: mean +/- SE over anchors; delta = FT - ZS (paired over anchors)", "",
         "## Metrics by scale", "",
         "| scale | metric | ZS (mean+/-SE) | FT (mean+/-SE) | delta (t) | n |",
         "|---|---|---|---|---|---|"]
    for s in scales:
        for mk in METRIC_KEYS:
            c = summary[s][mk]
            if has_ft and "delta_mean" in c:
                ft_cell = f"{_fmt(c['ft_mean'])}+/-{_fmt(c['ft_se'])}"
                d = f"{_fmt(c['delta_mean'])} ({_fmt(c['delta_t'], 2)})"
            else:
                ft_cell, d = "", ""
            L.append(f"| {s} | {mk} | {_fmt(c['zs_mean'])}+/-{_fmt(c['zs_se'])} | "
                     f"{ft_cell} | {d} | {c['n']} |")
    L += ["", "## Mean realised return by predicted-return quintile", "",
          "| quintile | ZS | FT |", "|---|---|---|"]
    for k in sorted(set(qz) | set(qf)):
        L.append(f"| Q{k+1} | {_fmt(qz.get(k, np.nan))} | {_fmt(qf.get(k, np.nan))} |")
    L += ["", f"Long-short (Q{q}-Q1): ZS={_fmt(lsz)}"
          + (f", FT={_fmt(lsf)}" if has_ft else "")]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def print_table(summary):
    has_ft = any("delta_mean" in summary[s][METRIC_KEYS[0]] for s in summary)
    print(f"\n{'scale':6} {'n':>5}  " + "  ".join(f"{mk:>13}" for mk in METRIC_KEYS))
    print("-" * (13 + 14 * len(METRIC_KEYS)))
    for s in [x for x in SCALES if x in summary] + (["ALL"] if "ALL" in summary else []):
        row = f"{s:6} {summary[s][METRIC_KEYS[0]]['n']:>5}  "
        for mk in METRIC_KEYS:
            c = summary[s][mk]
            if has_ft and "delta_mean" in c:
                cell = f"{c['zs_mean']:.2f}>{c['ft_mean']:.2f}"
            else:
                cell = f"{c['zs_mean']:.3f}"
            row += f"{cell:>13}  "
        print(row)
    if has_ft:
        print("(cells: ZS>FT)")


# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="ZS vs FT KiT comparison report")
    ap.add_argument("--ckpt-zs", required=True)
    ap.add_argument("--ckpt-ft", default=None)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", default="val", choices=["val", "test", "train"])
    ap.add_argument("--scales", default="")
    ap.add_argument("--n-anchors", type=int, default=20)
    ap.add_argument("--min-size", type=int, default=5)
    ap.add_argument("--max-inst", type=int, default=0)
    ap.add_argument("--n-paths", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--quintiles", type=int, default=5)
    ap.add_argument("--curve-scale", default="", help="scale for the equity curve (default: largest)")
    ap.add_argument("--preset", default="tiny")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    ap.add_argument("--out", type=Path, default=Path("runs/compare_zs_ft"))
    ap.add_argument("--no-charts", action="store_true")
    args = ap.parse_args(argv)

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    prec = args.precision if device.type == "cuda" else "fp32"
    scales = [s.strip() for s in args.scales.split(",") if s.strip()]

    def _pt(p):
        p = Path(p)
        if not p.is_dir():
            return str(p)
        latest = p / "latest.pt"
        if latest.is_file():
            return str(latest)
        pts = sorted(p.glob("*.pt"))
        if len(pts) != 1:
            raise SystemExit(f"{p}: expected latest.pt or a single .pt, found {len(pts)}")
        return str(pts[0])

    args.ckpt_zs = _pt(args.ckpt_zs)
    if args.ckpt_ft:
        args.ckpt_ft = _pt(args.ckpt_ft)
    model_zs, cfg = load_model_and_config(args.ckpt_zs, args.preset, args.seed)
    model_zs.to(device).eval()
    models = {"ZS": model_zs}
    if args.ckpt_ft:
        model_ft, _ = load_model_and_config(args.ckpt_ft, args.preset, args.seed)
        model_ft.to(device).eval()
        models["FT"] = model_ft

    stats = load_stats(cfg.data.stats_path or (args.data_root / "stats.json"))
    ds = KlineWindowDataset(cfg, split=args.split, data_root=args.data_root)
    sections = build_cross_sections(ds, scales, args.n_anchors, args.min_size,
                                    args.max_inst)
    print(f"[cmp] device={device} precision={prec} split={args.split} "
          f"cross-sections={len(sections)} (exact-timestamp, shared noise)")
    if not sections:
        raise SystemExit("[cmp] no cross-sections with >= min-size instruments; "
                         "lower --min-size or check the data/split")

    recs, maps = {}, {}
    for name, model in models.items():
        recs[name] = evaluate(model, ds, sections, stats, device, args.steps,
                              args.n_paths, prec)
        maps[name] = per_scale_rows(recs[name])
    summary = summarize(maps["ZS"], maps.get("FT"))
    print_table(summary)

    meta = {"ckpt_zs": args.ckpt_zs, "ckpt_ft": args.ckpt_ft,
            "data_root": str(args.data_root), "split": args.split,
            "n_anchors": len(sections), "n_paths": args.n_paths, "steps": args.steps,
            "quintiles": args.quintiles, "device": str(device), "precision": prec,
            "grouping": "exact target-bar timestamp", "shared_noise": True}
    write_report(summary, recs["ZS"], recs.get("FT", []), maps["ZS"],
                 maps.get("FT", {}), args.out, args.quintiles, meta)
    if not args.no_charts:
        make_charts(summary, recs["ZS"], recs.get("FT", []), args.out,
                    args.quintiles, args.curve_scale)
        print(f"[cmp] charts written to {args.out}")
    print(f"[cmp] report written to {args.out}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
