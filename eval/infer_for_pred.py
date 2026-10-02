#!/usr/bin/env python
"""KiT inference for a chosen stock, time scale and anchor time; predicts the following K-lines and plots them.

A case is one stock x one scale x one anchor time. The model sees the last Lc bars up to and
including the anchor and predicts the next Lh bars. Lc/Lh per scale (from the checkpoint):

  1m 1200/120   5m 960/96   15m 800/64   30m 640/40   1h 400/20   2h 360/20   1d 250/20

Usage (no arguments runs the 7 built-in cases):
  python eval/infer_for_pred.py
  python eval/infer_for_pred.py --case 601899.SH,5m,2026-03-04 --case 000001.SZ,1d,2026-02-27
  python eval/infer_for_pred.py --case 300750.SZ,30m,"2026-03-10 11:30" --paths 64 --steps 16
  python eval/infer_for_pred.py --no-plot

--case is code,scale,anchor and can be repeated. A date-only anchor means that day's close;
with a time it means the last bar at or before that time.

Errors out if:
  - the code is not in instruments.json or has no data file for the scale
  - fewer than Lc bars exist before the anchor
  - fewer than Lh real bars exist after the anchor
  - --ctx-len / --pred-len differ from the model window for that scale

Other settings match infer_for_eval.py (K=16 paths, 32 Euler steps, bf16, no CFG) and can be changed.
Output in results/infer_for_pred/: one PNG and one npz per case, plus summary.csv and infer_meta.json.
"""
from __future__ import annotations

import os

# avoid duplicate libiomp abort on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # make kit_v1 importable


def _rel(p: Path) -> str:
    """Path relative to the repo root."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.name
SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")
MARKET = "ashare"

# default cases: one per scale
DEFAULT_CASES = [
    "600681.SH,1m,2026-02-04",
    "300591.SZ,5m,2026-01-23",
    "600989.SH,15m,2026-02-06",
    "300260.SZ,30m,2026-01-26",
    "300733.SZ,1h,2026-02-09",
    "603590.SH,2h,2026-01-19",
    "300368.SZ,1d,2026-01-05",
]


def parse_case(text: str) -> tuple[str, str, pd.Timestamp, bool]:
    """'code,scale,anchor' -> (code, scale, anchor cutoff time, whether only a date was given)."""
    parts = [p.strip() for p in text.split(",", 2)]
    if len(parts) != 3 or not all(parts):
        raise SystemExit(f"--case format must be code,scale,anchor; got {text!r}")
    code, scale, when = parts
    if scale not in SCALES:
        raise SystemExit(f"--case {text!r}: unknown scale {scale!r}; choices: {SCALES}")
    try:
        ts = pd.Timestamp(when)
    except Exception:
        raise SystemExit(f"--case {text!r}: cannot parse anchor time {when!r}")
    date_only = ts == ts.normalize() and len(when.strip()) <= 10
    end = ts + pd.Timedelta(hours=23, minutes=59, seconds=59) if date_only else ts
    return code, scale, end, date_only


def parse_len_table(text: str, what: str) -> dict[str, int]:
    """'1m:1200,5m:960' -> {scale: length}; empty string -> {}."""
    out = {}
    for item in filter(None, (x.strip() for x in text.split(","))):
        try:
            k, v = item.split(":")
            out[k.strip()] = int(v)
        except ValueError:
            raise SystemExit(f"{what} format must be scale:length[,scale:length]; got {text!r}")
        if k.strip() not in SCALES:
            raise SystemExit(f"{what}: unknown scale {k.strip()!r}; choices: {SCALES}")
    return out


def load_case_frame(data_root: Path, code: str, scale: str, end: pd.Timestamp,
                    Lc: int, Lh: int, keep: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet and check range; return (history up to the anchor incl. EMA warmup, Lh real bars after it)."""
    path = data_root / MARKET / scale / f"{code}.parquet"
    if not path.is_file():
        raise SystemExit(f"{code} has no {scale} data: {path}")
    df = pd.read_parquet(path, columns=["datetime", "open", "high", "low", "close", "volume"])
    ts = pd.DatetimeIndex(df["datetime"])
    first, last = ts[0], ts[-1]
    rng = f"local {code} {scale} data range {first} .. {last}"
    if end < first:
        raise SystemExit(f"anchor {end} is earlier than the data start. {rng}")
    n_hist = int(ts.searchsorted(end, side="right"))
    if n_hist < Lc:
        raise SystemExit(
            f"{code} {scale} anchor {end} has only {n_hist} bars before it, the model needs {Lc} bars of history (padding not allowed). "
            f"Move the anchor later. {rng}")
    n_fut = len(ts) - n_hist
    if n_fut < Lh:
        raise SystemExit(
            f"{code} {scale} anchor {end} has only {n_fut} real bars after it, the prediction window needs {Lh}, beyond the end of local data. "
            f"Move the anchor earlier. {rng}")
    hist = df.iloc[max(0, n_hist - keep):n_hist].reset_index(drop=True)
    fut = df.iloc[n_hist:n_hist + Lh].reset_index(drop=True)
    return hist, fut


def draw_candles(ax, xs, o, h, l, c, width=0.34, up="#d43a3a", dn="#2f9e5e",
                 alpha=1.0, lw=0.8, zorder=3):
    """Draw candles (red up / green down by default; caller overrides colors)."""
    from matplotlib.patches import Rectangle
    for xi, oi, hi, li, ci in zip(xs, o, h, l, c):
        color = up if ci >= oi else dn
        ax.plot([xi, xi], [li, hi], color=color, lw=lw, alpha=alpha, zorder=zorder,
                solid_capstyle="butt", snap=False)
        span = max(abs(ci - oi), (hi - li) * 1e-3 + 1e-12)   # keep doji visible
        ax.add_patch(Rectangle((xi - width / 2, min(oi, ci)), width, span, facecolor=color,
                               edgecolor=color, lw=0.4, alpha=alpha, zorder=zorder, snap=False))


def plot_case(ax_p, ax_v, res: dict, ctx_show: int) -> None:
    """Price panel + volume panel. Gray=recent history, red/green=actual future, blue/purple=representative predicted path, blue band=predicted close quantiles."""
    ctx = res["ctx_tail"].tail(ctx_show)
    fut, pred = res["fut"], res["pred"]
    n_ctx, n_fut = len(ctx), len(fut)
    x_ctx = np.arange(n_ctx)
    x_fut = np.arange(n_ctx, n_ctx + n_fut)
    k = res["k_rep"]

    q05, q25, q50, q75, q95 = np.quantile(pred["close"], [.05, .25, .5, .75, .95], axis=0)
    draw_candles(ax_p, x_ctx, ctx["open"], ctx["high"], ctx["low"], ctx["close"],
                 up="#8a8a8a", dn="#b9b9b9", alpha=0.9)
    ax_p.fill_between(x_fut, q05, q95, color="#1f77b4", alpha=0.10, zorder=1, label="pred close q05-q95")
    ax_p.fill_between(x_fut, q25, q75, color="#1f77b4", alpha=0.18, zorder=1, label="pred close q25-q75")
    ax_p.plot(x_fut, q50, color="#1f77b4", lw=1.0, ls="--", alpha=0.8, zorder=2, label="pred close median")
    draw_candles(ax_p, x_fut - 0.21, fut["open"], fut["high"], fut["low"], fut["close"])
    draw_candles(ax_p, x_fut + 0.21, pred["open"][k], pred["high"][k], pred["low"][k], pred["close"][k],
                 up="#2b6cb0", dn="#9f5fd4", alpha=0.95)
    ax_p.axvline(n_ctx - 0.5, color="k", ls=":", lw=1.0)
    ax_p.set_xlim(-1, n_ctx + n_fut)
    ax_p.set_title(f"{res['code']} {res['scale']}  anchor={res['anchor']}   "
                   f"actual {res['act_ret']:+.2%} / pred median {res['pred_ret_med']:+.2%} "
                   f"(actual @ p{res['act_pct']:.0f})", fontsize=10)
    ax_p.grid(alpha=0.25)
    ax_p.tick_params(labelbottom=False)
    ax_p.legend(loc="upper left", fontsize=8, framealpha=0.9)

    ax_v.bar(x_ctx, ctx["volume"], width=0.6, color="#bdbdbd", alpha=0.8)
    ax_v.bar(x_fut - 0.21, fut["volume"], width=0.38, color="#d43a3a", alpha=0.85, label="actual vol")
    ax_v.bar(x_fut + 0.21, pred["volume"][k], width=0.38, color="#2b6cb0", alpha=0.85,
             label="pred vol (representative path)")
    ax_v.axvline(n_ctx - 0.5, color="k", ls=":", lw=1.0)
    ax_v.set_xlim(-1, n_ctx + n_fut)
    ax_v.grid(alpha=0.25)
    ax_v.set_yticks([])
    ax_v.legend(loc="upper left", fontsize=8, framealpha=0.9, ncol=2)

    all_ts = list(ctx["datetime"]) + list(fut["datetime"])
    ticks = np.linspace(0, len(all_ts) - 1, 6).astype(int)
    fmt = "%Y-%m-%d" if res["scale"] == "1d" else "%m-%d %H:%M"
    ax_v.set_xticks(ticks)
    ax_v.set_xticklabels([pd.Timestamp(all_ts[i]).strftime(fmt) for i in ticks], fontsize=8, rotation=20)


def setup_font() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False


def render(results: list[dict], out_dir: Path, ctx_show: int, suptitle: str) -> None:
    import matplotlib.pyplot as plt
    for r in results:
        fig = plt.figure(figsize=(11, 6.5))
        inner = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
        ax_p = fig.add_subplot(inner[0])
        ax_v = fig.add_subplot(inner[1], sharex=ax_p)
        plot_case(ax_p, ax_v, r, ctx_show)
        fig.suptitle(suptitle, fontsize=9, y=0.995)
        fig.savefig(r["png"], dpi=140, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(
        description="KiT inference: predict the K-line segment after a given instrument/scale/time point and plot it",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="--case format: code,scale,anchor  e.g. --case 601899.SH,5m,2026-03-04  "
               "--case 300750.SZ,30m,\"2026-03-10 11:30\" (repeatable; if omitted, the seven built-in cases are run)")
    ap.add_argument("--case", action="append", default=None, metavar="code,scale,anchor",
                    help="position to predict, repeatable. anchor = end of the history window; a date alone means that day's close")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "ckpt" / "step_market_284M.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "infer_for_pred")
    ap.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True,
                    help="whether to plot (default on, --no-plot disables; matplotlib not needed then)")
    ap.add_argument("--ctx-show", type=int, default=60, help="number of trailing history bars shown in the plot")
    ap.add_argument("--ctx-len", default="",
                    help="for validation: scale:Lc,... Must equal the model window, otherwise an error is raised (the model supports only its training windows)")
    ap.add_argument("--pred-len", default="", help="for validation: scale:Lh,... Same as above")
    ap.add_argument("--paths", type=int, default=16, help="number of sampled paths K per case (same as infer_for_eval)")
    ap.add_argument("--steps", type=int, default=32, help="number of integration steps (same as infer_for_eval)")
    ap.add_argument("--solver", choices=("euler", "heun"), default="euler")
    ap.add_argument("--cfg-w", type=float, default=1.0, help="identity CFG weight, 1.0=off (hurts calibration, use with care)")
    ap.add_argument("--hist-cfg-w", type=float, default=1.0, help="history CFG weight, 1.0=off")
    ap.add_argument("--precision", choices=("fp32", "bf16", "fp16"), default=None,
                    help="forward precision, default bf16 on CUDA (same as infer_for_eval), fp32 on CPU")
    ap.add_argument("--chunk", type=int, default=0, help="forward batch chunk size (sequences), 0=auto by GPU memory")
    ap.add_argument("--seed", type=int, default=0, help="random seed; case i uses seed+i")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cases = [parse_case(c) for c in (args.case or DEFAULT_CASES)]
    ctx_req = parse_len_table(args.ctx_len, "--ctx-len")
    pred_req = parse_len_table(args.pred_len, "--pred-len")

    from kit_v1.data.calendar import bars_per_day, load_trading_days
    from kit_v1.data.normalize import load_stats
    from kit_v1.infer import (denormalize_paths, flow_sample, load_model_and_config,
                                 paths_to_ohlcv, prepare_inference_window)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is False")
    precision = args.precision or ("bf16" if device.type == "cuda" else "fp32")
    if args.chunk <= 0:
        gb = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else 0
        args.chunk = 256 if gb >= 70 else 128 if gb >= 38 else 48 if gb >= 15 else 16

    data_root = args.data_root
    instruments = json.loads((data_root / "instruments.json").read_text(encoding="utf-8"))
    stats = load_stats(data_root / "stats.json")
    tdays = load_trading_days(data_root / "trade_dates.csv")
    if tdays is None:
        raise SystemExit(f"missing trading-day table: {data_root / 'trade_dates.csv'}")

    model, cfg = load_model_and_config(str(args.ckpt), "tiny", args.seed)

    # validate all cases before loading weights
    plans = []
    for code_, scale, end, date_only in cases:
        if code_ not in instruments:
            raise SystemExit(f"{code_} is not in {data_root / 'instruments.json'} (must be one of the {len(instruments)} stocks with local data)")
        Lc, Lh = cfg.data.window_for(scale)
        if scale in ctx_req and ctx_req[scale] != Lc:
            raise SystemExit(f"--ctx-len {scale}:{ctx_req[scale]} does not match the model: the {scale} history window must be {Lc} bars")
        if scale in pred_req and pred_req[scale] != Lh:
            raise SystemExit(f"--pred-len {scale}:{pred_req[scale]} does not match the model: the {scale} prediction window must be {Lh} bars")
        keep = Lc + int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days * bars_per_day(MARKET, scale))) + 8
        hist, fut = load_case_frame(data_root, code_, scale, end, Lc, Lh, keep)
        plans.append((code_, scale, end, Lc, Lh, hist, fut))

    model.to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"device={device} precision={precision} K={args.paths} steps={args.steps} solver={args.solver} "
          f"cfg_w={args.cfg_w} hist_cfg_w={args.hist_cfg_w} ckpt={args.ckpt.name}", flush=True)

    results, rows = [], []
    for i, (code_, scale, end, Lc, Lh, hist, fut) in enumerate(plans):
        t1 = time.time()
        inst = instruments[code_]
        win = prepare_inference_window(
            hist, MARKET, scale, Lc=Lc, Lh=Lh, stats=stats, soft_clip_c=cfg.data.soft_clip_c,
            half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
            sector_id=int(inst.get("sector", 0)), inst_id=int(inst.get("vocab_index", 0)),
            trading_days=tdays)
        if win["n_pad"] != 0:
            raise SystemExit(f"{code_} {scale}: valid history {Lc - win['n_pad']} bars < {Lc}, padding not allowed")
        # future timestamps must match the actual following bars
        if not pd.DatetimeIndex(fut["datetime"]).equals(pd.DatetimeIndex(win["future_timestamps"])):
            raise SystemExit(
                f"{code_} {scale} anchor {end}: timestamps of the real following bars differ from the calendar projection (possible suspension/missing bars in the range); "
                "this window is unsuitable for prediction, choose another anchor")
        paths = flow_sample(
            model, win["x_ctx"], win["cond"], Lh=Lh, steps=args.steps, solver=args.solver,
            n_paths=args.paths, cfg_w=args.cfg_w, hist_cfg_w=args.hist_cfg_w,
            generator=torch.Generator().manual_seed(args.seed + i), device=device,
            chunk_size=args.chunk, precision=precision)
        feats = denormalize_paths(paths, MARKET, scale, stats, c=cfg.data.soft_clip_c)[0]   # [K,Lh,5]
        pred = paths_to_ohlcv(feats, win["prev_close"], win["prev_ema"], win["half_life_bars"])
        pred = {k: np.asarray(v) for k, v in pred.items()}

        prev_close = win["prev_close"]
        term = np.log(pred["close"][:, -1] / prev_close)                 # [K] terminal cumulative return
        act_ret = float(np.log(fut["close"].iloc[-1] / prev_close))
        k_rep = int(np.argsort(term)[len(term) // 2])                    # representative path = median terminal return
        pred_cum = np.log(pred["close"] / prev_close)                    # [K,Lh]
        act_cum = np.log(fut["close"].to_numpy(dtype=np.float64) / prev_close)
        med_cum = pred_cum.mean(0)
        curve_corr = float(np.corrcoef(act_cum, med_cum)[0, 1]) if act_cum.std() > 0 and med_cum.std() > 0 else float("nan")
        tag = f"{code_}_{scale}_{end.strftime('%Y%m%d')}" + ("" if end.hour == 23 else end.strftime("_%H%M"))
        res = dict(code=code_, scale=scale, anchor=str(win["ctx_timestamps"][-1]), Lc=Lc, Lh=Lh,
                   ctx_tail=hist.tail(max(args.ctx_show, 1)), fut=fut, pred=pred, k_rep=k_rep,
                   act_ret=act_ret, pred_ret_med=float(np.median(term)),
                   act_pct=float((term <= act_ret).mean() * 100), png=args.out / f"{tag}.png")
        results.append(res)
        np.savez_compressed(
            args.out / f"{tag}.npz", pred_feat=np.asarray(feats, dtype=np.float32),
            **{f"pred_{k}": v.astype(np.float32) for k, v in pred.items()},
            act_ohlcv=fut[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float32),
            fut_ts=pd.DatetimeIndex(fut["datetime"]).to_numpy(dtype="datetime64[ns]"),
            prev_close=np.float64(prev_close), k_rep=k_rep)
        rows.append(dict(
            code=code_, scale=scale, ctx_start=str(win["ctx_timestamps"][0]), ctx_end=res["anchor"],
            pred_start=str(fut["datetime"].iloc[0]), pred_end=str(fut["datetime"].iloc[-1]), Lc=Lc, Lh=Lh,
            act_ret=act_ret, pred_ret_mean=float(term.mean()), pred_ret_q05=float(np.quantile(term, .05)),
            pred_ret_q50=float(np.median(term)), pred_ret_q95=float(np.quantile(term, .95)),
            act_percentile=res["act_pct"], curve_corr=curve_corr,
            act_in_q05_q95=float(np.mean((act_cum >= np.quantile(pred_cum, .05, axis=0))
                                         & (act_cum <= np.quantile(pred_cum, .95, axis=0)))),
            seconds=round(time.time() - t1, 1)))
        r = rows[-1]
        print(f"[{i + 1}/{len(plans)}] {code_} {scale} history {r['ctx_start']} .. {r['ctx_end']} ({Lc} bars)  "
              f"predicted {r['pred_start']} .. {r['pred_end']} ({Lh} bars)\n"
              f"      terminal return actual {act_ret:+.2%}  predicted median {r['pred_ret_q50']:+.2%} "
              f"[q05 {r['pred_ret_q05']:+.2%}, q95 {r['pred_ret_q95']:+.2%}]  "
              f"cumulative-return curve corr {curve_corr:.3f}  {r['seconds']}s", flush=True)

    pd.DataFrame(rows).to_csv(args.out / "summary.csv", index=False)
    (args.out / "infer_meta.json").write_text(json.dumps({
        "ckpt": _rel(args.ckpt), "data_root": _rel(data_root), "cases": args.case or DEFAULT_CASES,
        "k_paths": args.paths, "steps": args.steps, "solver": args.solver, "precision": precision,
        "cfg_w": args.cfg_w, "hist_cfg_w": args.hist_cfg_w, "seed": args.seed, "device": str(device),
        "device_type": device.type,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.plot:
        setup_font()
        title = (f"{args.ckpt.name} | gray=history tail ({args.ctx_show})  red/green=actual  "
                 f"blue/purple=predicted representative path (median terminal of K={args.paths})  "
                 f"band=pred close quantiles")
        render(results, args.out, args.ctx_show, title)
        print(f"Plots saved to {args.out}")
    print(f"Done, results in {args.out}")


if __name__ == "__main__":
    main()
