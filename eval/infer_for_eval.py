#!/usr/bin/env python
"""KiT inference on 100 A-share stocks over all anchors in the val window; saves full per-anchor results.

Inference only, no metrics (see cal_metrics.py). Defaults run the full set:
100 stocks x 7 scales x val window 2026-01-01 .. 2026-04-10, K=16 paths, 32 Euler steps, bf16.

Usage:
  python eval/infer_for_eval.py

Re-running the same command skips finished anchors, so it resumes after an interruption.

Anchors: per scale, one anchor every ceil(Lh / bars_per_day) trading days, taken at the day's close.
The last Lh worth of days is left so the future window stays inside the val window.

One npz per anchor: results/infer_for_eval/raw/<scale>/<scale>_<YYYYMMDD>_k16_s32.npz
  inst         [N]          stock codes
  prev_close   [N]          close of the last context bar
  prev_ema     [N]          volume EMA at the first future bar
  half_life_bars            volume EMA half-life (bars)
  pred_feat    [N,K,Lh,5]   denormalized features r_gap,r_body,r_up,r_dn,v (float16)
  pred_bar_ret [N,K,Lh]     per-bar log return = r_gap + r_body (float32)
  act_ohlcv    [N,Lh,5]     actual future open,high,low,close,volume (float32, adjusted)
  act_bar_ret  [N,Lh]       actual per-bar log return
  fut_ts       [N,Lh]       future bar timestamps (datetime64[ns])
Stocks with missing bars on the anchor day, fewer than Lh future bars, or bad data are dropped from that anchor.
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


def plan_anchors(days: np.ndarray, lo: str, hi: str, horizon_days: int, stride: int) -> list[str]:
    """Pick one anchor every `stride` trading days in [lo, hi], leaving horizon_days at the end."""
    span = days[(days >= np.datetime64(lo)) & (days <= np.datetime64(hi))]
    usable = span[:-horizon_days] if 0 < horizon_days < len(span) else span[:0]
    return [str(d) for d in usable[::max(1, int(stride))]]


def load_frames(data_root: Path, codes: list[str], scale: str,
                earliest_end: pd.Timestamp, keep_bars: int) -> dict[str, pd.DataFrame]:
    """Load each stock's parquet; keep keep_bars before the first anchor and everything after."""
    cols = ["datetime", "open", "high", "low", "close", "volume"]
    out = {}
    for code in codes:
        df = pd.read_parquet(data_root / MARKET / scale / f"{code}.parquet", columns=cols)
        ts = pd.DatetimeIndex(df["datetime"])
        hi = int(ts.searchsorted(earliest_end, side="right"))
        out[code] = df.iloc[max(0, hi - int(keep_bars)):].reset_index(drop=True)
    return out


def run_anchor(model, frames, instruments, codes, scale, anchor, Lc, Lh, stats, cfg,
               tdays, val_hi, k_paths, steps, chunk, seed, device):
    from kit_v1.infer import denormalize_paths, flow_sample, prepare_inference_window

    end = pd.Timestamp(anchor) + pd.Timedelta(hours=15)
    hi = pd.Timestamp(val_hi).normalize()
    wins = []
    for code in codes:
        df = frames[code]
        try:
            win = prepare_inference_window(
                df, MARKET, scale, Lc=Lc, Lh=Lh, stats=stats,
                soft_clip_c=cfg.data.soft_clip_c,
                half_life_days=cfg.data.vol_ema_half_life_days, end_ts=end,
                sector_id=int(instruments[code].get("sector", 0)),
                inst_id=int(instruments[code].get("vocab_index", 0)),
                trading_days=tdays,
            )
        except ValueError:
            continue
        ctx_last = pd.Timestamp(win["ctx_timestamps"][-1])
        if ctx_last.normalize() != pd.Timestamp(anchor):
            continue                        # no bars on the anchor day (e.g. suspended)
        fut = df.loc[pd.DatetimeIndex(df["datetime"]) > ctx_last].head(Lh)
        if len(fut) < Lh or pd.Timestamp(fut["datetime"].iloc[-1]).normalize() > hi:
            continue
        prev_close = float(win["prev_close"])
        closes = fut["close"].to_numpy(dtype=np.float64)
        if prev_close <= 0 or np.any(closes <= 0) or not np.isfinite(closes).all():
            continue
        act_bar_ret = np.diff(np.log(np.concatenate([[prev_close], closes])))
        if not np.isfinite(act_bar_ret).all():
            continue
        wins.append((code, win, fut, act_bar_ret))
    if not wins:
        return None

    x = torch.cat([w["x_ctx"] for _, w, _, _ in wins], dim=0)
    cond = {k: torch.cat([w["cond"][k] for _, w, _, _ in wins], dim=0)
            for k in wins[0][1]["cond"]}
    paths = flow_sample(
        model, x, cond, Lh=Lh, steps=int(steps), solver="euler", n_paths=int(k_paths),
        generator=torch.Generator().manual_seed(int(seed)), device=device,
        chunk_size=int(chunk), precision="bf16",
    )
    feats = denormalize_paths(paths, MARKET, scale, stats, c=cfg.data.soft_clip_c)  # [N,K,Lh,5]
    ohlcv = ["open", "high", "low", "close", "volume"]
    return {
        "inst": np.array([c for c, _, _, _ in wins]),
        "prev_close": np.array([w["prev_close"] for _, w, _, _ in wins], dtype=np.float64),
        "prev_ema": np.array([w["prev_ema"] for _, w, _, _ in wins], dtype=np.float64),
        "half_life_bars": np.float64(wins[0][1]["half_life_bars"]),
        "pred_feat": np.asarray(feats, dtype=np.float16),
        "pred_bar_ret": np.asarray(feats[..., 0] + feats[..., 1], dtype=np.float32),
        "act_ohlcv": np.stack([f[ohlcv].to_numpy(dtype=np.float32) for _, _, f, _ in wins]),
        "act_bar_ret": np.stack([a for _, _, _, a in wins]).astype(np.float32),
        "fut_ts": np.stack([pd.DatetimeIndex(f["datetime"]).to_numpy(dtype="datetime64[ns]")
                            for _, _, f, _ in wins]),
    }


def save_npz(path: Path, row: dict) -> None:
    """Write to a temp file, then replace atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **row)
    os.replace(tmp, path)


def is_valid_npz(path: Path) -> bool:
    try:
        with np.load(path) as z:
            return "pred_bar_ret" in z.files
    except Exception:
        return False


def fmt_dur(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="KiT inference: all anchors in the val window, saving per-anchor results")
    ap.add_argument("--data-root", type=Path, default=ROOT / "data")
    ap.add_argument("--ckpt", type=Path, default=ROOT / "ckpt" / "step_market_284M.pt")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "infer_for_eval")
    ap.add_argument("--scales", default=",".join(SCALES))
    ap.add_argument("--val-lo", default="2026-01-01", help="val window start")
    ap.add_argument("--val-hi", default="2026-04-10", help="val window end (the future window must also fall within it)")
    ap.add_argument("--anchor-stride", default="auto",
                    help="anchor stride (trading days). auto=ceil(Lh/bars per day), tiling non-overlapping prediction windows; an integer is also accepted")
    ap.add_argument("--max-anchors", type=int, default=0, help="take at most the first N anchors per scale, 0 = all (for smoke tests)")
    ap.add_argument("--paths", type=int, default=16, help="number of sampled paths K per sample")
    ap.add_argument("--steps", type=int, default=32, help="number of Euler integration steps")
    ap.add_argument("--chunk", type=int, default=0,
                    help="forward batch chunk size (sequences). 0=auto by GPU memory (8GB:16, 16GB:48, 40GB:128, 70GB+:256)")
    ap.add_argument("--max-inst", type=int, default=0, help="take only the first N instruments, 0 = all (for smoke tests)")
    ap.add_argument("--shard", default="0/1",
                    help="shard i/n: split all anchor jobs into n cost-balanced parts; this process runs only part i (for multi-GPU)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from kit_v1.data.calendar import bars_per_day, load_trading_days
    from kit_v1.data.normalize import load_stats
    from kit_v1.infer import load_model_and_config

    scales = tuple(s.strip() for s in args.scales.split(",") if s.strip())
    bad = [s for s in scales if s not in SCALES]
    if bad:
        raise SystemExit(f"unknown scales: {bad}; choices: {SCALES}")

    data_root = args.data_root
    instruments = json.loads((data_root / "instruments.json").read_text(encoding="utf-8"))
    codes = [c for c in sorted(instruments)
             if (data_root / MARKET / "1d" / f"{c}.parquet").is_file()]
    if args.max_inst > 0:
        codes = codes[: args.max_inst]
    if not codes:
        raise SystemExit(f"{data_root} has no usable instruments")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested, but torch.cuda.is_available() is False")

    t0 = time.time()
    model, cfg = load_model_and_config(str(args.ckpt), "tiny", args.seed)
    model.to(device).eval()
    stats = load_stats(data_root / "stats.json")
    tdays = load_trading_days(data_root / "trade_dates.csv")
    if tdays is None:
        raise SystemExit(f"missing trading-day table: {data_root / 'trade_dates.csv'}")

    if args.chunk <= 0:
        gb = torch.cuda.get_device_properties(device).total_memory / 2**30 if device.type == "cuda" else 0
        args.chunk = 256 if gb >= 70 else 128 if gb >= 38 else 48 if gb >= 15 else 16
    K, S = int(args.paths), int(args.steps)
    print(f"device={device} inst={len(codes)} K={K} steps={S} chunk={args.chunk} "
          f"ckpt={args.ckpt.name} val={args.val_lo}..{args.val_hi}", flush=True)

    jobs = []   # (scale, anchor, ai, Lc, Lh)
    for scale in scales:
        Lc, Lh = cfg.data.window_for(scale)
        hdays = int(math.ceil(Lh / bars_per_day(MARKET, scale)))
        stride = hdays if args.anchor_stride == "auto" else int(args.anchor_stride)
        anchors = plan_anchors(tdays, args.val_lo, args.val_hi, hdays, stride)
        if args.max_anchors > 0:
            anchors = anchors[: args.max_anchors]
        if not anchors:
            raise SystemExit(f"{scale} has no usable anchors within {args.val_lo}..{args.val_hi}")
        print(f"  {scale:3s} Lc={Lc} Lh={Lh} anchors {len(anchors)} (stride={stride}d)")
        jobs += [(scale, a, i, Lc, Lh) for i, a in enumerate(anchors)]

    si, sn = (int(x) for x in args.shard.split("/"))
    if sn > 1:
        # greedy split by sequence length (cost ~ Lc+Lh) to balance shards
        load, owner = [0.0] * sn, {}
        for j in sorted(jobs, key=lambda j: -(j[3] + j[4])):
            m = load.index(min(load))
            owner[j] = m
            load[m] += j[3] + j[4]
        jobs = [j for j in jobs if owner[j] == si]
        print(f"shard {si}/{sn}: {len(jobs)} anchor jobs", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / (f"infer_meta_shard{si}.json" if sn > 1 else "infer_meta.json")).write_text(json.dumps({
        "ckpt": _rel(args.ckpt), "ckpt_name": args.ckpt.name, "data_root": _rel(data_root),
        "market": MARKET, "codes": codes, "scales": scales,
        "val_lo": args.val_lo, "val_hi": args.val_hi, "anchor_stride": args.anchor_stride,
        "k_paths": K, "steps": S, "solver": "euler", "precision": "bf16",
        "seed": args.seed, "train_end": cfg.data.train_end, "val_start": cfg.data.val_start,
        "n_jobs_this_shard": len(jobs), "shard": args.shard, "device": str(device),
        "device_type": device.type,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    def npz_path(scale, anchor):
        return args.out / "raw" / scale / f"{scale}_{anchor.replace('-', '')}_k{K}_s{S}.npz"

    todo = [j for j in jobs if not (npz_path(j[0], j[1]).is_file() and is_valid_npz(npz_path(j[0], j[1])))]
    print(f"{len(jobs)} anchor jobs in total, {len(jobs) - len(todo)} done, {len(todo)} to run", flush=True)

    loaded_scale, frames = None, {}
    t_run, done = time.time(), 0
    for ji, (scale, anchor, ai, Lc, Lh) in enumerate(jobs, 1):
        path = npz_path(scale, anchor)
        if (scale, anchor, ai, Lc, Lh) not in todo:
            continue
        if loaded_scale != scale:
            earliest = min(pd.Timestamp(j[1]) for j in todo if j[0] == scale)
            keep = Lc + int(math.ceil(12.0 * cfg.data.vol_ema_half_life_days
                                      * bars_per_day(MARKET, scale))) + 8
            print(f"load {scale} keep_bars={keep} codes={len(codes)}", flush=True)
            frames = load_frames(data_root, codes, scale, earliest, keep)
            loaded_scale = scale
        seed = args.seed + 10007 * (SCALES.index(scale) + 1) + ai
        t1 = time.time()
        row = run_anchor(model, frames, instruments, codes, scale, anchor, Lc, Lh, stats, cfg,
                         tdays, args.val_hi, K, S, args.chunk, seed, device)
        if row is None:
            print(f"[{ji}/{len(jobs)}] {scale} {anchor} no valid samples, skipped", flush=True)
            continue
        save_npz(path, row)
        done += 1
        avg = (time.time() - t_run) / done
        print(f"[{ji}/{len(jobs)}] {scale} {anchor} n={len(row['inst'])} "
              f"{time.time() - t1:.1f}s  ran {done}/{len(todo)} "
              f"ETA {fmt_dur(avg * (len(todo) - done))}", flush=True)

    print(f"\nInference done, results in {args.out / 'raw'}, took {fmt_dur(time.time() - t0)}")
    print(f"Next: python eval/cal_metrics.py --results {args.out}")


if __name__ == "__main__":
    main()
