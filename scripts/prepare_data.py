#!/usr/bin/env python
"""Build the KiT training data tree from raw OHLCV.

Produces the layout consumed by the training/eval code::

    <out>/<market>/<timescale>/<code>.parquet
    <out>/instruments.json
    <out>/stats.json
    <out>/trade_dates.csv

Steps
-----
1. (optional) aggregate a raw fine-scale series (default 1m) into the seven
   timescales with :func:`kit_v1.data.aggregate.aggregate_tree`; use
   ``--no-aggregate`` when the raw root is already a multi-timescale tree.
2. ``instruments.json``: one entry per instrument with ``sector`` and a
   1-based ``vocab_index`` (0 is reserved for [UNK]).  Pass ``--instruments``
   to reuse a trained mapping (e.g. the one bundled with the checkpoint).
3. ``trade_dates.csv``: unique ``YYYYMMDD`` from the 1d bars.
4. ``stats.json``: per ``<market>|<timescale>|<feature>`` robust MAD scales
   fitted on rows up to ``--train-end`` (pass ``--stats`` to reuse a fitted
   file instead).

Examples
--------
::

    # from a directory of raw 1m parquet files
    python scripts/prepare_data.py --raw-root raw/ashare_1m --out data --market ashare

    # reuse the metadata bundled with a pre-trained checkpoint
    python scripts/prepare_data.py --raw-root raw/ashare_1m --out data \
        --instruments ckpt/step_market_284M/instruments.json \
        --stats ckpt/step_market_284M/stats.json
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.data.aggregate import aggregate_tree  # noqa: E402
from kit_v1.data.calendar import MARKET_IDS, TIMESCALE_IDS  # noqa: E402
from kit_v1.data.normalize import fit_stats, save_stats  # noqa: E402

DEFAULT_SCALES = ("1m", "5m", "15m", "30m", "1h", "2h", "1d")


def parse_scales(text: str):
    scales = tuple(s.strip() for s in text.split(",") if s.strip())
    bad = [s for s in scales if s not in TIMESCALE_IDS]
    if bad:
        raise SystemExit(f"unknown timescales: {bad}; choose from {sorted(TIMESCALE_IDS)}")
    return scales


def sync_tree(raw_base: Path, out_market_dir: Path, scales) -> int:
    """Copy an existing multi-timescale tree into ``out_market_dir``."""
    n = 0
    for ts in scales:
        src = raw_base / ts
        if not src.is_dir():
            continue
        dst = out_market_dir / ts
        dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(src.glob("*.parquet")):
            shutil.copy2(p, dst / p.name)
            n += 1
    return n


def find_existing_tree(raw_root: Path, market: str):
    """Locate an existing ``<market>/<scale>`` or ``<scale>`` tree."""
    if (raw_root / market).is_dir():
        return raw_root / market
    if any((raw_root / ts).is_dir() for ts in TIMESCALE_IDS):
        return raw_root
    return None


def build_instruments(out_market_dir: Path, out_path: Path,
                      instruments_path=None, sectors=None) -> dict:
    """Write ``instruments.json`` (reuse a trained mapping when provided)."""
    if instruments_path is not None:
        data = json.loads(Path(instruments_path).read_text(encoding="utf-8"))
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        return data

    codes = set()
    if out_market_dir.is_dir():
        for scale_dir in out_market_dir.iterdir():
            if scale_dir.is_dir():
                codes.update(p.stem for p in scale_dir.glob("*.parquet"))
    codes = sorted(codes)

    existing = {}
    if out_path.is_file():
        existing = json.loads(out_path.read_text(encoding="utf-8"))
    used = {int(v.get("vocab_index", 0)) for v in existing.values()} - {0}
    next_vocab = 1

    daily_dir = out_market_dir / "1d"
    out = {}
    for code in codes:
        entry = dict(existing.get(code, {}))
        if not entry.get("vocab_index"):
            while next_vocab in used:
                next_vocab += 1
            entry["vocab_index"] = next_vocab
            used.add(next_vocab)
            next_vocab += 1
        entry.setdefault("name", code)
        entry.setdefault("market", code.split(".")[-1] if "." in code else out_market_dir.name)
        entry.setdefault("sector", int(sectors.get(code, 0)) if sectors else 0)
        p = daily_dir / f"{code}.parquet"
        if p.is_file() and "list_date" not in entry:
            try:
                ts = pd.read_parquet(p, columns=["datetime"])["datetime"]
                if len(ts):
                    entry["list_date"] = str(pd.Timestamp(ts.iloc[0]).date())
            except Exception:
                pass
        out[code] = entry
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return out


def build_trade_dates(out_market_dir: Path, out_path: Path, market: str) -> int:
    """Write the trading calendar from the union of 1d bar dates."""
    days = set()
    daily = out_market_dir / "1d"
    if daily.is_dir():
        for p in daily.glob("*.parquet"):
            try:
                s = pd.read_parquet(p, columns=["datetime"])["datetime"]
            except Exception:
                continue
            days.update(pd.DatetimeIndex(s).normalize())
    if not days:
        return 0
    idx = pd.DatetimeIndex(sorted(days)).strftime("%Y%m%d")
    pd.DataFrame({"date": idx}).to_csv(out_path, index=False, encoding="utf-8-sig")
    return len(idx)


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Build the KiT data tree and metadata")
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="raw OHLCV root: flat <code>.parquet (source scale) or a <market>/<scale> tree")
    ap.add_argument("--out", type=Path, default=ROOT / "data", help="output data root")
    ap.add_argument("--market", default="ashare", help=f"market key ({sorted(MARKET_IDS)})")
    ap.add_argument("--source-scale", default="1m", help="source timescale for aggregation")
    ap.add_argument("--scales", default=",".join(DEFAULT_SCALES),
                    help="target timescales (comma separated)")
    ap.add_argument("--no-aggregate", action="store_true",
                    help="raw-root is already a multi-timescale tree; copy instead of aggregate")
    ap.add_argument("--workers", type=int, default=8, help="aggregation workers")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing parquet files")
    ap.add_argument("--instruments", type=Path, default=None,
                    help="existing instruments.json to reuse (keeps trained vocab_index/sector)")
    ap.add_argument("--stats", type=Path, default=None,
                    help="existing stats.json to reuse instead of refitting")
    ap.add_argument("--sectors", type=Path, default=None,
                    help="optional JSON {code: sector} used when building instruments.json")
    ap.add_argument("--train-end", default="2023-12-31",
                    help="fit normalisation stats on rows up to this date (inclusive)")
    ap.add_argument("--vol-halflife-days", type=float, default=5.0,
                    help="volume-EMA half-life in trading days for stats fitting")
    ap.add_argument("--skip-stats", action="store_true", help="do not build stats.json")
    args = ap.parse_args(argv)

    if args.market not in MARKET_IDS:
        raise SystemExit(f"unknown market {args.market!r}; choose from {sorted(MARKET_IDS)}")
    scales = parse_scales(args.scales)
    out = args.out
    out_market_dir = out / args.market
    out_market_dir.mkdir(parents=True, exist_ok=True)

    # 1. build / copy the parquet tree
    if args.no_aggregate:
        base = find_existing_tree(args.raw_root, args.market)
        if base is None:
            raise SystemExit(f"--no-aggregate but no <market>/<scale> tree found under {args.raw_root}")
        n = sync_tree(base, out_market_dir, scales)
        print(f"copied {n} parquet files into {out_market_dir}")
    else:
        codes = aggregate_tree(args.raw_root, out, args.market, scales,
                               source_timescale=args.source_scale,
                               workers=args.workers, overwrite=args.overwrite)
        print(f"aggregated {len(codes)} instruments x {len(scales)} timescales into {out_market_dir}")

    # 2. instruments.json
    sectors = None
    if args.sectors is not None:
        sectors = json.loads(Path(args.sectors).read_text(encoding="utf-8"))
    instruments = build_instruments(out_market_dir, out / "instruments.json",
                                    args.instruments, sectors)
    print(f"instruments.json: {len(instruments)} instruments")

    # 3. trade_dates.csv
    n_days = build_trade_dates(out_market_dir, out / "trade_dates.csv", args.market)
    print(f"trade_dates.csv: {n_days} trading days")

    # 4. stats.json
    if args.stats is not None:
        shutil.copy2(args.stats, out / "stats.json")
        print(f"stats.json: copied from {args.stats}")
    elif not args.skip_stats:
        stats = fit_stats(out, args.train_end, half_life_days=args.vol_halflife_days)
        save_stats(stats, out / "stats.json")
        print(f"stats.json: {len(stats)} buckets fitted on data <= {args.train_end}")

    print(f"done. data root: {out}")


if __name__ == "__main__":
    main()
