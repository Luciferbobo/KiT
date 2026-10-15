#!/usr/bin/env python
"""Aggregate a raw 1m (or other fine-scale) tree into the KiT timescales.

Thin CLI around :func:`kit_v1.data.aggregate.aggregate_tree`.  Writes only the
parquet tree; metadata (instruments/stats/calendar) is built by
``scripts/prepare_data.py``.

Example::

    python scripts/aggregate_1m.py --raw-root raw/ashare_1m --out data --market ashare
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.data.aggregate import aggregate_tree  # noqa: E402
from kit_v1.data.calendar import MARKET_IDS, TIMESCALE_IDS  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate fine-scale bars into KiT timescales")
    ap.add_argument("--raw-root", type=Path, required=True,
                    help="directory with <code>.parquet at the source timescale")
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    ap.add_argument("--market", default="ashare", help=f"market key ({sorted(MARKET_IDS)})")
    ap.add_argument("--source-scale", default="1m")
    ap.add_argument("--scales", default="1m,5m,15m,30m,1h,2h,1d")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if args.market not in MARKET_IDS:
        raise SystemExit(f"unknown market {args.market!r}")
    scales = tuple(s.strip() for s in args.scales.split(",") if s.strip())
    bad = [s for s in scales if s not in TIMESCALE_IDS]
    if bad:
        raise SystemExit(f"unknown timescales: {bad}")
    codes = aggregate_tree(args.raw_root, args.out, args.market, scales,
                           source_timescale=args.source_scale, workers=args.workers,
                           overwrite=args.overwrite)
    print(f"aggregated {len(codes)} instruments into {args.out / args.market}")


if __name__ == "__main__":
    main()
