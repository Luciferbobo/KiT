#!/usr/bin/env python
"""Print a KiT checkpoint's config, parameter count and available metadata.

Useful before fine-tuning to confirm the architecture and which files the
bundle ships.

    python finetune/inspect_ckpt.py --ckpt ckpt/step_market_284M
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import sys
from pathlib import Path

import torch  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.config import Config  # noqa: E402
from kit_v1.model import KlineDiT  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Inspect a KiT checkpoint")
    ap.add_argument("--ckpt", required=True, help=".pt file or bundle directory")
    args = ap.parse_args(argv)

    path = Path(args.ckpt)
    bundle = path if path.is_dir() else path.parent
    pt = path
    if path.is_dir():
        pts = sorted(path.glob("*.pt"))
        if not pts:
            raise SystemExit(f"no .pt in {path}")
        pt = pts[0]

    ckpt = torch.load(pt, map_location="cpu", weights_only=False)
    print(f"checkpoint : {pt}")
    print(f"top-level  : {sorted(ckpt.keys())}")
    print(f"step       : {ckpt.get('step')}")
    print(f"has ema    : {'ema' in ckpt}")
    if "config" in ckpt:
        cfg = Config.from_dict(ckpt["config"])
        m = KlineDiT(cfg.model)
        print(f"params     : {m.count_params()/1e6:.2f}M")
        print(f"model      : d={cfg.model.d_model} layers={cfg.model.n_layers} "
              f"heads={cfg.model.n_heads} ffn={cfg.model.ffn_hidden} "
              f"inst_vocab={cfg.model.inst_vocab} registers={cfg.model.n_registers}")
        print(f"data wins  : 1m={cfg.data.window_for('1m')} 1d={cfg.data.window_for('1d')}")
    else:
        print("config     : (absent)")
    for name in ("instruments.json", "stats.json", "trade_dates.csv"):
        f = bundle / name
        n = ""
        if f.is_file() and f.suffix == ".json":
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                n = f" ({len(d)} entries)"
            except Exception:
                pass
        print(f"{name:14}: {'yes' if f.is_file() else 'no'}{n}")
    if "model" in ckpt:
        print(f"model keys : {len(ckpt['model'])} tensors; "
              f"first={next(iter(ckpt['model']))}")


if __name__ == "__main__":
    main()
