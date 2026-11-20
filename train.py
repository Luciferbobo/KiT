#!/usr/bin/env python
"""KiT training entry point.

Examples
--------
Smoke test on CPU (tiny model, no real data needed beyond a small tree)::

    python train.py --preset tiny --data-root data --out-dir runs/tiny

Full model (single GPU)::

    python train.py --config configs/dit_300m.yaml \
        --data-root data --out-dir runs/dit_300m \
        --set data.train_end=2023-12-31 --set train.total_steps=400000

Multi-GPU / multi-node (FSDP)::

    torchrun --nproc_per_node=8 train.py --config configs/dit_300m.yaml \
        --data-root data --out-dir runs/dit_300m

Any field can be overridden with ``--set section.key=value``; see
``configs/dit_300m.yaml`` for the full list of knobs.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

# import torch before anything that pulls in pandas: some environments fail to
# load the torch DLLs when pandas was imported first.
import torch  # noqa: F401,E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from kit_v1.config import Config, get_config  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Train KiT (K-line Diffusion Transformer) with flow matching")
    ap.add_argument("--config", default=None,
                    help="YAML config, e.g. configs/dit_300m.yaml")
    ap.add_argument("--preset", default=None, choices=["tiny"],
                    help="built-in preset for smoke tests (tiny)")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value",
                    help="override a config field (repeatable)")
    ap.add_argument("--data-root", type=Path, default=None,
                    help="data directory (instruments.json/stats.json/<market>/<scale>)")
    ap.add_argument("--out-dir", default=None, help="checkpoint / log directory")
    ap.add_argument("--resume", default=None, help="checkpoint path to resume from")
    ap.add_argument("--num-workers", type=int, default=0,
                    help="DataLoader workers (0 = load in the main process)")
    ap.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return ap.parse_args(argv)


def build_config(args) -> Config:
    if args.preset:
        cfg = get_config(args.preset)
        if args.config:
            cfg = Config.load_yaml(args.config, base=cfg)
    elif args.config:
        cfg = Config.load_yaml(args.config)
    else:
        raise SystemExit("provide --config configs/dit_300m.yaml (or --preset tiny)")
    if args.overrides:
        cfg.apply_overrides(args.overrides)
    return cfg


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    args = parse_args(argv)
    cfg = build_config(args)
    if args.data_root is not None:
        cfg.data.data_root = str(args.data_root)
    if args.out_dir is not None:
        cfg.train.out_dir = args.out_dir
    if args.resume is not None:
        cfg.train.resume = args.resume
    cfg.validate()

    device = None
    if args.device is not None:
        import torch
        device = torch.device(args.device)

    from kit_v1.train import train
    train(cfg, data_root=None, num_workers=int(args.num_workers), device=device)


if __name__ == "__main__":
    main()
    # force a clean process exit: joining the OpenMP/autograd thread pools at
    # interpreter shutdown can hang on some Windows/WDDM systems after a backward.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
