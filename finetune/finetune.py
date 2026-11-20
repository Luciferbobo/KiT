#!/usr/bin/env python
"""Fine-tune a released KiT checkpoint on custom data (LoRA / head / full).

The pretrained checkpoint is never overwritten: its architecture, normalisation
stats and instrument mapping are reused, and the fine-tuned weights are written
to ``--out-dir`` as standard checkpoints the eval scripts read directly.

Examples
--------
LoRA (recommended on a single consumer GPU)::

    python finetune/finetune.py \
        --ckpt ckpt/step_market_284M \
        --data-root data --out-dir runs/ft_lora \
        --mode lora --lora-rank 16 --lora-alpha 32 \
        --steps 2000 --lr 2e-4 --grad-accum 8

Full fine-tune of KiT-L on a single H200 (bf16, activations checkpointed)::

    python finetune/finetune.py \
        --ckpt ckpt/step_market_284M \
        --data-root data --out-dir runs/ft_full_L \
        --mode full --micro-batch 8 --grad-accum 8 \
        --steps 5000 --lr 1e-5

Small-GPU variant (8-bit Adam, micro-batch 1, coarse scales only)::

    python finetune/finetune.py --ckpt ckpt/step_market_284M --data-root data \
        --out-dir runs/ft_full_small --mode full --micro-batch 1 --grad-accum 16 \
        --optimizer adamw8bit --grad-checkpointing on --steps 5000 --lr 1e-5 \
        --set data.timescales=1h,2h,1d

Head-only (freeze the backbone, train the head + identity embeddings)::

    python finetune/finetune.py --ckpt ckpt/step_market_284M --data-root data \
        --out-dir runs/ft_head --mode head --steps 3000 --lr 5e-5

Any config field can be overridden with ``--set section.key=value``.
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import shutil
import sys
from pathlib import Path

# torch must load before pandas in some wheel combos
import torch  # noqa: F401,E402
import torch.nn as nn  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kit_v1.config import Config  # noqa: E402
from kit_v1.model import KlineDiT  # noqa: E402
from kit_v1.train import train  # noqa: E402
from kit_v1.train.checkpoint import save_checkpoint  # noqa: E402
from finetune.lora import (inject_lora, mark_only_lora_trainable,  # noqa: E402
                           merge_lora_state_dict)

DEFAULT_HEAD_MODULES = ("in_proj", "role_emb", "register", "calendar", "cond_emb",
                        "t_emb", "adaln", "final_norm", "head_bias", "out_proj")
DEFAULT_LORA_TARGETS = ("qkv", "out_proj", "w1", "w2", "w3")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def resolve_ckpt(path: Path):
    """Return (checkpoint file, bundle directory)."""
    path = Path(path)
    if path.is_dir():
        pts = sorted(path.glob("*.pt"))
        if len(pts) != 1:
            raise SystemExit(f"{path} should contain exactly one .pt, found {len(pts)}")
        return pts[0], path
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    return path, path.parent


def load_pretrained(model: nn.Module, ckpt_file: Path, use_ema: bool) -> int:
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if use_ema and "ema" in ckpt:
        sd = ckpt["ema"].get("shadow", ckpt["ema"])
        print("[ft] initialising from EMA weights")
    else:
        sd = ckpt["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ft] warning: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"[ft] warning: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    return int(ckpt.get("step", 0))


def backup_once(path: Path) -> None:
    """Keep a single .ftorig backup of a file the fine-tuner is about to replace."""
    if path.is_file():
        bak = path.with_name(path.name + ".ftorig")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"[ft] backed up {path.name} -> {bak.name}")


def merge_instruments(data_root: Path, bundle_path: Path | None, extend: bool,
                      model_vocab: int):
    """Write data_root/instruments.json combining the pretrained mapping with custom codes.

    Codes present in the pretrained bundle keep their trained ``vocab_index``.
    Unknown codes are mapped to slot 0 ([UNK]) unless ``extend`` is set, in which
    case they get fresh indices starting at ``base`` = max(model_vocab,
    bundle_max+1) so they never collide with existing embedding rows.

    Returns (n_new, base) where ``base`` is the first new embedding index.
    """
    custom_path = data_root / "instruments.json"
    custom = json.loads(custom_path.read_text(encoding="utf-8")) if custom_path.is_file() else {}
    base = int(model_vocab)
    if bundle_path is None or not Path(bundle_path).is_file():
        print("[ft] no pretrained instruments.json given; using the custom mapping as-is. "
              "If these codes are not the ones the checkpoint was trained on, the "
              "instrument conditioning will be wrong. Pass --instruments to fix.")
        return 0, base

    bundle = json.loads(Path(bundle_path).read_text(encoding="utf-8"))
    base = max(base, max([int(v.get("vocab_index", 0)) for v in bundle.values()] + [0]) + 1)
    merged = dict(bundle)
    new_codes = sorted(c for c in custom if c not in bundle)
    for i, code in enumerate(new_codes):
        entry = dict(custom[code])
        entry["vocab_index"] = (base + i) if extend else 0
        entry.setdefault("_finetune_new", True)
        merged[code] = entry
    backup_once(custom_path)
    custom_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[ft] instruments.json: {len(bundle)} pretrained + {len(new_codes)} new "
          f"({'extended vocab from ' + str(base) if extend else 'mapped to [UNK]'}) "
          f"= {len(merged)}")
    return len(new_codes), base


def expand_instrument_embedding(model: KlineDiT, cfg: Config, new_vocab: int) -> None:
    old = model.cond_emb.inst
    if new_vocab <= old.num_embeddings:
        return
    new = nn.Embedding(new_vocab, old.embedding_dim)
    nn.init.normal_(new.weight, std=0.02)
    with torch.no_grad():
        new.weight[:old.num_embeddings] = old.weight.data
        if old.num_embeddings > 1:
            mean = old.weight.data[1:].mean(0, keepdim=True)
            new.weight[old.num_embeddings:] = mean + 0.02 * torch.randn(
                new_vocab - old.num_embeddings, old.embedding_dim)
    model.cond_emb.inst = new
    cfg.model.inst_vocab = new_vocab
    print(f"[ft] instrument embedding expanded {old.num_embeddings} -> {new_vocab}")


def freeze_except(model: nn.Module, keep: tuple) -> int:
    keep = set(keep)
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if any(part in keep for part in n.split(".")):
            p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_optimizer_factory(name: str, cfg: Config):
    name = (name or "adamw").lower()
    if name == "adamw":
        return None
    if name == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except Exception as e:  # pragma: no cover
            raise SystemExit(
                "--optimizer adamw8bit needs bitsandbytes (pip install bitsandbytes): "
                f"{e!r}")
        t = cfg.train

        def factory(decay, no_decay):
            return bnb.optim.AdamW8bit(
                [{"params": decay, "weight_decay": t.weight_decay},
                 {"params": no_decay, "weight_decay": 0.0}],
                lr=t.lr, betas=(t.beta1, t.beta2), eps=t.eps)
        return factory
    raise SystemExit(f"unknown --optimizer {name!r} (adamw | adamw8bit)")


def make_lora_save_hook(export_dir: Path):
    export_dir = Path(export_dir)

    def hook(path, model, ema, cfg, step, optimizer):
        # resume-able checkpoint (keeps the LoRA adapters)
        save_checkpoint(path, model, ema, cfg, step, optimizer)
        # merged export that the eval scripts can load
        ck = {
            "model": {k: v.detach().cpu() for k, v in
                      merge_lora_state_dict(model.state_dict()).items()},
            "ema": {"shadow": {k: v.detach().float().cpu() for k, v in
                               merge_lora_state_dict(ema.state_dict()).items()}},
            "config": cfg.to_dict(),
            "step": int(step),
        }
        export_dir.mkdir(parents=True, exist_ok=True)
        for dst in (export_dir / f"step_{step:07d}.pt", export_dir / "latest.pt"):
            tmp = dst.with_name(dst.name + ".tmp")
            torch.save(ck, tmp)
            os.replace(tmp, dst)
    return hook


def mem_estimate(n_params: int, n_trainable: int, optimizer: str):
    params = n_params * 4 / 1e9            # fp32 master params (default dtype)
    grads = n_trainable * 4 / 1e9
    opt = n_trainable * (2 if optimizer == "adamw8bit" else 8) / 1e9
    return params, grads, opt


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def build_cfg(args) -> Config:
    ckpt_file, bundle_dir = resolve_ckpt(Path(args.ckpt))
    ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
    if "config" not in ckpt:
        raise SystemExit(f"{ckpt_file} has no 'config'; pass a KiT checkpoint")
    base = Config.from_dict(ckpt["config"])
    cfg = Config.load_yaml(args.config, base=base) if args.config else base
    if args.overrides:
        cfg.apply_overrides(args.overrides)

    # fine-tuning overrides (only set fields that were explicitly provided; the
    # per-mode defaults are applied in main() before global_batch is computed)
    cfg.data.data_root = str(args.data_root)
    cfg.train.out_dir = str(args.out_dir)
    cfg.train.parallel = args.parallel
    cfg.train.seed = args.seed
    cfg.train.total_steps = args.steps
    if args.lr is not None:
        cfg.train.lr = args.lr
    cfg.train.warmup_steps = args.warmup
    cfg.train.decay_frac = args.decay_frac
    if args.micro_batch is not None:
        cfg.train.micro_batch = args.micro_batch
    if args.precision:
        cfg.train.precision = args.precision
    cfg.train.log_every = args.log_every
    cfg.train.val_every = args.val_every
    cfg.train.val_batches = args.val_batches
    cfg.train.ckpt_every = args.ckpt_every
    # never inherit train.resume from the checkpoint's saved config
    cfg.train.resume = args.resume or ""
    if args.grad_checkpointing == "on":
        cfg.model.use_activation_checkpointing = True
    elif args.grad_checkpointing == "off":
        cfg.model.use_activation_checkpointing = False
    elif args.mode in ("full", "head"):
        cfg.model.use_activation_checkpointing = True
    cfg.validate()
    return cfg, ckpt_file, bundle_dir


def main(argv=None) -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="Fine-tune a released KiT checkpoint")
    ap.add_argument("--ckpt", required=True,
                    help="checkpoint .pt or the bundle directory (with instruments/stats)")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="custom data dir built by scripts/prepare_data.py")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["lora", "head", "full"], default="lora")

    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=float, default=32.0)
    ap.add_argument("--lora-targets", default=",".join(DEFAULT_LORA_TARGETS))
    ap.add_argument("--lora-last-n", type=int, default=0, help="adapt only the last N blocks")
    ap.add_argument("--head-modules", default=",".join(DEFAULT_HEAD_MODULES))
    ap.add_argument("--freeze-inst-embedding", action="store_true",
                    help="do not train the instrument embedding in lora mode")

    ap.add_argument("--instruments", type=Path, default=None,
                    help="pretrained instruments.json (default: <bundle>/instruments.json)")
    ap.add_argument("--stats", type=Path, default=None,
                    help="pretrained stats.json (default: <bundle>/stats.json)")
    ap.add_argument("--no-stats-copy", action="store_true",
                    help="do not copy the pretrained stats.json into --data-root")
    ap.add_argument("--extend-instruments", action="store_true",
                    help="give new instruments fresh embedding rows instead of [UNK]")
    ap.add_argument("--use-ema-init", action="store_true",
                    help="start from the checkpoint's EMA weights instead of the raw ones")

    ap.add_argument("--config", default=None, help="optional YAML overlay")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    metavar="section.key=value")

    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--decay-frac", type=float, default=0.2)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    ap.add_argument("--optimizer", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--parallel", choices=["none", "ddp", "fsdp"], default="none")
    ap.add_argument("--grad-checkpointing", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=14)
    ap.add_argument("--ckpt-every", type=int, default=200)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (params / memory / data) and exit")
    args = ap.parse_args(argv)

    cfg, ckpt_file, bundle_dir = build_cfg(args)

    # sensible per-mode defaults (applied before global_batch is computed)
    if args.lr is None:
        cfg.train.lr = 2e-4 if args.mode == "lora" else 1e-5
    if args.micro_batch is None:
        # targeted at a single H200; pass --micro-batch / --grad-accum for smaller GPUs
        cfg.train.micro_batch = {"full": 8, "head": 8, "lora": 8}[args.mode]
    cfg.train.global_batch = int(cfg.train.micro_batch) * max(1, int(args.grad_accum))
    cfg.validate()

    # ----- normalisation stats (reuse the pretrained ones by default) -----
    if not args.no_stats_copy:
        stats = args.stats or (bundle_dir / "stats.json")
        if Path(stats).is_file():
            backup_once(args.data_root / "stats.json")
            shutil.copy2(stats, args.data_root / "stats.json")
            print(f"[ft] stats.json <- {stats}")

    # ----- model -----
    model = KlineDiT(cfg.model)
    pretrained_step = load_pretrained(model, ckpt_file, args.use_ema_init)

    # ----- instruments -----
    inst = args.instruments or (bundle_dir / "instruments.json")
    n_new, new_base = merge_instruments(args.data_root, inst, args.extend_instruments,
                                        int(cfg.model.inst_vocab))
    if args.extend_instruments and n_new > 0:
        expand_instrument_embedding(model, cfg, new_base + n_new)

    # ----- trainable parameters per mode -----
    if args.mode == "lora":
        targets = tuple(t for t in args.lora_targets.split(",") if t)
        wrapped = inject_lora(model, targets=targets, r=args.lora_rank,
                              alpha=args.lora_alpha, last_n_layers=args.lora_last_n,
                              include_ffn=False)
        n_train = mark_only_lora_trainable(model)
        if not args.freeze_inst_embedding:
            model.cond_emb.inst.weight.requires_grad = True
            n_train += model.cond_emb.inst.weight.numel()
        print(f"[ft] LoRA on {len(wrapped)} layers (r={args.lora_rank}, "
              f"alpha={args.lora_alpha}); trainable={n_train/1e6:.2f}M")
    elif args.mode == "head":
        keep = tuple(m.strip() for m in args.head_modules.split(",") if m.strip())
        n_train = freeze_except(model, keep)
        print(f"[ft] head-only; training {keep}; trainable={n_train/1e6:.2f}M")
    else:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[ft] full fine-tune; trainable={n_train/1e6:.2f}M")

    params_gb, grads_gb, opt_gb = mem_estimate(model.count_params(), n_train, args.optimizer)
    print(f"[ft] approx GPU state: params {params_gb:.2f}G + grads {grads_gb:.2f}G "
          f"+ optimizer {opt_gb:.2f}G = {params_gb+grads_gb+opt_gb:.2f}G "
          f"(+ activations; grad-checkpointing={'on' if cfg.model.use_activation_checkpointing else 'off'})")
    if args.mode == "full" and not cfg.model.use_activation_checkpointing:
        print("[warn] full fine-tune without gradient checkpointing needs a lot of memory; "
              "consider --grad-checkpointing on --optimizer adamw8bit --set data.timescales=1h,2h,1d")

    if args.dry_run:
        print("[ft] dry run; exiting before training")
        return

    # ----- optimizer + save hook -----
    factory = make_optimizer_factory(args.optimizer, cfg)
    save_hook = make_lora_save_hook(args.out_dir / "export") if args.mode == "lora" else None

    device = torch.device(args.device) if args.device else None
    print(f"[ft] fine-tuning from step {pretrained_step} for {cfg.train.total_steps} steps; "
          f"mode={args.mode}; out={cfg.train.out_dir}")
    train(cfg, data_root=None, num_workers=args.num_workers, device=device,
          model=model, save_hook=save_hook, optimizer_factory=factory)
    if args.mode == "lora":
        print(f"[ft] merged (eval-ready) checkpoints in {args.out_dir / 'export'}")


if __name__ == "__main__":
    main()
    # force a clean process exit (OpenMP/autograd thread join can hang on shutdown)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
