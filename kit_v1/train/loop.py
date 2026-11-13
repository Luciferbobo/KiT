"""Training loop (single-process, DDP or FSDP)."""
from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch

from kit_v1.config import Config
from kit_v1.data.loader import build_dataloader, set_loader_epoch
from kit_v1.model import KlineDiT
from kit_v1.train.checkpoint import load_checkpoint, save_checkpoint
from kit_v1.train.ema import EMA
from kit_v1.train.flow import flow_matching_loss
from kit_v1.train.parallel import (distributed_info, init_distributed, is_main,
                                   pick_device, unwrap, wrap_model)
from kit_v1.train.schedule import lr_at


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _autocast(precision: str, device: torch.device):
    if device.type == "cuda" and precision in ("bf16", "fp16"):
        dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    if device.type == "cpu" and precision == "bf16":
        return torch.autocast("cpu", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _make_scaler(precision: str, device: torch.device):
    enabled = precision == "fp16" and device.type == "cuda"
    try:
        return torch.amp.GradScaler(device.type, enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    batch["x_ctx"] = batch["x_ctx"].to(device, non_blocking=True)      # type: ignore[union-attr]
    batch["x_tgt"] = batch["x_tgt"].to(device, non_blocking=True)      # type: ignore[union-attr]
    batch["cond"] = {k: v.to(device, non_blocking=True)
                     for k, v in batch["cond"].items()}                # type: ignore[union-attr]
    return batch


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, device: torch.device,
             precision: str, max_batches: int) -> float:
    """Validation loss averaged **equally over (market, timescale) buckets**.

    Windows are scale-homogeneous, so each batch belongs to a single bucket; the
    per-bucket losses are averaged and then averaged over buckets, following the
    paper's "averaged equally over the monitored (market, timescale) buckets".
    """
    model.eval()
    buckets: Dict[str, list] = {}
    it = iter(loader)
    for _ in range(max_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        bucket = f"{batch['market'][0]}|{batch['scale'][0]}"            # type: ignore[index]
        batch = _to_device(batch, device)
        Lh = batch["x_tgt"].shape[1]                                   # type: ignore[union-attr]
        with _autocast(precision, device):
            out = flow_matching_loss(model, batch["x_ctx"], batch["x_tgt"],  # type: ignore[arg-type]
                                     batch["cond"], Lh)
        buckets.setdefault(bucket, []).append(float(out["loss"].detach().float()))
    model.train()
    if not buckets:
        return float("nan")
    per_bucket = [float(np.mean(v)) for v in buckets.values()]
    return float(np.mean(per_bucket))


def train(cfg: Config, data_root: Optional[Union[str, Path]] = None,
          num_workers: int = 0, device: Optional[torch.device] = None,
          model: Optional[torch.nn.Module] = None,
          save_hook=None, optimizer_factory=None) -> str:
    """Run training and return the final checkpoint path.

    ``model`` (optional) supplies a prebuilt module (e.g. a fine-tuned or
    pretrained-initialised ``KlineDiT`` with frozen / LoRA parameters); when
    ``None`` a fresh ``KlineDiT(cfg.model)`` is constructed.  Only parameters
    with ``requires_grad=True`` are optimised.

    ``save_hook(path, model, ema, cfg, step, optimizer)`` overrides checkpoint
    writing (used by LoRA fine-tuning to also emit a merged export).
    ``optimizer_factory(decay_params, no_decay_params)`` overrides the optimiser.
    """
    rank, world, local_rank = distributed_info()
    init_distributed(rank, world)
    seed_everything(int(cfg.train.seed) + rank)
    if device is None:
        device = pick_device(local_rank)

    out_dir = Path(cfg.train.out_dir)
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)

    raw_model = KlineDiT(cfg.model) if model is None else model
    n_params = raw_model.count_params()
    n_trainable = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    if is_main(rank):
        print(f"[kit] device={device} world={world} params={n_params/1e6:.1f}M "
              f"trainable={n_trainable/1e6:.2f}M micro_batch={cfg.train.micro_batch} "
              f"global_batch={cfg.train.global_batch}", flush=True)
    raw_model.to(device)
    if cfg.train.compile and model is None \
            and (cfg.train.parallel or "none") in ("none", "ddp"):
        raw_model = torch.compile(raw_model)  # type: ignore[assignment]
    model = wrap_model(raw_model, cfg.train, local_rank, world)

    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 else decay).append(p)
    if optimizer_factory is not None:
        optimizer = optimizer_factory(decay, no_decay)
    else:
        optimizer = torch.optim.AdamW(
            [{"params": decay, "weight_decay": float(cfg.train.weight_decay)},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=float(cfg.train.lr), betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
            eps=float(cfg.train.eps))
    scaler = _make_scaler(cfg.train.precision, device)
    ema = EMA(raw_model, decay=float(cfg.train.ema_decay))

    micro = int(cfg.train.micro_batch)
    accum = max(1, int(cfg.train.global_batch) // (micro * max(1, world)))
    if is_main(rank) and accum * micro * max(1, world) != int(cfg.train.global_batch):
        print(f"[warn] global_batch={cfg.train.global_batch} is not a multiple of "
              f"micro_batch*world={micro * max(1, world)}; using accum={accum} "
              f"(effective {accum * micro * max(1, world)})", flush=True)

    pin = device.type == "cuda"
    train_ds, train_loader = build_dataloader(cfg, "train", data_root, rank, world,
                                              num_workers, pin_memory=pin)
    val_loader = None
    if int(cfg.train.val_every) > 0:
        try:
            _, val_loader = build_dataloader(cfg, "val", data_root, rank, world,
                                             num_workers, pin_memory=pin)
        except (RuntimeError, FileNotFoundError) as e:
            if is_main(rank):
                print(f"[warn] no validation windows ({e}); validation disabled",
                      flush=True)
            val_loader = None
    if is_main(rank):
        print(f"[kit] train windows={len(train_ds)}"
              + (f" val windows={len(val_loader.dataset)}" if val_loader else ""),
              flush=True)

    step = 0
    if cfg.train.resume:
        step = load_checkpoint(cfg.train.resume, unwrap(model), ema, optimizer,
                               map_location=device)
        if is_main(rank):
            print(f"[kit] resumed from {cfg.train.resume} at step {step}", flush=True)

    total = int(cfg.train.total_steps)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    epoch = 0
    data_iter = iter(train_loader)
    t0 = time.time()
    running = 0.0
    running_n = 0
    ckpt_name = Path(cfg.train.resume).name if cfg.train.resume else "latest.pt"
    final_path = out_dir / ckpt_name

    while step < total:
        lr = lr_at(step, cfg.train)
        for pg in optimizer.param_groups:
            pg["lr"] = lr
        micro_loss = 0.0
        for _ in range(accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                set_loader_epoch(train_loader, epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = _to_device(batch, device)
            Lh = batch["x_tgt"].shape[1]                               # type: ignore[union-attr]
            with _autocast(cfg.train.precision, device):
                out = flow_matching_loss(
                    model, batch["x_ctx"], batch["x_tgt"], batch["cond"], Lh,  # type: ignore[arg-type]
                    inst_dropout=float(cfg.train.inst_dropout),
                    allcond_dropout=float(cfg.train.allcond_dropout),
                    hist_dropout=float(cfg.train.hist_dropout))
            loss = out["loss"] / accum
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()
            micro_loss += float(out["loss"].detach().float())

        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            float(cfg.train.grad_clip))
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        ema.update(raw_model)
        step += 1
        running += micro_loss / accum
        running_n += 1

        if is_main(rank) and (step % max(1, int(cfg.train.log_every)) == 0 or step == total):
            elapsed = time.time() - t0
            seen = step * accum * micro * max(1, world)
            print(f"step {step}/{total}  loss {running / running_n:.5f}  "
                  f"lr {lr:.3e}  grad {float(gnorm):.3f}  "
                  f"{seen / max(elapsed, 1e-6):.0f} samp/s  {elapsed:.0f}s", flush=True)
            running, running_n = 0.0, 0

        if val_loader is not None and step % max(1, int(cfg.train.val_every)) == 0:
            vloss = evaluate(model, val_loader, device, cfg.train.precision,
                             int(cfg.train.val_batches))
            if is_main(rank):
                print(f"step {step}  val_loss {vloss:.5f}", flush=True)

        if step % max(1, int(cfg.train.ckpt_every)) == 0 or step == total:
            if is_main(rank):
                if save_hook is not None:
                    save_hook(out_dir / f"step_{step:07d}.pt", unwrap(model), ema,
                              cfg, step, None)
                    save_hook(final_path, unwrap(model), ema, cfg, step, optimizer)
                else:
                    save_checkpoint(out_dir / f"step_{step:07d}.pt", unwrap(model),
                                    ema, cfg, step)
                    save_checkpoint(final_path, unwrap(model), ema, cfg, step,
                                    optimizer)
                print(f"[kit] saved checkpoint at step {step}", flush=True)

    if world > 1:
        import torch.distributed as dist
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    if is_main(rank):
        print(f"[kit] done; final checkpoint {final_path}", flush=True)
    return str(final_path)


__all__ = ["train", "evaluate", "seed_everything"]
