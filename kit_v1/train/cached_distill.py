"""Distill the original KiT model into the cached-context attention layout.

Teacher: original full bidirectional KlineDiT, frozen.
Student: context encoded once, target-only self-attention plus cached cross-attention.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, DistributedSampler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kit_v1.data.calendar import (  # noqa: E402
    MARKET_IDS,
    TIMESCALE_IDS,
    attach_window_calendar,
    bars_per_day,
    load_trading_days,
)
from kit_v1.data.features import FEATURE_NAMES, compute_features  # noqa: E402
from kit_v1.data.normalize import normalize, stats_key  # noqa: E402
from kit_v1.infer import load_model_and_config  # noqa: E402
from kit_v1.model.cached_dit import CachedKlineDiT  # noqa: E402


@dataclass(frozen=True)
class WindowRef:
    code: str
    end_index: int
    end_timestamp: str


class WindowDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        stats: Dict[str, float],
        market: str,
        scale: str,
        context_len: int,
        horizon: int,
        soft_clip_c: float,
        half_life_days: float,
        lo: str,
        hi: str,
        stride: int,
        limit_stocks: int,
        allow_empty: bool = False,
    ) -> None:
        self.market = market
        self.scale = scale
        self.context_len = int(context_len)
        self.horizon = int(horizon)
        self.trading_days = load_trading_days(data_root / "trade_dates.csv")
        instruments = json.loads(
            (data_root / "instruments.json").read_text(encoding="utf-8")
        )
        codes = sorted(instruments)
        if limit_stocks > 0:
            codes = codes[:limit_stocks]

        self.series: Dict[str, dict] = {}
        self.items: list[WindowRef] = []
        half_life_bars = float(half_life_days) * bars_per_day(market, scale)
        lo_ts, hi_ts = pd.Timestamp(lo), pd.Timestamp(hi)

        for code in codes:
            path = data_root / market / scale / f"{code}.parquet"
            frame = pd.read_parquet(
                path,
                columns=["datetime", "open", "high", "low", "close", "volume"],
            )
            frame["datetime"] = pd.to_datetime(frame["datetime"])
            raw = compute_features(
                frame["open"].to_numpy(np.float64),
                frame["high"].to_numpy(np.float64),
                frame["low"].to_numpy(np.float64),
                frame["close"].to_numpy(np.float64),
                frame["volume"].to_numpy(np.float64),
                half_life_bars=half_life_bars,
            )
            features = np.stack(
                [
                    normalize(
                        raw[name],
                        float(stats[stats_key(market, scale, name)]),
                        c=soft_clip_c,
                    )
                    for name in FEATURE_NAMES
                ],
                axis=1,
            ).astype(np.float32)
            timestamps = pd.DatetimeIndex(frame["datetime"])
            meta = instruments.get(code, {})
            self.series[code] = {
                "datetime": timestamps,
                "features": features,
                "sector": int(meta.get("sector", 0)),
                "inst": int(meta.get("vocab_index", 0)),
            }
            first = self.context_len - 1
            last = len(frame) - self.horizon - 1
            for index in range(first, last + 1, max(1, stride)):
                end_ts = timestamps[index]
                future_ts = timestamps[index + self.horizon]
                if end_ts < lo_ts or end_ts > hi_ts or future_ts > hi_ts:
                    continue
                self.items.append(
                    WindowRef(code, index, end_ts.isoformat())
                )
        if not self.items and not allow_empty:
            raise ValueError(f"no windows in {lo}..{hi}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, item_index: int) -> dict:
        ref = self.items[item_index]
        series = self.series[ref.code]
        index = ref.end_index
        start = index - self.context_len + 1
        timestamps = series["datetime"][start : index + 1 + self.horizon]
        prev_ts = (
            series["datetime"][start - 1] if start > 0 else None
        )
        calendar = attach_window_calendar(
            timestamps,
            self.market,
            self.scale,
            n_ctx=self.context_len,
            prev_ts=prev_ts,
            trading_days=self.trading_days,
        )
        return {
            "x_ctx": torch.from_numpy(
                series["features"][start : index + 1].copy()
            ),
            "x_target": torch.from_numpy(
                series["features"][
                    index + 1 : index + 1 + self.horizon
                ].copy()
            ),
            "clock": torch.from_numpy(calendar["clock"]),
            "sess": torch.from_numpy(calendar["sess"]),
            "dow": torch.from_numpy(calendar["dow"]),
            "event": torch.from_numpy(calendar["event"]),
            "month": torch.from_numpy(calendar["month"]),
            "yday": torch.from_numpy(calendar["yday"]),
            "market": torch.tensor(MARKET_IDS[self.market], dtype=torch.long),
            "sector": torch.tensor(series["sector"], dtype=torch.long),
            "inst": torch.tensor(series["inst"], dtype=torch.long),
            "scale": torch.tensor(
                TIMESCALE_IDS[self.scale], dtype=torch.long
            ),
            "pad_mask": torch.ones(
                self.context_len + self.horizon, dtype=torch.bool
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--teacher-ckpt",
        type=Path,
        default=ROOT / "ckpt" / "step_market_284M" / "step_market_284M.pt",
    )
    parser.add_argument("--init-ckpt", type=Path, default=None)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="resume a cached_distill_train_state_*.pt checkpoint",
    )
    parser.add_argument(
        "--resume-optimizer",
        action="store_true",
        help="also restore AdamW moments; may need extra GPU memory",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs" / "cached_distill"
    )
    parser.add_argument("--market", default="ashare")
    parser.add_argument("--scale", default="1d")
    parser.add_argument("--train-lo", default="2018-01-01")
    parser.add_argument("--train-hi", default="2025-06-30")
    parser.add_argument("--val-lo", default="2025-07-01")
    parser.add_argument("--val-hi", default="2025-12-31")
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument("--limit-stocks", type=int, default=0)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--lr-step-scale", type=float, default=1.0)
    parser.add_argument("--lr-total-steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument(
        "--train-scope",
        choices=("all", "no_mlp", "attention"),
        default="no_mlp",
        help="freeze MLP or everything except attention/AdaLN to reduce memory",
    )
    parser.add_argument("--truth-loss-weight", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.0)
    parser.add_argument("--student-steps", type=int, default=8)
    parser.add_argument("--teacher-steps", type=int, default=32)
    parser.add_argument("--shortcut-ratio", type=float, default=0.75)
    parser.add_argument("--shortcut-loss-weight", type=float, default=1.0)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.25)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-batches", type=int, default=8)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-inference-ckpts", type=int, default=3)
    parser.add_argument("--keep-training-states", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--manual-gloo",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--manual-nccl",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--local-sgd-sync-every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def distributed_context(device_arg: str):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if not dist.is_initialized():
            backend = (
                "nccl"
                if torch.cuda.is_available() and dist.is_nccl_available()
                else "gloo"
            )
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_arg)
    return world_size, rank, local_rank, device


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def slice_batch(batch: dict, indices: torch.Tensor) -> dict:
    return {key: value.index_select(0, indices) for key, value in batch.items()}


def lr_at_step(
    step: int,
    max_steps: int,
    warmup: int,
    base_lr: float,
    min_ratio: float,
) -> float:
    if warmup > 0 and step < warmup:
        return base_lr * (step + 1) / warmup
    progress = min(
        1.0, max(0.0, (step - warmup) / max(1, max_steps - warmup))
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * cosine)


def save_inference_checkpoint(
    path: Path,
    model,
    cfg,
    step: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    save_inference_checkpoint_state(
        path,
        {key: value.detach() for key, value in model.state_dict().items()},
        cfg,
        step,
        world_size,
        args,
    )


def save_inference_checkpoint_state(
    path: Path,
    state: dict[str, torch.Tensor],
    cfg,
    step: int,
    world_size: int,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "config": cfg.to_dict(),
        "model": {key: value.detach() for key, value in state.items()},
        "world_size": int(world_size),
        "accum": int(args.grad_accum),
        "global_batch": int(
            world_size * args.micro_batch * args.grad_accum
        ),
        "slim": {
            "kind": "cached_distill_student",
            "teacher": str(args.teacher_ckpt),
            "scale": args.scale,
            "context": "cached",
        },
    }
    temporary = path.with_name(path.stem + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_training_state(
    path: Path,
    model,
    optimizer,
    step: int,
    cfg,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": int(step),
        "model": {
            key: value.detach() for key, value in model.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "train_scope": args.train_scope,
    }
    temporary = path.with_name(path.stem + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def configure_trainable(model, scope: str) -> tuple[int, int]:
    if scope == "all":
        trainable = [parameter for parameter in model.parameters()]
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        selected = []
        for name, parameter in model.named_parameters():
            if scope == "no_mlp":
                keep = ".mlp." not in name
            else:
                keep = (
                    ".attn." in name
                    or ".adaln." in name
                    or name.endswith(".bias_i")
                )
            parameter.requires_grad_(keep)
            if keep:
                selected.append(parameter)
        trainable = selected
    count = sum(parameter.numel() for parameter in trainable)
    total = sum(parameter.numel() for parameter in model.parameters())
    return count, total


class ModelEMA:
    """Decay a full model state without gradients."""

    def __init__(self, model, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model) -> None:
        if self.decay <= 0:
            return
        for key, value in model.state_dict().items():
            shadow = self.shadow[key]
            if torch.is_floating_point(value):
                shadow.mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                shadow.copy_(value)


def rotate_checkpoints(out: Path, pattern: str, keep: int) -> None:
    files = sorted(out.glob(pattern))
    if keep <= 0:
        return
    for path in files[:-keep]:
        path.unlink(missing_ok=True)


def sync_gradients_manual(
    model: torch.nn.Module,
    world_size: int,
) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return
    flat = torch._utils._flatten_dense_tensors(gradients)
    sync_dtype_name = os.environ.get("KIT_MANUAL_SYNC_DTYPE", "float32")
    sync_dtype = getattr(torch, sync_dtype_name)
    sync_scale = float(os.environ.get("KIT_MANUAL_SYNC_SCALE", "1.0"))
    buffer = flat.to(
        device="cpu",
        dtype=sync_dtype,
        non_blocking=False,
    )
    if sync_scale != 1.0:
        buffer.mul_(sync_scale)
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    buffer.div_(world_size)
    if sync_scale != 1.0:
        buffer.div_(sync_scale)
    synced = torch._utils._unflatten_dense_tensors(
        buffer.to(device=flat.device, dtype=flat.dtype),
        gradients,
    )
    for gradient, updated in zip(gradients, synced):
        gradient.copy_(updated)


def sync_gradients_nccl(
    model: torch.nn.Module,
    world_size: int,
) -> None:
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return
    flat = torch._utils._flatten_dense_tensors(gradients)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    synced = torch._utils._unflatten_dense_tensors(flat, gradients)
    for gradient, updated in zip(gradients, synced):
        gradient.copy_(updated)


def sync_model_weights_manual(
    model: torch.nn.Module,
    world_size: int,
) -> None:
    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not parameters:
        return
    flat = torch._utils._flatten_dense_tensors(parameters)
    sync_dtype_name = os.environ.get("KIT_MANUAL_SYNC_DTYPE", "float32")
    sync_dtype = getattr(torch, sync_dtype_name)
    sync_scale = float(os.environ.get("KIT_MANUAL_SYNC_SCALE", "1.0"))
    buffer = flat.to(
        device="cpu",
        dtype=sync_dtype,
        non_blocking=False,
    )
    if sync_scale != 1.0:
        buffer.mul_(sync_scale)
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    buffer.div_(world_size)
    if sync_scale != 1.0:
        buffer.div_(sync_scale)
    synced = torch._utils._unflatten_dense_tensors(
        buffer.to(device=flat.device, dtype=flat.dtype),
        parameters,
    )
    with torch.no_grad():
        for parameter, updated in zip(parameters, synced):
            parameter.copy_(updated)


def main() -> None:
    args = parse_args()
    if args.student_steps < 1 or args.teacher_steps < args.student_steps:
        raise SystemExit("teacher-steps must be >= student-steps >= 1")
    if args.teacher_steps % args.student_steps != 0:
        raise SystemExit("teacher-steps must be divisible by student-steps")
    if not (0.0 <= args.shortcut_ratio <= 1.0):
        raise SystemExit("shortcut-ratio must be in [0, 1]")
    world_size, rank, local_rank, device = distributed_context(args.device)
    manual_gloo = world_size > 1 and args.manual_gloo
    manual_nccl = world_size > 1 and args.manual_nccl
    local_sgd = (
        manual_gloo
        and args.local_sgd_sync_every > 0
    )
    use_ddp = world_size > 1 and not (manual_gloo or manual_nccl)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    random.seed(args.seed + rank)
    torch.set_float32_matmul_precision("high")

    stats = json.loads(
        (args.data_root / "stats.json").read_text(encoding="utf-8")
    )
    train_set = WindowDataset(
        args.data_root,
        stats,
        args.market,
        args.scale,
        250,
        20,
        6.0,
        5.0,
        args.train_lo,
        args.train_hi,
        args.window_stride,
        args.limit_stocks,
    )
    val_set = WindowDataset(
        args.data_root,
        stats,
        args.market,
        args.scale,
        250,
        20,
        6.0,
        5.0,
        args.val_lo,
        args.val_hi,
        max(1, args.window_stride),
        args.limit_stocks,
        allow_empty=True,
    )
    train_sampler = (
        DistributedSampler(
            train_set,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.micro_batch,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.micro_batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    teacher, cfg = load_model_and_config(
        str(args.teacher_ckpt), "tiny", args.seed, device=device
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    init_ckpt = args.init_ckpt or args.teacher_ckpt
    student, student_cfg = load_model_and_config(
        str(init_ckpt), "tiny", args.seed, device=device
    )
    student.train()
    student.cfg.use_activation_checkpointing = args.checkpointing
    trainable_count, total_count = configure_trainable(
        student, args.train_scope
    )
    student_cfg.train.lr = args.lr
    student_cfg.train.global_batch = (
        args.micro_batch * args.grad_accum
        if local_sgd
        else world_size * args.micro_batch * args.grad_accum
    )
    student_cfg.train.micro_batch = args.micro_batch
    student_cfg.train.total_steps = args.max_steps
    student_cfg.train.seed = args.seed
    student_cfg.train.compile = False

    if use_ddp:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [local_rank]
            ddp_kwargs["output_device"] = local_rank
        student = DistributedDataParallel(student, **ddp_kwargs)

    def raw_student() -> torch.nn.Module:
        return student.module if use_ddp else student

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-15,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    if args.resume is not None:
        training_state = torch.load(
            args.resume, map_location=device, weights_only=False
        )
        base_model = raw_student()
        base_model.load_state_dict(training_state["model"], strict=True)
        if args.resume_optimizer:
            optimizer.load_state_dict(training_state["optimizer"])
        start_step = int(training_state["step"])
        training_state.pop("optimizer", None)
        training_state.pop("model", None)
        del training_state
        if rank == 0:
            print(
                f"resumed step={start_step} from {args.resume} "
                f"optimizer_state={args.resume_optimizer}",
                flush=True,
            )
    ema = (
        ModelEMA(raw_student(), args.ema_decay)
        if args.ema_decay > 0
        else None
    )
    args.out.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (args.out / "run_config.json").write_text(
            json.dumps(vars(args), ensure_ascii=False, indent=2, default=str)
            + "\n",
            encoding="utf-8",
        )
        print(
            f"train_windows={len(train_set)} val_windows={len(val_set)} "
            f"world={world_size} micro_batch={args.micro_batch} "
            f"grad_accum={args.grad_accum} device={device} "
            f"trainable={trainable_count / 1e6:.1f}M/"
            f"{total_count / 1e6:.1f}M",
            flush=True,
        )

    def teacher_velocity(batch: dict, z_target: torch.Tensor, t: torch.Tensor):
        z = torch.cat([batch["x_ctx"], z_target], dim=1)
        with torch.no_grad(), torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            return teacher(z, t, batch, batch["x_target"].shape[1])

    def teacher_integrate(
        batch: dict,
        z_start: torch.Tensor,
        t_start: torch.Tensor,
        substeps: int,
    ):
        z_current = z_start
        step_size = 1.0 / args.teacher_steps
        for index in range(substeps):
            t_current = t_start - index * step_size
            velocity = teacher_velocity(batch, z_current, t_current)
            z_current = z_current - step_size * velocity.float()
        return z_current

    def student_velocity(batch: dict, z_target: torch.Tensor, t: torch.Tensor):
        base_model = raw_student()
        cached = CachedKlineDiT(base_model)
        caches, valid = cached.prepare_context(batch["x_ctx"], batch)
        return cached.forward_target(
            z_target,
            t,
            batch,
            caches,
            valid,
            batch["x_ctx"].shape[1],
        )

    def validation_loss() -> float:
        if not len(val_set) or args.val_batches <= 0:
            return float("nan")
        base_model = raw_student()
        base_model.eval()
        values = []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if index >= args.val_batches:
                    break
                batch = move_batch(batch, device)
                t = torch.ones(
                    batch["x_target"].shape[0], device=device
                )
                noise = torch.randn_like(batch["x_target"])
                z_target = noise
                teacher_end = teacher_integrate(
                    batch,
                    z_target,
                    t,
                    args.teacher_steps // args.student_steps,
                )
                target_velocity = (
                    z_target - teacher_end
                ) / (1.0 / args.student_steps)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    student_v = student_velocity(batch, z_target, t)
                values.append(
                    float(
                        F.mse_loss(
                            student_v.float(), target_velocity.float()
                        )
                    )
                )
        base_model.train()
        return float(np.mean(values)) if values else float("nan")

    step = start_step
    micro_in_step = 0
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    stop = False
    while not stop:
        if train_sampler is not None:
            train_sampler.set_epoch(step)
        for batch in train_loader:
            batch = move_batch(batch, device)
            noise = torch.randn_like(batch["x_target"])
            target_velocity = noise - batch["x_target"]
            batch_size = batch["x_target"].shape[0]
            permutation = torch.randperm(batch_size, device=device)
            shortcut_count = min(
                batch_size,
                max(0, round(batch_size * args.shortcut_ratio)),
            )
            shortcut_indices = permutation[:shortcut_count]
            velocity_indices = permutation[shortcut_count:]

            shortcut_mse = torch.zeros((), device=device)
            velocity_mse = torch.zeros((), device=device)
            truth_mse = torch.zeros((), device=device)

            if shortcut_count > 0:
                shortcut_batch = slice_batch(batch, shortcut_indices)
                shortcut_noise = noise.index_select(0, shortcut_indices)
                segment = torch.randint(
                    0,
                    args.student_steps,
                    (shortcut_count,),
                    device=device,
                )
                t_start = (segment + 1).float() / args.student_steps
                t_view = t_start[:, None, None]
                z_start = (
                    (1.0 - t_view) * shortcut_batch["x_target"]
                    + t_view * shortcut_noise
                )
                teacher_end = teacher_integrate(
                    shortcut_batch,
                    z_start,
                    t_start,
                    args.teacher_steps // args.student_steps,
                )
                shortcut_target = (
                    z_start - teacher_end
                ) / (1.0 / args.student_steps)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    shortcut_student = student_velocity(
                        shortcut_batch, z_start, t_start
                    )
                    shortcut_mse = F.mse_loss(
                        shortcut_student.float(),
                        shortcut_target.float(),
                    )

            if len(velocity_indices) > 0:
                velocity_batch = slice_batch(batch, velocity_indices)
                velocity_noise = noise.index_select(0, velocity_indices)
                velocity_target = target_velocity.index_select(
                    0, velocity_indices
                )
                t = torch.rand(
                    len(velocity_indices), device=device
                )
                t_view = t[:, None, None]
                z_target = (
                    (1.0 - t_view) * velocity_batch["x_target"]
                    + t_view * velocity_noise
                )
                teacher_v = teacher_velocity(
                    velocity_batch, z_target, t
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    student_v = student_velocity(
                        velocity_batch, z_target, t
                    )
                    velocity_mse = F.mse_loss(
                        student_v.float(), teacher_v.float()
                    )
                    truth_mse = F.mse_loss(
                        student_v.float(), velocity_target.float()
                    )

            loss = (
                args.shortcut_loss_weight * shortcut_mse
                + args.velocity_loss_weight * velocity_mse
                + args.truth_loss_weight * truth_mse
            )

            if use_ddp and (micro_in_step + 1) < args.grad_accum:
                sync_context = student.no_sync()
            else:
                sync_context = torch.enable_grad()
            with sync_context:
                (loss / args.grad_accum).backward()
            micro_in_step += 1
            if micro_in_step < args.grad_accum:
                continue

            if manual_nccl:
                sync_gradients_nccl(raw_student(), world_size)
            elif manual_gloo and not local_sgd:
                sync_gradients_manual(raw_student(), world_size)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.grad_clip
            )
            schedule_total = args.lr_total_steps or args.max_steps
            schedule_step = (
                start_step
                + (step - start_step) * args.lr_step_scale
            )
            learning_rate = lr_at_step(
                schedule_step,
                schedule_total,
                args.warmup_steps,
                args.lr,
                args.min_lr_ratio,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            optimizer.step()
            if ema is not None:
                ema.update(raw_student())
            optimizer.zero_grad(set_to_none=True)
            micro_in_step = 0
            step += 1

            if (
                local_sgd
                and (
                    step % args.local_sgd_sync_every == 0
                    or step >= args.max_steps
                )
            ):
                sync_model_weights_manual(raw_student(), world_size)

            if step == 1 or step % 5 == 0 or step >= args.max_steps:
                print(
                    f"rank={rank} step={step}/{args.max_steps} "
                    f"loss={float(loss.detach()):.6f} "
                    f"shortcut={float(shortcut_mse.detach()):.6f} "
                    f"velocity={float(velocity_mse.detach()):.6f} "
                    f"truth={float(truth_mse.detach()):.6f} "
                    f"lr={learning_rate:.2e} "
                    f"lr_progress={schedule_step:.1f}/{schedule_total} "
                    f"grad={float(grad_norm):.3f} "
                    f"elapsed={(time.time() - started) / 60:.1f}m",
                    flush=True,
                )

            if args.val_every > 0 and step % args.val_every == 0:
                value = validation_loss()
                if rank == 0:
                    print(f"validation step={step} mse={value:.6f}", flush=True)

            if (
                rank == 0
                and args.save_every > 0
                and step % args.save_every == 0
            ):
                model_to_save = raw_student()
                save_inference_checkpoint(
                    args.out / f"cached_distill_step_{step:07d}.pt",
                    model_to_save,
                    student_cfg,
                    step,
                    world_size,
                    args,
                )
                if ema is not None:
                    save_inference_checkpoint_state(
                        args.out / f"cached_distill_ema_step_{step:07d}.pt",
                        ema.shadow,
                        student_cfg,
                        step,
                        world_size,
                        args,
                    )
                save_training_state(
                    args.out / f"cached_distill_train_state_{step:07d}.pt",
                    model_to_save,
                    optimizer,
                    step,
                    student_cfg,
                    args,
                )
                rotate_checkpoints(
                    args.out,
                    "cached_distill_step_*.pt",
                    args.keep_inference_ckpts,
                )
                rotate_checkpoints(
                    args.out,
                    "cached_distill_train_state_*.pt",
                    args.keep_training_states,
                )
                print(f"saved step={step}", flush=True)
            if step >= args.max_steps:
                stop = True
                break

    model_to_save = raw_student()
    if rank == 0:
        save_inference_checkpoint(
            args.out / "cached_distill_last.pt",
            model_to_save,
            student_cfg,
            step,
            world_size,
            args,
        )
        save_training_state(
            args.out / "cached_distill_train_last.pt",
            model_to_save,
            optimizer,
            step,
            student_cfg,
            args,
        )
        print(f"done steps={step} out={args.out}", flush=True)
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
