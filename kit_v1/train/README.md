# kit_v1/train/ — training stack

Flow-matching training for KiT. It reuses the exact model (`kit_v1.model.KlineDiT`),
feature/normalisation/calendar code (`kit_v1.data`) and flow convention
(`kit_v1.infer.sampler`) that ship with the released checkpoints, so a trained
model is a drop-in for the eval scripts.

| File | Role |
|---|---|
| `flow.py` | rectified-flow objective: `z_t=(1-t)x+tε`, `t~sigmoid(N(0,1))`, `L=‖v−（ε−x)‖²` on target bars |
| `ema.py` | fp32 exponential moving average of parameters (written to `ckpt["ema"]["shadow"]`) |
| `schedule.py` | linear warmup → plateau → cosine decay to `min_lr` (+ optional two-stage drop) |
| `checkpoint.py` | save/load in the layout `kit_v1.infer.loader` expects |
| `parallel.py` | process-group setup, `none` / `ddp` / `fsdp` wrapping |
| `loop.py` | the training loop (`train()`), validation, logging, checkpointing, resume |

Data pipeline lives in `kit_v1/data/`:

| File | Role |
|---|---|
| `dataset.py` | `KlineWindowDataset`: builds every valid (history + horizon) window, feature cache, random history truncation |
| `sampler.py` | `ScaleBatchSampler`: scale-homogeneous batches + market temperature (`temperature_tau`) |
| `loader.py` | `build_dataloader` + numpy→torch collate |
| `aggregate.py` | session-aware fine→coarse bar aggregation |
| `io.py` | canonical OHLCV column handling |

## Objective

Verified against `paper.pdf` (§3.1, §3.5, App. D) and `kit_v1.infer.sampler`:

```
z_t = (1 - t) * x + t * eps,      t ~ sigmoid(N(0, 1)),   eps ~ N(0, I)   (Eq. 1)
L   = (1/Lh) * sum_i || v_theta(z_t, t)_i - (eps_i - x_i) ||^2             (Eq. 8)
```

Only the `Lh` target positions contribute to the loss; the history tokens attend
but do not enter the loss and are returned bit-identical. Three stochastic
condition-dropout schemes are applied during training (§3.5 / App. D), matching
the inference-time CFG branches in `infer/sampler.py`:

| Training dropout | Replaces | `TrainConfig` | Inference branch |
|---|---|---|---|
| instrument + sector -> `[UNK]` (slot 0) | `inst`, `sector` | `inst_dropout` | identity CFG |
| all identity -> `[NULL]` (slot 0) | `market`, `sector`, `inst`, `scale` | `allcond_dropout` | identity CFG |
| zero the history context | `x_ctx`, context calendar, `pad_mask` | `hist_dropout` | history CFG |

Validation loss is averaged **equally over (market, timescale) buckets** as in
the paper.

## Quick start

```bash
# 1. build the data tree (instruments.json / stats.json / trade_dates.csv + parquet)
python scripts/prepare_data.py --raw-root raw/ashare_1m --out data --market ashare

# 2. smoke test the whole stack on CPU (tiny model, a few steps)
python train.py --preset tiny --data-root data --out-dir runs/tiny

# 3. real run on one GPU
python train.py --config configs/dit_300m.yaml --data-root data --out-dir runs/dit_300m

# 4. multi-GPU / multi-node (FSDP)
torchrun --nproc_per_node=8 train.py --config configs/dit_300m.yaml \
    --data-root data --out-dir runs/dit_300m
```

Override any config field from the CLI:

```bash
python train.py --config configs/dit_300m.yaml --data-root data \
    --set train.total_steps=400000 --set train.lr=1e-5 \
    --set data.train_end=2023-12-31 --set data.val_end=2024-06-30
```

Resume:

```bash
python train.py --config configs/dit_300m.yaml --data-root data \
    --out-dir runs/dit_300m --resume runs/dit_300m/latest.pt
```

## Checkpoints

`<out-dir>/step_0000123.pt` every `train.ckpt_every` steps, and
`<out-dir>/latest.pt` (includes the optimizer state) for resuming. Each file is
`{"model", "ema": {"shadow"}, "config", "step"}` — exactly what the eval scripts
read:

```bash
python eval/infer_for_pred.py --ckpt runs/dit_300m/latest.pt --data-root data
```

## Config knobs

All fields live in `kit_v1/config.py`; the three presets are in `configs/`.

- **model**: architecture (verified against the released `dit_300m` checkpoint:
  21 layers / d1024 / 8 heads → 283.7M params).
- **data**: window table (`scale_windows`), strides, truncation (`trunc_prob`,
  `trunc_min`), normalisation (`soft_clip_c`, `vol_ema_half_life_days`), sampling
  (`temperature_tau`, `per_inst_cap`, `market_weights`), splits (`train_end`,
  `val_start/end`, `test_start`).
- **train**: batch sizes, optimiser, LR schedule, precision, `parallel`
  (`none`/`ddp`/`fsdp`), `fsdp_sharding` (`grad_op`/`full`), dropouts
  (`inst_dropout`/`allcond_dropout`/`hist_dropout`), EMA, logging/ckpt cadence.
- **infer**: sampler defaults (only used by the eval scripts).

## Fidelity to the paper

Everything the paper states explicitly is matched:

- **Architecture / family** (Table 3): KiT-S 31.0M (8L, 512, 4H, 1408), KiT-M
  101.0M (13L, 768, 6H, 2048), KiT-L 283.7M (21L, 1024, 8H, 2816) — reproduced
  exactly by `configs/dit_*.yaml` (`KlineDiT(...).count_params()` = 31.05 / 100.99
  / 283.69M).
- **Windows** (Table 4) equal the `data.scale_windows` defaults.
- **Objective** (Eq. 1/2/8), logit-normal time, target-only loss with the `1/Lh`
  factor.
- **Conditioning**: identity via shared AdaLN (evaluated at `t` for target and
  `t=0` for context), per-bar calendar added to token embeddings, bidirectional
  attention, RoPE + QK-Norm, SwiGLU, zero-init AdaLN gates.
- **EMA** decay `0.9999`; CFG weights default to 1; inference draws K samples.

### Paper recipe vs. the released config files

Where the paper (§4.1) and the authors' `configs/*.yaml` differ, the code uses
the **config files** (they are the released artefacts); set the paper values via
`--set` if you want the paper's description:

| Setting | Paper §4.1 | `configs/dit_300m.yaml` |
|---|---|---|
| global batch | 4096 | 1024 |
| peak LR | 1e-4 | 1e-5 (2e-4 / 1e-4 for S / M) |
| warmup | 1000 | 2000 |
| steps | 50k reported (checkpoints ~63–64k; `total_steps` is the cap) | 400000 |
| validation | 2024-01-01 … 2025-12-31 | 2024-01-01 … 2024-06-30 |
| test | 2026-01-01 … 2026-04-10 | from 2024-07-01 |
| markets | A-share + US + crypto | (data-dependent) |

Optimizer settings that agree: AdamW `β1=0.9 β2=0.95`, weight decay `0.1`,
gradient clip `1.0`, BF16, EMA `0.9999`.

## Reconstruction notes

The original training code was **not** part of the public release (the README
listed it as "TBD"). This directory is a faithful reconstruction derived from the
released model, the `Config`/`DataConfig`/`TrainConfig` dataclasses, the flow
convention in `infer/sampler.py`, the feature/calendar code, and `paper.pdf`.
Consequences:

- The model forward, features, normalisation, calendar, flow convention and
  checkpoint layout are bit-for-bit what inference consumes.
- The objective, CFG dropouts, logit-normal time and bucket-equal validation
  follow the paper's explicit equations/sections.
- The exact optimiser/dataset code is not recoverable byte-for-byte; scheduling
  shape, sampler policy and the LR/warmup/batch values come from the config
  dataclasses (see the table above).
- `train.py` is exercised end-to-end on synthetic data (preparation → training →
  bf16/fp16 → checkpoint → EMA inference → resume) on both CPU and CUDA.
