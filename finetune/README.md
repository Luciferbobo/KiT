# finetune/

Fine-tune a released KiT checkpoint on your own K-line data. The pretrained
checkpoint is reused exactly: same architecture, same MAD normalisation stats,
same instrument vocabulary. Fine-tuned weights are written as standard
checkpoints the eval scripts load directly.

| File | Role |
|---|---|
| `finetune.py` | main CLI (`--mode lora` / `head` / `full`) |
| `lora.py` | dependency-free LoRA + merge-back-into-standard-weights |
| `inspect_ckpt.py` | print a checkpoint's config / parameter count / metadata |

## 0. Prepare

1. Check the checkpoint and download the bundle (checkpoint + `instruments.json`
   + `stats.json`):

```bash
python finetune/inspect_ckpt.py --ckpt ckpt/step_market_284M
```

2. Build your data tree (parquet + `instruments.json` / `stats.json` /
   `trade_dates.csv`):

```bash
python scripts/prepare_data.py --raw-root raw/my_1m --out data --market ashare
```

The fine-tuner will **reuse the checkpoint's `stats.json`** (so features are
normalised the same way the model was trained) and **merge instrument ids**:
codes known to the checkpoint keep their trained embedding; unknown codes map to
`[UNK]` (or get new embedding rows with `--extend-instruments`). A one-time
`.ftorig` backup is kept of any file it touches.

## 1. Full fine-tune of KiT-L on a single H200 (the main workpath)

KiT-L is 283.7M params; in bf16 with fp32 AdamW the optimiser state is ~4.5 GB,
so a single H200 (141 GB) fits comfortably. Gradient checkpointing is on by
default for `full`/`head`.

```bash
python finetune/finetune.py \
    --ckpt ckpt/step_market_284M \
    --data-root data --out-dir runs/ft_full_L \
    --mode full \
    --micro-batch 8 --grad-accum 8 \     # global batch 64
    --steps 5000 --lr 1e-5 --warmup 200 \
    --precision bf16 --grad-checkpointing on \
    --val-every 200 --ckpt-every 200
```

- Raise `--micro-batch` / `--grad-accum` to the global batch you want; scale
  `--lr` accordingly (5e-6 … 2e-5 works well for full fine-tuning).
- Restrict to fewer timescales with `--set data.timescales=1h,2h,1d`, or change
  the train/val split with `--set data.train_end=... --set data.val_end=...`.
- Multi-GPU: launch with `torchrun --nproc_per_node=N` and `--parallel fsdp`.

### Smaller GPU variant

```bash
python finetune/finetune.py --ckpt ckpt/step_market_284M --data-root data \
    --out-dir runs/ft_full_small --mode full \
    --micro-batch 1 --grad-accum 16 --optimizer adamw8bit \
    --grad-checkpointing on --set data.timescales=1h,2h,1d
```

`--optimizer adamw8bit` needs `bitsandbytes`. `--dry-run` prints the parameter
count, memory estimate and data sizes without training — use it to check a
configuration before a long run.

## 2. LoRA (cheapest, safest)

```bash
python finetune/finetune.py \
    --ckpt ckpt/step_market_284M \
    --data-root data --out-dir runs/ft_lora \
    --mode lora --lora-rank 16 --lora-alpha 32 \
    --steps 2000 --lr 2e-4 --grad-accum 8
```

LoRA adapts `attn.qkv`, `attn.out_proj` and the SwiGLU FFN (`w1/w2/w3`) of every
block (`--lora-last-n N` limits it to the last N blocks). The instrument
embedding is also trained by default (`--freeze-inst-embedding` to disable)
so new products can adapt. Because LoRA changes the state-dict layout, training
checkpoints (resume-able) live in `--out-dir`, while **eval-ready merged
checkpoints** are written to `<out-dir>/export/{latest.pt,step_*.pt}`.

## 3. Head-only

Freeze the backbone; train the head + identity embeddings (fast, few params):

```bash
python finetune/finetune.py --ckpt ckpt/step_market_284M --data-root data \
    --out-dir runs/ft_head --mode head --steps 3000 --lr 5e-5
```

The trainable modules default to `in_proj, role_emb, register, calendar,
cond_emb, t_emb, adaln, final_norm, head_bias, out_proj`
(`--head-modules` to change).

## 4. Evaluate / infer

`full` and `head` checkpoints are written directly to `--out-dir`
(`latest.pt`, `step_*.pt`). For LoRA use the merged export:

```bash
python eval/infer_for_pred.py --ckpt runs/ft_lora/export/latest.pt --data-root data
python eval/infer_for_pred.py --ckpt runs/ft_full_L/latest.pt       --data-root data
```

## 5. Zero-shot vs fine-tuned report (quant-style)

After fine-tuning, compare the pretrained (ZS) and fine-tuned (FT) checkpoints on
your data with charts and tables a data scientist (or an LLM) can read:

```bash
python finetune/compare_zs_ft.py \
    --ckpt-zs ckpt/step_market_284M \
    --ckpt-ft runs/ft_full_L/latest.pt \
    --data-root data --split val \
    --n-anchors 20 --n-paths 16 --steps 32 \
    --out runs/ft_full_L/report
```

It writes to `--out`:

| file | content |
|---|---|
| `metrics_by_scale.csv` | ret/vol/price IC **and** RankIC by scale, `zs_*` / `ft_*` / `delta_*` |
| `anchors.csv` | per-anchor metrics for both models (for scatter/time plots) |
| `quintiles.csv` | mean realised return per predicted-return quintile (ZS vs FT) |
| `summary.json` | everything, machine-readable (hand to an LLM) |
| `summary.md` | the same as markdown tables |
| `quintiles.png` | quintile bar chart (monotone = signal); long-short in the title |
| `rankic_by_scale.png` | Return & Volatility RankIC bars per scale (ZS vs FT) |
| `ic_over_time.png` | per-anchor Return RankIC over time (scatter + rolling mean) |
| `equity_curve.png` | cumulative long-short (Q_top − Q_bottom) return |

Metrics match `eval/cal_metrics.py`: return = terminal close-to-close log-return
(=∑`r_gap`+`r_body`), volatility = per-bar sample std of those returns, price =
per-horizon cross-sectional correlation of cumulative returns. All sampling is
batched **one call per cross-section** (all instruments share `Lc`/`Lh`), so the
GPU does one forward per anchor instead of one per stock.

Useful flags: `--scales 1h,2h,1d` (restrict), `--min-size 5` (quintiles need ≥5
names), `--quintiles 5`, `--precision bf16`, `--no-charts`.

## GPU notes

- Everything runs on CUDA by default (`--device cuda`, `--precision bf16`).
- Full fine-tune of KiT-L needs only ~5 GB of optimiser state, so a single H200
  (or a 24 GB card with `--micro-batch 1 --optimizer adamw8bit`) is enough.
- On Windows/WDDM machines where the GPU is shared with desktop apps, keep
  batches small; the batched sampler/section inference above avoids many tiny
  CUDA launches, which is what makes WDDM thrash.

## Notes

- Resume: pass `--resume <out-dir>/latest.pt` (LoRA resumes from the un-merged
  training checkpoint; re-pass the same mode/target flags).
- The fine-tuner never overwrites your data: it backs up `instruments.json` /
  `stats.json` to `*.ftorig` the first time it replaces them.
- `--use-ema-init` starts from the checkpoint's EMA weights; `--extend-instruments`
  grows the instrument embedding instead of mapping new codes to `[UNK]`.
