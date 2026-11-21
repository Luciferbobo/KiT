# Cached-context distillation

`cached_distill.py` adapts a pretrained KiT checkpoint to the cached-context
attention layout in `CachedKlineDiT`.

The original full-attention model is kept frozen as the teacher. The student:

- encodes registers and history once per window;
- applies target-only self-attention with cross-attention to cached history
  keys and values;
- learns the average teacher velocity over fixed 32-to-8 step shortcut
  segments;
- retains a plain teacher-velocity loss to preserve local flow behavior.

## Inputs

The data root must contain the normal KiT inference assets:

```text
data/
  instruments.json
  stats.json
  trade_dates.csv
  <market>/<scale>/<instrument>.parquet
ckpt/
  step_market_284M/step_market_284M.pt
```

`--teacher-ckpt`, `--data-root`, and `--out` can be overridden from the command
line. The default paths are relative to the repository root.

## Training

Single GPU:

```bash
python -m kit_v1.train.cached_distill \
  --micro-batch 4 \
  --grad-accum 8 \
  --max-steps 20000 \
  --student-steps 8 \
  --teacher-steps 32 \
  --shortcut-ratio 0.75
```

Multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m kit_v1.train.cached_distill \
  --micro-batch 4 \
  --grad-accum 8 \
  --max-steps 20000 \
  --student-steps 8 \
  --teacher-steps 32 \
  --shortcut-ratio 0.75
```

The default global batch is `world_size * micro_batch * grad_accum`. Wall time
per optimizer step is similar across GPU counts while throughput scales with
the global batch. Adjust `--grad-accum` if a run needs to match a smaller
single-GPU batch size.

On Windows, `launch_windows_ddp.py` avoids the libuv rendezvous used by some
PyTorch builds:

```powershell
python -m kit_v1.train.launch_windows_ddp --nproc 2 -- `
  --micro-batch 2 --grad-accum 16 --max-steps 20000
```

## Checkpoints

The trainer writes:

- `cached_distill_step_*.pt`: inference-compatible student checkpoint;
- `cached_distill_train_state_*.pt`: student and AdamW state.

Resume with:

```bash
--resume runs/cached_distill/cached_distill_train_last.pt
```

Resume restores model weights by default and starts fresh AdamW moments. Add
`--resume-optimizer` when sufficient device memory is available.

## Inference

Use the same student step count at inference:

```python
from kit_v1.infer.cached_sampler import flow_sample_cached

paths = flow_sample_cached(
    model,
    x_ctx,
    cond,
    Lh=Lh,
    steps=8,
    n_paths=32,
)
```

The returned tensor has shape `[batch, paths, horizon, features]` in normalized
KiT feature space.
