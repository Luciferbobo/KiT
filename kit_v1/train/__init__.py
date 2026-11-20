"""Training subpackage.

- flow:      rectified-flow / flow-matching objective (matches infer.sampler)
- ema:       fp32 exponential moving average of parameters
- schedule:  warmup + cosine learning-rate schedule
- checkpoint: save/load in the layout the inference loader expects
- parallel:  DDP / FSDP setup
- loop:      the training loop
"""
from kit_v1.train.checkpoint import load_checkpoint, save_checkpoint
from kit_v1.train.ema import EMA
from kit_v1.train.flow import (apply_condition_dropout, flow_matching_loss,
                               sample_logit_normal_t)
from kit_v1.train.loop import evaluate, seed_everything, train
from kit_v1.train.schedule import lr_at

__all__ = [
    "flow_matching_loss", "sample_logit_normal_t", "apply_condition_dropout",
    "EMA", "lr_at", "save_checkpoint", "load_checkpoint",
    "train", "evaluate", "seed_everything",
]
