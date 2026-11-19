"""Checkpoint loading for inference."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from kit_v1.config import Config, get_config
from kit_v1.model import KlineDiT


def _mapped_ema_model(cfg: Config, weights: dict) -> Optional[KlineDiT]:
    """Assign a complete, ordinary CPU FP32 EMA without allocating a second set of parameters."""
    if torch.empty(0).device.type != "cpu" or torch.get_default_dtype() != torch.float32:
        return None
    with torch.device("meta"):
        model = KlineDiT(cfg.model)
    expected = model.state_dict()
    if weights.keys() != expected.keys() or any(
        not isinstance(value, torch.Tensor) or value.layout != torch.strided
        or value.device.type != "cpu" or value.dtype != torch.float32
        or value.shape != expected[key].shape or value.stride() != expected[key].stride()
        or value.is_neg()
        for key, value in weights.items()
    ):
        return None
    # The normal load copies each parameter independently, including checkpoints with tied storage.
    if len({t.untyped_storage().data_ptr() for t in weights.values()}) != len(weights):
        return None
    model.load_state_dict(weights, assign=True)
    # The checkpoint excludes these derived, nonpersistent positional buffers.
    for block in model.blocks:
        rope = block.attn.rope
        rope.cos_cached, rope.sin_cached = rope._build(rope.cos_cached.size(0))
    return None if any(buf.is_meta for buf in model.buffers()) else model


def load_model_and_config(
    ckpt_path: str, preset: str, seed: int, config_path: Optional[str] = None, *, mmap: bool = False
) -> Tuple[KlineDiT, Config]:
    """Load EMA weights when available, or initialize a model for the special path "none".

    With mmap=True, a complete FP32 EMA can back the CPU parameters directly from the file.
    This skips random model initialization. Pass a seeded generator to the sampler for
    reproducible noise across both loading modes.
    """
    if ckpt_path.lower() == "none":
        torch.manual_seed(seed)
        cfg = get_config(preset)
        if config_path is not None:
            cfg = Config.load_yaml(config_path, base=cfg)
        model = KlineDiT(cfg.model)
        print(f"[warn] --ckpt none: using randomly initialized {preset!r} model"
              + (f"(YAML override: {config_path})" if config_path else "")
              + " (smoke test only)")
        return model, cfg
    p = Path(ckpt_path)
    if not p.is_file():
        raise FileNotFoundError(
            f"checkpoint not found: {p} (without training artifacts, use --ckpt none --preset tiny for a smoke test)")
    use_mmap = None
    if mmap:
        # torch.save's legacy pickle format cannot be memory-mapped.
        with p.open("rb") as stream:
            use_mmap = stream.read(4) == b"PK\x03\x04"
    ckpt = torch.load(p, map_location="cpu", weights_only=False, mmap=use_mmap)
    if "yday_dim" not in ckpt.get("config", {}).get("model", {}):
        raise RuntimeError(
            f"checkpoint {p} predates the month/yday calendar features; an old "
            "ckpt run on new code gets untrained month/yday embeddings, so behavior differs from "
            "the trained version; run inference with the old code or retrain.")
    cfg = Config.from_dict(ckpt["config"])
    ema = ckpt.get("ema")
    if ema:
        # Accept either {"shadow": {...}} or a flat parameter dictionary.
        shadow = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
        if use_mmap and isinstance(shadow, dict):
            model = _mapped_ema_model(cfg, shadow)
            if model is not None:
                print(f"[info] loaded EMA weights (step={ckpt.get('step')})")
                return model, cfg
    model = KlineDiT(cfg.model)
    # A partial EMA inherits missing parameters from the normally initialized model.
    state = dict(model.state_dict())
    if ema:
        state.update({k: v.to(torch.float32) for k, v in shadow.items()})
        print(f"[info] loaded EMA weights (step={ckpt.get('step')})")
    else:
        state.update(ckpt["model"])
        print(f"[warn] checkpoint has no ema key, falling back to model weights (step={ckpt.get('step')})")
    model.load_state_dict(state)
    return model, cfg
