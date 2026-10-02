"""Checkpoint loading for inference."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch

from kit_v1.config import Config, get_config
from kit_v1.model import KlineDiT


def load_model_and_config(
    ckpt_path: str, preset: str, seed: int, config_path: Optional[str] = None
) -> Tuple[KlineDiT, Config]:
    """Load a checkpoint (EMA weights preferred), or build a random-init model if ckpt_path is "none"."""
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
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    if "yday_dim" not in ckpt.get("config", {}).get("model", {}):
        raise RuntimeError(
            f"checkpoint {p} predates the month/yday calendar features; an old "
            "ckpt run on new code gets untrained month/yday embeddings, so behavior differs from "
            "the trained version; run inference with the old code or retrain.")
    cfg = Config.from_dict(ckpt["config"])
    model = KlineDiT(cfg.model)
    # EMA holds parameters only; merge it over the model state
    state = dict(model.state_dict())
    ema = ckpt.get("ema")
    if ema:
        # accept either {"shadow": {...}} or a flat param dict
        shadow = ema["shadow"] if isinstance(ema, dict) and "shadow" in ema else ema
        state.update({k: v.to(torch.float32) for k, v in shadow.items()})
        print(f"[info] loaded EMA weights (step={ckpt.get('step')})")
    else:
        state.update(ckpt["model"])
        print(f"[warn] checkpoint has no ema key, falling back to model weights (step={ckpt.get('step')})")
    model.load_state_dict(state)
    return model, cfg
