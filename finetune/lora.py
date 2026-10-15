"""Dependency-free LoRA for KiT, plus merging back into a standard checkpoint.

A targeted ``nn.Linear`` is replaced by :class:`LoRALinear`, which keeps the
frozen base layer and adds ``lora_A`` [r, in] / ``lora_B`` [out, r] (B zero-init,
so the adapter starts as an identity at step 0):

    y = base(x) + scaling * (x @ A.T @ B.T),   scaling = alpha / r

Because the wrapped state-dict keys (``<prefix>.base.weight`` + ``lora_*``) do
not match a plain ``KlineDiT``, :func:`merge_lora_state_dict` folds the adapters
back into the base weights, producing a checkpoint the eval scripts load
directly.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Frozen ``base`` linear + trainable low-rank adapter."""

    def __init__(self, base: nn.Linear, r: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        self.r = int(r)
        self.alpha = float(alpha)
        self.lora_A = nn.Parameter(torch.empty(self.r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("lora_scaling", torch.tensor(self.alpha / max(1, self.r)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        if self.r > 0:
            out = out + (x @ self.lora_A.t() @ self.lora_B.t()) * self.lora_scaling.to(x.dtype)
        return out


def _parent_and_key(model: nn.Module, dotted: str) -> Tuple[nn.Module, object]:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    last = parts[-1]
    return parent, (int(last) if last.isdigit() else last)


def _set_child(parent: nn.Module, key, value: nn.Module) -> None:
    if isinstance(key, int):
        parent[key] = value
    else:
        setattr(parent, key, value)


def _block_index(name: str):
    if name.startswith("blocks."):
        part = name.split(".", 2)[1]
        if part.isdigit():
            return int(part)
    return None


def inject_lora(model: nn.Module, targets: Iterable[str] = ("qkv", "out_proj"),
                r: int = 16, alpha: float = 32.0, last_n_layers: int = 0,
                include_ffn: bool = True, blocks_only: bool = True) -> List[str]:
    """Replace matching ``nn.Linear`` modules with :class:`LoRALinear`.

    ``targets`` are the leaf attribute names to adapt (e.g. ``qkv``, ``out_proj``,
    ``w1``, ``w2``, ``w3``).  ``last_n_layers`` > 0 restricts adaptation to the
    last N transformer blocks (0 = all).  With ``blocks_only`` (default) only
    modules inside ``model.blocks`` are adapted, so the AdaLN trunk and the output
    head are left untouched.  Returns the adapted module names.
    """
    targets = set(targets)
    if include_ffn:
        targets |= {"w1", "w2", "w3"}
    n_layers = len(getattr(model, "blocks", []))
    lo = max(0, n_layers - last_n_layers) if last_n_layers > 0 else 0

    to_wrap: List[Tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        leaf = name.rsplit(".", 1)[-1]
        if leaf not in targets:
            continue
        bi = _block_index(name)
        if blocks_only and bi is None:
            continue
        if bi is not None and bi < lo:
            continue
        to_wrap.append((name, module))

    wrapped = []
    for name, module in to_wrap:
        parent, key = _parent_and_key(model, name)
        _set_child(parent, key, LoRALinear(module, r, alpha))
        wrapped.append(name)
    return wrapped


def mark_only_lora_trainable(model: nn.Module) -> int:
    """Freeze everything, then unfreeze LoRA parameters. Returns the count."""
    for p in model.parameters():
        p.requires_grad = False
    n = 0
    for name, p in model.named_parameters():
        if "lora_" in name:
            p.requires_grad = True
            n += p.numel()
    return n


def lora_parameter_names(model: nn.Module) -> List[str]:
    return [n for n, _ in model.named_parameters() if "lora_" in n]


def merge_lora_(model: nn.Module) -> nn.Module:
    """In-place fold every LoRA adapter into its base weight; returns ``model``."""
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for name, module in targets:
        with torch.no_grad():
            delta = (module.lora_B @ module.lora_A) * module.lora_scaling.to(module.base.weight.dtype)
            module.base.weight.add_(delta)
        parent, key = _parent_and_key(model, name)
        _set_child(parent, key, module.base)
    return model


def merge_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Return a copy of ``state_dict`` with LoRA adapters folded into weights."""
    out: Dict[str, torch.Tensor] = {}
    prefixes = set()
    for k, v in state_dict.items():
        if k.endswith(".lora_A") or k.endswith(".lora_B") or k.endswith(".lora_scaling"):
            prefixes.add(k.rsplit(".", 1)[0])
            continue
        out[k] = v
    for p in prefixes:
        A = state_dict[p + ".lora_A"]
        B = state_dict[p + ".lora_B"]
        W = state_dict[p + ".base.weight"]
        s = float(state_dict[p + ".lora_scaling"])
        out[p + ".weight"] = W + s * (B.to(W.dtype) @ A.to(W.dtype))
        out.pop(p + ".base.weight", None)
    return out


__all__ = ["LoRALinear", "inject_lora", "mark_only_lora_trainable",
           "lora_parameter_names", "merge_lora_", "merge_lora_state_dict"]
