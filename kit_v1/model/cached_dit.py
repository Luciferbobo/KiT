"""Inference prototype with a one-shot context cache and target-only blocks."""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from kit_v1.model.dit import KlineDiT


ContextCache = List[Tuple[torch.Tensor, torch.Tensor]]


def _modulation(block, c_mod: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return (c_mod + block.bias_i).chunk(6, dim=-1)


def _qkv(attn, h: torch.Tensor, positions: torch.Tensor):
    batch, length, width = h.shape
    qkv = attn.qkv(h).view(
        batch, length, 3, attn.n_heads, attn.head_dim
    )
    q, k, v = qkv.unbind(dim=2)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    if attn.q_norm is not None:
        q = attn.q_norm(q)
        k = attn.k_norm(k)
    q, k = attn.rope.apply_rope(q, k, positions)
    return q, k, v


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    key_valid: torch.Tensor,
) -> torch.Tensor:
    mask = key_valid[:, None, None, :]
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class CachedKlineDiT:
    """Reuse a trained KlineDiT while separating context and target computation."""

    def __init__(self, model: KlineDiT) -> None:
        self.model = model
        self.cfg = model.cfg
        self.n_reg = self.cfg.n_registers

    def prepare_context(
        self,
        x_ctx: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        path_repeat: int = 1,
    ) -> Tuple[ContextCache, torch.Tensor]:
        """Encode the fixed context once and optionally replicate it per sampled path."""
        model = self.model
        batch, context_len, _ = x_ctx.shape
        device = x_ctx.device

        context_tokens = (
            model.in_proj(x_ctx)
            + model.role_emb(
                torch.zeros(
                    batch, context_len, dtype=torch.long, device=device
                )
            )
            + model.calendar(
                cond["clock"][:, :context_len],
                cond["sess"][:, :context_len],
                cond["dow"][:, :context_len],
                cond["event"][:, :context_len],
                cond["month"][:, :context_len],
                cond["yday"][:, :context_len],
            )
        )
        registers = model.register.unsqueeze(0).expand(batch, -1, -1)
        x = torch.cat([registers, context_tokens], dim=1)
        prefix_len = self.n_reg + context_len
        positions = torch.arange(prefix_len, device=device)

        pad_context = cond.get("pad_mask")
        if pad_context is None:
            pad_context = torch.ones(
                batch, context_len, dtype=torch.bool, device=device
            )
        else:
            pad_context = pad_context[:, :context_len]
        full_pad = torch.cat(
            [
                torch.ones(batch, self.n_reg, dtype=torch.bool, device=device),
                pad_context,
            ],
            dim=1,
        )

        c_base = model.cond_emb(
            cond["market"], cond["sector"], cond["inst"], cond["scale"]
        )
        c_mod_ctx, _ = model.adaln(
            model.t_emb(torch.zeros(batch, device=device)), c_base
        )

        caches: ContextCache = []
        for block in model.blocks:
            def context_block(block, x, c_mod, valid, positions):
                shift1, scale1, gate1, shift2, scale2, gate2 = _modulation(
                    block, c_mod
                )
                h = (
                    block.norm1(x) * (1 + scale1.unsqueeze(1))
                    + shift1.unsqueeze(1)
                )
                q, k, v = _qkv(block.attn, h, positions)
                attended = _attend(q, k, v, valid)
                attended = attended.transpose(1, 2).reshape(
                    x.shape[0], x.shape[1], -1
                )
                x = x + gate1.unsqueeze(1) * block.attn.out_proj(attended)
                h = (
                    block.norm2(x) * (1 + scale2.unsqueeze(1))
                    + shift2.unsqueeze(1)
                )
                x = x + gate2.unsqueeze(1) * block.mlp(h)
                return x, k, v

            if model.cfg.use_activation_checkpointing and model.training:
                x, k, v = torch.utils.checkpoint.checkpoint(
                    context_block,
                    block,
                    x,
                    c_mod_ctx,
                    full_pad,
                    positions,
                    use_reentrant=False,
                )
            else:
                x, k, v = context_block(
                    block, x, c_mod_ctx, full_pad, positions
                )
            caches.append((k, v))

        repeat = int(path_repeat)
        if repeat > 1:
            caches = [
                (
                    k.repeat_interleave(repeat, dim=0),
                    v.repeat_interleave(repeat, dim=0),
                )
                for k, v in caches
            ]
            full_pad = full_pad.repeat_interleave(repeat, dim=0)
        return caches, full_pad

    def forward_target(
        self,
        z_target: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        caches: ContextCache,
        context_valid: torch.Tensor,
        context_len: int,
    ) -> torch.Tensor:
        """Predict velocity for target tokens only, using the cached context K/V."""
        model = self.model
        batch, target_len, _ = z_target.shape
        device = z_target.device
        prefix_len = context_valid.shape[1]

        target_tokens = (
            model.in_proj(z_target)
            + model.role_emb(
                torch.ones(
                    batch, target_len, dtype=torch.long, device=device
                )
            )
            + model.calendar(
                cond["clock"][:, context_len:],
                cond["sess"][:, context_len:],
                cond["dow"][:, context_len:],
                cond["event"][:, context_len:],
                cond["month"][:, context_len:],
                cond["yday"][:, context_len:],
            )
        )
        x = target_tokens
        positions = torch.arange(
            prefix_len, prefix_len + target_len, device=device
        )
        c_base = model.cond_emb(
            cond["market"], cond["sector"], cond["inst"], cond["scale"]
        )
        c_mod_target, head_mod = model.adaln(model.t_emb(t), c_base)

        target_valid = cond.get("pad_mask")
        if target_valid is None:
            target_valid = torch.ones(
                batch, target_len, dtype=torch.bool, device=device
            )
        else:
            target_valid = target_valid[:, context_len:]

        for layer_index, block in enumerate(model.blocks):
            context_k, context_v = caches[layer_index]

            def target_block(
                block,
                x,
                c_mod,
                positions,
                context_k,
                context_v,
                context_valid,
                target_valid,
            ):
                shift1, scale1, gate1, shift2, scale2, gate2 = _modulation(
                    block, c_mod
                )
                h = (
                    block.norm1(x) * (1 + scale1.unsqueeze(1))
                    + shift1.unsqueeze(1)
                )
                q, k_target, v_target = _qkv(block.attn, h, positions)
                all_k = torch.cat([context_k, k_target], dim=2)
                all_v = torch.cat([context_v, v_target], dim=2)
                all_valid = torch.cat([context_valid, target_valid], dim=1)
                attended = _attend(q, all_k, all_v, all_valid)
                attended = attended.transpose(1, 2).reshape(
                    x.shape[0], x.shape[1], -1
                )
                x = x + gate1.unsqueeze(1) * block.attn.out_proj(attended)
                h = (
                    block.norm2(x) * (1 + scale2.unsqueeze(1))
                    + shift2.unsqueeze(1)
                )
                x = x + gate2.unsqueeze(1) * block.mlp(h)
                return x

            if model.cfg.use_activation_checkpointing and model.training:
                x = torch.utils.checkpoint.checkpoint(
                    target_block,
                    block,
                    x,
                    c_mod_target,
                    positions,
                    context_k,
                    context_v,
                    context_valid,
                    target_valid,
                    use_reentrant=False,
                )
            else:
                x = target_block(
                    block,
                    x,
                    c_mod_target,
                    positions,
                    context_k,
                    context_v,
                    context_valid,
                    target_valid,
                )

        shift_f, scale_f = (head_mod + model.head_bias).chunk(2, dim=-1)
        h = (
            model.final_norm(x) * (1 + scale_f.unsqueeze(1))
            + shift_f.unsqueeze(1)
        )
        return model.out_proj(h)
