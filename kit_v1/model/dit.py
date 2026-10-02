"""KlineDiT: single-stream DiT over [registers | context | target] tokens.

Global conditions modulate blocks through a shared AdaLN; per-bar calendar
features are added to the token embeddings.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

from kit_v1.config import ModelConfig
from kit_v1.model.adaln import SharedAdaLN
from kit_v1.model.block import DiTBlock
from kit_v1.model.embeddings import (CalendarEmbed, CondEmbeddings,
                                        TimestepEmbedder, make_register_tokens,
                                        make_role_embedding)
from kit_v1.model.norms import RMSNorm


class KlineDiT(nn.Module):
    """v_pred = model(z, t, cond, Lh); outputs the velocity for the target segment only."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model

        self.in_proj = nn.Linear(cfg.in_features, d)
        self.role_emb = make_role_embedding(d)
        self.register = make_register_tokens(cfg.n_registers, d)
        self.calendar = CalendarEmbed(d, cfg.clock_dim, cfg.sess_hidden,
                                      cfg.yday_dim)
        self.cond_emb = CondEmbeddings(d, cfg.market_slots, cfg.sector_slots,
                                       cfg.scale_slots, cfg.inst_vocab, cfg.inst_dim)
        self.t_emb = TimestepEmbedder(cfg.t_emb_dim, d)
        self.adaln = SharedAdaLN(d)
        self.blocks = nn.ModuleList([
            DiTBlock(d, cfg.n_heads, cfg.ffn_hidden, qk_norm=cfg.qk_norm,
                     rope_theta=cfg.rope_theta, backend=cfg.attn_backend)
            for _ in range(cfg.n_layers)
        ])
        # output head
        self.final_norm = RMSNorm(d, affine=False)
        self.head_bias = nn.Parameter(torch.zeros(2 * d))
        self.out_proj = nn.Linear(d, cfg.in_features)

        # Linear/Embedding: normal(0, 0.02), bias 0
        self.apply(self._init_weights)
        # re-zero AdaLN / output layers (apply() above overwrote them)
        self._zero_init()

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _zero_init(self) -> None:
        nn.init.zeros_(self.adaln.w2.weight)
        nn.init.zeros_(self.adaln.w2.bias)
        nn.init.zeros_(self.adaln.w_head.weight)
        nn.init.zeros_(self.adaln.w_head.bias)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        nn.init.zeros_(self.head_bias)
        for blk in self.blocks:
            nn.init.zeros_(blk.bias_i)

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        Lh: int,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """z [B,L,F] (clean ctx + noisy tgt); t [B] float in [0,1]; Lh = target length.

        cond: clock [B,L,16], sess [B,L,2], yday [B,L,8] float;
        dow/event/month [B,L] long; market/sector/inst/scale [B] long;
        pad_mask [B,L] bool (True=valid, optional).
        Returns v_pred [B,Lh,F] (plus intermediate modulations if return_aux).
        """
        B, L, _ = z.shape
        Lc = L - Lh
        n_reg = self.cfg.n_registers
        device = z.device

        pad_mask: Optional[torch.Tensor] = cond.get("pad_mask")
        if pad_mask is None:
            pad_mask = torch.ones(B, L, dtype=torch.bool, device=device)

        role_tok = torch.cat([
            torch.zeros(Lc, dtype=torch.long, device=device),
            torch.ones(Lh, dtype=torch.long, device=device),
        ]).unsqueeze(0).expand(B, -1)                                   # [B,L]

        tok = (self.in_proj(z) + self.role_emb(role_tok)
               + self.calendar(cond["clock"], cond["sess"],
                               cond["dow"], cond["event"],
                               cond["month"], cond["yday"]))            # [B,L,d]

        reg = self.register.unsqueeze(0).expand(B, -1, -1)              # [B,n_reg,d]
        x = torch.cat([reg, tok], dim=1)                                # [B,n_reg+L,d]
        full_role = torch.cat([
            torch.zeros(B, n_reg, dtype=torch.long, device=device), role_tok,
        ], dim=1)                                                       # [B,n_reg+L]
        full_pad = torch.cat([
            torch.ones(B, n_reg, dtype=torch.bool, device=device), pad_mask,
        ], dim=1)                                                       # [B,n_reg+L]
        positions = torch.arange(n_reg + L, device=device)              # RoPE uses assembled indices

        c_base = self.cond_emb(cond["market"], cond["sector"],
                               cond["inst"], cond["scale"])             # [B,d]
        c_mod_tgt, head_mod = self.adaln(self.t_emb(t), c_base)
        c_mod_ctx, _ = self.adaln(self.t_emb(torch.zeros_like(t)), c_base)

        for blk in self.blocks:
            if self.cfg.use_activation_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, c_mod_ctx, c_mod_tgt, full_role, full_pad, positions,
                    use_reentrant=False)
            else:
                x = blk(x, c_mod_ctx, c_mod_tgt, full_role, full_pad, positions)

        x_tgt = x[:, n_reg + Lc:, :]                                    # [B,Lh,d]
        shift_f, scale_f = (head_mod + self.head_bias).chunk(2, dim=-1)  # each [B,d]
        h = (self.final_norm(x_tgt) * (1 + scale_f.unsqueeze(1))
             + shift_f.unsqueeze(1))
        v_pred = self.out_proj(h)                                       # [B,Lh,F]

        if return_aux:
            return v_pred, {"c_mod_ctx": c_mod_ctx, "c_mod_tgt": c_mod_tgt,
                            "head_mod": head_mod, "c_base": c_base}
        return v_pred

    def count_params(self, trainable_only: bool = False) -> int:
        return sum(p.numel() for p in self.parameters()
                   if (p.requires_grad or not trainable_only))
