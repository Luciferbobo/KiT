"""Prepack one immutable FP32 KiT model and sample independent Euler paths."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .metal import metal_ops


@dataclass
class LayerWeights:
    qkv: torch.Tensor
    output: torch.Tensor
    up: torch.Tensor
    down: torch.Tensor


@dataclass
class Window:
    prefix: torch.Tensor
    calendar: torch.Tensor
    ctx_mod: torch.Tensor
    target_mod: torch.Tensor
    head: torch.Tensor
    deltas: list[float]


@torch.inference_mode()
def prepare(model, history, conditions, steps):
    """Precompute the calendar, historical tokens and per-step conditioning for one window."""
    device = next(model.parameters()).device
    if history.ndim != 3 or history.shape[0] != 1 or history.dtype != torch.float32 or steps < 1:
        raise ValueError("Expected one FP32 history window and a positive Euler step count")
    if torch.is_autocast_enabled(device.type):
        raise ValueError("The FP32 Apple plan does not support autocast")
    calendar = model.calendar(
        *(conditions[k].to(device) for k in ("clock", "sess", "dow", "event", "month", "yday"))
    )
    length = history.shape[1]
    observed = model.in_proj(history.to(device)) + model.role_emb.weight[0] + calendar[:, :length]
    prefix = torch.cat([model.register[None], observed], dim=1)
    ident = model.cond_emb(
        *(conditions[k].to(device) for k in ("market", "sector", "inst", "scale"))
    )
    grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")
    ctx_mod, _ = model.adaln(model.t_emb(torch.zeros(1, device=device)), ident)
    target_mod, head = model.adaln(model.t_emb(grid[:-1].to(device)), ident.expand(steps, -1))
    return Window(
        prefix, calendar[:, length:], ctx_mod, target_mod, head, (grid[:-1] - grid[1:]).tolist()
    )


def _project(x, weight):
    return torch.mm(x.reshape(-1, x.shape[-1]), weight).reshape(*x.shape[:-1], weight.shape[1])


class TorchPlan:
    """Per-device weights and operations for an immutable eval-mode KiT model."""

    def __init__(self, model, cpu_ops=None):
        self.device = next(model.parameters()).device
        if (
            model.training
            or model.cfg.n_heads < 1
            or model.cfg.d_model != 128 * model.cfg.n_heads
            or not model.blocks
        ):
            raise ValueError(
                "The Apple plan requires an eval-mode model with 128-dimensional attention heads"
            )
        if any(p.device != self.device or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("The Apple plan requires FP32 weights on one device")
        for block in model.blocks:
            if (block.attn.n_heads, block.attn.head_dim) != (model.cfg.n_heads, 128):
                raise ValueError(
                    "Attention heads must match the configured count and have 128 channels"
                )
            norms = (block.attn.q_norm, block.attn.k_norm)
            if any(
                n is None
                or n.eps != 1e-6
                or n.weight is None
                or n.weight.shape != (128,)
                or not n.weight.is_contiguous()
                for n in norms
            ):
                raise ValueError("The Apple plan requires affine Q/K RMSNorm with epsilon 1e-6")
            if any(n.eps != 1e-6 or n.weight is not None for n in (block.norm1, block.norm2)):
                raise ValueError(
                    "The Apple plan requires non-affine block RMSNorm with epsilon 1e-6"
                )
            cos, sin = block.attn.rope.cos_cached, block.attn.rope.sin_cached
            if (
                cos.ndim != 2
                or cos.shape[1] != 128
                or sin.shape != cos.shape
                or any(
                    t.device != self.device or t.dtype != torch.float32 or not t.is_contiguous()
                    for t in (cos, sin)
                )
            ):
                raise ValueError("The Apple plan requires contiguous FP32 RoPE tables")

        self.gpu = self.device.type == "mps"
        if self.gpu:
            self.ops = metal_ops(model.cfg.d_model)
        elif self.device.type == "cpu" and cpu_ops is not None:
            self.ops = cpu_ops
        else:
            raise ValueError("Expected an MPS model or a CPU model with native CPU operations")

        # MPS prefers contiguous [in, out] GEMM operands. CPU BLAS can consume
        # the transpose of each checkpoint matrix without copying it.
        def pack(weight):
            transposed = weight.detach().T
            return transposed.contiguous() if self.gpu else transposed

        self.model = model
        self.weights = [
            LayerWeights(
                pack(block.attn.qkv.weight),
                pack(block.attn.out_proj.weight),
                pack(torch.cat([block.mlp.w1.weight, block.mlp.w3.weight])),
                pack(block.mlp.w2.weight),
            )
            for block in model.blocks
        ]
        self.biases = torch.stack([block.bias_i.detach() for block in model.blocks])
        self.tail_weights = None
        if self.gpu and len(self.weights) > 1 and self.ops.tail_kernel is not None:
            dim = model.cfg.d_model
            qkv = self.weights[-1].qkv
            self.tail_weights = qkv[:, :dim].contiguous(), qkv[:, dim:].contiguous()

    @property
    def target_only_queries(self):
        return self.tail_weights is not None

    def _first_input(self, x, modulation):
        shift, scale, *_ = modulation.chunk(6)
        normalized = (
            self.model.blocks[0].norm1(x) if self.gpu else F.rms_norm(x, (x.shape[-1],), eps=1e-6)
        )
        return normalized * (1 + scale) + shift

    @torch.inference_mode()
    def sample(self, window, noise):
        """Return [1, paths, horizon, features] on this device, preserving noise order."""
        model = self.model
        if torch.is_autocast_enabled(self.device.type) or noise.dtype != torch.float32:
            raise ValueError("The Apple plan requires FP32 noise and FP32 arithmetic")
        if window.prefix.device != self.device or window.prefix.dtype != torch.float32:
            raise ValueError("Prepare the window on the same device as the plan")
        paths, horizon, _ = noise.shape
        if paths < 1 or horizon < 1 or horizon != window.calendar.shape[1]:
            raise ValueError(
                "Noise must contain at least one path and match the prepared forecast horizon"
            )
        z = noise.to(self.device)
        prefix = window.prefix
        size = prefix.shape[1]
        seq = size + horizon
        dim, heads = model.cfg.d_model, model.cfg.n_heads
        ctx_parts = (window.ctx_mod + self.biases).unbind(0)
        target_mod = window.target_mod[:, None, :] + self.biases[None, :, :]
        # History QKV at the first block is independent of both noise and time.
        prefix_qkv = _project(self._first_input(prefix, ctx_parts[0]), self.weights[0].qkv).expand(
            paths, -1, -1
        )
        for step, dt in enumerate(window.deltas):
            target = model.in_proj(z) + model.role_emb.weight[1] + window.calendar
            x = torch.cat([prefix.expand(paths, -1, -1), target], dim=1)
            targets = target_mod[step].unbind(0)
            pairs = list(zip(ctx_parts, targets))
            hidden = self._first_input(target, targets[0])
            for j, (block, weights) in enumerate(zip(model.blocks, self.weights)):
                final = j == len(self.weights) - 1
                if final and self.tail_weights is not None:
                    q, k, v = self.ops.target_qkv(hidden, *self.tail_weights, block.attn, size)
                else:
                    qkv = _project(hidden, weights.qkv)
                    if j == 0:
                        qkv = torch.cat([prefix_qkv, qkv], dim=1)
                    q, k = self.ops.qk(qkv, block.attn)
                    v = qkv.view(paths, seq, 3, heads, dim // heads)[:, :, 2].transpose(1, 2)
                    if final:
                        q = q[:, :, size:]
                if final:
                    x = x[:, size:].contiguous()
                att = F.scaled_dot_product_attention(q, k, v)
                update = _project(att.transpose(1, 2).reshape(paths, -1, dim), weights.output)
                # AdaLN uses [shift1, scale1, gate1, shift2, scale2, gate2].
                x, hidden = self.ops.gate_norm(
                    x, update, pairs[j], pairs[j], 0 if final else size, 2, 3
                )
                update = _project(self.ops.silu(_project(hidden, weights.up)), weights.down)
                if final:
                    x = x + targets[j][5 * dim :] * update
                else:
                    x, hidden = self.ops.gate_norm(x, update, pairs[j], pairs[j + 1], size, 5, 0)
            shift, scale = (window.head[step] + model.head_bias).chunk(2)
            velocity = model.out_proj(model.final_norm(x) * (1 + scale) + shift)
            z = z - dt * velocity
        return z[None]
