"""Run independent Euler trajectories on the MPS device and an optional CPU worker."""

from concurrent.futures import ThreadPoolExecutor, wait
from copy import deepcopy
from operator import index
from threading import RLock

import torch

from kit_v1.infer.sampler import flow_sample

from .metal import kernels_available
from .native import load_cpu_ops
from .plan import TorchPlan, prepare


def _copy_model(model, device):
    # Copy the module tree with each tensor transferred directly to its destination.
    # This keeps the CPU replica and avoids a temporary second copy of the CPU weights.
    tensors = {
        id(p): torch.nn.Parameter(p.detach().to(device), requires_grad=False)
        for p in model.parameters()
    }
    tensors.update({id(b): b.to(device) for b in model.buffers()})
    return deepcopy(model, tensors)


def _supports_model(model):
    cfg = model.cfg
    return (
        cfg.n_heads > 0
        and cfg.d_model == 128 * cfg.n_heads
        and cfg.ffn_hidden > 0
        and all(b.attn.backend == "sdpa" for b in model.blocks)
    )


def _supports_window(history, cond, horizon, cfg, path_batch):
    # The CLI chooses Lc and Lh from the checkpoint's data.window_for(scale).
    # The kernels only need a single dense window and aligned calendar tensors.
    if history.ndim != 3 or history.dtype != torch.float32 or horizon <= 0:
        return False
    batch, context, width = history.shape
    if batch != 1 or context <= 0 or width != cfg.in_features:
        return False
    tokens = context + horizon
    # Metal indexes packed QKV and gate/up tensors in float elements with uint offsets.
    max_elements = (
        path_batch * (tokens + cfg.n_registers) * max(3 * cfg.d_model, 2 * cfg.ffn_hidden)
    )
    if max_elements > 2**32:
        return False
    for key in ("clock", "sess", "yday", "dow", "event", "month"):
        value = cond.get(key)
        ndim = 3 if key in ("clock", "sess", "yday") else 2
        if (
            not isinstance(value, torch.Tensor)
            or value.ndim != ndim
            or value.shape[:2] != (batch, tokens)
        ):
            return False
    for key in ("scale", "market", "sector", "inst"):
        value = cond.get(key)
        if not isinstance(value, torch.Tensor) or value.shape != (batch,):
            return False
    mask = cond.get("pad_mask")
    return mask is None or (
        isinstance(mask, torch.Tensor)
        and mask.dtype == torch.bool
        and mask.shape == (batch, tokens)
        and bool(mask.all())
    )


class AppleSampler:
    """Bind an immutable eval-mode CPU model. Use a new session after changing its weights.

    The context manager releases packed weights and the worker. Results are CPU tensors.
    """

    def __init__(self, cpu_model, *, cpu_paths=4):
        if cpu_model.training or next(cpu_model.parameters()).device.type != "cpu":
            raise ValueError("AppleSampler expects an eval-mode CPU model")
        if isinstance(cpu_paths, bool) or not 0 <= index(cpu_paths) <= 16:
            raise ValueError("cpu_paths must be an integer between 0 and 16")
        self._cpu_model = cpu_model
        self.model = _copy_model(cpu_model, torch.device("mps"))
        self._cpu_paths = index(cpu_paths)
        self._eligible = _supports_model(self.model)
        self._gpu = self._cpu = self._pool = None
        self._lock = RLock()
        self._closed = False
        self.last_run = {}

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._lock:
            if self._pool is not None:
                self._pool.shutdown()
            self._gpu = self._cpu = self._pool = None
            self.model = self._cpu_model = None
            self._closed = True

    @torch.inference_mode()
    def sample(
        self,
        x_ctx,
        cond,
        Lh,
        steps=32,
        solver="euler",
        n_paths=64,
        cfg_w=1.0,
        hist_cfg_w=1.0,
        generator=None,
        chunk_size=256,
        precision="fp32",
    ):
        """Fall back before drawing noise for unsupported models, windows, or solvers."""
        with self._lock:
            if self._closed:
                raise RuntimeError("This AppleSampler has been closed")
            batch = min(16, chunk_size, n_paths)
            if (
                self._eligible
                and solver == "euler"
                and cfg_w == hist_cfg_w == 1.0
                and precision == "fp32"
                and min(steps, n_paths, chunk_size) > 0
                and _supports_window(x_ctx, cond, Lh, self.model.cfg, batch)
                and kernels_available(self.model.cfg.d_model)
                and torch.get_default_dtype() == torch.float32
                and not torch.is_autocast_enabled("mps")
                and not torch.is_autocast_enabled("cpu")
            ):
                try:
                    split = self._prepare_plans(n_paths, batch)
                except ValueError:
                    # The weights or normalization differ from the fused kernel contract.
                    self._eligible = False
                else:
                    return self._euler(x_ctx, cond, Lh, steps, n_paths, generator, batch, split)

            self.last_run = {"backend": "flow_sample"}
            return flow_sample(
                self.model,
                x_ctx,
                cond,
                Lh,
                steps=steps,
                solver=solver,
                n_paths=n_paths,
                cfg_w=cfg_w,
                hist_cfg_w=hist_cfg_w,
                generator=generator,
                device="mps",
                chunk_size=chunk_size,
                precision=precision,
            ).cpu()

    def _prepare_plans(self, paths, batch):
        split = self._cpu_paths if batch == 16 and paths >= 16 else 0
        if split and self._cpu is None:
            ops = load_cpu_ops()
            if ops is None:
                self._cpu_paths = split = 0
            else:
                self._cpu = TorchPlan(self._cpu_model, cpu_ops=ops)
                self._pool = ThreadPoolExecutor(1, thread_name_prefix="kit-cpu")

        if self._gpu is None and (split != 16 or paths % 16):
            self._gpu = TorchPlan(self.model)
        return split

    def _euler(self, history, cond, horizon, steps, paths, generator, batch, split):
        cpu_window = prepare(self._cpu_model, history, cond, steps) if split else None
        gpu_window = (
            prepare(self.model, history, cond, steps) if split != 16 or paths % 16 else None
        )
        # Match flow_sample's one draw, including its generator device and final RNG state.
        draws = torch.randn(
            paths,
            horizon,
            history.shape[-1],
            dtype=history.dtype,
            generator=generator,
            device=generator.device if generator is not None else "cpu",
        )
        groups = []
        cpu_total = 0
        for chunk in draws.split(batch):
            n_cpu = split if len(chunk) == 16 else 0
            n_gpu = len(chunk) - n_cpu
            cpu_total += n_cpu
            future = (
                self._pool.submit(self._cpu.sample, cpu_window, chunk[n_gpu:]) if n_cpu else None
            )
            try:
                if n_gpu:
                    groups.append(self._gpu.sample(gpu_window, chunk[:n_gpu]).cpu())
            except BaseException:
                if future is not None:
                    wait([future])
                raise
            if future is not None:
                groups.append(future.result())
        self.last_run = {
            "backend": "apple_fp32",
            "gpu_paths": paths - cpu_total,
            "cpu_paths": cpu_total,
        }
        return groups[0] if len(groups) == 1 else torch.cat(groups, dim=1)
