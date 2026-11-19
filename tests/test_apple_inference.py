"""Exercise Apple sampler dispatch, optional native arithmetic, and checkpoint compatibility."""

import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch

from kit_v1.config import ModelConfig
from kit_v1.data.calendar import TIMESCALE_IDS
from kit_v1.infer import flow_sample
from kit_v1.infer.apple import AppleSampler, runtime
from kit_v1.infer.apple.plan import TorchPlan, prepare
from kit_v1.model import KlineDiT


def small_config():
    return ModelConfig(
        d_model=8,
        n_layers=2,
        n_heads=2,
        ffn_hidden=16,
        t_emb_dim=8,
        sess_hidden=8,
        inst_vocab=8,
        inst_dim=4,
        attn_backend="sdpa",
    )


def daily_inputs(history_bars=250, horizon=20):
    bars = history_bars + horizon
    cond = {k: torch.zeros(1, bars, w) for k, w in [("clock", 16), ("sess", 2), ("yday", 8)]}
    cond.update({k: torch.zeros(1, bars, dtype=torch.long) for k in ("dow", "event", "month")})
    cond.update({k: torch.zeros(1, dtype=torch.long) for k in ("market", "sector", "inst")})
    cond["scale"] = torch.tensor([TIMESCALE_IDS["1d"]])
    return torch.zeros(1, history_bars, 5), cond


class TestAppleSampler(unittest.TestCase):
    def setUp(self):
        # The fresh DiT's output projection is zero. Every path therefore returns its original noise.
        self.model = KlineDiT(small_config()).eval()
        self.history, self.cond = daily_inputs()

    def test_draw_once_keep_path_order_and_generator_state(self):
        batches = []

        class EchoPlan:
            def __init__(self, model, cpu_ops=None):
                self.lane = "cpu" if cpu_ops is not None else "gpu"

            def sample(self, window, noise):
                batches.append((self.lane, noise.shape[0]))
                return noise.unsqueeze(0)

        actual_rng = torch.Generator().manual_seed(76)
        reference_rng = torch.Generator().manual_seed(76)
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "_copy_model", return_value=self.model))
            stack.enter_context(patch.object(runtime, "_supports_model", return_value=True))
            stack.enter_context(patch.object(runtime, "kernels_available", return_value=True))
            ops_loader = stack.enter_context(
                patch.object(runtime, "load_cpu_ops", return_value=object())
            )
            stack.enter_context(patch.object(runtime, "prepare", return_value=None))
            stack.enter_context(patch.object(runtime, "TorchPlan", EchoPlan))
            sampler = stack.enter_context(AppleSampler(self.model, cpu_paths=4))

            def check_chunk(sampler, chunk, sizes):
                batches.clear()
                output = sampler.sample(
                    self.history,
                    self.cond,
                    Lh=20,
                    n_paths=17,
                    steps=1,
                    chunk_size=chunk,
                    generator=actual_rng,
                )
                expected = torch.randn(17, 20, 5, generator=reference_rng).unsqueeze(0)
                self.assertEqual(sorted(batches), sorted(sizes))
                self.assertEqual(output.device.type, "cpu")
                self.assertTrue(torch.equal(output, expected))
                self.assertTrue(torch.equal(actual_rng.get_state(), reference_rng.get_state()))

            check_chunk(sampler, 16, [("gpu", 12), ("cpu", 4), ("gpu", 1)])
            check_chunk(sampler, 7, [("gpu", 7), ("gpu", 7), ("gpu", 3)])
            all_cpu = stack.enter_context(AppleSampler(self.model, cpu_paths=16))
            check_chunk(all_cpu, 16, [("cpu", 16), ("gpu", 1)])
            ops_loader.return_value = None
            no_native = stack.enter_context(AppleSampler(self.model, cpu_paths=4))
            check_chunk(no_native, 16, [("gpu", 16), ("gpu", 1)])
        with self.assertRaises(RuntimeError):
            sampler.sample(self.history, self.cond, Lh=20)

    def test_guidance_and_mask_defer_before_sampling(self):
        ref_rng = torch.Generator().manual_seed(4)
        call_rng = torch.Generator().manual_seed(4)
        mask = torch.ones(1, 270, dtype=torch.bool)
        mask[:, 8] = False
        # Exercise a nonzero velocity, so an uncalled model does not match the oracle.
        with torch.no_grad():
            self.model.out_proj.weight.fill_(0.125)

        def standard_on_cpu(model, history, cond, horizon, **kwargs):
            kwargs["device"] = "cpu"
            return flow_sample(model, history, cond, Lh=horizon, **kwargs)

        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "_copy_model", return_value=self.model))
            stack.enter_context(patch.object(runtime, "_supports_model", return_value=True))
            stack.enter_context(patch.object(runtime, "kernels_available", return_value=True))
            fallback = stack.enter_context(
                patch.object(runtime, "flow_sample", side_effect=standard_on_cpu)
            )
            sampler = stack.enter_context(AppleSampler(self.model))
            cases = [
                (self.cond, {"solver": "heun"}),
                (self.cond, {"cfg_w": 1.5}),
                (self.cond, {"hist_cfg_w": 1.25}),
                (dict(self.cond, pad_mask=mask), {}),
                (dict(self.cond, scale=torch.tensor([TIMESCALE_IDS["1m"]])), {}),
                (self.cond, {"precision": "bf16"}),
            ]
            for cond, options in cases:
                opts = dict(Lh=20, steps=2, n_paths=1, chunk_size=4, **options)
                expected = flow_sample(self.model, self.history, cond, generator=ref_rng, **opts)
                output = sampler.sample(self.history, cond, generator=call_rng, **opts)
                self.assertTrue(torch.equal(output, expected))
                self.assertTrue(torch.equal(ref_rng.get_state(), call_rng.get_state()))
                for name, value in options.items():
                    self.assertEqual(fallback.call_args.kwargs[name], value)
            self.assertEqual(fallback.call_count, len(cases))


class TestAppleNumerics(unittest.TestCase):
    def check_two_euler_steps(self, device, cpu_ops=None):
        config = replace(small_config(), d_model=1024, n_heads=8, ffn_hidden=64)
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.random.default_generator.manual_seed(321)
            model = KlineDiT(config).eval()
            # Upstream initializes the output and residual gates to zero. Enable both
            # attention and FFN updates, including time-dependent modulation.
            model.out_proj.weight.normal_(std=0.02)
            model.adaln.w2.weight.normal_(std=0.02)
            model.adaln.w2.bias.view(6, -1)[2].fill_(0.50)
            model.adaln.w2.bias.view(6, -1)[5].fill_(0.75)
            model.adaln.w_head.weight.normal_(std=0.02)
            for block in model.blocks:
                block.bias_i.normal_(std=0.05)
            history, cond = daily_inputs(history_bars=4, horizon=2)
            history.normal_()
            cond["clock"].normal_()
        model.to(device)
        noise = torch.randn(2, 2, 5, generator=torch.Generator().manual_seed(12))
        reference = flow_sample(
            model,
            history,
            cond,
            Lh=2,
            steps=2,
            n_paths=2,
            generator=torch.Generator().manual_seed(12),
            device=device,
        ).cpu()
        plan = TorchPlan(model, cpu_ops=cpu_ops)
        result = plan.sample(prepare(model, history, cond, steps=2), noise).cpu()
        self.assertGreater((reference - noise.unsqueeze(0)).abs().max().item(), 0.01)
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_metal_plan_matches_model(self):
        if not runtime.kernels_available():
            self.skipTest("Metal kernels are unavailable")
        self.check_two_euler_steps("mps")

    def test_native_cpu_plan_matches_model(self):
        ops = runtime.load_cpu_ops()
        if ops is None:
            self.skipTest("Native Apple CPU kernels are unavailable")
        self.check_two_euler_steps("cpu", cpu_ops=ops)


if __name__ == "__main__":
    unittest.main()
