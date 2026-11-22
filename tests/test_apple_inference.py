"""Exercise Apple sampler dispatch, optional native arithmetic, and checkpoint compatibility."""

import unittest
from contextlib import ExitStack
from dataclasses import replace
from unittest.mock import patch

import torch

from kit_v1.config import DEFAULT_SCALE_WINDOWS, DataConfig, ModelConfig
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


def window_inputs(history_bars=250, horizon=20, scale="1d"):
    bars = history_bars + horizon
    cond = {k: torch.zeros(1, bars, w) for k, w in [("clock", 16), ("sess", 2), ("yday", 8)]}
    cond.update({k: torch.zeros(1, bars, dtype=torch.long) for k in ("dow", "event", "month")})
    cond.update({k: torch.zeros(1, dtype=torch.long) for k in ("market", "sector", "inst")})
    cond["scale"] = torch.tensor([TIMESCALE_IDS[scale]])
    return torch.zeros(1, history_bars, 5), cond


PRESET_WIDTHS = (512, 768, 1024)


class TestAppleSampler(unittest.TestCase):
    def setUp(self):
        # The fresh DiT's output projection is zero. Every path therefore returns its original noise.
        self.model = KlineDiT(small_config()).eval()
        self.history, self.cond = window_inputs()

    def test_model_eligibility_presets_and_unsupported_geometry(self):
        # Inspect shapes without materializing even the small preset's weights.
        with torch.device("meta"):
            for width, ffn in ((512, 1408), (768, 2048), (1024, 2816), (512, 65)):
                cfg = replace(small_config(), d_model=width, n_heads=width // 128, ffn_hidden=ffn)
                with self.subTest(width=width):
                    self.assertTrue(runtime._supports_model(KlineDiT(cfg).eval()))
            wrong_head_dim = replace(small_config(), d_model=512, n_heads=2, ffn_hidden=64)
            self.assertFalse(runtime._supports_model(KlineDiT(wrong_head_dim).eval()))

    def test_draw_once_keep_path_order_and_generator_state(self):
        batches = []
        windows = []

        class EchoPlan:
            def __init__(self, model, cpu_ops=None):
                self.lane = "cpu" if cpu_ops is not None else "gpu"

            def sample(self, window, noise):
                batches.append((self.lane, noise.shape[0]))
                windows.append(window)
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

            def prepared_window(model, history, cond, steps):
                return history.shape[1], cond["clock"].shape[1], cond["scale"].item()

            stack.enter_context(patch.object(runtime, "prepare", side_effect=prepared_window))
            stack.enter_context(patch.object(runtime, "TorchPlan", EchoPlan))
            stack.enter_context(
                patch.object(
                    runtime, "flow_sample", side_effect=AssertionError("eligible window fell back")
                )
            )
            sampler = stack.enter_context(AppleSampler(self.model, cpu_paths=4))

            def check_chunk(sampler, chunk, sizes, history_bars=250, horizon=20, scale="1d"):
                batches.clear()
                windows.clear()
                history, cond = window_inputs(history_bars, horizon, scale)
                output = sampler.sample(
                    history,
                    cond,
                    Lh=horizon,
                    n_paths=17,
                    steps=1,
                    chunk_size=chunk,
                    generator=actual_rng,
                )
                expected = torch.randn(17, horizon, 5, generator=reference_rng).unsqueeze(0)
                self.assertEqual(sorted(batches), sorted(sizes))
                self.assertEqual(
                    set(windows), {(history_bars, history_bars + horizon, TIMESCALE_IDS[scale])}
                )
                self.assertEqual(output.device.type, "cpu")
                self.assertTrue(torch.equal(output, expected))
                self.assertTrue(torch.equal(actual_rng.get_state(), reference_rng.get_state()))

            data = DataConfig(scale_windows=DEFAULT_SCALE_WINDOWS)
            self.assertEqual(set(data.windows_map()), set(TIMESCALE_IDS))
            split = [("gpu", 12), ("cpu", 4), ("gpu", 1)]
            for scale in TIMESCALE_IDS:
                with self.subTest(scale=scale):
                    check_chunk(sampler, 16, split, *data.window_for(scale), scale)
            # The sampler API also accepts a custom window under an existing scale ID.
            check_chunk(sampler, 16, split, 13, 3, "30m")
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
            stack.enter_context(
                patch.object(
                    runtime,
                    "TorchPlan",
                    side_effect=AssertionError("unsupported configuration reached the fast path"),
                )
            )
            fallback = stack.enter_context(
                patch.object(runtime, "flow_sample", side_effect=standard_on_cpu)
            )
            sampler = stack.enter_context(AppleSampler(self.model))
            cases = [
                (self.cond, {"solver": "heun"}),
                (self.cond, {"cfg_w": 1.5}),
                (self.cond, {"hist_cfg_w": 1.25}),
                (dict(self.cond, pad_mask=mask), {}),
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

    def test_packed_ffn_index_limit_uses_actual_path_batch(self):
        # Includes four registers: 16 × 2049 × 2 × 65536 exceeds uint32 indexing.
        history, cond = window_inputs(2044, 1)
        cfg = replace(small_config(), d_model=512, n_heads=4, ffn_hidden=65536)
        self.assertFalse(runtime._supports_window(history, cond, 1, cfg, 16))
        self.assertTrue(runtime._supports_window(history, cond, 1, cfg, 1))

    def test_invalid_window_layouts_defer_without_drawing_noise(self):
        history, cond = window_inputs(13, 3, "30m")
        missing_identity = {name: val for name, val in cond.items() if name != "inst"}
        cases = {
            "batch": (history.expand(2, -1, -1), cond),
            "features": (history[:, :, :4], cond),
            "clock_length": (history, dict(cond, clock=cond["clock"][:, :-1])),
            "missing_identity": (history, missing_identity),
            "mask_type": (history, dict(cond, pad_mask=torch.ones(1, 16, dtype=torch.int8))),
            "mask_length": (history, dict(cond, pad_mask=torch.ones(1, 15, dtype=torch.bool))),
        }
        rng = torch.Generator().manual_seed(19)
        before = rng.get_state()
        with ExitStack() as stack:
            stack.enter_context(patch.object(runtime, "_copy_model", return_value=self.model))
            stack.enter_context(patch.object(runtime, "_supports_model", return_value=True))
            stack.enter_context(patch.object(runtime, "kernels_available", return_value=True))
            stack.enter_context(
                patch.object(
                    runtime,
                    "TorchPlan",
                    side_effect=AssertionError("invalid window reached the fast path"),
                )
            )
            fallback = stack.enter_context(
                patch.object(runtime, "flow_sample", return_value=torch.empty(0))
            )
            sampler = stack.enter_context(AppleSampler(self.model))
            for label, (x, conditions) in cases.items():
                with self.subTest(layout=label):
                    fallback.reset_mock()
                    sampler.sample(x, conditions, Lh=3, steps=1, n_paths=17, generator=rng)
                    fallback.assert_called_once()
                    self.assertTrue(torch.equal(rng.get_state(), before))


class TestAppleNumerics(unittest.TestCase):
    def check_two_euler_steps(self, device, width, cpu_ops=None):
        # Exercise 1-, 2-, and 3-element scalar tails after the SIMD groups of four.
        ffn = {512: 65, 768: 66, 1024: 67}[width]
        config = replace(small_config(), d_model=width, n_heads=width // 128, ffn_hidden=ffn)
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
        model.to(device)
        plan = TorchPlan(model, cpu_ops=cpu_ops)
        inputs_rng = torch.Generator().manual_seed(532)
        # Reuse the weights across distinct sequence lengths and a one-bar tail query.
        for history_bars, horizon, scale in ((4, 2, "1d"), (17, 5, "5m"), (2, 1, "2h")):
            with self.subTest(history_bars=history_bars, horizon=horizon, scale=scale):
                history, cond = window_inputs(history_bars, horizon, scale)
                history.normal_(generator=inputs_rng)
                cond["clock"].normal_(generator=inputs_rng)
                noise = torch.randn(2, horizon, 5, generator=torch.Generator().manual_seed(12))
                reference = flow_sample(
                    model,
                    history,
                    cond,
                    Lh=horizon,
                    steps=2,
                    n_paths=2,
                    generator=torch.Generator().manual_seed(12),
                    device=device,
                ).cpu()
                result = plan.sample(prepare(model, history, cond, steps=2), noise).cpu()
                self.assertGreater((reference - noise.unsqueeze(0)).abs().max().item(), 0.01)
                torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-5)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_metal_plan_matches_model(self):
        for width in PRESET_WIDTHS:
            with self.subTest(width=width):
                if not runtime.kernels_available(width):
                    self.skipTest(f"Metal kernels are unavailable for width {width}")
                self.check_two_euler_steps("mps", width)

    def test_native_cpu_plan_matches_model(self):
        ops = runtime.load_cpu_ops()
        if ops is None:
            self.skipTest("Native Apple CPU kernels are unavailable")
        for width in PRESET_WIDTHS:
            with self.subTest(width=width):
                self.check_two_euler_steps("cpu", width, cpu_ops=ops)


if __name__ == "__main__":
    unittest.main()
