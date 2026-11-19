"""Check the opt-in mapped loader against normal checkpoint initialization."""

from pathlib import Path
import tempfile
import unittest

import torch

from kit_v1.config import Config, ModelConfig
from kit_v1.infer import load_model_and_config
from kit_v1.model import KlineDiT


def small_config():
    return ModelConfig(
        d_model=8,
        n_layers=2,
        n_heads=2,
        ffn_hidden=16,
        inst_vocab=8,
        inst_dim=4,
        t_emb_dim=8,
        sess_hidden=8,
    )


class TestMappedCheckpoint(unittest.TestCase):
    def test_complete_ema_restores_derived_rotary_buffers(self):
        config = Config(model=small_config())
        reference = KlineDiT(config.model).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "complete.pt")
            torch.save({"config": config.to_dict(), "ema": reference.state_dict()}, path)
            loaded, _ = load_model_and_config(path, "tiny", seed=7, mmap=True)
            self.assertTrue(torch.equal(loaded.in_proj.weight, reference.in_proj.weight))
            for old, new in zip(reference.blocks, loaded.blocks):
                expected, actual = old.attn.rope, new.attn.rope
                self.assertEqual(actual.cos_cached.device.type, "cpu")
                self.assertTrue(torch.equal(expected.cos_cached, actual.cos_cached))
                self.assertTrue(torch.equal(expected.sin_cached, actual.sin_cached))
            del loaded, new, actual

    def test_partial_ema_keeps_standard_initialization_and_overlay(self):
        config = Config(model=small_config())
        checkpoint = {
            "config": config.to_dict(),
            "ema": {"shadow": {"out_proj.bias": torch.full((5,), 0.25)}},
        }
        with tempfile.TemporaryDirectory() as tmp, torch.random.fork_rng(devices=[]):
            path = str(Path(tmp) / "partial.pt")
            torch.save(checkpoint, path)
            torch.manual_seed(731)
            expected, _ = load_model_and_config(path, "tiny", seed=7)
            torch.manual_seed(731)
            mapped, _ = load_model_and_config(path, "tiny", seed=7, mmap=True)
        actual = mapped.state_dict()
        for name, value in expected.state_dict().items():
            with self.subTest(name=name):
                self.assertTrue(torch.equal(value, actual[name]))
        self.assertTrue(
            torch.equal(mapped.out_proj.bias, checkpoint["ema"]["shadow"]["out_proj.bias"])
        )


if __name__ == "__main__":
    unittest.main()
