"""Optional FP32 inference for a single KiT forecast window on Apple silicon."""

from .runtime import AppleSampler

__all__ = ["AppleSampler"]
