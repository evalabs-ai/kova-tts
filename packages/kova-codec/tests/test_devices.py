"""The default-device rule, which both packages depend on agreeing about."""

from __future__ import annotations

import torch

from kova_codec import devices


class TestDefaultDevice:
    def test_names_a_device_torch_can_use(self):
        device = devices.default_device()
        assert device.type in ("cuda", "mps", "cpu")
        torch.zeros(1, device=device)  # would raise on a device that is not really there

    def test_cuda_wins_when_there_is_any(self):
        expected = "cuda" if torch.cuda.is_available() else None
        if expected:
            assert devices.default_device().type == "cuda"

    def test_metal_is_chosen_when_it_is_the_only_accelerator(self):
        mps = getattr(torch.backends, "mps", None)
        if torch.cuda.is_available() or mps is None or not mps.is_available():
            return
        assert devices.default_device().type == "mps"

    def test_it_never_raises(self):
        # Called from constructors, which must not throw because a torch build reports its
        # Metal support in an unexpected way.
        for _ in range(2):
            devices.default_device()


class TestIsAccelerator:
    def test_the_cpu_is_not_one(self):
        assert not devices.is_accelerator("cpu")
        assert not devices.is_accelerator(torch.device("cpu"))

    def test_everything_else_is(self):
        assert devices.is_accelerator("mps")
        assert devices.is_accelerator(torch.device("cuda", 1))
