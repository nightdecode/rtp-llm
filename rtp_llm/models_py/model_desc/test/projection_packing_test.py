"""Packing several projections into one GEMM must not change their outputs.

Pure torch on CPU: loaded by file path so the test does not pull the engine's
compiled extensions in.
"""

import importlib.util
import unittest
from pathlib import Path

import torch
from torch import nn

path = Path(__file__).parents[1] / "projection_packing.py"
spec = importlib.util.spec_from_file_location("projection_packing", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

packed_projection_weight = module.packed_projection_weight


class _F16Linear(nn.Module):
    """Mirrors CudaF16Linear: the checkpoint's [in, out] weight stored as [out, in]."""

    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = weight.T
        self.bias = bias

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class PackedProjectionTest(unittest.TestCase):
    hidden = 256

    def _projection(self, out_width, dtype=torch.bfloat16, bias=None):
        return _F16Linear(torch.randn(self.hidden, out_width, dtype=dtype), bias=bias)

    def test_packed_matches_separate_bitwise(self):
        torch.manual_seed(37)
        widths = [8, 32, 32]
        projections = [self._projection(w) for w in widths]
        parts = [packed_projection_weight(p) for p in projections]
        self.assertTrue(all(p is not None for p in parts))
        self.assertEqual([p.shape[1] for p in parts], widths)

        packed = torch.cat(parts, dim=1).contiguous()
        x = torch.randn(7, self.hidden, dtype=torch.bfloat16)
        merged = torch.mm(x, packed).split(widths, dim=-1)
        for i, (got, projection) in enumerate(zip(merged, projections)):
            want = projection(x)
            self.assertEqual(got.shape, want.shape, f"part {i}")
            self.assertTrue(torch.equal(got, want), f"part {i} differs")

    def test_rejects_unpackable_projections(self):
        torch.manual_seed(37)
        with_bias = self._projection(4, bias=torch.zeros(4, dtype=torch.bfloat16))
        self.assertIsNone(packed_projection_weight(with_bias))
        self.assertIsNone(packed_projection_weight(self._projection(4, torch.float16)))
        self.assertIsNone(packed_projection_weight(self._projection(4, torch.float32)))
        self.assertIsNone(packed_projection_weight(nn.Identity()))

    def test_rejects_non_contiguous_weight(self):
        torch.manual_seed(37)
        wide = _F16Linear(torch.randn(self.hidden, 64, dtype=torch.bfloat16))
        # A column slice of the stored [out, in] weight is not a contiguous
        # [in, out] matrix once transposed back.
        wide.weight = wide.weight[:, ::2]
        self.assertIsNone(packed_projection_weight(wide))


if __name__ == "__main__":
    unittest.main()
