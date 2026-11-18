"""The fused MHC gating kernel must equal its torch reference.

`hc_pre` dispatches to this kernel on device and to `_hc_gates_torch` on CPU,
so the two have to agree; the Sinkhorn loop inside is iterative, which makes a
silent divergence easy to miss by eye.
"""

import unittest

import torch

from rtp_llm.models_py.model_desc.glm5_next import (
    _MHC_POST_MULT_VALUE,
    _hc_gates_torch,
)

HC_EPS = 1e-6
SINKHORN_ITERS = 20
RELATIVE_TOLERANCE = 2e-4


def _relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    denominator = want.abs().amax().clamp_min(1e-12)
    return ((got - want).abs().amax() / denominator).item()


@unittest.skipUnless(torch.cuda.is_available(), "fused MHC gating needs a device")
class MhcGatingTest(unittest.TestCase):
    def _check(self, tokens: int, hc_mult: int):
        from rtp_llm.models_py.triton_kernels.common.mhc_gating import mhc_gating

        torch.manual_seed(20261118)
        width = hc_mult * (hc_mult + 2)
        mixes = torch.randn(tokens, width, dtype=torch.float32, device="cuda")
        base = torch.randn(width, dtype=torch.float32, device="cuda")
        scale = torch.rand(3, dtype=torch.float32, device="cuda") * 2.0

        pre, post, comb = mhc_gating(
            mixes, base, scale, hc_mult, HC_EPS, SINKHORN_ITERS, _MHC_POST_MULT_VALUE
        )
        want_pre, want_post, want_comb = _hc_gates_torch(
            mixes, base, scale, hc_mult, HC_EPS, SINKHORN_ITERS
        )
        self.assertLess(_relative_error(pre, want_pre), RELATIVE_TOLERANCE)
        self.assertLess(_relative_error(post, want_post), RELATIVE_TOLERANCE)
        self.assertLess(_relative_error(comb, want_comb), RELATIVE_TOLERANCE)
        # Sinkhorn's purpose: the combination matrix is doubly stochastic.
        self.assertLess((comb.sum(-1) - 1).abs().max().item(), 1e-4)
        self.assertLess((comb.sum(-2) - 1).abs().max().item(), 1e-4)

    def test_decode_shape(self):
        self._check(tokens=1, hc_mult=4)

    def test_small_batch(self):
        self._check(tokens=8, hc_mult=4)

    def test_prefill_chunk(self):
        self._check(tokens=8192, hc_mult=4)

    def test_survives_cuda_graph_capture(self):
        """Decode replays this inside a CUDA graph, so capture must work."""
        from rtp_llm.models_py.triton_kernels.common.mhc_gating import mhc_gating

        torch.manual_seed(20261118)
        hc_mult = 4
        width = hc_mult * (hc_mult + 2)
        mixes = torch.randn(1, width, dtype=torch.float32, device="cuda")
        base = torch.randn(width, dtype=torch.float32, device="cuda")
        scale = torch.rand(3, dtype=torch.float32, device="cuda") * 2.0

        args = (mixes, base, scale, hc_mult, HC_EPS, SINKHORN_ITERS,
                _MHC_POST_MULT_VALUE)
        mhc_gating(*args)  # warm the JIT outside the capture
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = mhc_gating(*args)
        graph.replay()
        torch.cuda.synchronize()
        _, _, want_comb = _hc_gates_torch(
            mixes, base, scale, hc_mult, HC_EPS, SINKHORN_ITERS
        )
        self.assertLess(_relative_error(captured[2], want_comb), RELATIVE_TOLERANCE)


if __name__ == "__main__":
    unittest.main()
