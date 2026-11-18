"""KPool (GLM-5.3-Flash compressed indexer) math and index contracts.

Pure torch on CPU: the pooling, the pool-index expansion and the request-local
index contract are all device independent, and the reference they are compared
against lives in ``indexer_compressor.py``.
"""

import sys
import types
import unittest

import torch

from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    _hadamard_rotate_reference,
    compress_indexer_projection_reference,
)
from rtp_llm.models_py.modules.hybrid.kpool import (
    KPoolCacheLayout,
    KPoolIndexer,
    expand_pool_indices_to_tokens,
    gather_kpool_pool_keys,
    quantize_pool_keys,
    write_kpool_entries,
)

RATIO = 4
HEAD_DIM = 128
TOKENS_PER_PAGE = 256


def _write_request(layout, cache, block_table, request, length, generator):
    key = torch.randn(length, HEAD_DIM, dtype=torch.bfloat16, generator=generator)
    gate = torch.randn(length, HEAD_DIM, dtype=torch.bfloat16, generator=generator)
    token = torch.arange(length, dtype=torch.long)
    slots = (
        block_table[request, token // TOKENS_PER_PAGE].to(torch.long) * TOKENS_PER_PAGE
        + token % TOKENS_PER_PAGE
    )
    write_kpool_entries(layout, cache, slots, key, gate)
    return key, gate


class KPoolMathTest(unittest.TestCase):
    def setUp(self):
        self.generator = torch.Generator().manual_seed(20261118)
        self.layout = KPoolCacheLayout(HEAD_DIM, TOKENS_PER_PAGE)
        self.ape = torch.randn(
            RATIO, HEAD_DIM, dtype=torch.bfloat16, generator=self.generator
        )

    def test_pool_keys_match_reference(self):
        """Pooled keys equal the reference pooling, including the rotation."""
        lengths = [RATIO * 7 + 3, RATIO * 3 + 1]
        # Deliberately out-of-order pages: the gather must follow the block table.
        block_table = torch.tensor([[3, 0], [5, 1]], dtype=torch.int32)
        cache = torch.zeros(
            6, TOKENS_PER_PAGE * self.layout.entry_elems, dtype=torch.bfloat16
        )
        keys, gates = [], []
        for request, length in enumerate(lengths):
            key, gate = _write_request(
                self.layout, cache, block_table, request, length, self.generator
            )
            keys.append(key)
            gates.append(gate)

        pooled, pool_start = gather_kpool_pool_keys(
            self.layout,
            cache,
            block_table,
            torch.tensor(lengths, dtype=torch.long),
            self.ape,
            RATIO,
            rotate=True,
        )
        max_pools = (block_table.shape[1] * TOKENS_PER_PAGE) // RATIO
        self.assertEqual(pooled.shape[0], len(lengths) * max_pools)
        self.assertEqual(pool_start.tolist(), [0, max_pools])

        for request, length in enumerate(lengths):
            count = length // RATIO
            got = pooled[
                int(pool_start[request]) : int(pool_start[request]) + count
            ].float()
            want, _ = compress_indexer_projection_reference(
                keys[request], gates[request], self.ape, compress_ratio=RATIO
            )
            self.assertEqual(got.shape, want.shape)
            self.assertLess((got - want.float()).abs().max().item(), 2e-2)

    def test_hadamard_is_self_inverse(self):
        x = torch.randn(5, HEAD_DIM, generator=self.generator)
        back = _hadamard_rotate_reference(_hadamard_rotate_reference(x))
        self.assertLess((back - x).abs().max().item(), 1e-4)

    def test_expand_keeps_incomplete_tail_and_pads(self):
        context_lens = torch.tensor([RATIO * 7 + 3, RATIO * 3 + 1], dtype=torch.long)
        # Includes an out-of-range pool (99) and an invalid slot (-1).
        pool_indices = torch.tensor(
            [[0, 2, 6, -1, 99], [1, 0, 2, -1, -1]], dtype=torch.int32
        )
        width = 64
        out = expand_pool_indices_to_tokens(pool_indices, context_lens, RATIO, width)
        self.assertEqual(out.shape, (2, width))
        for request in range(2):
            row = out[request].tolist()
            valid = [v for v in row if v >= 0]
            self.assertEqual(len(valid), len(set(valid)), "no duplicate token ids")
            self.assertTrue(all(v < int(context_lens[request]) for v in valid))
            # Every complete pool contributes its `RATIO` tokens.
            for pool in pool_indices[request].tolist():
                if 0 <= pool < int(context_lens[request]) // RATIO:
                    for offset in range(RATIO):
                        self.assertIn(pool * RATIO + offset, valid)
            # The still-incomplete tail is always selected.
            complete = (int(context_lens[request]) // RATIO) * RATIO
            for token in range(complete, int(context_lens[request])):
                self.assertIn(token, valid)
        self.assertTrue(bool((out[:, -1] == -1).all()), "right padding is -1")

    def test_write_rejects_length_mismatch(self):
        cache = torch.zeros(
            2, TOKENS_PER_PAGE * self.layout.entry_elems, dtype=torch.bfloat16
        )
        key = torch.zeros(4, HEAD_DIM, dtype=torch.bfloat16)
        gate = torch.zeros(4, HEAD_DIM, dtype=torch.bfloat16)
        with self.assertRaises(RuntimeError):
            write_kpool_entries(
                self.layout, cache, torch.zeros(3, dtype=torch.long), key, gate
            )

    @unittest.skipUnless(
        hasattr(torch, "float8_e4m3fn"), "build has no float8_e4m3fn dtype"
    )
    def test_quantization_within_half_e4m3_ulp(self):
        pool_keys = torch.randn(16, HEAD_DIM, dtype=torch.bfloat16)
        quantized, scale = quantize_pool_keys(pool_keys)
        dequantized = quantized.float() * scale.unsqueeze(-1)
        denominator = pool_keys.float().abs().amax().clamp_min(1e-6)
        relative = ((dequantized - pool_keys.float()).abs().amax() / denominator).item()
        # Half an e4m3 ULP at the top of the range is 16/448 = 3.571e-2.
        self.assertLessEqual(relative, 4e-2)


class KPoolSelectIndexContractTest(unittest.TestCase):
    """The top-k backends return request-local indices; `select` must not shift.

    Regression for a defect where `select` subtracted the request's pool offset
    a second time: request 0 was unaffected (its offset is 0) while every later
    request in the batch lost all of its history.
    """

    def setUp(self):
        self.stub_calls = []
        module_name = "rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score"
        self.saved_module = sys.modules.get(module_name)
        stub = types.ModuleType(module_name)

        def legacy_fp8_mqa_logits(q, kv, weights, ks, ke, clean_logits=True):
            # Shape only; the stub top-k below decides what gets selected.
            return torch.zeros(
                (q.shape[0], int(ks.max()) + 1024), dtype=torch.float32
            )

        stub.legacy_fp8_mqa_logits = legacy_fp8_mqa_logits
        sys.modules[module_name] = stub
        self.module_name = module_name

    def tearDown(self):
        if self.saved_module is None:
            sys.modules.pop(self.module_name, None)
        else:
            sys.modules[self.module_name] = self.saved_module

    def test_second_request_keeps_its_history(self):
        pool_topk = 4
        sparse_width = 64
        indexer = KPoolIndexer(
            HEAD_DIM, RATIO, TOKENS_PER_PAGE, pool_topk, sparse_width
        )
        rows = 2
        context_lens = torch.tensor([RATIO * 6, RATIO * 6], dtype=torch.long)
        max_pools = 1024
        pool_ks = torch.tensor([0, max_pools], dtype=torch.int32)
        pool_ke = pool_ks + (context_lens // RATIO).to(torch.int32)

        def run_topk(logits, starts, ends, output, max_len):
            # Backend contract: indices are relative to starts[r]. Pick pools
            # 0..pool_topk-1 of each request.
            for row in range(output.shape[0]):
                output[row] = torch.arange(pool_topk, dtype=torch.int32)

        out = indexer.select(
            torch.zeros(rows, 1, HEAD_DIM),
            torch.zeros(rows, 1),
            torch.zeros(2 * max_pools, HEAD_DIM),
            torch.zeros(2 * max_pools, 1),
            pool_ks,
            pool_ke,
            context_lens,
            run_topk,
        )
        self.assertEqual(out.shape, (rows, sparse_width))
        for row in range(rows):
            valid = [v for v in out[row].tolist() if v >= 0]
            self.assertEqual(
                len(valid),
                pool_topk * RATIO,
                f"request {row} should expand {pool_topk} pools into tokens",
            )
            self.assertEqual(valid[: pool_topk * RATIO], list(range(pool_topk * RATIO)))


if __name__ == "__main__":
    unittest.main()
