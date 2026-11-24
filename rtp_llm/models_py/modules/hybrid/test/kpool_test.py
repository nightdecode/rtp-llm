"""KPool (GLM-5.3-Flash compressed indexer) math and index contracts.

Pure torch on CPU: the pooling, the pool-index expansion and the request-local
index contract are all device independent, and the reference they are compared
against lives in ``indexer_compressor.py``.
"""

import sys
import types
import os
import unittest

import torch

from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    _hadamard_rotate_reference,
    compress_indexer_projection_reference,
)
from rtp_llm.models_py.modules.hybrid.kpool import (
    KPoolCacheLayout,
    _hadamard_matrix,
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

    def test_hadamard_matmul_matches_butterfly(self):
        """The pooling path rotates with one matmul instead of 7 butterfly stages.

        The matrix is built by running the butterfly over the identity, so the
        two are the same linear map by construction. They are not bitwise equal
        in general: summing 128 terms in a different order moves the FP32
        result by a few 1e-6, which occasionally tips the BF16 rounding of an
        element whose value sits on a rounding boundary. That is far below the
        FP8 step the pooled key is finally stored at, so this pins the FP32
        agreement and only bounds how often BF16 may differ.
        """
        mat = _hadamard_matrix(HEAD_DIM, torch.device("cpu"))
        self.assertEqual(tuple(mat.shape), (HEAD_DIM, HEAD_DIM))
        self.assertTrue(
            torch.allclose(mat.abs(), torch.full_like(mat, HEAD_DIM**-0.5)),
            "entries must be +-1/sqrt(n)",
        )
        self.assertTrue(
            torch.allclose(mat @ mat.T, torch.eye(HEAD_DIM), atol=1e-5),
            "matrix must be orthogonal",
        )
        for shape in ((1, HEAD_DIM), (9, HEAD_DIM), (3, 5, HEAD_DIM), (16, 97, HEAD_DIM)):
            x = torch.randn(*shape, generator=self.generator)
            want = _hadamard_rotate_reference(x)
            got = x @ mat
            self.assertLess(
                (want - got).abs().max().item(),
                1e-4,
                f"shape {shape}: FP32 results must agree up to reassociation",
            )
            differing = (
                want.to(torch.bfloat16) != got.to(torch.bfloat16)
            ).float().mean().item()
            self.assertLess(
                differing,
                1e-3,
                f"shape {shape}: too many elements land on a different BF16 value",
            )

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

    @staticmethod
    def _scoring_takes_cpu_tensors() -> bool:
        """打分后端收不收 CPU 张量。

        装了 DeepGEMM 的机器上，打分走它的原生 FP8 MQA 入口，而那个入口断言
        输入必须是 FP8/BF16/INT8 的**设备**张量，拿 CPU 张量进去直接断言失败。
        没装 DeepGEMM 时退回自写 Triton，能在 CPU 上跑。
        这个用例验的是「第二个请求保留自己的历史」这条契约，与打分后端无关，
        所以后端吃不下 CPU 张量时跳过，而不是把它改成能过。
        """
        try:
            from rtp_llm.models_py.modules.base.cuda import indexer_op
        except Exception:  # noqa: BLE001
            return True
        return getattr(indexer_op, "deep_gemm", None) is None

    def test_second_request_keeps_its_history(self):
        if not self._scoring_takes_cpu_tensors():
            self.skipTest("打分后端是 DeepGEMM，它不收 CPU 张量；这条契约与后端无关")
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


class IncrementalPoolingTest(unittest.TestCase):
    """写入时增量池化必须与「每步重算所有池」算出同一批池 key。

    两条路的数学完全相同（同一个 softmax 加权、同一个 Hadamard、同一套 FP8 量化），
    差别只在什么时候算，所以这里要求**逐位相同**，不留容差。
    """

    def _cache(self, pages, device="cpu"):
        return torch.zeros(
            (pages, TOKENS_PER_PAGE * 2 * HEAD_DIM), dtype=torch.bfloat16, device=device
        )

    def setUp(self):
        # 分页打分默认开着，开着时 build_pools 直接返回空张量（池由打分算子按页读），
        # 而这个用例比的就是**取出来的池**，所以必须走搬运路。
        self._paged = os.environ.get("RTP_GLM_KPOOL_PAGED_SCORE")
        os.environ["RTP_GLM_KPOOL_PAGED_SCORE"] = "0"

    def tearDown(self):
        if self._paged is None:
            os.environ.pop("RTP_GLM_KPOOL_PAGED_SCORE", None)
        else:
            os.environ["RTP_GLM_KPOOL_PAGED_SCORE"] = self._paged

    def test_matches_recompute(self):
        torch.manual_seed(11)
        pages_per_req, batch = 3, 2
        pages = pages_per_req * batch
        ape = torch.randn(RATIO, HEAD_DIM) * 0.1
        block_table = torch.arange(pages, dtype=torch.int32).view(batch, pages_per_req)

        # 每个请求写满 2 页零 7 个 token：最后一个池凑不满，正好覆盖尾部分支。
        tokens = TOKENS_PER_PAGE * 2 + 7
        base = self._cache(pages)
        inc = self._cache(pages)
        old = KPoolIndexer(HEAD_DIM, RATIO, TOKENS_PER_PAGE, 8, 2176)
        old.incremental = False
        new = KPoolIndexer(HEAD_DIM, RATIO, TOKENS_PER_PAGE, 8, 2176)
        new.incremental = True

        # 分两次写，模拟 prefill 分块加 decode 逐 token：第二段每次只写 1 个 token。
        for req in range(batch):
            first = TOKENS_PER_PAGE * 2
            offset = req * pages_per_req * TOKENS_PER_PAGE
            chunk_slots = torch.arange(offset, offset + first, dtype=torch.long)
            key = torch.randn(first, HEAD_DIM, dtype=torch.bfloat16)
            gate = torch.randn(first, HEAD_DIM, dtype=torch.bfloat16)
            old.write(base, chunk_slots, key, gate)
            new.write(inc, chunk_slots, key, gate, ape)
            for i in range(first, tokens):
                slot = torch.tensor([offset + i], dtype=torch.long)
                k1 = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16)
                g1 = torch.randn(1, HEAD_DIM, dtype=torch.bfloat16)
                old.write(base, slot, k1, g1)
                new.write(inc, slot, k1, g1, ape)

        self.assertTrue(torch.equal(base, inc), "两条路写进 cache 的 (key, gate) 应当相同")

        ctx = torch.tensor([tokens] * batch, dtype=torch.long)
        want_fp8, want_scale, want_start = old.build_pools(base, block_table, ctx, ape)
        got_fp8, got_scale, got_start = new.build_pools(inc, block_table, ctx, ape)
        self.assertTrue(torch.equal(want_start, got_start))
        self.assertEqual(want_fp8.shape, got_fp8.shape)

        # 只有 [ks, ke) 里的池参与打分；没凑满的池两条路都算的是垃圾，不比。
        max_pools = int(block_table.shape[1]) * (TOKENS_PER_PAGE // RATIO)
        full_pools = tokens // RATIO
        for req in range(batch):
            lo = req * max_pools
            hi = lo + full_pools
            self.assertTrue(
                torch.equal(
                    want_fp8[lo:hi].view(torch.uint8), got_fp8[lo:hi].view(torch.uint8)
                ),
                f"请求 {req} 的池 key 不逐位相同",
            )
            self.assertTrue(
                torch.equal(want_scale[lo:hi], got_scale[lo:hi]),
                f"请求 {req} 的池 scale 不逐位相同",
            )

    def test_store_backed_by_external_buffer(self):
        """池缓存组那条路：KPoolStore 包住外部存储，不另开显存。

        PD 分离下池必须住在随 KV cache 跨机传输的缓存组里，模块私有缓冲传不过去
        （decode 机读到的是那些物理页上别的请求留下的池）。这里只验存储层的契约：
        写进去的东西落在外部那块内存上，读回来一致，块间有对齐填充时直接报错。
        """
        from rtp_llm.models_py.modules.hybrid.kpool import KPoolStore

        pools_per_page = TOKENS_PER_PAGE // RATIO
        page_bytes = pools_per_page * (HEAD_DIM + 4)
        pages = 3
        # **基址带偏移**：缓存组给的张量是大 arena 里的一段，storage_offset 非零。
        # 取用路里 scale 的 as_strided 如果传了显式 storage_offset，会把张量自身的
        # 偏移顶掉，于是 key 读对、scale 从别的行读出来。私有缓冲偏移为 0，
        # 不带偏移测就测不出这个错。
        backing = torch.zeros(pages * page_bytes + 4096, dtype=torch.uint8)
        arena = backing[4096:].view(pages, page_bytes)
        self.assertNotEqual(arena.storage_offset(), 0)
        store = KPoolStore(
            HEAD_DIM, TOKENS_PER_PAGE, RATIO, pages, torch.device("cpu"), base=arena
        )
        self.assertEqual(store.pool_packed.data_ptr(), arena.data_ptr())
        self.assertEqual(int(arena.max()), 0, "建 store 不应当改写这块存储")

        rows = torch.randn(
            pages * TOKENS_PER_PAGE, 2 * HEAD_DIM, dtype=torch.bfloat16
        )
        ape = torch.randn(RATIO, HEAD_DIM)
        slots = torch.arange(0, pages * TOKENS_PER_PAGE, RATIO, dtype=torch.long)
        store.update_from_slots(rows, slots, ape, _hadamard_matrix(HEAD_DIM, rows.device))
        self.assertGreater(int(arena.max()), 0, "写入应当落在外部那块内存上")

        block_table = torch.arange(pages, dtype=torch.int32).view(1, pages)
        pool_fp8, pool_scale, _ = store.gather(block_table, pages * pools_per_page)
        self.assertEqual(int(pool_fp8.shape[0]), pages * pools_per_page)
        self.assertTrue(torch.isfinite(pool_scale).all())
        # 取回来的 scale 必须和写进去的那一份逐位相同：按写入路的下标直接读原始存储
        ppp = pools_per_page
        want = store.pool_scale.reshape(-1)[
            torch.arange(pages).repeat_interleave(ppp) * (page_bytes // 4)
            + (ppp * HEAD_DIM) // 4
            + torch.arange(ppp).repeat(pages)
        ]
        self.assertTrue(
            torch.equal(pool_scale, want),
            "取用路读到的 scale 与写入路的下标对不上（多半是 storage_offset 被顶掉）",
        )

        # 块间有对齐填充时分页打分会读错页，必须当场报错而不是默默算错
        padded = torch.zeros(pages, page_bytes + 64, dtype=torch.uint8)
        with self.assertRaises(RuntimeError):
            KPoolStore(
                HEAD_DIM, TOKENS_PER_PAGE, RATIO, pages, torch.device("cpu"), base=padded
            )

    @unittest.skipUnless(torch.cuda.is_available(), "融合写入 kernel 是 Triton，需要设备")
    def test_butterfly_write_matches_torch_and_matmul(self):
        """池写入的蝶形版（每 token 一个 program）与 torch 逐算子版、旧矩阵乘版写出同样的池。

        旧版每个 program 固定 16 个 token、旋转是关掉 TF32 的 FP32 矩阵乘；decode 每卡几个
        token 时整个 kernel 只有 1 个 program，一次 61 μs。蝶形与 _hadamard_rotate_reference
        逐级同序，池 key 是 FP8，个别元素落在量化边界上允许差一个刻度。
        """
        from rtp_llm.models_py.modules.hybrid.kpool import KPoolStore

        dev = torch.device("cuda")
        gen = torch.Generator(device="cpu").manual_seed(7)
        for pages, tokens in ((2, 1), (2, 8), (8, 64), (64, 2048)):
            rows = (torch.randn(pages * TOKENS_PER_PAGE, 2 * HEAD_DIM, generator=gen) * 1.5).to(
                dev, torch.bfloat16
            )
            ape = (torch.randn(RATIO, HEAD_DIM, generator=gen) * 0.1).to(dev)
            slots = torch.clamp(
                torch.arange(tokens, device=dev) * RATIO + (RATIO - 1),
                max=pages * TOKENS_PER_PAGE - 1,
            )
            stores = {}
            for name, fused, butterfly in (("torch", "0", "1"), ("矩阵乘", "1", "0"), ("蝶形", "1", "1")):
                env = {"RTP_GLM_KPOOL_FUSED_WRITE": fused, "RTP_GLM_KPOOL_WRITE_BUTTERFLY": butterfly}
                old = {k: os.environ.get(k) for k in env}
                os.environ.update(env)
                try:
                    store = KPoolStore(HEAD_DIM, TOKENS_PER_PAGE, RATIO, pages, dev)
                    store.update_from_slots(rows, slots, ape, _hadamard_matrix(HEAD_DIM, dev))
                finally:
                    for k, v in old.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                stores[name] = store
            ref = stores["torch"]
            touched = (ref.pool_scale != 0).nonzero().flatten()
            self.assertGreater(int(touched.numel()), 0)
            for name in ("矩阵乘", "蝶形"):
                got = stores[name]
                qa = got.pool_u8[touched].to(torch.int16)
                qb = ref.pool_u8[touched].to(torch.int16)
                same = (qa == qb).float().mean().item()
                off1 = ((qa - qb).abs() <= 1).float().mean().item()
                rel = (got.pool_scale[touched] - ref.pool_scale[touched]).abs().max().item() / max(
                    ref.pool_scale[touched].abs().max().item(), 1e-30
                )
                self.assertGreater(same, 0.99, f"{name} {tokens} token：逐位相同只有 {same:.4f}")
                self.assertGreater(off1, 0.9999, f"{name} {tokens} token：超出一个刻度")
                self.assertLess(rel, 1e-5, f"{name} {tokens} token：scale 相对误差 {rel:.2e}")

    def test_pool_row_from_slot(self):
        """池行号只由槽位算出：页内偏移整除 ratio，加上页号乘每页池数。"""
        from rtp_llm.models_py.modules.hybrid.kpool import _pool_rows_for_slots

        pools_per_page = TOKENS_PER_PAGE // RATIO
        slots = torch.tensor([0, 3, 4, TOKENS_PER_PAGE - 1, TOKENS_PER_PAGE, TOKENS_PER_PAGE + 5])
        want = torch.tensor([0, 0, 1, pools_per_page - 1, pools_per_page, pools_per_page + 1])
        got = _pool_rows_for_slots(slots, TOKENS_PER_PAGE, RATIO)
        self.assertTrue(torch.equal(want, got), f"{got.tolist()} != {want.tolist()}")


if __name__ == "__main__":
    unittest.main()
