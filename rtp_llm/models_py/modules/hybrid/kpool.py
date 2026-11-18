"""GLM-5.3-Flash 的 indexer KPool 压缩（纯 torch 版，先保证正确）。

官方语义（`glm5_next_modeling_glm5_next.py` 的 `Glm5NextTextIndexer`）：

    k      = k_norm(wk(x))                         # [T, 128]，不做 rope
    gate   = linear(x, index_kpool_compress_gate)  # [T, 128]
    每 ratio 个连续 token 一组（只有凑满的组可被选）：
        w      = softmax(gate[组] + ape, dim=组内位置)   # 逐通道
        pool_k = Σ_组 k * w
    scores = relu(q · pool_kᵀ · softmax_scale)
    index  = Σ_head weights_proj(x) * scores
    选 top (index_topk // ratio) 个池，展开回 token，再追加未满一池的尾部

与分支实现的对照（`ref/branch-port/.../hybrid/indexer_compressor.py`
的 `compress_indexer_projection_reference`）逐项一致，只多一步 Hadamard 旋转：
分支对池化后的 key 做 128 点归一化 Hadamard，并同时旋转 q。Hadamard 是正交变换，
在 `q·k` 里严格抵消，纯粹为 FP8 量化精度服务。本实现沿用基线 indexer 已有的
`_rotate_activation`（q 本来就旋转过），所以池 key 也必须旋转，否则 q·k 对不上。

**本版本的取舍（写进验收报告）**：
- 池 key 不进 cache，每次前向现算。cache 里存的是**每 token 的 BF16 (key, gate)**，
  也就是说 indexer cache 的 entry 从 132 字节（FP8 key + scale）变成 512 字节
  （128 个 BF16 key + 128 个 BF16 gate）。好处是不需要压缩后的 slot mapping、
  不需要 state 环、池化用的是精确 BF16 值（与 reference 同精度）；
  代价是 indexer cache 大 3.9 倍、每步重算池 key。
- 对方分支的做法是把池 key 量化成 FP8 存进压缩 cache（每池 132 字节，省 4 倍），
  未满一池的尾部存 FP32 环。那是性能更优的形态，列为后续优化（见 notes）。
"""

from typing import Optional, Tuple

import torch

from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    _hadamard_rotate_reference,
)

__all__ = [
    "KPoolCacheLayout",
    "write_kpool_entries",
    "gather_kpool_pool_keys",
    "expand_pool_indices_to_tokens",
]


class KPoolCacheLayout:
    """每 token 一条 entry：前 head_dim 个 BF16 是 indexer key，后 head_dim 个是门控分数。

    页内布局与基线的 indexer cache 相同（每页 ``tokens_per_page`` 条 entry），
    所以可以直接复用这一组现成的 slot mapping 与 block table，不需要新的几何。
    """

    def __init__(self, head_dim: int, tokens_per_page: int) -> None:
        self.head_dim = int(head_dim)
        self.tokens_per_page = int(tokens_per_page)
        self.entry_elems = 2 * self.head_dim

    def view(self, kv_cache_base: torch.Tensor) -> torch.Tensor:
        """把 cache 张量看成 ``[总槽位数, 2 * head_dim]``。

        传进来的可能是 ``[pages, tokens_per_page * entry_elems]``，
        也可能已经是 ``[pages, tokens_per_page, entry_elems]``；两种都接受。
        """
        if kv_cache_base.dtype != torch.bfloat16:
            raise RuntimeError(
                "KPool indexer cache must be BF16 (key|gate pairs), got "
                f"{kv_cache_base.dtype}"
            )
        if not kv_cache_base.is_contiguous():
            raise RuntimeError("KPool indexer cache must be contiguous")
        flat = kv_cache_base.reshape(-1)
        if flat.numel() % self.entry_elems:
            raise RuntimeError(
                f"KPool indexer cache size {flat.numel()} is not a multiple of "
                f"entry width {self.entry_elems}"
            )
        return flat.view(-1, self.entry_elems)


def write_kpool_entries(
    layout: KPoolCacheLayout,
    kv_cache_base: torch.Tensor,
    slot_mapping: torch.Tensor,
    key: torch.Tensor,
    gate: torch.Tensor,
) -> None:
    """把当前这批 token 的 (key, gate) 写进 cache。

    全程不做 device→host 同步：decode 侧在 CUDA Graph 捕获里跑，
    一个 ``.item()`` 或 ``bool(tensor)`` 就会报
    ``operation not permitted when stream is capturing``。
    所以 slot < 0 的 padding 行不做条件分支，而是重定向到物理 block 0
    ——该 block 是引擎保留的 dummy，不属于任何请求，写进去无害。

    Args:
        slot_mapping: ``[T]``，每个 token 的物理槽位，<0 表示 padding。
        key:  ``[T, head_dim]`` BF16，k_norm(wk(x)) 的结果，**未旋转**。
        gate: ``[T, head_dim]`` BF16，门控分数。
    """
    rows = layout.view(kv_cache_base)
    slots = slot_mapping.reshape(-1).to(torch.long)
    n = int(slots.shape[0])
    # A mismatch here means some tokens would never reach the cache and the
    # indexer would score a stale key. Shapes are static per forward, so this
    # is a host-side check on metadata, not a device sync.
    if n != int(key.shape[0]) or n != int(gate.shape[0]):
        raise RuntimeError(
            "KPool slot mapping and projections disagree: "
            f"slots={n}, key={int(key.shape[0])}, gate={int(gate.shape[0])}"
        )
    if n == 0:
        return
    safe = torch.clamp(slots, min=0)
    rows.index_copy_(
        0,
        safe,
        torch.cat((key.to(torch.bfloat16), gate.to(torch.bfloat16)), dim=-1),
    )


def _pool_slot_table(
    block_table: torch.Tensor, max_pools: int, ratio: int, tokens_per_page: int
) -> torch.Tensor:
    """给每个请求的前 ``max_pools`` 个池算出组内 ratio 个 token 的物理槽位。

    返回 ``[B, max_pools, ratio]`` int64。越界的池也会算出一个槽位，
    但调用方用 ks/ke 把它们排除在打分之外，所以内容无所谓，只要不越界。
    形状全部是静态的，没有 ``.item()``，可以在 CUDA Graph 捕获里跑。
    """
    batch, max_pages = int(block_table.shape[0]), int(block_table.shape[1])
    device = block_table.device
    token = torch.arange(max_pools * ratio, device=device, dtype=torch.long)
    page = torch.clamp(token // tokens_per_page, max=max_pages - 1)
    physical_page = block_table.to(torch.long).index_select(1, page)
    slots = physical_page * tokens_per_page + (token % tokens_per_page).unsqueeze(0)
    return slots.view(batch, max_pools, ratio)


def gather_kpool_pool_keys(
    layout: KPoolCacheLayout,
    kv_cache_base: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    ape: torch.Tensor,
    ratio: int,
    rotate: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从 cache 里取出每个请求的池，算出池 key。``rotate`` 为真时顺带做 Hadamard。

    池缓冲用**固定步长**布局：第 b 个请求占 ``[b*max_pools, (b+1)*max_pools)``，
    ``max_pools`` 由 block table 的宽度决定，是静态值。
    这样整条路不需要知道「实际有多少个池」，也就不需要 ``.item()``，
    可以在 decode 的 CUDA Graph 捕获里跑；超出实际长度的池是垃圾值，
    由打分时的 ks/ke 排除。

    Returns:
        pool_keys ``[B * max_pools, head_dim]`` BF16（**未旋转**），
        pool_start ``[B]`` int64，每个请求的起始行（= b * max_pools）。
    """
    if ratio < 2:
        raise ValueError(f"kpool ratio must be >= 2, got {ratio}")
    device = kv_cache_base.device
    batch = int(block_table.shape[0])
    head_dim = layout.head_dim
    max_pools = (int(block_table.shape[1]) * layout.tokens_per_page) // ratio
    pool_start = (
        torch.arange(batch, device=device, dtype=torch.long) * max_pools
    )
    if batch == 0 or max_pools == 0:
        return (
            torch.empty((0, head_dim), dtype=torch.bfloat16, device=device),
            pool_start,
        )

    rows = layout.view(kv_cache_base)
    slots = _pool_slot_table(block_table, max_pools, ratio, layout.tokens_per_page)
    slots = torch.clamp(slots, min=0, max=int(rows.shape[0]) - 1)
    ape_f = ape.to(device).float().view(1, 1, ratio, head_dim)
    pooled = torch.empty(
        (batch, max_pools, head_dim), dtype=torch.bfloat16, device=device
    )
    rotate = bool(rotate)
    # 按池分块算，别一次性把 [B, max_pools, ratio, 2*head_dim] 物化出来：
    # decode 的 CUDA Graph 按最大 batch 捕获，block table 又是按 max_seq_len 开的，
    # 一次性物化在 batch=64 / 40960 token 下是 1.3 GB，实测会把 PPU 的分配器逼到
    # 一条它没实现的驱动接口上（Can't find cuDeviceGet）。
    # 块数只由静态的 max_pools 决定，所以仍然可以在捕获里跑。
    bytes_per_pool_row = ratio * 2 * head_dim * 2
    chunk = max(1, min(max_pools, (64 << 20) // max(1, batch * bytes_per_pool_row)))
    for begin in range(0, max_pools, chunk):
        end = min(begin + chunk, max_pools)
        part = rows.index_select(0, slots[:, begin:end].reshape(-1)).view(
            batch, end - begin, ratio, 2 * head_dim
        )
        key = part[..., :head_dim].float()
        gate = part[..., head_dim:].float()
        # 逐通道在组内位置上做 softmax，权重乘回 key 求和。与官方
        # Glm5NextTextIndexer.get_pooled_states 的三行一致。
        weight = torch.softmax(gate + ape_f, dim=2)
        part_pooled = (weight.to(torch.bfloat16).float() * key).sum(dim=2)
        if rotate:
            # Hadamard 也放进分块里做：它每一级都要 stack 出一份同样大的张量，
            # 在整块上做同样会把 PPU 的分配器逼到 cuDeviceGet 那条路上。
            part_pooled = _hadamard_rotate_reference(part_pooled)
        pooled[:, begin:end] = part_pooled.to(torch.bfloat16)
    return pooled.reshape(batch * max_pools, head_dim), pool_start


def expand_pool_indices_to_tokens(
    pool_indices: torch.Tensor,
    context_lens: torch.Tensor,
    ratio: int,
    width: int,
) -> torch.Tensor:
    """池索引展开成请求内的 token 索引，追加未满一池的尾部，右侧补 -1 到 ``width``。

    Args:
        pool_indices: ``[T, pool_topk]`` int32，**请求内**的池编号，<0 表示无效。
        context_lens: ``[T]``，每个查询 token 能看到的上下文长度。
        width: 输出宽度（稀疏注意力 kernel 要求 128 的倍数，GLM-5.3-Flash 是 2176）。
    """
    device = pool_indices.device
    lengths = context_lens.to(device).to(torch.long).reshape(-1, 1)
    pools = torch.div(lengths, ratio, rounding_mode="floor")

    lanes = torch.arange(ratio, device=device, dtype=torch.long)
    ids = pool_indices.to(torch.long)
    ok = (ids >= 0) & (ids < pools)
    expanded = ids.unsqueeze(-1) * ratio + lanes
    expanded = torch.where(ok.unsqueeze(-1), expanded, torch.full_like(expanded, -1))
    expanded = expanded.flatten(start_dim=-2)

    tail_width = ratio - 1
    tail_lanes = torch.arange(tail_width, device=device, dtype=torch.long)
    tail_count = lengths - pools * ratio
    tail = pools * ratio + tail_lanes
    tail = torch.where(
        tail_lanes.unsqueeze(0) < tail_count, tail, torch.full_like(tail, -1)
    )

    out = torch.cat((expanded, tail), dim=-1)
    if out.shape[-1] > width:
        raise ValueError(
            f"expanded index width {out.shape[-1]} exceeds kernel width {width}"
        )
    if out.shape[-1] < width:
        pad = out.new_full((out.shape[0], width - out.shape[-1]), -1)
        out = torch.cat((out, pad), dim=-1)
    return out.to(torch.int32)


def quantize_pool_keys(
    pool_keys: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """按 legacy_fp8_mqa_logits 的约定量化：每行一个 FP32 scale，128 宽一组。

    同样分块做：整块转 FP32 在 batch=64 / 40960 token 下是几百 MB，
    会把 PPU 的分配器逼到它没实现的驱动接口上。
    """
    rows = int(pool_keys.shape[0])
    device = pool_keys.device
    quantized = torch.empty(pool_keys.shape, dtype=torch.float8_e4m3fn, device=device)
    scale = torch.empty((rows,), dtype=torch.float32, device=device)
    if rows == 0:
        return quantized, scale
    width = int(pool_keys.shape[-1])
    chunk = max(1, (32 << 20) // max(1, width * 4))
    for begin in range(0, rows, chunk):
        end = min(begin + chunk, rows)
        values = pool_keys[begin:end].float()
        amax = values.abs().amax(dim=-1).clamp_min(1e-10)
        scale[begin:end] = amax / 448.0
        quantized[begin:end] = (
            (values * (448.0 / amax).unsqueeze(-1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )
    return quantized, scale


class KPoolIndexer:
    """把 kpool 的「写 cache → 组池 → 打分 → 选 top-k 池 → 展开成 token 索引」串起来。

    为什么不复用 IndexerOp 的 `_get_topk_ragged` / `_get_topk_paged`：
    那两条路都假定 cache 里存的是**每 token 的 FP8 key + scale**（132 字节），
    用 `cp_gather_indexer_k_quant_cache` 这个 C++ 算子按那个布局取数。
    kpool 这边 cache 存的是 BF16 的 (key, gate) 对，池 key 是现算的，
    所以自己走一条 ragged 打分路，prefill 与 decode 共用同一条。
    """

    def __init__(self, head_dim: int, ratio: int, tokens_per_page: int, pool_topk: int,
                 sparse_width: int) -> None:
        self.layout = KPoolCacheLayout(head_dim, tokens_per_page)
        self.ratio = int(ratio)
        self.pool_topk = int(pool_topk)
        self.sparse_width = int(sparse_width)

    # ---- 写 cache ----
    def write(self, kv_cache_base, slot_mapping, key_bf16, gate_bf16) -> None:
        write_kpool_entries(self.layout, kv_cache_base, slot_mapping, key_bf16, gate_bf16)

    # ---- 组池 ----
    def build_pools(self, kv_cache_base, block_table, context_lens, ape):
        rotated, pool_start = gather_kpool_pool_keys(
            self.layout,
            kv_cache_base,
            block_table,
            context_lens,
            ape,
            self.ratio,
            rotate=True,
        )
        pool_fp8, pool_scale = quantize_pool_keys(rotated)
        return pool_fp8, pool_scale, pool_start

    # ---- 打分 + 选池 + 展开 ----
    def select(self, q_fp8, weights, pool_fp8, pool_scale, pool_ks, pool_ke,
               row_context_lens, run_topk):
        """返回 ``[rows, sparse_width]`` int32 的**请求内 token 索引**（-1 为无效）。"""
        import torch as _torch

        from rtp_llm.models_py.triton_kernels.legacy_fp8_indexer_score import (
            legacy_fp8_mqa_logits,
        )

        rows = int(q_fp8.shape[0])
        if rows == 0 or int(pool_fp8.shape[0]) == 0:
            return _torch.full(
                (rows, self.sparse_width), -1, dtype=_torch.int32, device=q_fp8.device
            )
        logits = legacy_fp8_mqa_logits(
            q_fp8, (pool_fp8, pool_scale), weights, pool_ks, pool_ke, clean_logits=False
        )
        local = logits.new_empty((rows, self.pool_topk), dtype=_torch.int32)
        # The top-k backends already return request-local indices: both the PPU
        # Triton path (``kernels/ppu/indexer_topk.py`` subtracts ``start`` in all
        # three of ``_finish``, ``_stable_topk_small_kernel`` and
        # ``_all_indices``) and the CUDA op (contract comment in
        # ``bindings/cuda/kernels/dsv4_top_k_per_row_prefill.h``: "Indices are
        # relative to row_starts[r]"). Subtracting ``pool_ks`` a second time
        # would turn every request after the first into large negatives.
        run_topk(logits, pool_ks, pool_ke, local, int(pool_fp8.shape[0]))
        return expand_pool_indices_to_tokens(
            local, row_context_lens, self.ratio, self.sparse_width
        )
