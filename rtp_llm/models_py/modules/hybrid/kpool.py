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

**cache 形态**：每 token 一条 BF16 的 (key, gate)，共 512 字节。不改成压缩 cache
是为了不动引擎的 cache 组声明与 slot mapping；代价是 indexer cache 比对方分支大 3.9 倍，
容量上够用（实测每 rank 449 万 token，需求 65.5 万）。

**池 key 什么时候算（2026-11-20 改）**：原来每次前向把所有池从 BF16 cache 里重算一遍，
实测这是 decode 的主要开销——池数按 ``max_seq_len`` 开（40960 token 即 10240 个池），
每个池要取 4 条 512 字节的 entry 再做 softmax 加权、旋转、量化，11 个 DSA 层每步合计约 75 毫秒。
现在改成**写入时增量池化**：token 写进 cache 之后顺带把它所在的那个池算出来、量化成 FP8
存进一块按物理页索引的池 cache（``KPoolStore``），打分时只做一次 gather。

增量池化成立的两个前提，都已核实：

1. **池的物理位置只由 token 槽位决定**，不需要知道 token 在请求里的绝对位置。
   页大小 256 是 ratio 4 的整数倍，所以页内偏移 ``slot % 256`` 与绝对位置 ``p % 256`` 相等，
   池在页内的序号就是 ``(slot % 256) // 4``，池行 = ``(slot // 256) * 64 + (slot % 256) // 4``。
2. **一个池的 4 个 token 一定都在本卡**。CP prefill 下 ``kv_cache_sharded`` 默认关闭
   （``server_args/parallel_group_args.py`` 的 ``--prefill_cp_kv_cache_sharded`` 默认 False），
   各 rank 持有完整的 indexer cache，写入前还做过一次 all-gather 还原全序列。

未凑满的池不需要特殊处理：``expand_pool_indices_to_tokens`` 把尾部不足一池的 token
无条件追加进结果，它们从来不参与打分，所以也不需要 state 环。写入时对未凑满的池
算出来的是垃圾值，但打分用的 ``ke = 上下文长度 // ratio`` 把它排除在外，
等那个池凑满时会被正确值覆盖。

用 ``RTP_GLM_KPOOL_INCREMENTAL=0`` 退回每步重算的老路，便于 A/B。
"""

import os
from typing import Dict, Optional, Tuple

import torch

try:  # Triton 只在融合写入路上用到；没有它时退回逐算子的 torch 版
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from rtp_llm.models_py.modules.hybrid.indexer_compressor import (
    _hadamard_rotate_reference,
)

__all__ = [
    "KPoolCacheLayout",
    "KPoolStore",
    "write_kpool_entries",
    "gather_kpool_pool_keys",
    "expand_pool_indices_to_tokens",
    "incremental_pooling_enabled",
]


def paged_scoring_enabled() -> bool:
    """打分是否按页直接读池 cache，而不是先把池搬出来再打分。

    需要增量池化开着（池 cache 才存在）。默认开：上卡实测各并发档都更快
    （并发 64 省 2.98 毫秒、并发 128 省 2.30 毫秒），数值与搬运路逐位相同。
    置 0 可退回搬运路做 A/B。
    """
    return os.environ.get("RTP_GLM_KPOOL_PAGED_SCORE", "1") == "1"


def incremental_read_enabled() -> bool:
    """打分是否真的去用旁路缓冲里存着的池。

    这是把「写」和「读」分开的二分开关，只为定位精度问题，不是性能开关：
    置 0 时旁路缓冲照常维护（写入路一字不改），但打分改回从 kv cache 现算。
      精度恢复 → 问题在「用了存下来的池」
      精度仍坏 → 问题在 update_from_slots 这次写调用本身，与读哪条路无关
    所有同进程审计都显示两条读路选出的池下标逐位相同，这条二分是剩下唯一
    能把两者分清的办法。默认开。
    """
    return os.environ.get("RTP_GLM_KPOOL_INCREMENTAL_READ", "1") == "1"


def incremental_pooling_enabled() -> bool:
    """写入时增量池化（默认开）。置 0 退回每步重算所有池的老路，用于 A/B。"""
    return os.environ.get("RTP_GLM_KPOOL_INCREMENTAL", "1") != "0"


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


def _capturing(device: torch.device) -> bool:
    """当前是否在 CUDA Graph 捕获中。CPU 上（单元测试）直接返回假。"""
    return (
        device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_current_stream_capturing()
    )


_SLOT_CONST_CACHE: Dict[Tuple[int, int, int, str], Tuple[torch.Tensor, torch.Tensor]] = {}


def _pool_slot_constants(
    tokens: int, tokens_per_page: int, max_pages: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``token -> (页号, 页内偏移)``，只取决于形状，与请求内容无关，按形状缓存。

    这两个张量每层都一样，而且是 ``[max_pools * ratio]`` 的 int64，
    batch 64 / 40960 token 下每次重算要写几十 MB，11 层就重复 11 次。
    """
    key = (int(tokens), int(tokens_per_page), int(max_pages), str(device))
    cached = _SLOT_CONST_CACHE.get(key)
    if cached is not None:
        return cached
    token = torch.arange(tokens, device=device, dtype=torch.long)
    page = torch.clamp(token // tokens_per_page, max=max_pages - 1)
    built = (page.contiguous(), (token % tokens_per_page).contiguous())
    # 捕获期间建出来的张量属于那张 graph 的私有显存池，缓存下来会被别的 graph
    # 或 eager 调用读到不属于自己的内存。这种情况下只用不存，等预热时再缓存。
    if not _capturing(device):
        _SLOT_CONST_CACHE[key] = built
    return built


def _pool_slot_table(
    block_table: torch.Tensor,
    max_pools: int,
    ratio: int,
    tokens_per_page: int,
    num_pages: int,
) -> torch.Tensor:
    """给每个请求的前 ``max_pools`` 个池算出组内 ratio 个 token 的物理槽位。

    返回 ``[B, max_pools, ratio]`` int64。越界的池也会算出一个槽位，
    但调用方用 ks/ke 把它们排除在打分之外，所以内容无所谓，只要不越界。
    形状全部是静态的，没有 ``.item()``，可以在 CUDA Graph 捕获里跑。

    越界保护放在 ``block_table`` 上做（``[B, max_pages]``，几 KB），
    而不是放在展开后的槽位上（``[B, max_pools * ratio]``，几十 MB）：
    页号落在 ``[0, num_pages)`` 内，页内偏移本来就小于 ``tokens_per_page``，
    所以槽位必然落在缓存行数之内。
    """
    batch, max_pages = int(block_table.shape[0]), int(block_table.shape[1])
    if num_pages <= 0:
        # 缓存连一页都放不下（只在构造得很小的测试里出现）。页号钳到 0 并不能
        # 保证槽位落在行数之内，直接返回全 0，调用方的 ks/ke 会把它们排除掉。
        return block_table.new_zeros(
            (batch, max_pools, ratio), dtype=torch.long
        )
    page, offset = _pool_slot_constants(
        max_pools * ratio, tokens_per_page, max_pages, block_table.device
    )
    # 非原地：block_table 已经是 int64 时 .to(torch.long) 不会复制，
    # 原地 clamp 会改到调用方的块表。
    safe_pages = block_table.to(torch.long).clamp(0, max(num_pages - 1, 0))
    physical_page = safe_pages.index_select(1, page)
    slots = physical_page * tokens_per_page + offset.unsqueeze(0)
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
    slots = _pool_slot_table(
        block_table,
        max_pools,
        ratio,
        layout.tokens_per_page,
        int(rows.shape[0]) // layout.tokens_per_page,
    )
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
            # Hadamard 是线性变换，用一次矩阵乘代替 7 级蝶形：
            # 蝶形每级都要 reshape/unbind/stack，128 宽就是约 21 个算子、
            # 7 份与输入等大的临时张量；矩阵乘一个 kernel，精度口径不变（都在 FP32 做）。
            part_pooled = part_pooled @ _hadamard_matrix(head_dim, device)
        pooled[:, begin:end] = part_pooled.to(torch.bfloat16)
    return pooled.reshape(batch * max_pools, head_dim), pool_start


_HADAMARD_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _hadamard_matrix(width: int, device: torch.device) -> torch.Tensor:
    """行向量约定下的归一化 Hadamard 矩阵，满足 ``rotate(x) == x @ M``。

    直接对单位阵跑一遍 ``_hadamard_rotate_reference`` 得到，所以与参考实现
    逐项一致，也就与 q 侧 ``_rotate_activation``（fast_hadamard_transform）
    的排列一致——两侧旋转必须是同一个变换，点积才抵消得掉。
    矩阵是 width×width 的常量，按 (宽度, 设备) 缓存；在预热阶段就会建好，
    decode 的 Graph 捕获里不会再分配。
    """
    key = (int(width), str(device))
    mat = _HADAMARD_CACHE.get(key)
    if mat is not None:
        return mat
    eye = torch.eye(width, dtype=torch.float32, device=device)
    mat = _hadamard_rotate_reference(eye).contiguous()
    # 同 _pool_slot_constants：捕获期间建的只用不存。
    if not _capturing(device):
        _HADAMARD_CACHE[key] = mat
    return mat


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


if triton is not None:

    @triton.jit
    def _pool_write_kernel(
        ROWS, SLOTS, APE, HAD, POOL_U8, POOL_S,
        n_tokens,
        TPP: tl.constexpr, R: tl.constexpr, D: tl.constexpr,
        PPP: tl.constexpr, BT: tl.constexpr,
        PAGE_BYTES: tl.constexpr,
    ):
        """一个 kernel 走完「取组内 key/gate → 加权平均 → 旋转 → 量化 → 写池」。

        拆成 torch 算子时这条路每层要发约 30 个 kernel，decode 的 11 个 DSA 层就是每步
        三百多个；每个 kernel 处理的数据只有几 KB，耗时几乎全是启动开销。
        每一步都照着 torch 版逐条对应写（含权重先落一次 BF16、旋转后再落一次 BF16），
        所以两条路的结果应当只差浮点归约顺序那一点。
        """
        pid = tl.program_id(0)
        t = pid * BT + tl.arange(0, BT)
        live = t < n_tokens
        slot = tl.load(SLOTS + t, mask=live, other=0)
        gbase = (slot // R) * R

        d = tl.arange(0, D)
        lane = tl.arange(0, R)
        base = (gbase[:, None, None] + lane[None, :, None]) * (2 * D) + d[None, None, :]
        m3 = live[:, None, None]
        key = tl.load(ROWS + base, mask=m3, other=0.0).to(tl.float32)
        gate = tl.load(ROWS + base + D, mask=m3, other=0.0).to(tl.float32)

        # 逐通道在组内位置上 softmax（与 torch.softmax 同样先减最大值）
        x = gate + tl.load(APE + lane[None, :, None] * D + d[None, None, :])
        x = x - tl.max(x, 1)[:, None, :]
        e = tl.exp(x)
        w = e / tl.sum(e, 1)[:, None, :]
        # torch 版这里把权重落一次 BF16 再回 FP32，少这一步结果会差到 FP8 量化之上
        w = w.to(tl.bfloat16).to(tl.float32)
        pooled = tl.sum(w * key, 1)

        # 归一化 Hadamard 是正交变换，用矩阵乘实现；精度要按 FP32 走，
        # 让它退化成 TF32 会把误差放到 FP8 量化步长之上。
        had = tl.load(HAD + d[:, None] * D + d[None, :])
        rot = tl.dot(pooled, had, allow_tf32=False)

        # 没凑满的组读到的是旧值或 0，算出来是垃圾但不会被打分读到；真出 NaN 会污染
        # 同一个打分 kernel 里的别的行，这里压掉。
        rot = tl.where(rot == rot, rot, 0.0)
        rot = tl.where(tl.abs(rot) == float("inf"), 0.0, rot)
        rot = rot.to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(rot), 1), 1e-10)
        scale = amax / 448.0
        q = tl.clamp(rot * (448.0 / amax)[:, None], -448.0, 448.0).to(tl.float8e4nv)

        prow = (slot // TPP) * PPP + ((slot % TPP) // R)
        # 页内分区布局（与 DeepGEMM 分页打分算子一致，空卡对拍逐位相同）：
        # 一页里先连续放 PPP 条 D 字节的 FP8 key，再放 PPP 个 FP32 scale。
        # 不是「每条 key 后面紧跟自己的 scale」那种交错摆法——交错摆法喂进去
        # 读出来的是垃圾（实测相对误差 1e8）。
        # POOL_S 是同一块存储按 FP32 重解释的别名视图。
        page = prow // PPP
        islot = prow % PPP
        tl.store(POOL_U8 + page[:, None] * PAGE_BYTES + islot[:, None] * D + d[None, :],
                 q.to(tl.uint8, bitcast=True), mask=live[:, None])
        tl.store(POOL_S + page * (PAGE_BYTES // 4) + (PPP * D) // 4 + islot,
                 scale, mask=live)

    @triton.jit
    def _hadamard_stage(x, D: tl.constexpr, GROUPS: tl.constexpr, STRIDE: tl.constexpr):
        """蝶形一级：长度 D 的向量看成 [GROUPS, 2, STRIDE]，相距 STRIDE 的两两做加和减。"""
        x3 = tl.reshape(x, (GROUPS, 2, STRIDE))
        x3 = tl.trans(x3, 0, 2, 1)
        a, b = tl.split(x3)
        x3 = tl.join(a + b, a - b)
        x3 = tl.trans(x3, 0, 2, 1)
        return tl.reshape(x3, (D,))

    @triton.jit
    def _pool_write_row_kernel(
        ROWS, SLOTS, APE, POOL_U8, POOL_S,
        TPP: tl.constexpr, R: tl.constexpr, D: tl.constexpr, LOG_D: tl.constexpr,
        PPP: tl.constexpr, PAGE_BYTES: tl.constexpr,
    ):
        """与 _pool_write_kernel 同一条写入路，但每个 token 一个 program，旋转用蝶形。

        旧 kernel 每个 program 固定 16 个 token（tl.dot 的 M 至少 16），旋转是一次
        16×128 乘 128×128 的 FP32 矩阵乘、关掉了 TF32；decode 每卡只有几个 token 时
        整个 kernel 只有 1 个 program，在一个计算单元上串行做完这次矩阵乘，实测每次 61 μs
        （sglang 同作用的 kernel 4.6 μs，见结果文档 2026-11-22 timeline 一节）。
        蝶形是精确的 FP32 加减法，步长顺序 1、2、4……与 ``_hadamard_rotate_reference``
        逐级相同（sglang ``_hadamard128`` 也是这个写法），所以与参考实现的累加顺序一致。
        """
        t = tl.program_id(0)
        slot = tl.load(SLOTS + t)
        gbase = (slot // R) * R
        d = tl.arange(0, D)
        lane = tl.arange(0, R)
        base = (gbase + lane[:, None]) * (2 * D) + d[None, :]
        key = tl.load(ROWS + base).to(tl.float32)
        gate = tl.load(ROWS + base + D).to(tl.float32)

        x = gate + tl.load(APE + lane[:, None] * D + d[None, :])
        x = x - tl.max(x, 0)[None, :]
        e = tl.exp(x)
        w = e / tl.sum(e, 0)[None, :]
        w = w.to(tl.bfloat16).to(tl.float32)
        rot = tl.sum(w * key, 0)

        for s in tl.static_range(LOG_D):
            rot = _hadamard_stage(rot, D, D >> (s + 1), 1 << s)
        rot = rot * (1.0 / tl.sqrt(D * 1.0))

        rot = tl.where(rot == rot, rot, 0.0)
        rot = tl.where(tl.abs(rot) == float("inf"), 0.0, rot)
        rot = rot.to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(rot), 0), 1e-10)
        scale = amax / 448.0
        q = tl.clamp(rot * (448.0 / amax), -448.0, 448.0).to(tl.float8e4nv)

        prow = (slot // TPP) * PPP + ((slot % TPP) // R)
        page = prow // PPP
        islot = prow % PPP
        tl.store(POOL_U8 + page * PAGE_BYTES + islot * D + d, q.to(tl.uint8, bitcast=True))
        tl.store(POOL_S + page * (PAGE_BYTES // 4) + (PPP * D) // 4 + islot, scale)

else:  # pragma: no cover
    _pool_write_kernel = None
    _pool_write_row_kernel = None


def _pool_rows_for_slots(
    slots: torch.Tensor, tokens_per_page: int, ratio: int
) -> torch.Tensor:
    """token 槽位 → 池 cache 里的行号。依据见模块说明的前提 1。"""
    pools_per_page = tokens_per_page // ratio
    page = torch.div(slots, tokens_per_page, rounding_mode="floor")
    in_page = slots - page * tokens_per_page
    return page * pools_per_page + torch.div(in_page, ratio, rounding_mode="floor")


class KPoolStore:
    """按物理页索引的池 cache：FP8 池 key 加每池一个 FP32 scale。

    行数 = 页数 × 每页池数，与 indexer cache 的页一一对应，所以用引擎现成的 block table
    就能定位，不需要另一套分配器。页被别的请求回收后，旧池值会在新 token 写进来时覆盖；
    覆盖之前也读不到，因为打分的 ``[ks, ke)`` 只覆盖本请求已经凑满的池。

    显存开销是每 token ``(head_dim + 4) / ratio`` 字节（GLM-5.3-Flash 是 33 字节），
    相对每 token 512 字节的 indexer cache 多 6.4%。
    """

    def __init__(
        self,
        head_dim: int,
        tokens_per_page: int,
        ratio: int,
        num_pages: int,
        device: torch.device,
        base: Optional[torch.Tensor] = None,
    ) -> None:
        self.head_dim = int(head_dim)
        self.tokens_per_page = int(tokens_per_page)
        self.ratio = int(ratio)
        self.pools_per_page = self.tokens_per_page // self.ratio
        self.num_pages = int(num_pages)
        # 调用方按 indexer cache 行数算出来的页数。缓存组那条路会把 num_pages
        # 改成 arena 自己的块数，而 _ensure_store 的复用判据比的是调用方算出来的
        # 这个值——不单独记一份的话，两者永远不等，每次前向都会重建 store。
        self.requested_pages = int(num_pages)
        rows = max(1, self.num_pages * self.pools_per_page)
        # 池 key 按 uint8 存，取用时按位重解释成 FP8（两者都是 1 字节，view 不复制）。
        # 直接用 float8_e4m3fn 存会踩到 index_copy_ / index_select 在这个 dtype 上
        # 没实现的问题（CPU 上实测报 "index_copy_cpu not implemented"），uint8 没有这个限制。
        # 零初始化而不是 torch.empty：未写过的行也会被 gather 碰到（越界的池只是不参与
        # 打分，gather 还是会读），未初始化显存可能是 NaN 位型，一旦进了 FP8 乘加
        # 就会把同一个 kernel 里的别的行也污染成 NaN。
        # 一个池按 DeepGEMM 分页打分算子要的槽位格式存：128 字节 FP8 key
        # 紧跟 4 字节 FP32 scale，整块形状 [页数, 每页池数, 1, 132]。
        # 这样打分可以直接把这块 cache 按页读，不必先把池整段搬出来——
        # batch 1 时那次搬运每层就是 1.31 MB、11 个 DSA 层每步 14.4 MB。
        # 每页池数正好是 256 / 4 = 64，与该算子写死的 block_kv == 64 对上。
        self.slot_bytes = self.head_dim + 4
        self.page_bytes = self.pools_per_page * self.slot_bytes
        if base is None:
            # 模块私有的旁路缓冲。注意它**不随 KV cache 跨机传输**：PD 分离下
            # decode 机读到的提示词范围的池是别的请求留下的旧值，长上下文会偏。
            # 正确的做法是把 base 传进来（池缓存组），见 models/glm5_next.py。
            self.pool_packed = torch.zeros(
                (self.num_pages, self.page_bytes), dtype=torch.uint8, device=device
            )
        else:
            # 池缓存组的存储，由引擎按块分配、随 KV cache 一起传输。
            # 只重解释形状，不复制也不另开显存。
            # 分页打分要求整块 cache 逐页紧挨着：一页 64 条 key 连续，后面跟 64 个
            # scale，页与页之间不能有对齐填充。声明里没有给
            # block_stride_bytes_alignment，所以这里直接要求每块的字节数正好等于
            # page_bytes；真有填充就立刻报出来，不要默默算错。
            if base.dim() >= 2:
                # 页数以池缓存组自己的 arena 为准：块号来自引擎的块表，
                # 按 indexer cache 的行数反推出来的页数不一定等于这一组的块数。
                pages_in_base = int(base.shape[0])
                if pages_in_base < self.num_pages:
                    raise RuntimeError(
                        f"kpool 缓存组只有 {pages_in_base} 块，少于按 indexer cache "
                        f"推出的 {self.num_pages} 块；块表会指到不存在的页"
                    )
                self.num_pages = pages_in_base
                stride_bytes = int(base.numel() // base.shape[0]) * base.element_size()
                if stride_bytes != self.page_bytes:
                    raise RuntimeError(
                        f"kpool 缓存组每块 {stride_bytes} 字节，期望 "
                        f"{self.page_bytes}（{self.pools_per_page} 池 × "
                        f"{self.slot_bytes} 字节）；块间有对齐填充时分页打分会读错页"
                    )
            flat = base.reshape(-1).view(torch.uint8)
            want = self.num_pages * self.page_bytes
            if int(flat.numel()) < want:
                raise RuntimeError(
                    "kpool 缓存组太小：按 indexer cache 推出 "
                    f"{self.num_pages} 页 × {self.page_bytes} 字节 = {want}，"
                    f"实际只有 {int(flat.numel())} 字节；"
                    "检查 models/glm5_next.py 里 entry_elems 与 compression_ratio"
                )
            self.pool_packed = flat[:want].view(self.num_pages, self.page_bytes)
            # **不要在这里清零。** 建 store 的时机不只有起服务：探测前向用的是很小的
            # 假 cache，真实 cache 第一次出现时会重建一次 store，而在 decode 机上
            # 「真实 cache 第一次出现」正是收到跨机传输之后的第一次前向——
            # 那一次清零会把刚传过来的池全抹掉，修复直接失效。
            #
            # 代价是这块 arena 不保证清过零，搬运回退路 gather 会读到还没写过的行，
            # 万一正好是 FP8 的 NaN 位型会污染同一个 kernel 里的别的行。
            # 默认走的是分页打分，它按 pool_lens 只读每个请求自己那段，碰不到这些行；
            # 搬运路只在分页路被拒时才走。这个取舍记在结果文档里。
        # 两个别名视图，和打包缓冲共享同一块存储，不复制：
        # 写入路按行寻址用 flat，scale 按 FP32 重解释用 flat_f32
        # （rows * 132 能被 4 整除，所以这个重解释是合法的）。
        self.pool_u8 = self.pool_packed.view(-1)
        self.pool_scale = self.pool_u8.view(torch.float32)
        # 越界审计的剩余次数，见 _audit_row_bounds。默认那条路直接返回，不花代价。
        # skip 把窗口移出预热段：起服务时会先用一个很小的假 cache 跑探测前向，
        # 不跳过的话八次配额全花在那上面，拿不到真实请求的数据（已经犯过两次）。
        self._row_audit_left = 8
        try:
            self._row_audit_skip = int(os.environ.get("RTP_GLM_KPOOL_ROW_AUDIT_SKIP", "0"))
        except ValueError:
            self._row_audit_skip = 0


    def _fused_write(self, rows, slots, ape, hadamard, total) -> bool:
        """一个 Triton kernel 走完整条写入路；形状不合适就返回 False 退回 torch 版。

        置 ``RTP_GLM_KPOOL_FUSED_WRITE=0`` 可强制走 torch 版，用来 A/B 或排查。
        """
        # **默认开**。这个 kernel 确实与 torch 版有微小数值偏差
        # （生产内审计：约 1.56% 的池行不同，离线复现见
        # scripts/pla/check_kpool_fused_write_prod.py，全部在一个 FP8 刻度以内），
        # 但**关掉它并不能改善端到端精度**——v20（开）与 v21（关）的完整统计判据
        # 完全相同，32768 档首个分歧步都是 1.0。
        # 而关掉的代价是确凿的：并发 1 的 decode 从 25.70 涨到 27.11 毫秒。
        # 代价确凿、收益为零的开关不该默认关着。
        # 那个数值偏差本身仍是未解决的缺陷，记在结果文档里，但它不是精度退化的原因。
        if triton is None or os.environ.get("RTP_GLM_KPOOL_FUSED_WRITE", "1") == "0":
            return False
        # Triton 只能吃设备上的指针，CPU 张量会在 launch 时报
        # "Pointer argument cannot be accessed from Triton"。单元测试在 CPU 上跑，
        # 这里按设备退回 torch 版，而不是让它崩在 kernel 启动里。
        if rows.device.type != "cuda":
            return False
        d, r = self.head_dim, self.ratio
        # tl.arange 要求 2 的幂；tl.dot 的 M 至少 16，所以每个 program 固定处理 16 个 token
        if d & (d - 1) or r & (r - 1) or self.tokens_per_page % r:
            return False
        # 默认走每 token 一个 program 的蝶形版；RTP_GLM_KPOOL_WRITE_BUTTERFLY=0 退回
        # 旧的矩阵乘版（16 个 token 一个 program），用来 A/B
        if os.environ.get("RTP_GLM_KPOOL_WRITE_BUTTERFLY", "1") != "0":
            _pool_write_row_kernel[(total,)](
                rows,
                slots,
                ape,
                self.pool_u8,
                self.pool_scale,
                TPP=self.tokens_per_page,
                R=r,
                D=d,
                LOG_D=d.bit_length() - 1,
                PPP=self.pools_per_page,
                PAGE_BYTES=self.page_bytes,
                num_warps=1,
            )
            return True
        block_t = 16
        _pool_write_kernel[(triton.cdiv(total, block_t),)](
            rows,
            slots,
            ape,
            hadamard,
            self.pool_u8,
            self.pool_scale,
            total,
            TPP=self.tokens_per_page,
            R=r,
            D=d,
            PPP=self.pools_per_page,
            BT=block_t,
            PAGE_BYTES=self.page_bytes,
            num_warps=4,
        )
        return True

    def _audit_row_bounds(self, rows: torch.Tensor, slots: torch.Tensor) -> None:
        """诊断用：看写入的池行号有没有超出池缓冲，以及读的成员槽位有没有超出 cache。

        两条写入路（torch 的 ``index_copy_`` 和融合 Triton kernel）都只按「这个 token
        有没有数据」掩码，**都没有对行号做上界约束**。CUDA 上越界写不报错，会打到
        ``pool_packed`` 之后的别的显存上；打坏的数据两条读路会一起读到，所以读路对拍
        永远一致——这正是现在「审计全一致但开关一变输出就变」的形状。

        置 ``RTP_GLM_KPOOL_ROW_AUDIT=1`` 打开。会同步一次取标量，只在前若干次调用上做，
        所以不能常开。默认关。
        """
        if os.environ.get("RTP_GLM_KPOOL_ROW_AUDIT", "0") != "1":
            return
        if self._row_audit_left <= 0:
            return
        if self._row_audit_skip > 0:
            self._row_audit_skip -= 1
            return
        if _capturing(rows.device):
            # 取标量要同步，捕获期间同步会让整张图捕不成、八个 rank 一起起不来。
            # decode 的 write 正是在捕获里跑的，所以这里必须躲开；
            # 捕获前的 eager 预热和 prefill 都会走到下面，不影响拿到结论。
            return
        self._row_audit_left -= 1
        import logging

        pool_rows = _pool_rows_for_slots(slots, self.tokens_per_page, self.ratio)
        # 读的那一侧：同组 ratio 个成员，最大的是 group_base + ratio - 1
        member_max = (
            torch.div(slots, self.ratio, rounding_mode="floor") * self.ratio
            + (self.ratio - 1)
        ).max()
        row_limit = self.num_pages * self.pools_per_page
        pr_max = int(pool_rows.max())
        logging.warning(
            "KPOOL_ROW_AUDIT 槽位 [%d, %d] / cache 行数 %d；池行号 [%d, %d] / 池行数 %d；"
            "成员上界 %d；写越界=%s 读越界=%s",
            int(slots.min()), int(slots.max()), int(rows.shape[0]),
            int(pool_rows.min()), pr_max, row_limit,
            int(member_max),
            pr_max >= row_limit,
            int(member_max) >= int(rows.shape[0]),
        )

    def update_from_slots(
        self,
        rows: torch.Tensor,
        slots: torch.Tensor,
        ape: torch.Tensor,
        hadamard: torch.Tensor,
    ) -> None:
        """把本批 token 所在的池重新算一遍并存进来。

        Args:
            rows: ``[总槽位数, 2 * head_dim]``，indexer cache 的 BF16 (key, gate) 视图。
            slots: ``[T]``，本批 token 的物理槽位，已经钳到非负。
            ape:   ``[ratio, head_dim]``，池内位置偏置。
            hadamard: ``[head_dim, head_dim]``，行向量约定的归一化 Hadamard 矩阵。
        """
        ratio, head_dim = self.ratio, self.head_dim
        total = int(slots.numel())
        if total == 0:
            return
        self._audit_row_bounds(rows, slots)
        if self._fused_write(rows, slots, ape, hadamard, total):
            return
        # 不按「这个 token 是不是让某组凑满」筛选：筛选要按数据取子集，形状就不再静态，
        # eager 下还要同步一次。同一组里的 ratio 个 token 算出的池值完全相同，所以
        # 重复算只是浪费，不会算错；没凑满的组算出来的是会被覆盖的垃圾值，打分时由
        # ``ke = 上下文长度 // ratio`` 排除在外。代价是 prefill 一个 8192 token 的
        # 分块里有 4 倍重复，相对该分块 550 毫秒的总耗时可以忽略。
        lanes = torch.arange(ratio, device=slots.device, dtype=slots.dtype)
        ape_f = ape.to(torch.float32).view(1, ratio, head_dim)
        # 分块只为了压住临时张量的峰值：一次 8192 个 token 的 FP32 中间量约 100 MB，
        # 实测 PPU 的分配器在这个量级上会走到一条它没实现的驱动接口。
        # 块数只由静态的 token 数决定，捕获期间也能跑。
        chunk = max(1, (32 << 20) // max(1, ratio * 2 * head_dim * 4))
        for begin in range(0, total, chunk):
            end = min(begin + chunk, total)
            part_slots = slots[begin:end]
            group_base = torch.div(part_slots, ratio, rounding_mode="floor") * ratio
            member = (group_base.unsqueeze(1) + lanes).reshape(-1)
            part = rows.index_select(0, member).view(-1, ratio, 2 * head_dim)
            key = part[..., :head_dim].float()
            gate = part[..., head_dim:].float()
            # 与 gather_kpool_pool_keys 逐项同口径：逐通道在组内位置上 softmax，
            # 权重先落一次 BF16 再回 FP32 乘回 key——少这一步会和老路差到 FP8 量化之上。
            weight = torch.softmax(gate + ape_f, dim=1)
            pooled = (weight.to(torch.bfloat16).float() * key).sum(dim=1)
            pooled = pooled @ hadamard
            # 没凑满的组读到的是别的请求留下的旧值或刚初始化的 0，算出来是垃圾但不会
            # 被打分读到。真出现 NaN 的话会留在池里，下一次 gather 就可能把同一个
            # 打分 kernel 里的别的行也带成 NaN，所以这里压掉。
            pooled = torch.nan_to_num(pooled, nan=0.0, posinf=0.0, neginf=0.0)
            quantized, scale = quantize_pool_keys(pooled.to(torch.bfloat16))
            pool_rows = _pool_rows_for_slots(part_slots, self.tokens_per_page, ratio)
            # 与融合 kernel 同一套页内分区布局：一页里先连续放所有 key，再放所有 scale。
            dev = pool_rows.device
            ppp, d = self.pools_per_page, self.head_dim
            page, islot = pool_rows // ppp, pool_rows % ppp
            key_pos = (
                page.unsqueeze(1) * self.page_bytes
                + islot.unsqueeze(1) * d
                + torch.arange(d, device=dev)
            ).reshape(-1)
            self.pool_u8.index_copy_(
                0, key_pos, quantized.view(torch.uint8).reshape(-1)
            )
            self.pool_scale.index_copy_(
                0, page * (self.page_bytes // 4) + (ppp * d) // 4 + islot, scale
            )

    def gather(
        self, block_table: torch.Tensor, max_pools: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """按请求取出前 ``max_pools`` 个池，拼成固定步长的打分缓冲。

        返回值与 ``KPoolIndexer.build_pools`` 的老实现完全同形：
        ``pool_fp8 [B * max_pools, head_dim]``、``pool_scale [B * max_pools]``、
        ``pool_start [B]``（第 b 个请求从第 ``b * max_pools`` 行开始）。
        """
        device = self.pool_u8.device
        batch = int(block_table.shape[0])
        pool_start = (
            torch.arange(batch, device=device, dtype=torch.long) * int(max_pools)
        )
        if batch == 0 or max_pools == 0:
            return (
                torch.empty(
                    (0, self.head_dim), dtype=torch.float8_e4m3fn, device=device
                ),
                torch.empty((0,), dtype=torch.float32, device=device),
                pool_start,
            )
        max_pages = int(block_table.shape[1])
        # 池在块表里的页号与页内序号只取决于形状，与请求内容无关，复用同一份按形状
        # 缓存的常量（这里的「每页条数」是每页的池数，不是 token 数）。
        page, offset = _pool_slot_constants(
            int(max_pools), self.pools_per_page, max_pages, device
        )
        safe_pages = block_table.to(torch.long).clamp(0, max(self.num_pages - 1, 0))
        rows = safe_pages.index_select(1, page) * self.pools_per_page + offset.unsqueeze(
            0
        )
        flat = rows.reshape(-1)
        # 回退路：按页内分区布局把 key 与 scale 取出来。
        # 用 as_strided 把两个区各自看成 [页数, 每页条数, ...] 的视图再做高级索引，
        # 代价和改布局之前那次整行 index_select 相当。
        # 不要按元素位置自己拼索引：那样要造一个 [条数, head_dim] 的 int64 索引张量，
        # 并发 128 时就是约 1.3 GB，实测把 decode 整体拖慢三成。
        # 打分走分页算子时根本不会走到这里——那条路直接按页读这块 cache。
        ppp, d = self.pools_per_page, self.head_dim
        fpage, fslot = flat // ppp, flat % ppp
        keys_view = self.pool_u8.as_strided(
            (self.num_pages, ppp, d), (self.page_bytes, d, 1)
        )
        # storage_offset 是**相对整块存储**的，不是相对这个张量的。
        # 这块缓冲来自缓存组时自身就带偏移，只传 (ppp*d)//4 会把它顶掉，
        # 于是 scale 从别的行读出来——key 读对（上面那个 as_strided 没传 offset，
        # 沿用张量自身的偏移），scale 读错，表现是「key 逐位相同、scale 几乎全不同，
        # 而且两侧出现的是同一批数值只是落在不同行」。
        # 私有缓冲时偏移为 0，这个错一直看不出来。
        scales_view = self.pool_scale.as_strided(
            (self.num_pages, ppp),
            (self.page_bytes // 4, 1),
            self.pool_scale.storage_offset() + (ppp * d) // 4,
        )
        keys = keys_view[fpage, fslot]
        scales = scales_view[fpage, fslot]
        return keys.view(torch.float8_e4m3fn), scales, pool_start


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
        self.incremental = incremental_pooling_enabled()
        self.store: Optional[KPoolStore] = None
        self._pool_audit_left = 4
        try:
            self._pool_audit_min_rows = int(
                os.environ.get("RTP_GLM_KPOOL_AUDIT_MIN_ROWS", "2048")
            )
        except ValueError:
            self._pool_audit_min_rows = 2048

    def _ensure_store(
        self, rows: torch.Tensor, pool_base: Optional[torch.Tensor] = None
    ) -> KPoolStore:
        """池 cache 按 indexer cache 的页数开，首次写入时建好。

        不能在 CUDA Graph 捕获里建：那样分配出来的显存属于那张图的私有池，
        别的图或 eager 调用读到的就不是同一块。decode 的捕获前有两次 eager 预热
        （``cuda_graph_runner.cc`` 的 ``captureOneGraphInstance``），所以正常路径下
        这里总是在 eager 阶段就建好了。
        """
        num_pages = int(rows.shape[0]) // self.layout.tokens_per_page
        if self.store is not None and self.store.requested_pages == num_pages:
            return self.store
        if _capturing(rows.device):
            raise RuntimeError(
                "KPool 池 cache 还没建好就进了 CUDA Graph 捕获；"
                "捕获前应当有 eager 预热把它建出来"
            )
        self.store = KPoolStore(
            self.layout.head_dim,
            self.layout.tokens_per_page,
            self.ratio,
            num_pages,
            rows.device,
            base=pool_base,
        )
        return self.store

    # ---- 写 cache ----
    def write(
        self, kv_cache_base, slot_mapping, key_bf16, gate_bf16, ape=None, pool_base=None
    ) -> None:
        write_kpool_entries(self.layout, kv_cache_base, slot_mapping, key_bf16, gate_bf16)
        if not self.incremental:
            return
        if ape is None:
            raise ValueError("增量池化需要 ape（池内位置偏置）")
        rows = self.layout.view(kv_cache_base)
        store = self._ensure_store(rows, pool_base)
        slots = torch.clamp(slot_mapping.reshape(-1).to(torch.long), min=0)
        store.update_from_slots(
            rows,
            slots,
            ape.to(rows.device).float().view(self.ratio, self.layout.head_dim),
            _hadamard_matrix(self.layout.head_dim, rows.device),
        )

    # ---- 组池 ----
    def paged_score(self, q_fp8, weights, block_table, pool_lens, max_pools,
                    block_table_i32=None):
        """按页直接在池 cache 上打分，不把池搬出来。

        池本来就是每页 64 条，正好是 DeepGEMM 分页打分算子写死的 ``block_kv``，
        所以这块 cache 可以原样喂给它。省掉的是 ``gather`` 那次整段搬运——
        每副本 batch 1 时每个 DSA 层 1.31 MB、11 层每步 14.4 MB。

        返回 ``[rows, max_pools]`` 的 FP32 分数，已经是「每行只含自己那段」的形状，
        所以调用方也不再需要打分后那次按行切片。
        """
        from rtp_llm.models_py.modules.base.cuda.indexer_op import (
            _fp8_paged_mqa_logits_compat,
        )

        store = self.store
        # 页号必须钳进池 cache 的实际页数，与搬运路同一口径
        # （那边是 `block_table.clamp(0, num_pages - 1)`）。
        # 预热和图捕获阶段 cache 只有很少几页，而块表带的是真实页号——
        # 搬运路钳位之后算出的是垃圾但不崩（那一步的输出本来就丢弃），
        # 分页算子不钳位，会直接在 host 端断言，把整个图捕获带崩。
        # block_table_i32：调用方已经按同一口径转好、钳好（一次前向内各 DSA 层共用）
        block_table = (
            block_table_i32
            if block_table_i32 is not None
            else block_table.to(torch.int32).clamp(0, max(store.num_pages - 1, 0))
        )
        return _fp8_paged_mqa_logits_compat(
            q_fp8.unsqueeze(1),
            # 算子要 [页数, 每页条数, 1, 槽位字节]；底层存的是 [页数, 页字节数]，
            # 页内按「先所有 key 再所有 scale」分区摆，view 不复制。
            store.pool_packed.view(
                store.num_pages, store.pools_per_page, 1, store.slot_bytes
            ),
            weights,
            pool_lens.reshape(-1, 1),
            block_table,
            store.pools_per_page,
            int(max_pools),
        )

    def audit_stored_vs_recomputed(
        self, kv_cache_base, block_table, ape, pool_ks, pool_ke
    ):
        """比「存下来的池」与「从 cache 现算的池」在打分窗口内差多少行。

        置 ``RTP_GLM_KPOOL_AUDIT=1`` 打开，``RTP_GLM_KPOOL_AUDIT_SKIP=N`` 把窗口
        移出预热段——不跳过的话配额全花在起服务时那几次很小的假 cache 上，
        对真实请求没有说明力（这个坑踩过两次）。会同步取标量，不能常开。

        窗口必须是**打分真正读的那一段** ``[pool_ks, pool_ke)``，由调用方传进来。
        不要拿 ``build_pools`` 收到的 ``context_lens`` 自己算：它在预热前向里是按
        最大序列长度填充的，而且 ``gather_kpool_pool_keys`` 根本不用它——
        用它当窗口会把两千多行死区算进来，得出「scale 几乎每行都不同」这种假结论
        （2026-11-21 犯过）。

        另外 decode 的真实请求走 CUDA Graph 回放，这段 Python 根本不执行，
        所以这条审计只在 prefill 侧（不走图）和 decode 的 eager 预热里能采到；
        要看真实请求就看 prefill 侧的数字。
        """
        if os.environ.get("RTP_GLM_KPOOL_AUDIT", "0") != "1":
            return
        store = self.store
        if store is None or self._pool_audit_left <= 0:
            return
        if _capturing(block_table.device):
            return
        import logging

        max_pools = int(block_table.shape[1]) * store.pools_per_page
        got_fp8, got_scale, _ = store.gather(block_table, max_pools)
        rotated, _ = gather_kpool_pool_keys(
            self.layout, kv_cache_base, block_table, None, ape,
            self.ratio, rotate=True,
        )
        want_fp8, want_scale = quantize_pool_keys(rotated)
        # [ks, ke) 是扁平行号区间，逐行展开成掩码
        total = int(got_scale.numel())
        col = torch.arange(total, device=block_table.device)
        ks = pool_ks.to(torch.int64).reshape(-1)
        ke = pool_ke.to(torch.int64).reshape(-1)
        live = torch.zeros((total,), dtype=torch.bool, device=block_table.device)
        for a, b in zip(ks.tolist(), ke.tolist()):
            if b > a:
                live |= (col >= a) & (col < min(b, total))
        if got_scale.shape != want_scale.shape:
            raise RuntimeError(
                f"池审计两路形状不同：存下来的 {tuple(got_scale.shape)}、"
                f"现算的 {tuple(want_scale.shape)}；形状不同会被广播成逐对比较，"
                "算出来的差异数字没有意义"
            )
        key_diff = (
            got_fp8.view(torch.uint8) != want_fp8.view(torch.uint8)
        ).any(dim=1) & live
        scale_diff = (got_scale != want_scale) & live
        n_live = int(live.sum())
        # 按**窗口大小**跳过预热，而不是按调用次数：计数器是每个 DSA 层一份的，
        # 每个请求每层只被调用四五次，用「跳过前 N 次」那种写法根本到不了真实请求
        # （设成 30 的那一轮 prefill 一行都没记上）。预热前向用的是很小的假 cache，
        # 窗口只有几十到几百行，真实请求是几千行，按窗口判自动分得开。
        if n_live < self._pool_audit_min_rows:
            return
        self._pool_audit_left -= 1
        live_f = live.float()
        # 只在窗口内比 scale 的相对差，绝对差看不出量级（scale 本身在 1e-3 量级）
        rel = ((got_scale - want_scale).abs() / want_scale.abs().clamp_min(1e-30))
        rel = torch.where(live, rel, torch.zeros_like(rel))
        msg = (
            "KPOOL_POOL_AUDIT 窗口 %d 行：key 不同 %d 行（%.3f%%），"
            "scale 不同 %d 行（%.3f%%），scale 相对差 最大 %.3e 均值 %.3e"
            % (
                n_live, int(key_diff.sum()),
                100.0 * float(key_diff.sum()) / max(1, n_live),
                int(scale_diff.sum()),
                100.0 * float(scale_diff.sum()) / max(1, n_live),
                float(rel.max()), float(rel.sum() / live_f.sum().clamp_min(1)),
            )
        )
        # 比值是常数还是随数据变，直接决定下一步往哪查：
        # 常数说明某一路少乘/少除了一个因子，变动说明是数值路径不同。
        idx = torch.nonzero(live, as_tuple=False).reshape(-1)[:5]
        if int(idx.numel()):
            g = got_scale.index_select(0, idx).tolist()
            w = want_scale.index_select(0, idx).tolist()
            msg += "  前 5 个活行 存/算 = " + ", ".join(
                f"{a:.6g}/{b:.6g}={a/b:.4f}" if b else f"{a:.6g}/0" for a, b in zip(g, w)
            )
        logging.warning("%s", msg)
        # 可以额外落到一个专用文件：引擎日志会按轮归档，多个 rank 又写同一个文件，
        # 从里面分不清哪一批是哪一轮（已经误读过一次）。给 RTP_GLM_KPOOL_AUDIT_LOG
        # 一个路径就写，默认不写——审计本身只在排查时打开，不该假设某个固定目录存在。
        path = os.environ.get("RTP_GLM_KPOOL_AUDIT_LOG", "")
        if path:
            try:
                with open(path, "a") as fh:
                    fh.write(msg + "\n")
            except OSError:
                pass

    def build_pools(self, kv_cache_base, block_table, context_lens, ape, pool_base=None):
        if self.incremental and incremental_read_enabled():
            rows = self.layout.view(kv_cache_base)
            store = self._ensure_store(rows, pool_base)
            max_pools = int(block_table.shape[1]) * store.pools_per_page
            if paged_scoring_enabled():
                # 打分按页直接读池 cache，不需要把池搬出来——这次搬运正是要省掉的
                # 那一项。只有每请求的起始行还要给出来，那是纯形状计算。
                batch = int(block_table.shape[0])
                device = store.pool_packed.device
                empty = torch.empty(
                    (0, store.head_dim), dtype=torch.float8_e4m3fn, device=device
                )
                return (
                    empty,
                    torch.empty((0,), dtype=torch.float32, device=device),
                    torch.arange(batch, device=device, dtype=torch.long) * max_pools,
                )
            return store.gather(block_table, max_pools)
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
               row_context_lens, run_topk, max_pools=None, logits=None,
               window_is_relative=False):
        """返回 ``[rows, sparse_width]`` int32 的**请求内 token 索引**（-1 为无效）。"""
        import torch as _torch

        # 走和 indexer_op 同一个封装：有 DeepGEMM 就用它的原生 FP8 MQA 打分，
        # 没有才退回自写的 Triton。GLM 的 index_n_heads=32、index_head_dim=128
        # 落在 DeepGEMM 在 M890 上支持的头数（32/64）与头宽（64/128）里；
        # Legacy FP8 家族收 FP32 的路由权重、出 FP32 logits，与这里的约定一致。
        # 实测自写 Triton 在 prefill 每个 DSA 层要 2.8 毫秒，是改之前最大的单项。
        from rtp_llm.models_py.modules.base.cuda.indexer_op import (
            _fp8_mqa_logits_compat,
        )

        rows = int(q_fp8.shape[0])
        if rows == 0 or (logits is None and (pool_fp8 is None or int(pool_fp8.shape[0]) == 0)):
            return _torch.full(
                (rows, self.sparse_width), -1, dtype=_torch.int32, device=q_fp8.device
            )
        if logits is None:
            logits = _fp8_mqa_logits_compat(
                q_fp8, (pool_fp8, pool_scale), weights, pool_ks, pool_ke,
                clean_logits=False,
            )
        elif not window_is_relative:
            # 分页打分直接给出 [行数, max_pools]，每行已经只含自己那段，
            # 所以窗口从 0 开始、宽度就是该请求的池数，下面那次切片不需要了。
            # window_is_relative=True 表示调用方已经按这个口径给好了（共用的 decode 元数据）。
            pool_ke = (pool_ke - pool_ks).to(pool_ke.dtype)
            pool_ks = _torch.zeros_like(pool_ks)
        # 池缓冲是固定步长的：第 b 个请求占 [b*max_pools, (b+1)*max_pools)，
        # 所以 logits 是 [查询行数, 请求数 * max_pools]，而每一行真正要看的只有自己
        # 那 max_pools 列。并发 128 时列数 16*10240 = 163,840，top-k 要白扫 16 倍数据，
        # 也超出了 PPU 那条按列全展开快路的 34,816 列上限。
        # 行数不多时（decode：每个请求一行）先把各行切回自己那段再选，两样都解决。
        # prefill 的行数是上千，切出来的张量反而更大，所以按行数设闸。
        if max_pools and rows <= 64 and int(logits.shape[1]) > int(max_pools):
            window = _torch.arange(int(max_pools), device=logits.device)
            index = pool_ks.to(_torch.long).unsqueeze(1) + window.unsqueeze(0)
            logits = logits.gather(1, index)
            # 切完之后每行的窗口都从 0 开始；top-k 后端返回的本来就是相对 starts 的
            # 请求内下标，所以这一步不改变输出语义。
            pool_ke = (pool_ke - pool_ks).to(pool_ke.dtype)
            pool_ks = _torch.zeros_like(pool_ks)
        local = logits.new_empty((rows, self.pool_topk), dtype=_torch.int32)
        # The top-k backends already return request-local indices: both the PPU
        # Triton path (``kernels/ppu/indexer_topk.py`` subtracts ``start`` in all
        # three of ``_finish``, ``_stable_topk_small_kernel`` and
        # ``_all_indices``) and the CUDA op (contract comment in
        # ``bindings/cuda/kernels/dsv4_top_k_per_row_prefill.h``: "Indices are
        # relative to row_starts[r]"). Subtracting ``pool_ks`` a second time
        # would turn every request after the first into large negatives.
        # 最后一个参数是「最长能看到多少列」，切过之后要跟着变，
        # 否则后端会按原来的全宽去判断该不该走快路。
        # 打分带的是 clean_logits=False，越界列不清零，交给 top-k 后端按 ks/ke 排除。
        # 但两条取池路在越界列上的内容**完全不同**：
        #   重算路 —— 从当前 cache 临时算出来的垃圾，量级通常很低
        #   增量路 —— 那些物理页里存的是**别的请求的真实池**，量级正常、打分不低
        # 只要后端对边界的处理有任何缝隙，增量路就会选中别的请求的上下文块，
        # 表现正是「局部流畅但不召回本请求上下文里的具体符号」。
        # 这里显式把越界列压成 -inf，不依赖后端的边界语义。
        # **默认关**：这是一次失败的修复尝试。它确实把越界列排除干净了
        # （top-20 Jaccard 从 0.739 升到 0.818），但**没有修好首个分歧步**
        # （32768 档仍是 1.0），而每层一次扫满 max_pools 列的 torch.where
        # 实测让并发 1 的 decode 从 25.70 涨到 27.27 毫秒。
        # 代价确凿、收益没有，所以默认不开；置 1 可用于继续排查。
        if os.environ.get("RTP_GLM_KPOOL_MASK_OOR", "0") == "1":
            col = _torch.arange(int(logits.shape[1]), device=logits.device)
            inside = (col.unsqueeze(0) >= pool_ks.unsqueeze(1)) & (
                col.unsqueeze(0) < pool_ke.unsqueeze(1)
            )
            logits = _torch.where(
                inside, logits, _torch.full_like(logits, float("-inf"))
            )
        run_topk(logits, pool_ks, pool_ke, local, int(logits.shape[1]))
        return expand_pool_indices_to_tokens(
            local, row_context_lens, self.ratio, self.sparse_width
        )
