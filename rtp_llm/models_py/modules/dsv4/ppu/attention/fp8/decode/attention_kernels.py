"""PPU overrides; unchanged behavior is inherited from the shared DSV4 implementation."""

from __future__ import annotations
import os
from typing import Any, Optional
import torch
from rtp_llm.models_py.modules.dsv4.fp8.decode.fp8_sparse_attn_decode_op import (
    SparseAttnV4DecodeFp8Op,
)

# FlashMLA splits KV work over (request, SM part) pairs, and every query token
# of a request goes through the same split. DSpARK verify sends 8 requests x 6
# tokens per rank: as 48 one-token rows the scheduler spreads them like 48
# requests and needs fewer partial results to combine. Each token keeps its own
# indices and its request's length bound, so only the reduction order changes.
_DECODE_ATTN_TOKEN_ROWS = os.environ.get("DSV4_PPU_DECODE_ATTN_TOKEN_ROWS", "0") == "1"


def _use_token_rows(bsz: int, q_len: int) -> bool:
    # A single request keeps its [1, q_len] call: in the DP8 engine the
    # one-request warmup of the main model's SWA layer faults with 6 one-token
    # rows (all other batch sizes, capture and replay run), and the same call
    # with identical inputs does not fault on an isolated card. Cause unknown.
    return _DECODE_ATTN_TOKEN_ROWS and q_len > 1 and bsz > 1


def _token_rows(tensor: torch.Tensor) -> torch.Tensor:
    """``[B, q_len, ...]`` -> ``[B * q_len, 1, ...]``."""
    return tensor.reshape(-1, 1, *tensor.shape[2:])


def _token_lengths(lengths: Optional[torch.Tensor], q_len: int) -> Optional[torch.Tensor]:
    """Per-request ``[B]`` lengths -> one entry per query token ``[B * q_len]``."""
    if lengths is None:
        return None
    return lengths[:, None].expand(-1, q_len).reshape(-1)


def attn_fp8_swa_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, entries_per_block, 584] uint8
    attn_sink: torch.Tensor,  # [H] bf16 / fp32
    swa_topk_3d: torch.Tensor,  # [B, 1, win] int32 global slots into SWA pool
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadataFP8
    fp8_op: SparseAttnV4DecodeFp8Op,
    platform_provider=None,
    topk_length: Optional[torch.Tensor] = None,  # [B] int32 SWA effective length
) -> torch.Tensor:
    """SWA-only FP8 FlashMLA — returns ``[B, 1, H, D_v]`` bf16 output.

    Contract notes:
      * ``swa_topk_3d`` holds per-request **global pool slot ids** (``-1``
        padding for entries before sequence start). FlashMLA's sparse FP8
        kernel reads ``pool[indices[i]]`` with no block-table indirection,
        so the caller MUST translate abs positions via
        :func:`~paged_topk_translator.translate_local_to_global_slots`
        before invoking this op.
      * ``sched_meta`` is owned by :class:`DSv4DecodeAttnMetadataFP8` (see
        :func:`~decode_attn_metadata.get_or_build_sched_meta`); lifetime
        is tied to the decode step (eager) or the capture impl (CUDA graph).
      * ``topk_length`` (opt_flash_mla) is the per-request SWA effective length
        ``[B] int32``; FlashMLA only scans ``swa_topk_3d[:, :, :topk_length[b]]``.
        ``None`` keeps the legacy full-capture-width scan.
    """
    validate = getattr(platform_provider, "validate_swa_pool", None)
    if validate is not None:
        validate(q=q, swa_pool_3d=swa_pool_3d, swa_topk_3d=swa_topk_3d, attn_sink=attn_sink, topk_length=topk_length)
    bsz, q_len = q.shape[:2]
    if _use_token_rows(bsz, q_len):
        out = fp8_op.forward(
            _token_rows(q),
            swa_pool_3d,
            attn_sink,
            _token_rows(swa_topk_3d),
            sched_meta,
            block_table=swa_block_table,
            topk_length=_token_lengths(topk_length, q_len),
        )
        return out.view(bsz, q_len, *out.shape[2:])
    return fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=topk_length,
    )

def attn_fp8_dual_paged(
    *,
    q: torch.Tensor,  # [B, 1, H, D] bf16
    swa_pool_3d: torch.Tensor,  # [num_blocks, eb_swa, 584] uint8
    cmp_pool_3d: torch.Tensor,  # [num_blocks, eb_cmp, 584] uint8
    compress_ratio: Optional[int] = None,
    cmp_tokens_per_block: Optional[int] = None,
    attn_sink: torch.Tensor,
    swa_topk_3d: torch.Tensor,  # [B, 1, win] int32 global slots into SWA pool
    cmp_topk_3d: torch.Tensor,  # [B, 1, K_cmp] int32 global slots into cmp pool
    swa_block_table: torch.Tensor,  # [B, max_blocks_per_req] int32
    sched_meta: Any,  # FlashMLA sched_meta from DSv4DecodeAttnMetadataFP8
    fp8_op: SparseAttnV4DecodeFp8Op,
    platform_provider=None,
    topk_length: Optional[torch.Tensor] = None,  # [B] int32 SWA effective length
    extra_topk_length: Optional[torch.Tensor] = None,  # [B] int32 cmp effective length
) -> torch.Tensor:
    """Dual-pool (SWA + compressed) FP8 FlashMLA — returns ``[B, 1, H, D_v]``.

    The FlashMLA kernel merges the two pools' attention in-kernel via
    ``extra_k_cache`` / ``extra_indices_in_kvcache`` / merged-softmax.
    Eliminates the legacy "dequant both pools → BF16 cat → TileLang
    sparse_attn" round-trip.

    Callers must feed indices ``[B, 1, K]``-contiguous (iter3.2 — the
    translator already emits int32 so no dtype cast is needed). CSA and
    HCA layers use distinct scheduler entries, further keyed by the actual
    compressed index width, see
    :func:`~decode_attn_metadata.get_or_build_sched_meta`.

    ``topk_length`` / ``extra_topk_length`` (opt_flash_mla) are per-request
    ``[B] int32`` effective lengths on ``swa_topk_3d`` / ``cmp_topk_3d``;
    FlashMLA only scans the leftmost ``length[b]`` indices of each (``-1``
    entries inside that bound are skipped). This shrinks the HCA scan from the
    CUDA-graph capture width (e.g. 8192) to the true ``(seq_len)//ratio``.
    ``None`` keeps the legacy full-capture-width scan.
    """
    validate = getattr(platform_provider, "validate_dual_pool", None)
    if validate is not None:
        validate(q=q, swa_pool_3d=swa_pool_3d, swa_topk_3d=swa_topk_3d, cmp_pool_3d=cmp_pool_3d, cmp_topk_3d=cmp_topk_3d, attn_sink=attn_sink, topk_length=topk_length, extra_topk_length=extra_topk_length, compress_ratio=compress_ratio, cmp_tokens_per_block=cmp_tokens_per_block)
    bsz, q_len = q.shape[:2]
    if _use_token_rows(bsz, q_len):
        out = fp8_op.forward(
            _token_rows(q),
            swa_pool_3d,
            attn_sink,
            _token_rows(swa_topk_3d),
            sched_meta,
            block_table=swa_block_table,
            topk_length=_token_lengths(topk_length, q_len),
            extra_k_cache=cmp_pool_3d,
            extra_topk_idxs=_token_rows(cmp_topk_3d),
            extra_topk_length=_token_lengths(extra_topk_length, q_len),
        )
        return out.view(bsz, q_len, *out.shape[2:])
    return fp8_op.forward(
        q,
        swa_pool_3d,
        attn_sink,
        swa_topk_3d,
        sched_meta,
        block_table=swa_block_table,
        topk_length=topk_length,
        extra_k_cache=cmp_pool_3d,
        extra_topk_idxs=cmp_topk_3d,
        extra_topk_length=extra_topk_length,
    )
