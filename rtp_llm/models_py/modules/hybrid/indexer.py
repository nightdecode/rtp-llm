from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group
from rtp_llm.models_py.distributed.pynccl_cp import all_gather
from rtp_llm.models_py.modules import IndexerOp, LayerNorm
from rtp_llm.models_py.modules.base.cuda.indexer_op import (
    _physical_block_table,
    _rotate_activation,
)
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import AttentionConfigs, HWKernelConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import LayerKVCache
from rtp_llm.utils.model_weight import W

_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.sparse_mla.fused_logits_head_gate import (
        fused_logits_head_gate,
    )

    _FP8_GEMM_LINEAR = CudaFp8GEMMLinear
elif _DEVICE_TYPE == DeviceType.Ppu:
    from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_deepgemm_linear import (
        PpuFp8DeepGemmLinear,
    )

    fused_logits_head_gate = None  # type: ignore
    _FP8_GEMM_LINEAR = PpuFp8DeepGemmLinear
else:
    fused_logits_head_gate = None  # type: ignore
    _FP8_GEMM_LINEAR = None


def _is_decode_phase(attention_inputs: Any) -> bool:
    return (
        not attention_inputs.is_prefill
        or bool(getattr(attention_inputs, "is_target_verify", False))
        or bool(getattr(attention_inputs, "is_draft_extend", False))
    )


class Indexer(nn.Module):
    """
    Indexer for DeepSeek-V3.2 DSA (DeepSeek Sparse Attention) mechanism.
    Adapted from sglang's Indexer implementation.
    """

    def __init__(
        self,
        attn_config: AttentionConfigs,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        layernorm_eps: float,
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        parallelism_config: Optional[ParallelismConfig] = None,
        scale_fmt: Optional[str] = "none",
        stable_topk: bool = False,
        query_scale_fmt: Optional[str] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        # Resolve once at init: HWKernelConfig.enable_fuse_kernels (or env
        # ``ENABLE_FUSE_KERNELS``) → ``self._fuse_logits_head_gate``. Keep it
        # out of the forward path so it's free at decode (no env / config
        # lookup per token).
        self._fuse_kernels_enabled = fuse_kernels_enabled(hw_kernel_config)
        self._fuse_logits_head_gate = (
            self._fuse_kernels_enabled and fused_logits_head_gate is not None
        )

        self.index_n_heads = attn_config.indexer_head_num
        self.index_head_dim = attn_config.indexer_head_dim
        self.index_topk = attn_config.indexer_topk

        self.rope_head_dim = attn_config.rope_head_dim
        self.block_size = 128  # quantization block size (128)
        self.head_kv = 1
        self.scale_fmt = scale_fmt  # FP8 quantization format
        self.softmax_scale = self.index_head_dim**-0.5
        self.weights_scale = self.index_n_heads**-0.5
        self.blocksize = attn_config.kernel_tokens_per_block  # page size, typically 64
        # Owner (physical) block size used by the C++ KVCacheAllocator / CPSlotMapper
        # to decide page-RR ownership. bpk = owner_tpb / kernel_tpb >= 1. Mirrors the
        # DSV4 indexer's _kv_owner_tokens_per_block contract; threaded into
        # _get_topk_ragged_cp so build_indexer_cp_chunk_plan computes per-rank padded
        # local KV lens and restore indices at the owner granularity that matches
        # how prefill writes were laid out via cp_params.sharded_slot_mapping.
        kernel_tpb = int(attn_config.kernel_tokens_per_block)
        owner_tpb = int(getattr(attn_config, "tokens_per_block", kernel_tpb))
        if owner_tpb <= 0 or kernel_tpb <= 0 or owner_tpb % kernel_tpb != 0:
            owner_tpb = kernel_tpb
        self._kv_owner_tokens_per_block = owner_tpb
        self.indexer_size = self.index_head_dim / 2 + self.index_head_dim / 128 * 2
        self.is_neox_style = attn_config.rope_config.indexer_is_neox_style
        self.parallelism_config = parallelism_config

        self.wq_b = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_qb_w,
            W.mla_indexer_qb_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.wk = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_k_w,
            W.mla_indexer_k_s,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        # The caller hands down FP8 activations whenever the *MLA* projections
        # are FP8, but the indexer keeps its own dtype: GLM-5.3-Flash ships
        # indexer.wq_b / indexer.wk in BF16 while the MLA weights are FP8 per
        # [128,128] block. A BF16 linear has no ``input_scales`` argument, so
        # decide per linear whether the FP8 fast path is usable at all.
        self._wq_b_accepts_fp8 = _FP8_GEMM_LINEAR is not None and isinstance(
            self.wq_b, _FP8_GEMM_LINEAR
        )
        self._wk_accepts_fp8 = _FP8_GEMM_LINEAR is not None and isinstance(
            self.wk, _FP8_GEMM_LINEAR
        )

        self.k_norm = LayerNorm(
            weights[W.mla_indexer_k_norm_w],
            weights[W.mla_indexer_k_norm_b],
            eps=layernorm_eps,
        )

        self.weights_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.mla_indexer_weights_proj_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # Pre-contiguify weight for the fused Triton kernel (one-time init
        # copy). Production weight is often a transposed view [N, K] of
        # underlying [K, N] storage; the small-T per-(t,n) kernel needs
        # contiguous weight for coalesced 1D loads. Use plain attribute
        # reassignment (not `.data = ...`) — the latter does an in-place
        # `set_` of the underlying storage that leaves the tensor in a
        # state where `F.linear` under cuda-graph capture + inference_mode
        # trips PyTorch's version-counter check.
        if (
            self._fuse_logits_head_gate
            and hasattr(self.weights_proj, "weight")
            and not self.weights_proj.weight.is_contiguous()
        ):
            self.weights_proj.weight = self.weights_proj.weight.contiguous()
        # NoPE models (e.g. glm5_next with qk_rope_head_dim=0) ship no rotary
        # cos/sin cache; the indexer then skips RoPE and keeps only the
        # Hadamard rotation in IndexerOp.
        self.cos_sin_cache = (
            global_weights.get(W.rope_cos_sin_cache)
            if global_weights is not None
            else None
        )

        # Fused RoPE kernels use equal rotary/non-rotary halves and power-of-2
        # tiles. Other supported indexer geometries retain the unfused path.
        self._fuse_kernels_enabled = self._fuse_kernels_enabled and (
            self.index_head_dim == 128
            and self.rope_head_dim == 64
            and self.index_n_heads > 0
            and self.index_n_heads & (self.index_n_heads - 1) == 0
            and self.cos_sin_cache.shape[-1] == self.rope_head_dim
        )

        # ---- KPool（GLM-5.3-Flash 专有）----
        # ratio > 1 时 indexer 对「池」打分而不是对单个 token 打分，
        # cache 里存的是每 token 的 BF16 (key, 门控分数)，池 key 每次现算。
        # 语义与搬运依据见 modules/hybrid/kpool.py 的模块说明。
        self.compress_ratio = int(
            getattr(attn_config, "indexer_compress_ratio", 1) or 1
        )
        self.kpool = None
        if self.compress_ratio > 1:
            from rtp_llm.models_py.modules.hybrid.kpool import KPoolIndexer

            # 门控投影直接用 F.linear，不走 LinearFactory：ckpt 里
            # index_kpool_compress_gate 的形状是 [head_dim, hidden]（out, in），
            # 而 CudaF16Linear 约定传进去的权重已经是 [in, out]。
            # 官方与对方分支都是 `F.linear(hidden_states, wgate.weight)`，这里照搬。
            self.register_buffer(
                "kpool_gate_weight",
                weights[W.mla_indexer_kpool_gate],
                persistent=False,
            )
            self.register_buffer(
                "kpool_ape", weights[W.mla_indexer_kpool_ape], persistent=False
            )
            semantic_width = int(getattr(attn_config, "sparse_attention_topk", 0) or 0)
            if semantic_width <= 0:
                raise ValueError(
                    "KPool needs a positive sparse_attention_topk "
                    f"(indexer_compress_ratio={self.compress_ratio})"
                )
            # PPU 的稀疏 prefill kernel 断言 topk 是 128 的倍数（S4 实测 2*B_TOPK=128），
            # 2051 补到 2176，补位填 -1。
            kernel_width = ((semantic_width + 127) // 128) * 128
            self.kpool_semantic_width = semantic_width
            self.kpool_kernel_width = kernel_width
            self.kpool = KPoolIndexer(
                self.index_head_dim,
                self.compress_ratio,
                self.blocksize,
                self.index_topk,
                kernel_width,
            )

        self.indexer_op = IndexerOp(
            index_n_heads=self.index_n_heads,
            index_head_dim=self.index_head_dim,
            index_topk=self.index_topk,
            rope_head_dim=self.rope_head_dim,
            cos_sin_cache=self.cos_sin_cache,
            blocksize=self.blocksize,
            block_size=self.block_size,
            scale_fmt=self.scale_fmt,
            is_neox_style=self.is_neox_style,
            stable_topk=stable_topk,
            query_scale_fmt=query_scale_fmt,
            enable_fuse_kernels=self._fuse_kernels_enabled,
        )

    def _prefill_cp_enabled(self) -> bool:
        if self.parallelism_config is None:
            return False
        return self.parallelism_config.prefill_cp_config.is_enabled()

    def _prefill_cp_fused_quant_enabled(self) -> bool:
        return self._fuse_kernels_enabled

    def _is_sparse_prefill_cp(self, attention_inputs: Any) -> bool:
        return not _is_decode_phase(attention_inputs) and self._prefill_cp_enabled()

    def _get_logits_head_gate(
        self, x: torch.Tensor, q_scale: torch.Tensor
    ) -> torch.Tensor:
        # F3: fused (cast + GEMV + 2 elementwise muls) into one Triton kernel.
        # ``self._fuse_logits_head_gate`` is resolved at __init__ from
        # ``HWKernelConfig.enable_fuse_kernels``.
        scale = self.softmax_scale * self.weights_scale
        if self._fuse_logits_head_gate and x.is_contiguous():
            return fused_logits_head_gate(
                x,
                q_scale,
                self.weights_proj.weight,
                scale,
                fallback_proj=self.weights_proj,
            )
        x = x.float()
        weights = self.weights_proj(x)
        weights = weights.float()
        weights = weights.unsqueeze(-1) * q_scale * scale
        return weights

    def _fused_forward_decode(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: LayerKVCache,
        fmha_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused decode: QK kernel does K(RoPE+Had→bf16) + Q(RoPE+Had+FP8).

        Returns (q_fp8, q_scale).
        """
        if self._wq_b_accepts_fp8 and q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if self._wk_accepts_fp8 and x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        q_fp8, q_scale, key = self.indexer_op.fused_rope_quant_qk(
            q, k, fmha_params.positions_d
        )
        self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)

        return q_fp8, q_scale

    def _fused_forward_prefill_cp(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        kv_cache: LayerKVCache,
        fmha_params: Any,
        cp_params: Any,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._wq_b_accepts_fp8 and q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if self._wk_accepts_fp8 and x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        q_fp8, q_scale, key = self.indexer_op.fused_rope_quant_qk(
            q,
            k,
            cp_params.full_rope_pos_ids,
        )
        slot_mapping = (
            cp_params.sharded_slot_mapping
            if bool(getattr(cp_params, "kv_cache_sharded", False))
            else fmha_params.slot_mapping
        )
        self.indexer_op.quant_k_cp_only(
            key,
            kv_cache,
            slot_mapping,
            cp_params.kv_restore_unpad_indices,
        )
        return q_fp8, q_scale

    def _get_q_k_bf16(
        self,
        q_lora: torch.Tensor,
        x: torch.Tensor,
        flashmla_params: Any,
        cp_params: Optional[Any],
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._wq_b_accepts_fp8 and q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)

        if self._wk_accepts_fp8 and x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(x)
        k = self.k_norm(k)

        if cp_params is not None:
            query, key = self.indexer_op.apply_rope_and_rotate_q_k_cp(
                q,
                k,
                cp_params.full_rope_pos_ids,
            )
        else:
            positions = flashmla_params.positions_d
            query, key = self.indexer_op.apply_rope_and_rotate_q_k(q, k, positions)

        return query, key

    def _forward_all_keys_prefill_cp(
        self, x, kv_cache, fmha_params, cp_params, x_fp8, x_scale
    ):
        from rtp_llm.models_py.kernels.ppu.indexer_k_rope import fused_indexer_k_rope

        if self._wk_accepts_fp8 and x_fp8 is not None and x_scale is not None:
            key = self.wk(x_fp8, input_scales=x_scale)
        else:
            key = self.wk(x)
        key = fused_indexer_k_rope(
            self.k_norm(key),
            cp_params.full_rope_pos_ids,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        # Decode still needs exactly the same K cache, including CP restore,
        # all-gather and sharded physical slot mapping.
        slot_mapping = (
            cp_params.sharded_slot_mapping
            if bool(getattr(cp_params, "kv_cache_sharded", False))
            else fmha_params.slot_mapping
        )
        self.indexer_op.quant_k_cp_only(
            key, kv_cache, slot_mapping, cp_params.kv_restore_unpad_indices
        )
        return self.indexer_op._all_topk_ragged_cp(
            (
                0
                if cp_params.total_local_ids is None
                else int(cp_params.total_local_ids.numel())
            ),
            cp_params.total_kv_len,
            cp_params.precomputed_ks,
            cp_params.precomputed_ke,
            cp_params.precomputed_topk_off,
            x.device,
        )

    def _get_k_bf16(
        self,
        x: torch.Tensor,
        flashmla_params: Any,
    ) -> torch.Tensor:
        k = self.wk(x)
        k = self.k_norm(k)
        return self.indexer_op.apply_rope_and_rotate_k(k, flashmla_params.positions_d)

    def _quantize_q_k(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        kv_cache: LayerKVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None
            slot_mapping = (
                cp_params.sharded_slot_mapping
                if bool(getattr(cp_params, "kv_cache_sharded", False))
                else fmha_params.slot_mapping
            )
            return self.indexer_op.quant_q_k_cp(
                query,
                key,
                kv_cache,
                slot_mapping,
                cp_params.kv_restore_unpad_indices,
            )
        return self.indexer_op.quant_q_k(query, key, kv_cache, fmha_params.slot_mapping)

    def _compute_topk(
        self,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        kv_cache: LayerKVCache,
        fmha_params: Any,
        attention_inputs: Any,
        cp_params: Optional[Any],
    ) -> torch.Tensor:
        if _is_decode_phase(attention_inputs):
            return self.indexer_op._get_topk_paged(
                q_fp8, weights, kv_cache, fmha_params, attention_inputs
            )
        if self._prefill_cp_enabled():
            assert cp_params is not None
            return self.indexer_op._get_topk_ragged_cp(
                q_fp8,
                weights,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params.total_local_ids,
                cp_params.cu_kv_seqlens_global,
                cp_params.total_kv_len,
                cp_params.precomputed_ks,
                cp_params.precomputed_ke,
                cp_params.precomputed_lengths,
                cp_params.precomputed_topk_off,
                bool(getattr(cp_params, "kv_cache_sharded", False)),
                int(getattr(cp_params, "cp_size", 1)),
                int(getattr(cp_params, "cp_rank", 0)),
                kv_owner_tokens_per_block=self._kv_owner_tokens_per_block,
                indexer_cp_plan=getattr(cp_params, "indexer_cp_plan", None),
                indexer_cp_local_cu=getattr(cp_params, "indexer_cp_local_cu", None),
                indexer_copy_dst_idx=getattr(cp_params, "indexer_copy_dst_idx", None),
                indexer_src_for_padded=getattr(
                    cp_params, "indexer_src_for_padded", None
                ),
                total_local_ids_is_identity=bool(
                    getattr(cp_params, "total_local_ids_is_identity", False)
                ),
            )
        return self.indexer_op._get_topk_ragged(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs
        )

    # ------------------------------------------------------------------
    # KPool 路径
    # ------------------------------------------------------------------
    def _kpool_context_lengths(self, fmha_params, attention_inputs, cp_params):
        """返回 (每请求上下文长度, 每查询行上下文长度, 每行属于哪个请求, 参与打分的 q 行)。

        最后一项在 context parallel 下是 ``cp_params.total_local_ids``：
        CP 的 q 是本卡的局部张量（可能带 padding 行），只有这些行是真 token，
        ``precomputed_ks/ke`` 也正好是按这些行预先取好的
        （``fmha_params.ks[total_global_ids]``，见 indexer_op.py:1124-1126）。
        非 CP 时返回 None，表示所有行都参与。
        """
        if _is_decode_phase(attention_inputs):
            per_req = fmha_params.kvlen_d.reshape(-1).to(torch.long)
            return (
                per_req,
                per_req,
                torch.arange(per_req.numel(), device=per_req.device, dtype=torch.long),
                None,
            )
        if cp_params is not None and self._is_sparse_prefill_cp(attention_inputs):
            cu = cp_params.cu_kv_seqlens_global.reshape(-1).to(torch.long)
            ks = cp_params.precomputed_ks.reshape(-1).to(torch.long)
            ke = cp_params.precomputed_ke.reshape(-1).to(torch.long)
            rows = cp_params.total_local_ids
        else:
            cu = attention_inputs.cu_kv_seqlens_device.reshape(-1).to(torch.long)
            ks = fmha_params.ks.reshape(-1).to(torch.long)
            ke = fmha_params.ke.reshape(-1).to(torch.long)
            rows = None
        per_req = cu[1:] - cu[:-1]
        # ks[row] 正好等于该请求在扁平 KV 缓冲里的起点，据此反查行属于哪个请求
        req = torch.searchsorted(cu[1:].contiguous(), ks.contiguous(), right=True)
        return per_req, (ke - ks), req, rows

    def _forward_kpool(
        self,
        hidden_states,
        q_lora,
        kv_cache,
        fmha_params,
        attention_inputs,
        cp_params,
        x_fp8,
        x_scale,
        q_c_fp8,
        q_c_scale,
        use_fast_path,
    ):
        assert kv_cache is not None, "KPool indexer requires a KV cache"
        # 1) 投影。池化要在**未旋转**的 key 上做（官方就是对 k_norm(wk(x)) 分组），
        #    q 仍旧旋转（本模型无 rope，apply_rope_* 只剩 Hadamard）。
        if self._wq_b_accepts_fp8 and q_c_fp8 is not None and q_c_scale is not None:
            q = self.wq_b(q_c_fp8, input_scales=q_c_scale)
        else:
            q = self.wq_b(q_lora)
        q = q.view(-1, self.index_n_heads, self.index_head_dim)
        if self._wk_accepts_fp8 and x_fp8 is not None and x_scale is not None:
            k = self.wk(x_fp8, input_scales=x_scale)
        else:
            k = self.wk(hidden_states)
        k = self.k_norm(k)
        gate = torch.nn.functional.linear(
            hidden_states.to(self.kpool_gate_weight.dtype), self.kpool_gate_weight
        )

        # 2) 写 cache：每 token 一条 (key, gate)。CP 下先把各卡的局部结果汇总还原成
        #    全序列，再按整表 slot_mapping 写——与 quant_q_k_cp 的做法一致。
        pair = torch.cat(
            (k.to(torch.bfloat16), gate.to(torch.bfloat16)), dim=-1
        ).contiguous()
        if cp_params is not None and self._is_sparse_prefill_cp(attention_inputs):
            gathered = all_gather(pair, group=Group.TP, role="indexer_kpool_pair")
            pair = gathered.reshape(-1, pair.size(-1))[
                cp_params.kv_restore_unpad_indices
            ]
            slot_mapping = (
                cp_params.sharded_slot_mapping
                if bool(getattr(cp_params, "kv_cache_sharded", False))
                else fmha_params.slot_mapping
            )
        else:
            slot_mapping = fmha_params.slot_mapping
        half = self.index_head_dim
        self.kpool.write(
            kv_cache.kv_cache_base, slot_mapping, pair[:, :half], pair[:, half:]
        )
        if use_fast_path:
            # 全选层只需要把 K 落盘，不出 top-k
            return None

        # 3) 组池：从 cache 里按请求取出凑满的池，算池 key，旋转后量化
        per_req_ctx, row_ctx, row_req, score_rows = self._kpool_context_lengths(
            fmha_params, attention_inputs, cp_params
        )
        if _is_decode_phase(attention_inputs) and int(q.shape[0]) != int(
            per_req_ctx.numel()
        ):
            # The decode branch of `_kpool_context_lengths` maps query row i to
            # request i. Multi-token decode (MTP target-verify, draft-extend)
            # gives a request several query rows and would silently mis-map
            # them, so refuse rather than score against the wrong context.
            raise NotImplementedError(
                "KPool decode assumes one query row per request, got "
                f"{int(q.shape[0])} rows for {int(per_req_ctx.numel())} requests; "
                "multi-token decode is not supported yet"
            )
        block_table = _physical_block_table(attention_inputs)
        pool_fp8, pool_scale, pool_start = self.kpool.build_pools(
            kv_cache.kv_cache_base, block_table, per_req_ctx, self.kpool_ape
        )

        # 4) 打分 + 选池 + 展开成请求内的 token 索引
        q_fp8, q_scale = self.indexer_op.quant_q_only(_rotate_activation(q))
        weights = self._get_logits_head_gate(hidden_states, q_scale)
        # 未融合路径返回 [T, H, 1]，融合 kernel 直接返回 [T, H]
        if weights.dim() == 3:
            weights = weights.squeeze(-1)
        weights = weights.float().contiguous()
        if score_rows is not None:
            # CP：只有 total_local_ids 指到的行是真 token，上面的 ks/ke 也是按这些行取的。
            # 返回的 top-k 行数要与 SparseMlaCpImpl 里的 q0 = q[total_local_ids] 对齐。
            q_fp8 = q_fp8[score_rows].contiguous()
            weights = weights[score_rows].contiguous()
        pool_ks = pool_start.to(torch.int32)[row_req]
        pool_ke = pool_ks + torch.div(
            row_ctx, self.compress_ratio, rounding_mode="floor"
        ).to(torch.int32)
        return self.kpool.select(
            q_fp8,
            weights,
            pool_fp8,
            pool_scale,
            pool_ks,
            pool_ke,
            row_ctx,
            self.indexer_op._run_prefill_topk,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        q_lora: torch.Tensor,
        kv_cache: LayerKVCache,
        fmha_params: Any,
        attention_inputs: Any,
        use_fast_path: bool,
        cp_params: Any = None,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
        q_c_fp8: Optional[torch.Tensor] = None,
        q_c_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.kpool is not None:
            return self._forward_kpool(
                hidden_states,
                q_lora,
                kv_cache,
                fmha_params,
                attention_inputs,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
                use_fast_path,
            )

        if use_fast_path:
            key = self._get_k_bf16(hidden_states, fmha_params)
            self.indexer_op.quant_k_only(key, kv_cache, fmha_params.slot_mapping)
            return None

        if self._is_sparse_prefill_cp(attention_inputs):
            assert cp_params is not None, "cp_params is required for sparse prefill CP"

            if (
                self._prefill_cp_fused_quant_enabled()
                and cp_params.full_rope_pos_ids is not None
                and hidden_states.dtype == torch.bfloat16
                and self.cos_sin_cache.dtype == torch.float32
                and self.cos_sin_cache.stride(-1) == 1
                and cp_params.full_rope_pos_ids.is_contiguous()
                and self.indexer_op._prefill_all_keys_fit(
                    fmha_params, cp_params.total_kv_len
                )
            ):
                # Every causal key is selected; Q and head weights cannot
                # affect the result. Retain K production for later decode.
                return self._forward_all_keys_prefill_cp(
                    hidden_states,
                    kv_cache,
                    fmha_params,
                    cp_params,
                    x_fp8,
                    x_scale,
                )

        # Fused Q-RoPE-Hadamard-Quant path: single Triton kernel does
        # RoPE + 128-pt Hadamard + ue8m0 FP8 quant for Q (decode only).
        if (
            self._is_sparse_prefill_cp(attention_inputs)
            and self._prefill_cp_fused_quant_enabled()
            and cp_params.full_rope_pos_ids is not None
        ):
            q_fp8, q_scale = self._fused_forward_prefill_cp(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        elif (
            self._fuse_logits_head_gate
            and self._fuse_kernels_enabled
            and _is_decode_phase(attention_inputs)
            and cp_params is None
        ):
            q_fp8, q_scale = self._fused_forward_decode(
                q_lora,
                hidden_states,
                kv_cache,
                fmha_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
        else:
            query, key = self._get_q_k_bf16(
                q_lora,
                hidden_states,
                fmha_params,
                cp_params,
                x_fp8,
                x_scale,
                q_c_fp8,
                q_c_scale,
            )
            q_fp8, q_scale = self._quantize_q_k(
                query, key, kv_cache, fmha_params, attention_inputs, cp_params
            )

        weights = self._get_logits_head_gate(hidden_states, q_scale)
        return self._compute_topk(
            q_fp8, weights, kv_cache, fmha_params, attention_inputs, cp_params
        )
