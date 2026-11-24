"""GLM-5.3 (glm5_next) model description.

Hybrid architecture ported from the SGLang reference implementation
(sglang-for-sail: python/sglang/srt/models/glm5_next.py):

  - KDA (Kimi Delta Attention) linear-attention layers, reusing the
    kimi_linear model-desc modules and the kimi_kda Triton kernels.
  - Full-attention layers are GLM-5 style sparse MLA (MLA + DSA indexer),
    reusing ``MlaAttention`` from glm5.py.
  - Dense FFN for the first ``first_k_dense_replace`` layers, MoE afterwards.
  - MHC (multi-hyper-connection) residual streams: the residual is ``hc_mult``
    parallel streams ([T, hc_mult * H]) mixed by learned per-layer
    ``hc_attn_*`` / ``hc_ffn_*`` parameters.  The math below follows the
    SGLang torch reference (``_mhc_pre_torch`` / ``_mhc_post_torch`` in
    sglang/kernels/ops/layernorm/mhc.py) exactly; fused kernels are a
    hardware-side follow-up.
"""

import logging
import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from rtp_llm.config.mega_moe import MEGA_MOE_STRATEGIES
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import is_ppu
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_layer_cache_for_group,
    get_layer_caches_for_groups,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
)
from rtp_llm.models_py.model_desc.glm5 import Glm5MoeLayer
from rtp_llm.models_py.model_desc.glm5_mhc_ppu import (
    fused_hc_supported,
    hc_post_fused,
    hc_pre_norm_fused,
)
from rtp_llm.models_py.model_desc.kimi_linear import (
    KimiLinearKDA,
    KimiLinearMetadata,
    build_cp_linear_attn_metadata,
)
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    FMHAImplBase,
    MlaAttention,
    RMSNorm,
)
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeLayer
from rtp_llm.models_py.modules.hybrid import indexer as indexer_mod
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    prepare_causal_conv1d_metadata,
)
from rtp_llm.ops import (
    HWKernelConfig,
    HybridAttentionType,
    MoeConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)

# Matches sglang glm5_next._MHC_POST_MULT_VALUE: the fused and reference
# boundaries must agree on it.
_MHC_POST_MULT_VALUE = 2.0


def hc_expand(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """[T, H] -> [T, hc_mult * H]: replicate the embedding into hc_mult streams."""
    return x.repeat(1, hc_mult)


def hc_contract(x: torch.Tensor, hc_mult: int) -> torch.Tensor:
    """[T, hc_mult * H] -> [T, H]: mean over the streams."""
    return x.unflatten(-1, (hc_mult, -1)).mean(dim=-2)


def hc_pre(
    streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """MHC sublayer input mixing (port of sglang's _mhc_pre_torch).

    Args:
        streams: residual streams, [T, hc_mult * H].
        hc_fn: [hc_mult * (hc_mult + 2), hc_mult * H] mixing projection.
        hc_scale: [3] sigmoid scales for (pre, post, comb).
        hc_base: [hc_mult * (hc_mult + 2)] mixing biases.
    Returns:
        (layer_input [T, H], h_res (comb) [T, hc_mult, hc_mult], h_post
        [T, hc_mult]); h_res/h_post feed the matching hc_post write-back.
    """
    num_tokens = streams.shape[0]
    n = hc_mult
    hidden = streams.shape[-1] // n
    residual = streams.view(num_tokens, n, hidden)

    x_flat = residual.reshape(num_tokens, n * hidden).float()
    rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + rms_eps)
    mixes = F.linear(x_flat, hc_fn) * rsqrt

    if mixes.is_cuda:
        from rtp_llm.models_py.triton_kernels.common.mhc_gating import mhc_gating

        pre, post, comb = mhc_gating(
            mixes,
            hc_base,
            hc_scale,
            n,
            hc_eps,
            sinkhorn_iters,
            _MHC_POST_MULT_VALUE,
        )
    else:
        pre, post, comb = _hc_gates_torch(
            mixes, hc_base, hc_scale, n, hc_eps, sinkhorn_iters
        )

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
    return layer_input.to(streams.dtype), comb, post


def _hc_gates_torch(
    mixes: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    n: int,
    hc_eps: float,
    sinkhorn_iters: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference definition of the MHC gates; the fused kernel must match it.

    The Sinkhorn loop below is ~114 kernels for an [T, n, n] tensor with n = 4,
    which is why ``hc_pre`` prefers the fused kernel on device. This stays as
    the semantic reference and the CPU path.
    """
    num_tokens = mixes.shape[0]
    pre_raw = mixes[:, :n]
    post_raw = mixes[:, n : 2 * n]
    comb_raw = mixes[:, 2 * n :].view(num_tokens, n, n)
    pre_base = hc_base[:n]
    post_base = hc_base[n : 2 * n]
    comb_base = hc_base[2 * n :].view(n, n)

    pre = torch.sigmoid(pre_raw * hc_scale[0] + pre_base) + hc_eps
    post = _MHC_POST_MULT_VALUE * torch.sigmoid(post_raw * hc_scale[1] + post_base)
    comb = comb_raw * hc_scale[2] + comb_base

    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


def hc_post(
    x: torch.Tensor,
    streams: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    hc_mult: int,
) -> torch.Tensor:
    """MHC sublayer write-back (port of sglang's _mhc_post_torch + hc_post).

    out[to] = h_post[to] * x + sum_from h_res[from, to] * streams[from].
    """
    num_tokens = x.shape[0]
    hidden = streams.shape[-1] // hc_mult
    residual = streams.view(num_tokens, hc_mult, hidden)
    post = h_post.view(num_tokens, hc_mult, 1)
    comb = h_res.view(num_tokens, hc_mult, hc_mult)
    out = post * x.unsqueeze(1) + (comb.unsqueeze(-1) * residual.unsqueeze(2)).sum(dim=1)
    return out.type_as(x).view(num_tokens, -1)


class Glm5NextDecoderLayer(nn.Module):
    """One GLM-5.3 decoder layer: KDA or sparse-MLA attention, Dense/MoE FFN,
    with MHC hyper-connection residual streams instead of a plain residual."""

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        global_weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config: MoeConfig,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]
        self.hc_mult = max(int(config.hc_mult), 1)
        self.use_mhc = self.hc_mult > 1
        self.hc_eps = float(config.hc_eps)
        self.sinkhorn_iters = max(int(config.hc_sinkhorn_iters), 1)
        self.rms_eps = config.layernorm_eps

        quant_config = config.quant_config
        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = KimiLinearKDA(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
                gate_lower_bound=getattr(config, "linear_gate_lower_bound", None),
            )
        else:
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config,
                global_weights=global_weights,
                # Every GLM-5.3 full-attention layer owns a DSA indexer.
                has_indexer=True,
                stable_topk=True,
                # Match GLM's SGL indexer: UE8M0 Q, ordinary FP32 K scales.
                indexer_query_scale_fmt="ue8m0" if is_ppu() else None,
                # Indexer LayerNorm has its own epsilon; it must not inherit
                # the residual/MLA RMSNorm epsilon from the model config.
                indexer_layernorm_eps=1e-6,
            )

        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
                # First `first_k_dense_replace` layers are plain MLPs and clamp
                # before SwiGLU just like the experts do.
                swiglu_limit=float(getattr(config, "swiglu_limit", 0.0) or 0.0),
            )
        elif moe_config.moe_strategy in MEGA_MOE_STRATEGIES:
            # The pre-SwiGLU clamp is plumbed through the DenseMLP and the PPU
            # DeepGEMM executors only. Whether MegaMoE picks `swiglu_limit` up
            # for this model has not been verified, and dropping the clamp is a
            # silent accuracy loss, so refuse the combination instead.
            if float(getattr(config, "swiglu_limit", 0.0) or 0.0) > 0:
                raise NotImplementedError(
                    "glm5_next needs a pre-SwiGLU clamp "
                    f"(swiglu_limit={config.swiglu_limit}); the MegaMoE strategy "
                    f"{moe_config.moe_strategy} has not been verified to apply it"
                )
            self.mlp = Glm5MoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
                layer_idx=layer_idx,
            )
        else:
            self.mlp = GenericMoeLayer(
                config,
                parallelism_config,
                weights,
                moe_config,
                max_generate_batch_size,
                enable_cuda_graph=enable_cuda_graph,
                hw_kernel_config=hw_kernel_config,
            )

        # MHC owns the residual streams, so the sublayer norms are plain
        # RMSNorms (no fused residual add).
        self.input_layernorm = RMSNorm(weights[W.pre_ln_gamma], eps=config.layernorm_eps)
        self.post_attention_layernorm = RMSNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

        self.fused_hc = False
        if self.use_mhc:
            self.hc_attn_fn = weights[W.hc_attn_fn]
            self.hc_attn_scale = weights[W.hc_attn_scale]
            self.hc_attn_base = weights[W.hc_attn_base]
            self.hc_ffn_fn = weights[W.hc_ffn_fn]
            self.hc_ffn_scale = weights[W.hc_ffn_scale]
            self.hc_ffn_base = weights[W.hc_ffn_base]
            # Both sublayers share one decision: the two mixing projections
            # have the same shape and the two RMSNorm weights the same width.
            # RTP_GLM_FUSED_MHC=0 keeps the unfused torch path, i.e. the
            # behaviour before the fused kernels, for baselines and A/B runs.
            self.fused_hc = os.environ.get(
                "RTP_GLM_FUSED_MHC", "1"
            ) != "0" and fused_hc_supported(
                self.hc_mult,
                config.hidden_size,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm.weight.data,
                self.rms_eps,
                self.hc_eps,
                self.sinkhorn_iters,
                _MHC_POST_MULT_VALUE,
                self.input_layernorm.variance_epsilon,
            ) and fused_hc_supported(
                self.hc_mult,
                config.hidden_size,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm.weight.data,
                self.rms_eps,
                self.hc_eps,
                self.sinkhorn_iters,
                _MHC_POST_MULT_VALUE,
                self.post_attention_layernorm.variance_epsilon,
            )

    def _hc_pre_norm(
        self,
        streams: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm: RMSNorm,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MHC pre plus the sublayer RMSNorm; returns (normed, comb, post)."""
        if self.fused_hc:
            return hc_pre_norm_fused(
                streams,
                hc_fn,
                hc_scale,
                hc_base,
                self.hc_mult,
                self.rms_eps,
                self.hc_eps,
                self.sinkhorn_iters,
                _MHC_POST_MULT_VALUE,
                norm.weight.data,
                norm.variance_epsilon,
            )
        layer_input, h_res, h_post = hc_pre(
            streams,
            hc_fn,
            hc_scale,
            hc_base,
            self.hc_mult,
            self.rms_eps,
            self.hc_eps,
            self.sinkhorn_iters,
        )
        return norm(layer_input), h_res, h_post

    def _hc_post(
        self,
        x: torch.Tensor,
        streams: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ) -> torch.Tensor:
        if self.fused_hc:
            return hc_post_fused(x, streams, h_post, h_res, self.hc_mult)
        return hc_post(x, streams, h_post, h_res, self.hc_mult)

    def forward(
        self,
        streams: torch.Tensor,
        fmha_impl: Optional[FMHAImplBase] = None,
        kv_cache: Optional[Any] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: Optional[KimiLinearMetadata] = None,
    ) -> torch.Tensor:
        """streams: [T, hc_mult * H] in, [T, hc_mult * H] out."""
        if self.use_mhc:
            normed, h_res, h_post = self._hc_pre_norm(
                streams,
                self.hc_attn_fn,
                self.hc_attn_scale,
                self.hc_attn_base,
                self.input_layernorm,
            )
        else:
            # Plain-residual fallback (not used by the GLM-5.3 main model;
            # kept for structural completeness).
            normed = self.input_layernorm(streams)

        if self.layer_type == HybridAttentionType.LINEAR:
            attn_out = self.self_attn(
                hidden_states=normed,
                fmha_impl=None,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            attn_out = self.self_attn(
                hidden_states=normed,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )

        if self.use_mhc:
            streams = self._hc_post(attn_out, streams, h_post, h_res)
            normed, h_res, h_post = self._hc_pre_norm(
                streams,
                self.hc_ffn_fn,
                self.hc_ffn_scale,
                self.hc_ffn_base,
                self.post_attention_layernorm,
            )
        else:
            streams = streams + attn_out
            normed = self.post_attention_layernorm(streams)

        ffn_out = self.mlp(normed)

        if self.use_mhc:
            streams = self._hc_post(ffn_out, streams, h_post, h_res)
        else:
            streams = streams + ffn_out
        return streams


class Glm5NextModel(GptModelBase):
    """GLM-5.3 text model (hybrid KDA + sparse MLA, MoE, MHC residuals)."""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
        )
        self.moe_config = moe_config
        self.max_generate_batch_size = max_generate_batch_size
        self.hc_mult = max(int(model_config.hc_mult), 1)
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph=enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )
        # 池 key 是不是单独一个缓存组，由 models/glm5_next.py 的同一个开关决定。
        # 声明成缓存组之后它才会随 KV cache 跨机传输——PD 分离下这是正确性前提，
        # 模块私有的旁路缓冲传不过去。两处必须读同一个环境变量，否则这里会去取
        # 一个没被声明的组、直接抛「available tags=[...]」。
        self._full_attention_cache_tags = (
            ("default", "indexer_kv", "kpool")
            if os.environ.get("RTP_GLM_KPOOL_CACHE", "0") == "1"
            else ("default", "indexer_kv")
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # Layer-boundary values are MHC streams; the canonical hidden state is
        # the contracted (mean) stream.
        if self.hc_mult > 1:
            return hc_contract(hidden_states, self.hc_mult)
        return hidden_states

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return self.norm(hidden_states)

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        full_attention_layers = (
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.layer_type != HybridAttentionType.LINEAR
        )
        tags = get_group_tags_for_layers(self.kv_cache, full_attention_layers)
        # 池缓存组只是一块随 KV 一起传输的存储，没有对应的注意力实现。
        # 不剔掉的话上层会给它建一个 FMHA route，而稀疏 MLA 那边按
        # {default, indexer_kv} 严格比对 route 集合，多一个就直接抛。
        if tags is not None and "kpool" in tags:
            tags = [t for t in tags if t != "kpool"]
        return tags

    def _linear_attention_inputs(
        self, inputs: PyModelInputs, fallback: PyAttentionInputs
    ) -> PyAttentionInputs:
        """Attention inputs of the first KDA layer, which carry the CP info.

        All 34 KDA layers share one cache group ("linear") and therefore one
        set of context-parallel offsets, so the first one is representative.
        """
        if self.kv_cache is None:
            return fallback
        linear_layer_idx = next(
            (
                layer_idx
                for layer_idx, layer in enumerate(self.layers[: self.layer_num])
                if layer.layer_type == HybridAttentionType.LINEAR
            ),
            None,
        )
        if linear_layer_idx is None:
            return fallback
        selected = select_attention_inputs_for_layer(
            inputs, self.kv_cache, linear_layer_idx
        )
        if isinstance(selected, list):
            raise RuntimeError(
                "GLM-5.3-Flash LINEAR layer must map to exactly one attention-input tag"
            )
        return selected

    def prepare_fmha_impl(
        self,
        inputs: PyModelInputs,
        is_cuda_graph: bool = False,
        cuda_graph_selection_mode: Optional[str] = None,
    ) -> Any:
        fmha_impl = super().prepare_fmha_impl(
            inputs, is_cuda_graph, cuda_graph_selection_mode
        )
        if self.kv_cache is None and not isinstance(fmha_impl, dict):
            # Cacheless warmup shares one input and skips the indexer.
            return {"default": fmha_impl, "indexer_kv": fmha_impl}
        return fmha_impl

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # 新的一次前向：各 DSA 层 indexer 共用的 decode 下标换算从这里开始重算
        indexer_mod.begin_forward()
        input_ids: torch.Tensor = inputs.input_ids
        hidden_states = self.embed_tokens(input_ids)

        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        is_target_verify = bool(getattr(attention_inputs, "is_target_verify", False))
        prefill_conv1d_meta = None
        full_prefill_conv1d_meta = None
        full_prefill_cu_seqlens = None
        cp_restore_indices = None
        cp_unpad_restore_indices = None
        if attention_inputs.is_prefill and not is_target_verify:
            # KDA recurs along the sequence, so under context parallel every
            # rank has to see the full sequence; build the gather/restore
            # metadata once per forward and share it across the 34 KDA layers.
            if self.parallelism_config.prefill_cp_config.is_enabled():
                linear_attention_inputs = self._linear_attention_inputs(
                    inputs, attention_inputs
                )
                (
                    full_prefill_conv1d_meta,
                    full_prefill_cu_seqlens,
                    cp_restore_indices,
                    cp_unpad_restore_indices,
                ) = build_cp_linear_attn_metadata(
                    linear_attention_inputs,
                    self.parallelism_config,
                    hidden_states.device,
                )
            if cp_restore_indices is None:
                prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                    query_start_loc=attention_inputs.cu_seqlens_device,
                    device=hidden_states.device,
                )
        attn_meta = KimiLinearMetadata(
            prefill_conv1d_meta,
            is_target_verify,
            full_prefill_conv1d_meta=full_prefill_conv1d_meta,
            full_prefill_cu_seqlens=full_prefill_cu_seqlens,
            cp_restore_indices=cp_restore_indices,
            cp_unpad_restore_indices=cp_unpad_restore_indices,
        )

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        capture = self.capture_context(inputs.capture_hidden_states)
        streams = (
            hc_expand(hidden_states, self.hc_mult)
            if self.hc_mult > 1
            else hidden_states
        )

        for idx, layer in enumerate(self.layers[: self.layer_num]):
            if layer.layer_type == HybridAttentionType.LINEAR:
                layer_fmha_impl = None
                layer_kv_cache = (
                    get_layer_cache_for_group(self.kv_cache, idx, "linear")
                    if self.kv_cache is not None
                    else None
                )
                layer_attention_inputs = select_attention_inputs_for_layer(
                    inputs, self.kv_cache, idx
                )
            else:
                layer_fmha_impl = fmha_impl
                layer_kv_cache = (
                    get_layer_caches_for_groups(
                        self.kv_cache, idx, self._full_attention_cache_tags
                    )
                    if self.kv_cache is not None
                    else None
                )
                layer_attention_inputs = None
            streams = layer(
                streams,
                fmha_impl=layer_fmha_impl,
                kv_cache=layer_kv_cache,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )
            capture.capture_layer(idx, streams, None)

        hidden_states = (
            hc_contract(streams, self.hc_mult) if self.hc_mult > 1 else streams
        )
        return capture.finalize(hidden_states, None)


__all__ = [
    "Glm5NextModel",
    "Glm5NextDecoderLayer",
    "hc_expand",
    "hc_contract",
    "hc_pre",
    "hc_post",
]
