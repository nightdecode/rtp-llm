"""Kimi Linear model description.

Hybrid architecture:
  - KDA (Kimi Delta Attention) linear attention layers
  - MLA (Multi-head Latent Attention) full attention layers
  - MoE FFN with sigmoid routing (layer 1+), Dense FFN (layer 0)
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.distributed.collective_torch import (
    Group,
    all_gather,
    all_reduce,
    all_to_all,
)
from rtp_llm.models_py.model_desc.block_map import (
    get_group_tags_for_layers,
    get_primary_attention_inputs,
    select_attention_inputs_for_layer,
    select_fmha_impl_for_layer,
)
from rtp_llm.models_py.model_desc.generic_moe import DecodeLayerOutput, GenericMoeLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.model_desc.projection_packing import (
    MAX_PACKED_PROJECTION_WIDTH,
    packed_projection_weight,
)
from rtp_llm.models_py.modules import (
    DenseMLP,
    Embedding,
    FMHAImplBase,
    LinearFactory,
    MlaAttention,
    RMSResNorm,
)
from rtp_llm.models_py.triton_kernels.causal_conv1d import (
    CausalConv1dMetadata,
    causal_conv1d_fn,
    causal_conv1d_update,
    prepare_causal_conv1d_metadata,
)
from rtp_llm.models_py.triton_kernels.common.layernorm_gated import RmsNormGated
from rtp_llm.models_py.triton_kernels.fla.block import (
    load_initial_state_from_block_map,
    store_ssm_state_to_block_map,
)
# 一处读取，后面按 head/norm 的融合点共用，避免每次前向都查环境变量。
_KDA_FP32_STATE = os.environ.get("RTP_GLM_KDA_FP32_STATE", "0") != "0"

if _KDA_FP32_STATE:
    # FP32 recurrent-state variants of the same three entry points. The public
    # signatures are a strict superset of the default ones -- same positional
    # arguments, extra keyword options that all default to the old behaviour --
    # so the call sites below are unchanged and the switch is a pure A/B.
    from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state import (
        chunk_kda,
        fused_kda_gate,
        fused_recurrent_kda,
    )

    logging.info("KDA: using the FP32-state kernels (RTP_GLM_KDA_FP32_STATE=1)")

else:
    from rtp_llm.models_py.triton_kernels.kimi_kda import (
        chunk_kda,
        fused_kda_gate,
        fused_recurrent_kda,
    )
from rtp_llm.models_py.triton_kernels.fla.block import kda_pool_v_first
# Layout mismatch between the two roles is silent and produces wrong
# numbers rather than an error, so state it at startup on both sides.
logging.info(
    "KDA: SSM state pool layout is %s (RTP_GLM_KDA_STATE_V_FIRST)",
    "[HV, V, K]" if os.environ.get("RTP_GLM_KDA_STATE_V_FIRST", "0") == "1"
    else "[HV, K, V]",
)
if os.environ.get("RTP_GLM_KDA_PLA", "0") == "1":
    logging.info(
        "KDA: vendor decode kernel on for per-replica batch >= %s "
        "(RTP_GLM_KDA_PLA_MIN_BATCH); below that the Triton recurrence "
        "is faster end to end",
        os.environ.get("RTP_GLM_KDA_PLA_MIN_BATCH", "8"),
    )
if os.environ.get("RTP_GLM_KDA_PLA_PREFILL", "0") == "1":
    logging.info(
        "KDA: vendor prefill kernel requested (RTP_GLM_KDA_PLA_PREFILL); "
        "each layer logs 'PLA KDA prefill off' if it falls back to Triton"
    )
from rtp_llm.models_py.modules.kda_pla import (
    flashkda_prefill,
    fused_recurrent_kda_pla,
    pla_kda_min_batch,
    pla_kda_prefill_supported,
    pla_kda_supported,
    resolve_state_slots,
)
from rtp_llm.models_py.utils.typed_storage_view import LinearCacheConverter
from rtp_llm.ops import (
    AttentionConfigs,
    HWKernelConfig,
    HybridAttentionType,
    LinearAttentionConfig,
    ParallelismConfig,
)
from rtp_llm.ops.compute_ops import (
    LayerKVCache,
    PyAttentionInputs,
    PyModelInputs,
    PyModelOutputs,
)
from rtp_llm.utils.model_weight import W
from rtp_llm.utils.util import to_torch_dtype


class KimiLinearMetadata(object):
    def __init__(
        self,
        prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        is_target_verify: bool = False,
        full_prefill_conv1d_meta: Optional[CausalConv1dMetadata] = None,
        full_prefill_cu_seqlens: Optional[torch.Tensor] = None,
        cp_restore_indices: Optional[torch.Tensor] = None,
        cp_unpad_restore_indices: Optional[torch.Tensor] = None,
    ):
        self.prefill_conv1d_meta = prefill_conv1d_meta
        self.is_target_verify = is_target_verify
        # Context-parallel prefill metadata. KDA recurs along the sequence, so
        # each rank must see the whole sequence; these describe how to restore
        # the all-gathered tokens to sequence order and how to take this rank's
        # zigzag slice back out. Mirrors Qwen3NextMetadata.
        self.full_prefill_conv1d_meta = full_prefill_conv1d_meta
        self.full_prefill_cu_seqlens = full_prefill_cu_seqlens
        self.cp_restore_indices = cp_restore_indices
        # Sequence position -> row of the rank-major gathered layout, padding
        # rows dropped: [total new tokens], int64.
        self.cp_unpad_restore_indices = cp_unpad_restore_indices
        self._cp_tail_blocks: Optional[torch.Tensor] = None
        self._decode_state_slots: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None

    def decode_state_slots(
        self, attn_inputs: PyAttentionInputs, seq_size_per_block: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Physical SSM state slots this decode step reads from and writes to.

        Only needed by the PLA path, which takes the slot as an argument
        instead of resolving it inside the kernel. All 34 KDA layers share one
        block table, so this is resolved once per forward -- doing it per layer
        would add a few hundred tiny launches per step and eat the gain.
        """
        if self._decode_state_slots is None:
            self._decode_state_slots = resolve_state_slots(
                attn_inputs.kv_cache_kernel_block_id_device,
                attn_inputs.sequence_lengths_plus_1_device,
                seq_size_per_block,
            )
        return self._decode_state_slots

    def get_prefill_conv1d_meta(self) -> Optional[CausalConv1dMetadata]:
        return self.prefill_conv1d_meta

    def cp_tail_blocks(
        self, attn_inputs: PyAttentionInputs, seq_size_per_block: int
    ) -> torch.Tensor:
        """Kernel block id holding each request's last new token, on device.

        Built once per forward from the host block table and shared by all
        KDA layers, which use the same linear-attention block table.
        """
        if self._cp_tail_blocks is None:
            cp_info = attn_inputs.context_parallel_info
            prefix_lengths = attn_inputs.prefix_lengths.tolist()
            new_lengths = cp_info.prefill_actual_input_lengths_cpu.tolist()
            block_table = attn_inputs.kv_cache_kernel_block_id
            blocks = torch.tensor(
                [
                    int(block_table[i, (prefix + length - 1) // seq_size_per_block])
                    for i, (prefix, length) in enumerate(
                        zip(prefix_lengths, new_lengths)
                    )
                    if length > 0
                ],
                dtype=torch.int64,
            )
            # Clamp rather than filter. The count has to stay a pure function
            # of the request lengths, which are replicated across ranks: it
            # sizes an all-gather, and a rank-dependent size is a collective
            # mismatch, not a wrong answer. Ids <= 0 are the null block (-1)
            # or the engine's reserved dummy block (0); redirecting them to
            # block 0 is what the cache writers already do.
            self._cp_tail_blocks = blocks.clamp_(min=0).to(
                attn_inputs.kv_cache_kernel_block_id_device.device
            )
        return self._cp_tail_blocks

    @property
    def is_cp_linear_attn(self) -> bool:
        return self.cp_restore_indices is not None


def build_cp_linear_attn_metadata(
    attention_inputs: PyAttentionInputs,
    parallelism_config: ParallelismConfig,
    device: torch.device,
) -> tuple[
    Optional[CausalConv1dMetadata],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    """Precompute the per-layer all-gather metadata for CP linear attention.

    Returns (full_conv1d_meta, full_cu_seqlens, restore_indices,
             unpad_restore_indices).

    Originally ported from Qwen3NextModel._build_cp_linear_attn_metadata, which
    all-gathers the projections and slices this rank's zigzag tokens back out
    of the full-sequence result. The head-split recurrence does not need that
    slice -- the second all-to-all already returns this rank's tokens -- so the
    inverse-permutation scatter that produced it is gone. ``restore_indices``
    stays as the flag that says CP linear attention is active.
    """
    cp_info = attention_inputs.context_parallel_info
    if cp_info is None:
        return None, None, None, None

    full_new_lengths = cp_info.prefill_actual_input_lengths_cpu
    full_cu = torch.zeros(
        full_new_lengths.shape[0] + 1, dtype=torch.int32, device=device
    )
    full_cu[1:] = full_new_lengths.cumsum(0).to(device)
    full_conv1d_meta = prepare_causal_conv1d_metadata(
        query_start_loc=full_cu, device=device
    )

    restore_indices = cp_info.prefill_qkv_restore_indice
    padding_mask = cp_info.prefill_qkv_padding_mask
    unpad_restore = restore_indices[padding_mask == 1]

    return (
        full_conv1d_meta,
        full_cu,
        restore_indices,
        unpad_restore.to(device=device, dtype=torch.long),
    )


class KimiLinearKDABase(nn.Module):
    """Base class for KDA (Kimi Delta Attention) prefill/decode.

    KDA differs from GDN in:
      - qkv_size = q + k + v (no z component, since output gate is separate LoRA)
      - beta is [B, H] scalar per head (from b_proj), not packed with a
      - forget gate g is [B, H, D] per-dim vector (from LoRA + fused_kda_gate)
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config
        self.weights = weights
        # Optional decay lower bound (GLM-5.3 KDA uses -5.0; None keeps the
        # default -exp(A_log) * softplus(g + dt_bias) gate).
        self.gate_lower_bound = gate_lower_bound
        # params
        self.head_k_dim: int = linear_attn_config.linear_key_head_dim
        self.head_v_dim: int = linear_attn_config.linear_value_head_dim
        assert (
            self.head_k_dim == self.head_v_dim
        ), "head_k_dim and head_v_dim must be the same now"
        # Shard by the attention TP view, not the raw TP size: under prefill CP
        # the TP group is repurposed as the CP group, get_attn_tp_size() is 1 and
        # the loader (LoadConfig.tp_size, model_weight_info.py) keeps attention
        # weights unsharded on every rank. Without CP the two are equal.
        attn_tp_size = parallelism_config.get_attn_tp_size()
        self.local_num_k_heads: int = (
            linear_attn_config.linear_num_key_heads // attn_tp_size
        )
        self.local_num_v_heads: int = (
            linear_attn_config.linear_num_value_heads // attn_tp_size
        )
        self.linear_conv_kernel_dim: int = (
            self.linear_attn_config.linear_conv_kernel_dim
        )
        self.ssm_state_size: int = (
            self.local_num_v_heads * self.head_k_dim * self.head_v_dim
        )
        # KDA: qkv only (no z), since output gate is separate LoRA
        self.qkv_size: int = (
            self.head_k_dim * self.local_num_k_heads
            + self.head_k_dim * self.local_num_k_heads
            + self.head_v_dim * self.local_num_v_heads
        )
        self.conv_state_size: int = (self.linear_conv_kernel_dim - 1) * self.qkv_size
        self.ssm_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.ssm_state_dtype
        )
        self.conv_state_dtype: torch.dtype = to_torch_dtype(
            linear_attn_config.conv_state_dtype
        )
        self.linear_cache_converter = LinearCacheConverter(
            local_num_v_heads=self.local_num_v_heads,
            head_v_dim=self.head_v_dim,
            head_k_dim=self.head_k_dim,
            ssm_state_dtype=self.ssm_state_dtype,
            linear_conv_kernel_dim=self.linear_conv_kernel_dim,
            qkv_size=self.qkv_size,
            conv_state_dtype=self.conv_state_dtype,
        )
        # weights
        self.conv_weights = weights[W.linear_attn_conv1d_w].squeeze(1)
        self.dt_bias = weights[W.linear_attn_dt_b_kda]
        self.alog = weights[W.linear_attn_alog]
        # Tri-state: None until the vendor-kernel probe has run for this layer.
        self._pla_ok: Optional[bool] = None
        self._pla_prefill_ok: Optional[bool] = None
        # The vendor decode kernel reads A_log and dt_bias in the activation
        # dtype (BF16). Cast once here instead of on every decode step: inside
        # the captured graph each `.to()` is a kernel launch, 2 x 34 layers per
        # step (timeline 2026-11-22: 68 bfloat16_copy launches, 0.18 ms/step).
        self._alog_bf16 = self.alog.to(torch.bfloat16)
        self._dt_bias_bf16 = self.dt_bias.to(torch.bfloat16)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        raise NotImplementedError

    def _get_conv_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        conv_states = self.linear_cache_converter.get_conv_state_tensor(kv_cache_tensor)
        return conv_states

    def _get_ssm_states(self, kv_cache_tensor: torch.Tensor) -> torch.Tensor:
        ssm_states = self.linear_cache_converter.get_ssm_state_tensor(kv_cache_tensor)
        return ssm_states


class KimiLinearKDAPrefill(KimiLinearKDABase):
    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__(
            linear_attn_config,
            parallelism_config,
            weights,
            gate_lower_bound=gate_lower_bound,
        )

    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        metadata: Optional[CausalConv1dMetadata] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        channels: Optional[slice] = None,
    ) -> torch.Tensor:
        # Under context parallel the caller passes the full-sequence offsets;
        # attn_inputs still describes this rank's local slice. `channels`
        # restricts the conv to a contiguous range of qkv channels (weights
        # and cached conv state alike); conv1d is depthwise, so channels are
        # independent.
        cu_seqlen_without_padding = (
            attn_inputs.cu_seqlens_device if cu_seqlens is None else cu_seqlens
        )
        channels = slice(None) if channels is None else channels
        conv_states = (
            self._get_conv_states(kv_cache_tensor).transpose(1, 2)[:, channels]
            if kv_cache_tensor is not None
            else None
        )
        out = causal_conv1d_fn(
            x=mixed_qkv.transpose(0, 1),
            weight=self.conv_weights[channels],
            bias=None,
            conv_states=conv_states,
            query_start_loc=cu_seqlen_without_padding,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            prefix_lengths=attn_inputs.prefix_lengths_device,
            metadata=metadata,
        ).transpose(0, 1)
        return out

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_k_heads * self.head_k_dim,
                self.local_num_v_heads * self.head_v_dim,
            ],
            dim=-1,
        )
        return self._recur(
            query,
            key,
            value,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            cu_seqlens=cu_seqlens,
        )

    def _recur(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: Optional[torch.Tensor],
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        cu_seqlens: Optional[torch.Tensor] = None,
        heads: Optional[slice] = None,
    ) -> torch.Tensor:
        # `heads` restricts the recurrence to a contiguous range of heads: the
        # gate parameters and the cached SSM state are sliced to match, and
        # query/key/value/forget_gate/beta must already hold only those heads.
        if heads is None:
            num_k_heads, num_heads = self.local_num_k_heads, self.local_num_v_heads
            a_log, dt_bias, ssm_heads = self.alog, self.dt_bias, slice(None)
        else:
            assert self.local_num_k_heads == self.local_num_v_heads
            num_heads = num_k_heads = heads.stop - heads.start
            # A_log is [H] and dt_bias is [H * head_k_dim]; both stay 1-D.
            a_log = self.alog[heads]
            dt_bias = self.dt_bias[
                heads.start * self.head_k_dim : heads.stop * self.head_k_dim
            ]
            ssm_heads = heads
        # Compute gate: [T, H*D] -> [1, T, H, D]
        g = forget_gate.view(
            1, forget_gate.shape[0], num_heads, self.head_k_dim
        ).contiguous()
        # beta: [token, H] -> apply sigmoid in float32 -> [1, token, H]
        beta_reshaped = beta.float().sigmoid().unsqueeze(0)

        ssm_states = (
            self._get_ssm_states(kv_cache_tensor)[:, ssm_heads]
            if kv_cache_tensor is not None
            else None
        )
        context_batch_size = attn_inputs.input_lengths.shape[0]
        # Under context parallel this is the full sequence's offsets; the
        # request count is the same on every rank, only the token span differs.
        cu_seqlens_without_padding = (
            attn_inputs.cu_seqlens_device if cu_seqlens is None else cu_seqlens
        )
        initial_states: Optional[torch.Tensor] = None
        if ssm_states is not None:
            initial_states = torch.empty(
                context_batch_size,
                num_heads,
                self.head_k_dim,
                self.head_v_dim,
                device=query.device,
                dtype=self.ssm_state_dtype,
            )
            load_initial_state_from_block_map(
                attn_inputs.prefix_lengths_device,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                initial_states,
                seq_size_per_block,
            )

        # Vendor prefill kernel (pla.prefill.flashkdapro, sglang-for-sail's PPU path).
        # It takes the RAW beta and emits only the final state, which is all
        # the block-map store writes when the LINEAR group keeps its tail block
        # only (see kda_pla.py); flashkda_prefill audits that and returns None
        # to send this call down the Triton path otherwise.
        if self._pla_prefill_ok is None:
            self._pla_prefill_ok = pla_kda_prefill_supported(
                num_heads, self.head_k_dim, query.dtype, ssm_states, self.gate_lower_bound
            )
        if self._pla_prefill_ok and num_k_heads == num_heads:
            pla = flashkda_prefill(
                query.view(-1, num_heads, self.head_k_dim),
                key.view(-1, num_heads, self.head_k_dim),
                value.view(-1, num_heads, self.head_v_dim),
                forget_gate.view(-1, num_heads, self.head_k_dim),
                beta,
                a_log,
                dt_bias,
                initial_states,
                cu_seqlens_without_padding,
                self.gate_lower_bound,
                block_map=attn_inputs.kv_cache_kernel_block_id_device,
                prefix_lengths=attn_inputs.prefix_lengths_device,
                seq_size_per_block=seq_size_per_block,
            )
            if pla is not None:
                attn_out, final_state = pla
                if ssm_states is not None:
                    # One chunk per request (chunk_size far above any length):
                    # the store kernel then takes only its final-state branch,
                    # never the per-block-boundary one that would read `h`.
                    store_ssm_state_to_block_map(
                        final_state,
                        final_state,
                        attn_inputs.prefix_lengths_device,
                        cu_seqlens_without_padding,
                        attn_inputs.kv_cache_kernel_block_id_device,
                        ssm_states,
                        seq_size_per_block,
                        chunk_size=1 << 30,
                    )
                return attn_out

        query = query.view(1, query.shape[0], num_k_heads, self.head_k_dim).contiguous()
        key = key.view(1, key.shape[0], num_k_heads, self.head_k_dim).contiguous()
        value = value.view(1, value.shape[0], num_heads, self.head_v_dim).contiguous()

        # Prefill: use chunk_kda with fused gate (use_gate_in_kernel=True)
        attn_out, final_state, h = chunk_kda(
            query,
            key,
            value,
            g,
            beta_reshaped,
            initial_state=initial_states,
            output_final_state=True,
            cu_seqlens=cu_seqlens_without_padding,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            return_intermediate_states=True,
            A_log=a_log,
            dt_bias=dt_bias,
            lower_bound=self.gate_lower_bound,
        )
        h_from_chunk = h

        if ssm_states is not None and final_state is not None:
            h_for_store = (
                h_from_chunk
                if h_from_chunk is not None
                else final_state.unsqueeze(0).unsqueeze(0)
            )
            # store_ssm_state_to_block_map requires h and final_state in float32
            if h_for_store.dtype != torch.float32:
                h_for_store = h_for_store.float()
            if final_state.dtype != torch.float32:
                final_state = final_state.float()

            store_ssm_state_to_block_map(
                h_for_store,
                final_state,
                attn_inputs.prefix_lengths_device,
                cu_seqlens_without_padding,
                attn_inputs.kv_cache_kernel_block_id_device,
                ssm_states,
                seq_size_per_block,
                chunk_size=64,
            )

        return attn_out.squeeze_(0)

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block
        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            metadata=attn_meta.get_prefill_conv1d_meta(),
        )
        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
        )
        cache_store_inputs = attn_inputs.cache_store_inputs
        cache_store_writer = attn_inputs.cache_store_writer
        if (
            kv_cache is not None
            and cache_store_inputs is not None
            and cache_store_writer is not None
        ):
            cache_store_writer.write(cache_store_inputs, kv_cache)
        return attn_out

    def forward_cp(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        """Context-parallel prefill with the recurrence split by head.

        Context parallel hands each rank one head chunk and one tail chunk of
        the sequence (ZigzagProcessor), but KDA carries state along the
        sequence, so the recurrence needs every token of a head, in order. One
        all-to-all trades token ownership for head ownership: rank r receives
        heads [r*H/cp, (r+1)*H/cp) of every token, runs conv1d and chunk_kda
        over the full sequence for those heads only, and a second all-to-all
        hands every rank its own tokens back with all heads. Each rank does
        1/cp of the recurrence instead of all of it, and moves 1/cp of the
        bytes an all-gather of the projections would.

        Each rank writes only its own heads' SSM state and conv channels;
        _sync_cp_states then all-gathers each request's tail block, the one
        the PD transfer hands to decode.
        """
        cp_info = attn_inputs.context_parallel_info
        assert cp_info is not None, "CP prefill requires context_parallel_info"
        full_cu = attn_meta.full_prefill_cu_seqlens
        full_conv_meta = attn_meta.full_prefill_conv1d_meta
        unpad_restore = attn_meta.cp_unpad_restore_indices
        assert (
            full_cu is not None
            and full_conv_meta is not None
            and unpad_restore is not None
        )
        assert self.local_num_k_heads == self.local_num_v_heads
        # In CP mode the TP group is repurposed as the CP group.
        cp_size = self.parallelism_config.tp_size
        cp_rank = self.parallelism_config.tp_rank
        num_heads = self.local_num_v_heads
        assert num_heads % cp_size == 0, f"{num_heads} KDA heads, CP {cp_size}"
        heads_per_rank = num_heads // cp_size
        heads = slice(cp_rank * heads_per_rank, (cp_rank + 1) * heads_per_rank)
        # One head slice is this wide in every field: q, k, v and the gate all
        # carry head_k_dim == head_v_dim channels per head (asserted above).
        width = heads_per_rank * self.head_k_dim

        # 1. Token-major -> head-major. Send slot j carries this rank's tokens
        #    for head slice j, packed as [q_j | k_j | v_j | g_j | beta_j].
        n_local = mixed_qkv.shape[0]
        fields = list(mixed_qkv.split(num_heads * self.head_k_dim, dim=-1))
        fields.append(forget_gate)
        send = mixed_qkv.new_empty(cp_size, n_local, 4 * width + heads_per_rank)
        for i, field in enumerate(fields):
            send[:, :, i * width : (i + 1) * width].copy_(
                field.view(n_local, cp_size, width).transpose(0, 1)
            )
        send[:, :, 4 * width :].copy_(
            beta.view(n_local, cp_size, heads_per_rank).transpose(0, 1)
        )
        received = all_to_all(send, group=Group.TP)
        del send
        # Rows arrive rank-major, as from an all-gather; restore sequence order.
        received = received.view(cp_size * n_local, -1)
        bounds = [
            (0, width),
            (width, 2 * width),
            (2 * width, 3 * width),
            (3 * width, 4 * width),
            (4 * width, 4 * width + heads_per_rank),
        ]
        query, key, value, gate, beta_heads = [
            received[:, start:stop].index_select(0, unpad_restore)
            for start, stop in bounds
        ]
        del received

        kv_cache_tensor: Optional[torch.Tensor] = None
        seq_size_per_block = 1
        if kv_cache is not None:
            kv_cache_tensor = kv_cache.kv_cache_base.reshape(
                kv_cache.kv_cache_base.shape[0], -1
            )
            seq_size_per_block = kv_cache.seq_size_per_block

        # 2. conv1d per q/k/v segment: this rank's channels of each segment are
        #    contiguous, the three segments are not.
        segment = num_heads * self.head_k_dim
        convolved = []
        for i, part in enumerate((query, key, value)):
            convolved.append(
                self._conv1d(
                    part,
                    kv_cache_tensor,
                    seq_size_per_block,
                    attn_inputs,
                    metadata=full_conv_meta,
                    cu_seqlens=full_cu,
                    channels=slice(
                        i * segment + heads.start * self.head_k_dim,
                        i * segment + heads.stop * self.head_k_dim,
                    ),
                )
            )
        query, key, value = convolved
        attn_out = self._recur(
            query,
            key,
            value,
            gate,
            beta_heads,
            kv_cache_tensor,
            seq_size_per_block,
            attn_inputs,
            cu_seqlens=full_cu,
            heads=heads,
        )
        del query, key, value, gate, beta_heads

        # 3. Head-major -> token-major. Padding rows stay zero; they are never
        #    read downstream.
        rows = attn_out.new_zeros(cp_size * n_local, width)
        rows.index_copy_(0, unpad_restore, attn_out.reshape(-1, width))
        del attn_out
        returned = all_to_all(rows, group=Group.TP)
        del rows
        local_attn_out = (
            returned.view(cp_size, n_local, heads_per_rank, self.head_v_dim)
            .transpose(0, 1)
            .reshape(n_local, num_heads, self.head_v_dim)
        )

        if kv_cache_tensor is not None:
            self._sync_cp_states(
                kv_cache_tensor,
                attn_inputs,
                attn_meta,
                seq_size_per_block,
                cp_size,
                heads,
            )

        cache_store_inputs = attn_inputs.cache_store_inputs
        cache_store_writer = attn_inputs.cache_store_writer
        if (
            kv_cache is not None
            and cache_store_inputs is not None
            and cache_store_writer is not None
        ):
            cache_store_writer.write(cache_store_inputs, kv_cache)
        return local_attn_out

    def _sync_cp_states(
        self,
        kv_cache_tensor: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        attn_meta: KimiLinearMetadata,
        seq_size_per_block: int,
        cp_size: int,
        heads: slice,
    ) -> None:
        """All-gather the head slices of each request's tail block.

        Only the tail block needs every head, and it needs it on every rank.
        A linear-attention group publishes just its tail block to the cache
        store (CacheGroupType.h, active_tail_blocks = 1), and the decode side
        picks which prefill peer to read it from at runtime: with CP prefill
        it takes the "prefill worker has full KV cache on each rank" branch
        (DecodeRpcServer.cc, is_prefill_enabled), which maps a decode worker
        to a peer by worker index. Rather than depend on that mapping -- it
        varies with the decode worker count and the peer count -- make every
        rank's tail block identical, so any peer serves the same bytes.

        Earlier blocks keep only this rank's heads. That is all a CP prefill
        on this rank ever reads back: the next chunk's initial state and a
        prefix-cache hit both read the heads this rank wrote itself, and the
        head-to-rank map depends only on cp_rank, which is fixed per process.
        """
        blocks = attn_meta.cp_tail_blocks(attn_inputs, seq_size_per_block)
        # Empty only when the batch has no new tokens at all, which is the
        # same on every rank, so the early return stays collective-safe.
        if blocks.numel() == 0:
            return
        num_blocks = blocks.shape[0]
        num_heads = self.local_num_v_heads
        width = (heads.stop - heads.start) * self.head_k_dim

        ssm_states = self._get_ssm_states(kv_cache_tensor)
        gathered = all_gather(
            ssm_states[:, heads].index_select(0, blocks), group=Group.TP
        )
        ssm_states[blocks] = (
            gathered.view(cp_size, num_blocks, -1, self.head_v_dim, self.head_k_dim)
            .transpose(0, 1)
            .reshape(num_blocks, num_heads, self.head_v_dim, self.head_k_dim)
        )

        # Conv channels are [q | k | v], each H*D wide; this rank owns the
        # same head range inside each of the three segments.
        conv_states = self._get_conv_states(kv_cache_tensor)
        segment = num_heads * self.head_k_dim
        starts = [s * segment + heads.start * self.head_k_dim for s in range(3)]
        owned = torch.cat(
            [
                conv_states[:, :, start : start + width].index_select(0, blocks)
                for start in starts
            ],
            dim=-1,
        )
        gathered = all_gather(owned, group=Group.TP)
        conv_states[blocks] = (
            gathered.view(cp_size, num_blocks, -1, 3, width)
            .permute(1, 2, 3, 0, 4)
            .reshape(num_blocks, -1, 3 * segment)
        )


class KimiLinearKDADecode(KimiLinearKDABase):
    def _conv1d(
        self,
        mixed_qkv: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> torch.Tensor:
        conv_states = self._get_conv_states(kv_cache_tensor)
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )

        origin_shape = mixed_qkv.shape
        mixed_qkv = mixed_qkv.reshape(batch, seq, -1).transpose(1, 2)
        out = causal_conv1d_update(
            mixed_qkv,
            conv_states.transpose(1, 2),
            self.conv_weights,
            bias=None,
            activation="silu",
            cache_seqlens=None,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
        )
        out = out.transpose(1, 2).reshape(origin_shape)
        return out

    def _use_pla(
        self, ssm_states: torch.Tensor, attn_inputs: PyAttentionInputs
    ) -> bool:
        """Whether this layer's decode step goes through the vendor kernel.

        Probed once per layer and remembered: the probe logs its reason, and
        repeating it every step would spam the log.
        """
        if self._pla_ok is None:
            self._pla_ok = pla_kda_supported(
                self.local_num_v_heads,
                self.head_k_dim,
                ssm_states,
                attn_inputs.kv_cache_kernel_block_id_device,
                attn_inputs.sequence_lengths_plus_1_device,
            )
        return self._pla_ok

    def _fla(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        kv_cache_tensor: torch.Tensor,
        seq_size_per_block: int,
        attn_inputs: PyAttentionInputs,
        is_target_verify: bool,
        attn_meta: Optional[KimiLinearMetadata] = None,
    ) -> torch.Tensor:
        batch, seq = self._get_bs_from_attention_input(
            mixed_qkv, attn_inputs, is_target_verify
        )
        # Split qkv
        mixed_qkv = mixed_qkv.reshape(
            batch,
            seq,
            self.local_num_k_heads * 2 + self.local_num_v_heads,
            self.head_k_dim,
        )
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.local_num_k_heads,
                self.local_num_k_heads,
                self.local_num_v_heads,
            ],
            dim=2,
        )

        # Compute gate: reshape to [B, S, local_H, D] for fused_recurrent
        forget_gate_2d = forget_gate.reshape(batch * seq, -1)
        g = forget_gate_2d.view(
            batch, seq, self.local_num_v_heads, self.head_k_dim
        ).contiguous()
        ssm_states = self._get_ssm_states(kv_cache_tensor)

        # Each captured batch size gets its own graph, so a batch-dependent
        # branch resolves at capture time -- no host-side branch at replay.
        min_batch = pla_kda_min_batch()
        if (
            self._use_pla(ssm_states, attn_inputs)
            and attn_meta is not None
            and batch >= min_batch
        ):
            load_slot, store_slot = attn_meta.decode_state_slots(
                attn_inputs, seq_size_per_block
            )
            try:
                # PLA applies sigmoid(b) itself, so it takes the raw beta that
                # the Triton path receives already sigmoided.
                core_attn_out = fused_recurrent_kda_pla(
                    q=query.contiguous(),
                    k=key.contiguous(),
                    v=value.contiguous(),
                    g=g,
                    beta_raw=beta.reshape(
                        batch, seq, self.local_num_v_heads
                    ).contiguous(),
                    A_log=self._alog_bf16,
                    dt_bias=self._dt_bias_bf16,
                    initial_state=ssm_states,
                    load_slot=load_slot,
                    store_slot=store_slot,
                    lower_bound=self.gate_lower_bound,
                )
            except Exception:
                # The vendor kernel validates dtypes and shapes host-side and
                # raises. Crashing every rank over a contract mismatch is worse
                # than serving on the Triton path, so disable it for this layer
                # and fall through -- but say so loudly: a quiet fallback is
                # what made an earlier benchmark round unattributable.
                logging.exception(
                    "PLA KDA decode raised; falling back to the Triton kernel "
                    "for this layer for the rest of the process"
                )
                self._pla_ok = False
            else:
                return core_attn_out.reshape(
                    [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
                )

        # beta: [batch*seq, H] -> sigmoid in float32 -> [batch, seq, H]. Only the
        # Triton recurrence takes the activated beta (PLA applies sigmoid itself),
        # so it is computed here rather than ahead of the branch: on the PLA path
        # it was two dead launches per layer per step.
        beta_out = beta.reshape(batch * seq, -1).float().sigmoid()
        beta_out = beta_out.view(batch, seq, self.local_num_v_heads)
        core_attn_out, _ = fused_recurrent_kda(
            q=query.contiguous(),
            k=key.contiguous(),
            v=value.contiguous(),
            g=g,
            beta=beta_out,
            scale=None,
            initial_state=ssm_states,
            A_log=self.alog,
            dt_bias=self.dt_bias,
            inplace_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            lower_bound=self.gate_lower_bound,
            block_map=attn_inputs.kv_cache_kernel_block_id_device,
            seq_size_per_block=seq_size_per_block,
            sequence_lengths=attn_inputs.sequence_lengths_plus_1_device,
            # The state pool layout is shared with prefill and with the PLA
            # path; one switch drives all three. Only the FP32-state kernels
            # take this argument -- the stock ones are K-first only, which is
            # why the probe below refuses PLA without them.
            **({"state_v_first": True} if _KDA_FP32_STATE and kda_pool_v_first() else {}),
        )

        res = core_attn_out.reshape(
            [-1, core_attn_out.shape[2], core_attn_out.shape[3]]
        )
        return res

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        forget_gate: torch.Tensor,
        beta: torch.Tensor,
        attn_inputs: PyAttentionInputs,
        kv_cache: Optional[LayerKVCache],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert kv_cache is not None, "kv_cache is required for decode"
        assert (
            kv_cache.kv_cache_base is not None
        ), "kv_cache_tensor is required for decode"
        kv_cache_tensor: torch.Tensor = kv_cache.kv_cache_base.reshape(
            kv_cache.kv_cache_base.shape[0], -1
        )
        is_target_verify = attn_meta.is_target_verify

        mixed_qkv = self._conv1d(
            mixed_qkv,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
        )

        attn_out = self._fla(
            mixed_qkv,
            forget_gate,
            beta,
            kv_cache_tensor,
            kv_cache.seq_size_per_block,
            attn_inputs,
            is_target_verify,
            attn_meta,
        )
        return attn_out

    def _get_bs_from_attention_input(
        self,
        mixed_qkv: torch.Tensor,
        attention_inputs: PyAttentionInputs,
        is_target_verify: bool,
    ) -> tuple[int, int]:
        token, _ = mixed_qkv.shape
        if not is_target_verify:
            return token, 1
        assert (
            attention_inputs.prefix_lengths.size(0) > 0
        ), f"prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} <=0 when target verify"
        assert (
            token % attention_inputs.prefix_lengths.size(0) == 0
        ), f"token: {token} is not divisible by prefill_lengths size: {attention_inputs.prefix_lengths.size(0)} when target verify"
        b = attention_inputs.prefix_lengths.size(0)
        s = token // b
        return b, s


class KimiLinearKDA(nn.Module):
    """KDA (Kimi Delta Attention) module.

    Forward flow:
      1. qkv = in_proj_qkv(hidden)           # merged q+k+v projection
      2. beta_input = in_proj_b(hidden)       # beta gate input [token, H]
      3. forget_gate = f_b_proj(f_a_proj(hidden))  # LoRA forget gate [token, H*D]
      4. g_proj = g_b_proj(g_a_proj(hidden))  # LoRA output gate [token, H*D]
      5. conv1d on qkv
      6. FLA attention (chunk_kda for prefill, fused_recurrent_kda for decode)
         - fused_kda_gate applies: g = -exp(A_log) * softplus(forget_gate + dt_bias)
         - beta = sigmoid(beta_input)
      7. o_norm(attn_out, g_proj) with sigmoid activation
      8. o_proj(result) + all_reduce
    """

    def __init__(
        self,
        linear_attn_config: LinearAttentionConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layernorm_eps: float,
        quant_config: Optional[object] = None,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        gate_lower_bound: Optional[float] = None,
    ):
        super().__init__()
        self.linear_attn_config = linear_attn_config
        self.parallelism_config = parallelism_config

        # Projections
        self.in_proj_qkv = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_qkv_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.in_proj_b = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # LoRA forget gate
        self.f_a_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_f_a_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.f_b_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_f_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # LoRA output gate
        self.g_a_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_g_a_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        self.g_b_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_g_b_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )

        self.head_k_dim = linear_attn_config.linear_key_head_dim
        self.head_v_dim = linear_attn_config.linear_value_head_dim
        # Same attention-TP view as KimiLinearKDAPrefill above.
        self.local_num_v_heads = (
            linear_attn_config.linear_num_value_heads
            // parallelism_config.get_attn_tp_size()
        )

        self.prefill_kda = KimiLinearKDAPrefill(
            linear_attn_config,
            parallelism_config,
            weights,
            gate_lower_bound=gate_lower_bound,
        )
        self.decode_kda = KimiLinearKDADecode(
            linear_attn_config,
            parallelism_config,
            weights,
            gate_lower_bound=gate_lower_bound,
        )
        # o_norm with sigmoid activation (not SwiGLU)
        self.norm = RmsNormGated(
            weights[W.linear_attn_norm_w],
            eps=layernorm_eps,
            group_size=linear_attn_config.linear_value_head_dim,
            activation="sigmoid",
        )
        self.out_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.linear_attn_out_w,
            None,
            None,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
        )
        # One GEMM for the three narrow projections that read hidden_states:
        # beta [H] and the two LoRA down-projections. At decode each is a
        # launch-bound GEMV, so merging saves two launches per layer. The QKV
        # projection stays separate: at 3*H*D columns it is the widest weight
        # in the layer and reading it already dominates its own GEMV, while
        # packing it would duplicate that weight in memory.
        self.packed_low_weight: Optional[torch.Tensor] = None
        self.packed_low_widths: Optional[List[int]] = None
        # RTP_GLM_KDA_PACK_PROJ=0 keeps the three separate projections, i.e. the
        # pre-packing behaviour, so a baseline run and an A/B can share one build.
        parts = (
            [packed_projection_weight(proj) for proj in self.low_rank_projections()]
            if os.environ.get("RTP_GLM_KDA_PACK_PROJ", "1") != "0"
            else [None]
        )
        # Say why when packing was asked for and did not happen. The fused MHC
        # path had the same shape of bug -- a silent capability check that left
        # a whole benchmark round measuring the unoptimised path.
        if os.environ.get("RTP_GLM_KDA_PACK_PROJ", "1") == "0":
            pass
        elif not all(part is not None for part in parts):
            rejected = [
                type(proj).__name__
                for proj, part in zip(self.low_rank_projections(), parts)
                if part is None
            ]
            logging.info(
                "KDA projection packing off: %s is not a bias-free contiguous "
                "BF16 projection",
                ", ".join(rejected),
            )
        else:
            widths = [part.shape[1] for part in parts]
            # The packed copy is a duplicate; only accept it while narrow.
            if sum(widths) <= MAX_PACKED_PROJECTION_WIDTH:
                self.packed_low_widths = widths
                self.packed_low_weight = torch.cat(parts, dim=1).contiguous()
            else:
                logging.info(
                    "KDA projection packing off: packed width %d exceeds the "
                    "%d cap, the duplicated weight would not pay for itself",
                    sum(widths),
                    MAX_PACKED_PROJECTION_WIDTH,
                )

    def low_rank_projections(self):
        return (self.in_proj_b, self.f_a_proj, self.g_a_proj)

    def _project_inputs(
        self, hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """qkv, beta, forget gate and output gate from hidden_states."""
        if self.packed_low_weight is not None and hidden_states.ndim == 2:
            packed = torch.mm(hidden_states, self.packed_low_weight)
            beta_input, forget_low, gate_low = packed.split(
                self.packed_low_widths, dim=-1
            )
        else:
            beta_input = self.in_proj_b(hidden_states)
            forget_low = self.f_a_proj(hidden_states)
            gate_low = self.g_a_proj(hidden_states)
        return (
            self.in_proj_qkv(hidden_states),
            beta_input,
            self.f_b_proj(forget_low),
            self.g_b_proj(gate_low),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache],
        attention_inputs: Optional[PyAttentionInputs],
        attn_meta: KimiLinearMetadata,
    ) -> torch.Tensor:
        assert attention_inputs is not None, "attention_inputs is required"
        assert (
            attention_inputs.is_target_verify
            or not attention_inputs.is_prefill
            or attn_meta.get_prefill_conv1d_meta() is not None
            # CP prefill carries full_prefill_conv1d_meta instead; the local
            # per-rank conv1d metadata is never used on that path.
            or attn_meta.is_cp_linear_attn
        ), "prefill_conv1d_meta is required for prefill"

        # 1. Projections: qkv [token, 3*H*D], beta [token, H], forget gate and
        #    output gate [token, H*D] each.
        projected_qkv, beta_input, forget_gate, g_proj = self._project_inputs(
            hidden_states
        )

        # 2. Prefill or decode
        if attention_inputs.is_prefill and not attn_meta.is_target_verify:
            if attn_meta.is_cp_linear_attn:
                attn_output = self.prefill_kda.forward_cp(
                    projected_qkv,
                    forget_gate,
                    beta_input,
                    attention_inputs,
                    kv_cache,
                    attn_meta,
                )
            else:
                attn_output = self.prefill_kda(
                    projected_qkv,
                    forget_gate,
                    beta_input,
                    attention_inputs,
                    kv_cache,
                    attn_meta,
                )
        else:
            attn_output = self.decode_kda(
                projected_qkv,
                forget_gate,
                beta_input,
                attention_inputs,
                kv_cache,
                attn_meta,
            )

        # 3. o_norm with sigmoid gating: y = RMSNorm(attn_out) * sigmoid(g_proj)
        # 解码时走一个融合 kernel：RmsNormGated 这条路把归一化、sigmoid 和逐元素乘
        # 拆成好几个 kernel，每层都付一次，34 个 KDA 层累起来不小。
        if _KDA_FP32_STATE and not attention_inputs.is_prefill:
            from rtp_llm.models_py.triton_kernels.kimi_kda.fp32_state.rms_norm_gate import (
                kimi_kda_rms_norm_sigmoid_gate,
            )

            attn_output = kimi_kda_rms_norm_sigmoid_gate(
                attn_output.reshape(-1, self.head_v_dim),
                g_proj.reshape(-1, self.head_v_dim),
                self.norm.weight,
                self.norm.eps,
            )
        else:
            attn_output = self.norm(
                attn_output.reshape(-1, self.head_v_dim),
                g_proj.reshape(-1, self.head_v_dim),
            )

        # from [token * head, dim] -> [token, head * dim]
        attn_output = attn_output.reshape(-1, self.local_num_v_heads * self.head_v_dim)

        # 4. Output projection + all_reduce
        attn_output = self.out_proj(attn_output)

        if self.parallelism_config.get_attn_tp_size() > 1:
            attn_output = all_reduce(attn_output, group=Group.TP)

        return attn_output


class KimiLinearDecoderLayer(nn.Module):
    """Kimi Linear decoder layer.

    Hybrid KDA/MLA attention + MoE/Dense FFN.
    Uses RMSResNorm (fused residual add + layernorm) like GenericMoeDecoderLayer.
    """

    def __init__(
        self,
        config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        layer_idx: int,
        moe_config,
        max_generate_batch_size: int = 0,
        enable_cuda_graph: bool = False,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_type = config.hybrid_attention_config.hybrid_attention_types[
            layer_idx
        ]

        quant_config = config.quant_config

        if self.layer_type == HybridAttentionType.LINEAR:
            self.self_attn = KimiLinearKDA(
                config.linear_attention_config,
                parallelism_config,
                weights,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )
        else:
            # Full MLA attention layer
            self.self_attn = MlaAttention(
                config.attn_config,
                parallelism_config,
                weights,
                layer_idx,
                config.layernorm_eps,
                quant_config,
                hw_kernel_config=hw_kernel_config,
            )

        # FFN: Dense (layer 0) or MoE (layer 1+)
        if layer_idx not in config.moe_layer_index:
            self.mlp = DenseMLP(
                config.activation_type,
                parallelism_config,
                weights,
                quant_config,
                hw_kernel_config=hw_kernel_config,
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

        # RMSResNorm: fused residual add + layernorm
        self.input_layernorm = RMSResNorm(
            weights[W.pre_ln_gamma], eps=config.layernorm_eps
        )
        self.post_attention_layernorm = RMSResNorm(
            weights[W.post_ln_gamma], eps=config.layernorm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        fmha_impl: FMHAImplBase,
        kv_cache: Optional[LayerKVCache] = None,
        attention_inputs: Optional[PyAttentionInputs] = None,
        attn_meta: KimiLinearMetadata = KimiLinearMetadata(),
    ) -> DecodeLayerOutput:
        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self Attention (KDA or MLA)
        if self.layer_type == HybridAttentionType.LINEAR:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
                attention_inputs=attention_inputs,
                attn_meta=attn_meta,
            )
        else:
            hidden_states = self.self_attn(
                hidden_states=hidden_states,
                fmha_impl=fmha_impl,
                kv_cache=kv_cache,
            )

        # Fused: residual = residual + hidden_states, hidden_states = RMSNorm(residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # MLP (Dense or MoE)
        hidden_states = self.mlp(hidden_states)

        return DecodeLayerOutput(hidden_states, residual)


class KimiLinearModel(GptModelBase):
    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config,
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
                KimiLinearDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    enable_cuda_graph,
                    hw_kernel_config=py_hw_kernel_config,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )
        self._init_capture_context(
            self._capture_canonical_layer,
            self._capture_canonical_final,
        )

    def _capture_canonical_layer(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual capture requires a residual tensor")
        return hidden_states + residual

    def _capture_canonical_final(
        self, hidden_states: torch.Tensor, residual: torch.Tensor | None
    ) -> torch.Tensor:
        if residual is None:
            raise ValueError("residual finalization requires a residual tensor")
        return self.norm(hidden_states, residual)[0]

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        if self.kv_cache is None:
            return None
        full_attention_layers = (
            layer_idx
            for layer_idx, layer in enumerate(self.layers)
            if layer.layer_type != HybridAttentionType.LINEAR
        )
        return get_group_tags_for_layers(self.kv_cache, full_attention_layers)

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        attention_inputs = get_primary_attention_inputs(inputs, self.kv_cache)
        prefill_conv1d_meta = None
        is_target_verify = attention_inputs.is_target_verify
        if attention_inputs.is_prefill and not is_target_verify:
            cu_seqlen_without_padding = attention_inputs.cu_seqlens_device
            prefill_conv1d_meta = prepare_causal_conv1d_metadata(
                query_start_loc=cu_seqlen_without_padding,
                device=hidden_states.device,
            )

        attn_meta = KimiLinearMetadata(prefill_conv1d_meta, is_target_verify)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        capture = self.capture_context(inputs.capture_hidden_states)
        residual = torch.zeros_like(hidden_states)

        for i, decoder_layer in enumerate(self.layers):
            layer_attention_inputs = select_attention_inputs_for_layer(
                inputs, self.kv_cache, i
            )
            layer_fmha_impl = (
                None
                if decoder_layer.layer_type == HybridAttentionType.LINEAR
                else select_fmha_impl_for_layer(fmha_impl, self.kv_cache, i)
            )
            output = decoder_layer(
                hidden_states,
                residual,
                layer_fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
                attention_inputs=layer_attention_inputs,
                attn_meta=attn_meta,
            )
            hidden_states = output.hidden_states
            residual = output.residual
            capture.capture_layer(i, hidden_states, residual)

        return capture.finalize(hidden_states, residual)
