"""GLM-5.3-Flash (glm5_next) registration for RTP-LLM.

`Glm5NextForConditionalGeneration` is a VL wrapper; this port implements the
text path only (the vision tower is not loaded).  The checkpoint is a hybrid
KDA (linear attention) + sparse MLA (DSA indexer) model with MoE FFN and MHC
(multi-hyper-connection) residual streams.

Checkpoint layout (VL-wrapped): text weights live under
``model.language_model.``; ``lm_head.weight`` is top level; the MTP layer sits
at ``model.language_model.layers.<num_hidden_layers>`` and is skipped by the
main model.
"""

import functools
import json
import logging
import os
from typing import Dict, List, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.model_loader.attn_weight import MlaAttnAtomicWeight, MlaConfig
from rtp_llm.model_loader.linear_attn_weight import (
    LinearAttnAtomicWeight,
    LinearAttnConfig,
)
from rtp_llm.model_loader.model_weight_info import (
    ModelDeployWeightInfo,
    ModelWeightInfo,
)
from rtp_llm.model_loader.weight_module import AtomicWeight, WeightModule
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.deepseek_v2 import DeepSeekV2Weight, Glm5Mtp, Glm5MtpWeight
from rtp_llm.ops import (
    DataType,
    HybridAttentionType,
    KVCacheSpecDesc,
    KVCacheSpecType,
    OpaqueBlockEntryCountMode,
)
from rtp_llm.utils.model_weight import (
    CkptWeightInfo,
    W,
    identity,
    merge_qkv_hf,
    transpose,
    transpose_slice_k,
    transpose_slice_v,
)

logger = logging.getLogger(__name__)

_CKPT_BASE = "model."
_VL_TEXT_PREFIX = "model.language_model."


def merge_conv1d(ts: List[torch.Tensor]) -> torch.Tensor:
    """Merge separate q_conv1d, k_conv1d, v_conv1d into one tensor."""
    return torch.cat(ts, dim=0)


def _squeeze_alog(ts: List[torch.Tensor]) -> torch.Tensor:
    return ts[0].squeeze()


class Glm5NextWeight(DeepSeekV2Weight):
    """GLM-5.3 weight loading.

    Reuses DeepSeekV2Weight for the sparse-MLA layers and the MoE/dense FFN,
    adds KDA linear-attention weights (kimi_linear-style names) for the KDA
    layers, the MHC per-layer mixing parameters, and the DSA indexer kpool
    compression parameters.  Checkpoint tensors that ship unquantized BF16
    while their ``W`` name is FP8-wrap eligible (kv_b_proj, indexer wq_b/wk,
    KDA o_proj) are restored to their BF16 descriptors after the generic FP8
    pass.
    """

    # W names whose checkpoint tensors are BF16 in the GLM-5.3 checkpoint even
    # though the generic per-block FP8 wrapper would claim them.
    _KV_BF16_NAMES = (W.mla_kv_b_w, W.mla_kc, W.mla_vc)
    _INDEXER_BF16_NAMES = (W.mla_indexer_qb_w, W.mla_indexer_k_w)
    _KDA_BF16_NAMES = (W.linear_attn_out_w,)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = _VL_TEXT_PREFIX
        self._layer_types = (
            self.model_config.hybrid_attention_config.hybrid_attention_types
        )
        self._use_mhc = int(getattr(self.model_config, "hc_mult", 1)) > 1
        # (layer_id, W-name) -> original BF16 descriptor, stashed at build time
        self._bf16_descriptors = {}
        self._kv_b_fp8 = True
        self._indexer_fp8 = True
        self._kda_o_fp8 = True

    # ------------------------------------------------------------------
    # checkpoint inspection
    # ------------------------------------------------------------------
    def _process_meta(self, meta_dict, weight_keys):
        # Detect the text-weight prefix ("model." vs the VL "model.language_model.")
        suffix = "layers.0.self_attn.q_proj.weight"
        for key in weight_keys:
            if key.endswith(suffix):
                self._prefix = key[: -len(suffix)]
                break
        # GLM-5.3 always has q LoRA (q_lora_rank=1536) on the MLA layers.
        self.q_use_lora = True
        self.has_e_score_correction_bias = False
        for layer_id in range(self._num_layers):
            if (
                f"{self._prefix}layers.{layer_id}.mlp.gate.e_score_correction_bias"
                in weight_keys
            ):
                self.has_e_score_correction_bias = True
                break
        # FP8-vs-BF16 detection for the selectively quantized tensors.
        full_layers = [
            i
            for i in range(self._num_layers)
            if self._layer_types[i] != HybridAttentionType.LINEAR
        ]
        probe_layer = full_layers[0] if full_layers else 0
        self._kv_b_fp8 = (
            f"{self._prefix}layers.{probe_layer}.self_attn.kv_b_proj.weight_scale_inv"
            in weight_keys
        )
        self._indexer_fp8 = (
            f"{self._prefix}layers.{probe_layer}.self_attn.indexer.wq_b.weight_scale_inv"
            in weight_keys
        )
        kda_layers = [
            i
            for i in range(self._num_layers)
            if self._layer_types[i] == HybridAttentionType.LINEAR
        ]
        kda_probe = kda_layers[0] if kda_layers else 0
        self._kda_o_fp8 = (
            f"{self._prefix}layers.{kda_probe}.self_attn.o_proj.weight_scale_inv"
            in weight_keys
        )

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        # GLM-5.3 is NoPE: qk_rope_head_dim=0 and the indexer applies no rope.
        return None

    # ------------------------------------------------------------------
    # name plumbing
    # ------------------------------------------------------------------
    def _remap_to_prefix(self, weight_modules: List[WeightModule]) -> None:
        """Rewrite leading "model." ckpt names to the detected text prefix."""
        if self._prefix == _CKPT_BASE:
            return
        for wm in weight_modules:
            for comp in wm.get_components():
                ckpt_list = getattr(comp, "weights", None)
                if not ckpt_list:
                    continue
                for ckpt in ckpt_list:
                    if hasattr(ckpt, "name") and ckpt.name.startswith(_CKPT_BASE):
                        ckpt.name = self._prefix + ckpt.name[len(_CKPT_BASE) :]

    # ------------------------------------------------------------------
    # per-layer descriptors
    # ------------------------------------------------------------------
    def _get_hf_layer_weight_info(self, layer_id: int):
        if self._layer_types[layer_id] == HybridAttentionType.LINEAR:
            weights: List[WeightModule] = []
            weights.extend(self._create_kda_layer_norm_weight())
            weights.extend(self._create_kda_attention_weight())
            weights.extend(
                DeepSeekV2Weight._get_hf_ffn_layer_weight_info(self, layer_id)
            )
        else:
            weights = DeepSeekV2Weight._get_hf_layer_weight_info(self, layer_id)
            weights.extend(self._create_kpool_weight())
        if self._use_mhc:
            weights.extend(self._create_mhc_weight())
        self._stash_bf16_descriptors(layer_id, weights)
        self._remap_to_prefix(weights)
        return weights

    def _create_kda_layer_norm_weight(self) -> List[WeightModule]:
        return [
            AtomicWeight(
                W.pre_ln_gamma,
                [CkptWeightInfo("model.layers.{i}.input_layernorm.weight", identity)],
                identity,
            ),
            AtomicWeight(
                W.post_ln_gamma,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.post_attention_layernorm.weight", identity
                    )
                ],
                identity,
            ),
        ]

    def _create_kda_attention_weight(self) -> List[WeightModule]:
        """KDA linear-attention weights (same layout as kimi_linear)."""
        linear_attn_cfg = LinearAttnConfig(self.model_config.linear_attention_config)
        return [
            LinearAttnAtomicWeight(
                W.linear_attn_qkv_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_proj.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_proj.weight", identity
                    ),
                ],
                merge_qkv_hf,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_b_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.b_proj.weight", identity)],
                transpose,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_f_a_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.f_a_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_f_b_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.f_b_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_g_a_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.g_a_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_g_b_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.g_b_proj.weight", identity
                    )
                ],
                transpose,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_conv1d_w,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.q_conv1d.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.k_conv1d.weight", identity
                    ),
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.v_conv1d.weight", identity
                    ),
                ],
                merge_conv1d,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_norm_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_norm.weight", identity)],
                identity,
                linear_attn_cfg,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_dt_b_kda,
                [CkptWeightInfo("model.layers.{i}.self_attn.dt_bias", identity)],
                identity,
                linear_attn_cfg,
                data_type=torch.float32,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_alog,
                [CkptWeightInfo("model.layers.{i}.self_attn.A_log", identity)],
                _squeeze_alog,
                linear_attn_cfg,
                data_type=torch.float32,
            ),
            LinearAttnAtomicWeight(
                W.linear_attn_out_w,
                [CkptWeightInfo("model.layers.{i}.self_attn.o_proj.weight", identity)],
                transpose,
                linear_attn_cfg,
            ),
        ]

    def _create_kpool_weight(self) -> List[WeightModule]:
        """DSA indexer kpool compression parameters (GLM-5.3)."""
        return [
            AtomicWeight(
                W.mla_indexer_kpool_ape,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.indexer.index_kpool_compress_ape",
                        identity,
                    )
                ],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.mla_indexer_kpool_gate,
                [
                    CkptWeightInfo(
                        "model.layers.{i}.self_attn.indexer.index_kpool_compress_gate",
                        identity,
                    )
                ],
                identity,
            ),
        ]

    def _create_mhc_weight(self) -> List[WeightModule]:
        """MHC hyper-connection mixing parameters (fp32 in the runtime)."""
        return [
            AtomicWeight(
                W.hc_attn_base,
                [CkptWeightInfo("model.layers.{i}.hc_attn_base", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.hc_attn_scale,
                [CkptWeightInfo("model.layers.{i}.hc_attn_scale", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.hc_attn_fn,
                [CkptWeightInfo("model.layers.{i}.hc_attn_fn", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.hc_ffn_base,
                [CkptWeightInfo("model.layers.{i}.hc_ffn_base", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.hc_ffn_scale,
                [CkptWeightInfo("model.layers.{i}.hc_ffn_scale", identity)],
                identity,
                data_type=torch.float32,
            ),
            AtomicWeight(
                W.hc_ffn_fn,
                [CkptWeightInfo("model.layers.{i}.hc_ffn_fn", identity)],
                identity,
                data_type=torch.float32,
            ),
        ]

    # ------------------------------------------------------------------
    # BF16 restore after the generic FP8 wrap
    # ------------------------------------------------------------------
    def _stash_bf16_descriptors(
        self, layer_id: int, weights: List[WeightModule]
    ) -> None:
        keep = set()
        if not self._kv_b_fp8:
            keep.update(self._KV_BF16_NAMES)
        if not self._indexer_fp8:
            keep.update(self._INDEXER_BF16_NAMES)
        if not self._kda_o_fp8:
            keep.update(self._KDA_BF16_NAMES)
        if not keep:
            return
        for wm in weights:
            for comp in wm.get_components():
                if comp.name in keep:
                    self._bf16_descriptors[(layer_id, comp.name)] = comp

    @staticmethod
    def _wrapped_kernel_name(weight: WeightModule) -> Optional[str]:
        kernel = getattr(weight, "kernel", None)
        if kernel is not None and getattr(kernel, "name", None) is not None:
            return kernel.name
        return getattr(weight, "name", None)

    def get_weight_info(self, database=None) -> ModelWeightInfo:
        weight_info = super().get_weight_info(database)
        if not self._bf16_descriptors:
            return weight_info
        if self._quant_algo is None or not self._quant_algo.isQuant():
            return weight_info
        # The generic per-block FP8 pass wraps by W name alone; GLM-5.3 ships
        # these tensors unquantized, so restore the BF16 descriptors stashed
        # at build time.
        for layer_id, layer_weights in enumerate(weight_info.layer_weights):
            if not isinstance(layer_weights, list):
                continue
            for pos, weight in enumerate(layer_weights):
                kernel_name = self._wrapped_kernel_name(weight)
                original = self._bf16_descriptors.get((layer_id, kernel_name))
                if original is not None and weight is not original:
                    layer_weights[pos] = original
        return weight_info

    def _get_weight_info(self):
        weight_info = super()._get_weight_info()
        self._remap_to_prefix(weight_info.weights)
        return weight_info


class Glm5Next(BaseModel):
    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"config.json not found in {ckpt_path}")
        with open(config_path) as reader:
            config_json = json.loads(reader.read())
        # VL wrapper: the text model parameters live under text_config.
        text = config_json.get("text_config", config_json)

        config = ModelConfig()
        config.ckpt_path = ckpt_path
        cls._parse_basic_config(text, config)
        cls._parse_mla_config(text, config)
        cls._parse_rope_config(text, config)
        cls._parse_moe_config(text, config)
        cls._parse_hybrid_attention_config(text, config)
        cls._parse_linear_attention_config(text, config)
        cls._parse_mhc_config(text, config)
        return config

    # ------------------------------------------------------------------
    # config parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_basic_config(text: dict, config: ModelConfig):
        config.attn_config.head_num = text["num_attention_heads"]
        config.attn_config.kv_head_num = text.get(
            "num_key_value_heads", text["num_attention_heads"]
        )
        config.num_layers = text["num_hidden_layers"]
        config.hidden_size = text["hidden_size"]
        config.vocab_size = text["vocab_size"]
        config.inter_size = text["intermediate_size"]
        config.dense_inter_size = config.inter_size
        config.max_seq_len = text.get("max_position_embeddings", 8192)
        config.layernorm_eps = text.get("rms_norm_eps", 1e-5)
        config.norm_type = "rmsnorm"
        config.has_post_decoder_layernorm = True
        config.activation_type = "SiGLU"
        config.tie_word_embeddings = text.get("tie_word_embeddings", False)
        config.config_dtype = text.get("dtype", text.get("torch_dtype"))

    @staticmethod
    def _parse_mla_config(text: dict, config: ModelConfig):
        config.attn_config.use_mla = True
        q_lora_rank = text.get("q_lora_rank")
        config.attn_config.q_lora_rank = (
            int(q_lora_rank) if q_lora_rank is not None else 0
        )
        kv_lora_rank = text.get("kv_lora_rank")
        config.attn_config.kv_lora_rank = (
            int(kv_lora_rank) if kv_lora_rank is not None else 0
        )
        config.attn_config.nope_head_dim = text["qk_nope_head_dim"]
        config.attn_config.rope_head_dim = text.get("qk_rope_head_dim", 0)
        config.attn_config.v_head_dim = text["v_head_dim"]
        config.attn_config.size_per_head = (
            config.attn_config.nope_head_dim + config.attn_config.rope_head_dim
        )
        config.attn_config.rope_config.dim = config.attn_config.rope_head_dim
        config.attn_config.rope_config.offset = config.attn_config.nope_head_dim

    @staticmethod
    def _parse_rope_config(text: dict, config: ModelConfig):
        # GLM-5.3 is NoPE: qk_rope_head_dim=0 and the indexer applies no rope.
        # Keep the fields consistent for downstream consumers that read them.
        config.attn_config.rope_config.base = int(text.get("rope_theta", 10000))
        config.attn_config.rope_config.style = 0
        rope_scaling = text.get("rope_scaling")
        if rope_scaling is not None:
            from rtp_llm.utils.model_weight import yarn_get_mscale

            config.attn_config.rope_config.scale = rope_scaling["factor"]
            config.attn_config.rope_config.factor1 = float(
                rope_scaling.get("beta_slow", 1)
            )
            config.attn_config.rope_config.factor2 = float(
                rope_scaling.get("beta_fast", 32)
            )
            config.attn_config.rope_config.max_pos = rope_scaling[
                "original_max_position_embeddings"
            ]
            scaling_factor = rope_scaling["factor"]
            mscale = rope_scaling["mscale"]
            mscale_all_dim = rope_scaling["mscale_all_dim"]
            config.deepseek_rope_mscale = mscale
            config.deepseek_mscale_all_dim = mscale_all_dim
            config.attn_config.rope_config.mscale = yarn_get_mscale(
                scaling_factor, mscale
            ) / yarn_get_mscale(scaling_factor, mscale_all_dim)
            softmax_mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
            config.attn_config.softmax_extra_scale = softmax_mscale * softmax_mscale
        rope_interleave = text.get("rope_interleave", True)
        config.attn_config.rope_config.is_neox_style = not rope_interleave
        indexer_rope_interleave = text.get("indexer_rope_interleave", False)
        config.attn_config.rope_config.indexer_is_neox_style = (
            not indexer_rope_interleave
        )

    @staticmethod
    def _parse_moe_config(text: dict, config: ModelConfig):
        scoring_func = text.get("scoring_func")
        if scoring_func == "sigmoid":
            config.scoring_func = 1
        elif scoring_func == "softmax":
            config.scoring_func = 0
        elif scoring_func is not None:
            raise ValueError(f"Unknown scoring_func: {scoring_func}")

        config.routed_scaling_factor = text["routed_scaling_factor"]
        config.moe_k = text["num_experts_per_tok"]
        config.expert_num = text["n_routed_experts"]
        moe_intermediate_size = text["moe_intermediate_size"]
        config.moe_inter_size = moe_intermediate_size
        config.moe_n_group = text.get("n_group", 1)
        config.moe_topk_group = text.get("topk_group", 1)

        n_shared_experts = text.get("n_shared_experts", 0)
        config.n_shared_experts = n_shared_experts
        config.inter_size = n_shared_experts * moe_intermediate_size
        config.has_moe_norm = text.get("norm_topk_prob", False)
        config.moe_style = 2 if n_shared_experts > 0 else 1
        config.swiglu_limit = float(text.get("swiglu_limit", 0.0))

        moe_step = text.get("moe_layer_freq", 1)
        first_k_dense_replace = text.get("first_k_dense_replace", 0)
        config.moe_layer_index = [
            i
            for i in range(config.num_layers)
            if i >= first_k_dense_replace and i % moe_step == 0
        ]

        # DSA indexer config (sparse MLA).
        if text.get("index_topk") is not None:
            config.attn_config.is_sparse = True
            config.attn_config.indexer_head_dim = text["index_head_dim"]
            config.attn_config.indexer_head_num = text["index_n_heads"]
            config.attn_config.indexer_topk = text["index_topk"]
            config.index_topk_freq = int(text.get("index_topk_freq", 1))
            config.index_skip_topk_offset = text.get("index_skip_topk_offset", None)
            config.index_topk_pattern = text.get("index_topk_pattern", None)
            config.indexer_types = text.get("indexer_types", None)
            config.index_share_for_mtp_iteration = bool(
                text.get("index_share_for_mtp_iteration", False)
            )

    @staticmethod
    def _parse_hybrid_attention_config(text: dict, config: ModelConfig):
        layer_types = text.get("layer_types")
        config.hybrid_attention_config.enable_hybrid_attention = True
        hybrid_layer_types: List[HybridAttentionType] = []
        for i in range(config.num_layers):
            layer_type = layer_types[i] if layer_types else None
            if layer_type == "linear_attention":
                hybrid_layer_types.append(HybridAttentionType.LINEAR)
            else:
                hybrid_layer_types.append(HybridAttentionType.NONE)
        config.hybrid_attention_config.hybrid_attention_types = hybrid_layer_types

    @staticmethod
    def _parse_linear_attention_config(text: dict, config: ModelConfig):
        lac = text.get("linear_attn_config", {})
        head_dim = lac["head_dim"]
        num_heads = lac["num_heads"]
        config.linear_attention_config.linear_key_head_dim = head_dim
        config.linear_attention_config.linear_value_head_dim = head_dim
        config.linear_attention_config.linear_num_key_heads = num_heads
        config.linear_attention_config.linear_num_value_heads = num_heads
        config.linear_attention_config.linear_conv_kernel_dim = lac.get(
            "short_conv_kernel_size", 4
        )
        config.linear_gate_lower_bound = lac.get("gate_lower_bound", None)

    @staticmethod
    def _parse_mhc_config(text: dict, config: ModelConfig):
        if not text.get("mhc", False):
            return
        config.hc_mult = int(text.get("hc_mult", 1))
        config.hc_sinkhorn_iters = int(text.get("hc_sinkhorn_iters", 0))
        config.hc_eps = float(text.get("hc_eps", 1e-6))

    # ------------------------------------------------------------------
    # KV cache topology: KDA layers own a LINEAR cache; full-attention layers
    # own the MLA cache plus the DSA indexer cache (same as deepseek_v2).
    # ------------------------------------------------------------------
    @classmethod
    def _post_build_model_config(cls, model_config: ModelConfig) -> None:
        if model_config.kv_cache_spec_descs:
            return

        indexer_head_dim = int(model_config.attn_config.indexer_head_dim)
        if indexer_head_dim <= 0 or indexer_head_dim % 128 != 0:
            raise ValueError(
                "sparse MLA indexer_head_dim must be positive and divisible by "
                f"128, got {indexer_head_dim}"
            )

        linear_desc = KVCacheSpecDesc()
        linear_desc.tag = "linear"
        linear_desc.cache_type = KVCacheSpecType.LINEAR

        indexer_desc = KVCacheSpecDesc()
        indexer_desc.tag = "indexer_kv"
        indexer_desc.cache_type = KVCacheSpecType.OPAQUE_KV
        indexer_desc.entry_dtype = DataType.TYPE_UINT8
        indexer_desc.entry_elems = indexer_head_dim + indexer_head_dim // 128 * 4
        indexer_desc.entry_count_mode = (
            OpaqueBlockEntryCountMode.KERNEL_BLOCK_COMPRESSED
        )
        indexer_desc.compression_ratio = 1

        descs = []
        for layer_type in model_config.hybrid_attention_config.hybrid_attention_types:
            if layer_type == HybridAttentionType.LINEAR:
                descs.append([linear_desc])
            else:
                full_desc = KVCacheSpecDesc()
                full_desc.tag = "default"
                full_desc.cache_type = KVCacheSpecType.MLA
                descs.append([full_desc, indexer_desc])
        model_config.kv_cache_spec_descs = descs

    def support_cuda_graph(self) -> bool:
        return True

    def _create_python_model(self):
        from rtp_llm.models_py.model_desc.glm5_next import Glm5NextModel

        self.py_model = Glm5NextModel(
            self.model_config,
            self.parallelism_config,
            self.weight,
            self.moe_config,
            max_generate_batch_size=self.max_generate_batch_size,
            fmha_config=self.fmha_config,
            py_hw_kernel_config=self.hw_kernel_config,
        )
        return self.py_model

    @staticmethod
    def get_weight_cls():
        return Glm5NextWeight


register_model("glm5_next", Glm5Next, ["Glm5NextForConditionalGeneration"])


def _swap_wrapped_with(
    layer_weights: List[WeightModule], originals: Dict[str, WeightModule]
) -> None:
    """Replace FP8-wrapped descriptors whose kernel name has a BF16 original."""
    for pos, weight in enumerate(layer_weights):
        kernel = getattr(weight, "kernel", None)
        name = getattr(kernel, "name", None) or getattr(weight, "name", None)
        original = originals.get(name)
        if original is not None and weight is not original:
            layer_weights[pos] = original


class Glm5NextMtpWeight(Glm5MtpWeight):
    """GLM-5.3 MTP (NextN) draft-layer weight loading.

    The MTP draft layer sits at ``model.language_model.layers.<num_hidden_layers>``
    in the full GLM-5.3 checkpoint: it is a sparse-MLA + MoE layer with the DSA
    indexer (including kpool parameters) and no MHC parameters.  Layout follows
    Glm5MtpWeight (full-checkpoint layout B); this class adds the VL text
    prefix, the kpool descriptors, and the BF16 restore for the checkpoint's
    selectively-unquantized kv_b/indexer tensors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefix = _VL_TEXT_PREFIX
        self._kv_b_fp8 = True
        self._indexer_fp8 = True

    def _mtp_layer_idx(self) -> int:
        return int(getattr(self.model_config, "mtp_layer_offset", 0) or 0)

    def _process_meta(self, meta_dict, weight_keys):
        mtp_idx = self._mtp_layer_idx()
        suffix = f"layers.{mtp_idx}.eh_proj.weight"
        for key in weight_keys:
            if key.endswith(suffix):
                self._prefix = key[: -len(suffix)]
                break
        self.q_use_lora = True
        self.has_e_score_correction_bias = (
            f"{self._prefix}layers.{mtp_idx}.mlp.gate.e_score_correction_bias"
            in weight_keys
        )
        self._kv_b_fp8 = (
            f"{self._prefix}layers.{mtp_idx}.self_attn.kv_b_proj.weight_scale_inv"
            in weight_keys
        )
        self._indexer_fp8 = (
            f"{self._prefix}layers.{mtp_idx}.self_attn.indexer.wq_b.weight_scale_inv"
            in weight_keys
        )

    def _create_rope_w(self) -> Optional[AtomicWeight]:
        return None

    def _bf16_originals(self) -> Dict[str, WeightModule]:
        """Fresh unquantized descriptors for the MTP layer's BF16 tensors."""
        mtp_idx = self._mtp_layer_idx()
        prefix = f"{self._prefix}layers.{mtp_idx}.self_attn"
        attn_config = MlaConfig(
            head_num=self._head_num,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            ope_head_dim=self.nope_head_dim,
            v_head_dim=self.v_head_dim,
            use_mla=self.model_config.attn_config.use_mla,
            q_use_lora=self.q_use_lora,
        )
        kv_b_name = f"{prefix}.kv_b_proj.weight"
        originals: Dict[str, WeightModule] = {}
        if not self._kv_b_fp8:
            originals[W.mla_kv_b_w] = MlaAttnAtomicWeight(
                W.mla_kv_b_w,
                [CkptWeightInfo(kv_b_name, identity)],
                transpose,
                config=attn_config,
            )
            originals[W.mla_kc] = MlaAttnAtomicWeight(
                W.mla_kc,
                [CkptWeightInfo(kv_b_name, identity)],
                functools.partial(
                    transpose_slice_k,
                    head_num=self._head_num,
                    nope_head_dim=self.nope_head_dim,
                    v_head_dim=self.v_head_dim,
                    lora_rank=self.kv_lora_rank,
                ),
                config=attn_config,
            )
            originals[W.mla_vc] = MlaAttnAtomicWeight(
                W.mla_vc,
                [CkptWeightInfo(kv_b_name, identity)],
                functools.partial(
                    transpose_slice_v,
                    head_num=self._head_num,
                    nope_head_dim=self.nope_head_dim,
                    v_head_dim=self.v_head_dim,
                    lora_rank=self.kv_lora_rank,
                ),
                config=attn_config,
            )
        if not self._indexer_fp8:
            originals[W.mla_indexer_qb_w] = MlaAttnAtomicWeight(
                W.mla_indexer_qb_w,
                [CkptWeightInfo(f"{prefix}.indexer.wq_b.weight", identity)],
                transpose,
            )
            originals[W.mla_indexer_k_w] = MlaAttnAtomicWeight(
                W.mla_indexer_k_w,
                [CkptWeightInfo(f"{prefix}.indexer.wk.weight", identity)],
                transpose,
            )
        return originals

    def _remap_to_prefix(self, weight_modules: List[WeightModule]) -> None:
        if self._prefix == _CKPT_BASE:
            return
        for wm in weight_modules:
            for comp in wm.get_components():
                ckpt_list = getattr(comp, "weights", None)
                if not ckpt_list:
                    continue
                for ckpt in ckpt_list:
                    if hasattr(ckpt, "name") and ckpt.name.startswith(_CKPT_BASE):
                        ckpt.name = self._prefix + ckpt.name[len(_CKPT_BASE) :]

    def _get_weight_info(self):
        weight_info = super()._get_weight_info()
        # The MTP layer is a full sparse-MLA layer: add the kpool descriptors
        # (literal layer index, matching Glm5MtpWeight's pre-substitution).
        mtp_idx = self._mtp_layer_idx()
        for layer_weights in weight_info.layer_weights:
            layer_weights.extend(
                [
                    AtomicWeight(
                        W.mla_indexer_kpool_ape,
                        [
                            CkptWeightInfo(
                                f"{_CKPT_BASE}layers.{mtp_idx}.self_attn.indexer.index_kpool_compress_ape",
                                identity,
                            )
                        ],
                        identity,
                        data_type=torch.float32,
                    ),
                    AtomicWeight(
                        W.mla_indexer_kpool_gate,
                        [
                            CkptWeightInfo(
                                f"{_CKPT_BASE}layers.{mtp_idx}.self_attn.indexer.index_kpool_compress_gate",
                                identity,
                            )
                        ],
                        identity,
                    ),
                ]
            )
        self._remap_to_prefix(weight_info.weights)
        for layer_weights in weight_info.layer_weights:
            self._remap_to_prefix(layer_weights)
        return weight_info

    def get_weight_info(self, database=None) -> ModelWeightInfo:
        weight_info = super().get_weight_info(database)
        originals = self._bf16_originals()
        if not originals:
            return weight_info
        if self._quant_algo is None or not self._quant_algo.isQuant():
            return weight_info
        for layer_weights in weight_info.layer_weights:
            if isinstance(layer_weights, list):
                _swap_wrapped_with(layer_weights, originals)
        return weight_info


class Glm5NextMtp(Glm5Mtp):
    """GLM-5.3 MTP draft model: single sparse-MLA + MoE NextN layer.

    Reuses GenericMoeMTPModel; the draft layer has no KDA and no MHC, so the
    hybrid config is collapsed to one full-attention layer and hc_mult to 1.
    """

    @classmethod
    def _create_config(cls, ckpt_path: str):
        config = Glm5Next._create_config(ckpt_path)
        num_main_layers = config.num_layers
        num_mtp_layers = 1
        config_path = os.path.join(ckpt_path, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as reader:
                config_json = json.loads(reader.read())
            text = config_json.get("text_config", config_json)
            num_main_layers = int(text.get("num_hidden_layers", num_main_layers))
            num_mtp_layers = int(text.get("num_nextn_predict_layers", 1)) or 1
        # Layout B (full checkpoint): the draft layer is at num_hidden_layers.
        config.mtp_layer_offset = num_main_layers if num_main_layers > 1 else 0
        config.num_layers = num_mtp_layers
        config.moe_layer_index = [i for i in range(config.num_layers)]
        config.is_mtp = True
        config.hybrid_attention_config.hybrid_attention_types = [
            HybridAttentionType.NONE
        ] * num_mtp_layers
        # The MTP draft layer has no MHC parameters.
        config.hc_mult = 1
        config.hc_sinkhorn_iters = 0
        return config

    @staticmethod
    def get_weight_cls():
        return Glm5NextMtpWeight


register_model(
    "glm5_next_mtp", Glm5NextMtp, ["Glm5NextForConditionalGenerationNextN"]
)

