"""Qwen 4.5-Flash (``Qwen4_5ForCausalLM``) RTP-LLM model registration wrapper.

Heavy C++ dependencies (``rtp_llm.models.base_model``, ``rtp_llm.ops``) are
imported lazily so configuration parsing, weight validation, and KV cache
planning work cleanly in pure-Python test environments as well as in compiled
Bazel/RTP-LLM binaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Collection, Dict, Mapping, Optional, Sequence, Type, Union

from rtp_llm.config.qwen4_5_config import (
    EXPECTED_MODEL_TYPE,
    Qwen4_5ModelConfig,
    Qwen4_5ParamAndStateEstimate,
    estimate_qwen4_5_params_and_state,
    parse_qwen4_5_config_json,
    parse_qwen4_5_quant_config,
)
from rtp_llm.models.qwen4_5.qwen4_5_kv_cache import (
    Qwen4_5KVCachePlan,
    build_qwen4_5_kv_cache_plan,
    build_qwen4_5_kv_cache_spec_descs,
)
from rtp_llm.models.qwen4_5.qwen4_5_weight import (
    Qwen4_5Weight,
    validate_qwen4_5_weight_map,
)
from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

try:
    from rtp_llm.models.base_model import BaseModel as _BaseModel

    _HAS_RTP_BASE_MODEL = True
except Exception:  # pragma: no cover - pure-Python fallback without C++ ops
    _BaseModel = object  # type: ignore[assignment,misc]
    _HAS_RTP_BASE_MODEL = False


class Qwen4_5(_BaseModel):
    """RTP-LLM model wrapper for Qwen 4.5-Flash (``Qwen4_5ForCausalLM``).

    Architecture summary (1109-bf16-re-mxfp4-ple-fp8):
      - 40 hybrid layers (30 Gated DeltaNet linear + 10 MQA-1 full attention)
      - YoCo cross-layer attention sharing (layer 19 -> 5 full consumers;
        layer 20 -> 14 linear consumers)
      - 4-stream Hyper-Connections with ``hc_prev_mix`` and global mixer norm
      - Layer 1 FP8 per-tensor 128-shard PLE n-gram table
      - Layer 0 dense FFN; Layers 1..39 Latent MoE (640 MXFP4 routed experts,
        top-10, SiTU-GLU) + BF16 shared expert
      - ShortConv on FullAttn K/V, attention outputs, and FFN outputs
    """

    MODEL_TYPE: str = EXPECTED_MODEL_TYPE
    ARCHITECTURES = ("Qwen4_5ForCausalLM",)
    _model_config_cls = Qwen4_5ModelConfig

    @staticmethod
    def get_weight_cls() -> Type[Qwen4_5Weight]:
        return Qwen4_5Weight

    def support_pp(self) -> bool:
        return False

    def support_cuda_graph(self) -> bool:
        return getattr(self, "model_config", None) is not None

    def _py_model_class(self) -> Any:
        from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5Model

        return Qwen4_5Model

    @classmethod
    def create_config(
        cls,
        ckpt_path: Union[str, Path],
        *,
        strict: bool = True,
        **kwargs: Any,
    ) -> Qwen4_5ModelConfig:
        """Create a ``Qwen4_5ModelConfig`` from a checkpoint directory or ``config.json``."""
        p = Path(ckpt_path)
        config_path = p / "config.json" if p.is_dir() else p
        config_json = json.loads(config_path.read_text(encoding="utf-8"))
        return cls._create_config_from_json(
            str(config_path.parent), config_json, strict=strict
        )

    @classmethod
    def _create_config_from_json(
        cls,
        ckpt_path: str,
        config_json: Mapping[str, Any],
        *,
        strict: bool = True,
    ) -> Qwen4_5ModelConfig:
        arch_cfg = parse_qwen4_5_config_json(config_json, strict=strict)
        quant_cfg = parse_qwen4_5_quant_config(config_json)
        config = cls._model_config_cls(
            arch_config=arch_cfg,
            quant_config=quant_cfg,
            ckpt_path=ckpt_path,
            raw_config=config_json,
        )
        cls._populate_model_config(config, arch_cfg)
        return config

    @classmethod
    def _populate_model_config(
        cls,
        config: Qwen4_5ModelConfig,
        arch_cfg: Qwen4_5ArchConfig,
    ) -> None:
        config.model_type = cls.MODEL_TYPE
        config.vocab_size = arch_cfg.vocab_size
        config.hidden_size = arch_cfg.hidden_size
        config.num_layers = arch_cfg.num_hidden_layers
        config.hc_mult = arch_cfg.hc_count
        config.expert_num = arch_cfg.num_experts
        config.moe_k = arch_cfg.num_experts_per_tok
        config.tie_word_embeddings = arch_cfg.tie_word_embeddings

        # Attach pure-Python KV cache plan and exact sizing summary
        config.qwen4_5_kv_cache_plan = build_qwen4_5_kv_cache_plan(arch_cfg)
        config.qwen4_5_param_and_state_estimate = estimate_qwen4_5_params_and_state(
            arch_cfg
        )

        # Populate compiled RTP-LLM ModelConfig sub-structures when ops are available
        if _HAS_RTP_BASE_MODEL:
            try:
                from rtp_llm.ops import HybridAttentionType

                config.inter_size = arch_cfg.intermediate_size
                config.moe_inter_size = arch_cfg.moe_intermediate_size
                if hasattr(config, "attn_config"):
                    config.attn_config.head_num = arch_cfg.num_attention_heads
                    config.attn_config.kv_head_num = arch_cfg.num_key_value_heads
                    config.attn_config.size_per_head = arch_cfg.head_dim
                if hasattr(config, "hybrid_attention_config"):
                    config.hybrid_attention_config.enable_hybrid_attention = True
                    config.hybrid_attention_config.hybrid_attention_types = [
                        (
                            HybridAttentionType.LINEAR
                            if lt == "linear_attention"
                            else HybridAttentionType.NONE
                        )
                        for lt in arch_cfg.layer_types
                    ]
                config.kv_cache_spec_descs = build_qwen4_5_kv_cache_spec_descs(
                    config, arch_cfg
                )
            except Exception:
                pass

    @classmethod
    def build_kv_cache_plan(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        *,
        enable_yoco_decode: Optional[bool] = None,
        kv_dtype: str = "BF16",
    ) -> Qwen4_5KVCachePlan:
        return build_qwen4_5_kv_cache_plan(
            arch_cfg,
            enable_yoco_decode=enable_yoco_decode,
            kv_dtype=kv_dtype,
        )

    @classmethod
    def estimate_params_and_state(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        *,
        enable_yoco_decode: Optional[bool] = None,
        kv_dtype_bytes: int = 2,
    ) -> Qwen4_5ParamAndStateEstimate:
        return estimate_qwen4_5_params_and_state(
            arch_cfg,
            enable_yoco_decode=enable_yoco_decode,
            kv_dtype_bytes=kv_dtype_bytes,
        )

    @classmethod
    def validate_checkpoint_index(
        cls,
        arch_cfg: Qwen4_5ArchConfig,
        weight_map: Union[Mapping[str, Any], Collection[str]],
        *,
        prefix: str = "model.",
        allow_extra_prefixes: Sequence[str] = (),
    ) -> Dict[str, Any]:
        return validate_qwen4_5_weight_map(
            arch_cfg,
            weight_map,
            prefix=prefix,
            allow_extra_prefixes=allow_extra_prefixes,
        )


def register_qwen4_5_model() -> None:
    """Register ``qwen4_5`` / ``Qwen4_5ForCausalLM`` with ``model_factory_register``."""
    from rtp_llm.model_factory_register import register_model

    register_model(Qwen4_5.MODEL_TYPE, Qwen4_5, list(Qwen4_5.ARCHITECTURES))


try:
    register_qwen4_5_model()
except Exception:  # pragma: no cover - safe if registry is unavailable
    pass
