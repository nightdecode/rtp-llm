"""PPU DeepEP low-latency router implementations."""

import os
from typing import Any, Optional

import torch

from rtp_llm.models_py.distributed.deepep_wrapper import DeepEPWrapper
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
    ExpertTokensMetadata,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
    DEEPEP_QUANT_BLOCK_SIZE,
    SUPPORTED_HIDDEN_SIZES,
    DeepEpLowLatencyRouter,
    LowLatencyDispatchState,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class PpuDeepEpLowLatencyRouterNoQuant(DeepEpLowLatencyRouter):
    """BF16 DeepEP low-latency router without CUDA SM checks."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) is None)
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)


class PpuDeepEpLowLatencyRouterW8A8Int8(DeepEpLowLatencyRouter):
    """Let PPU DeepEP quantize BF16 dispatch inputs to per-token INT8."""

    QUANT_METHOD = "W8A8_INT8_PER_CHANNEL_COMPRESSED"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)
        self._use_int8_dispatch = True

    def _dispatch_recv(
        self, state: LowLatencyDispatchState
    ) -> ExpertForwardPayload:
        self._wait_transfer(state.event, state.hook)
        expert_x = state.expert_x
        if not isinstance(expert_x, tuple) or len(expert_x) != 2:
            raise TypeError(
                "PPU DeepEP INT8 dispatch must return (expert_x, expert_x_scale)"
            )
        expert_x, expert_x_scale = expert_x
        if expert_x.dtype != torch.int8 or expert_x_scale.dtype != torch.float32:
            raise TypeError(
                "PPU DeepEP INT8 dispatch returned unexpected dtypes: "
                f"{expert_x.dtype}/{expert_x_scale.dtype}"
            )
        return ExpertForwardPayload(
            expert_x=expert_x,
            expert_x_scale=expert_x_scale,
            expert_x_origin_dtype=state.origin_dtype,
            expert_topk_ids=state.topk_ids,
            expert_topk_weights=state.topk_weights,
            expert_tokens_meta=ExpertTokensMetadata(
                expected_m=state.expected_m,
                expert_num_tokens=state.expert_num_tokens,
            ),
        )

    def prepare_send(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> LowLatencyDispatchState:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("W8A8 DeepEP LL expects an unquantized BF16 input")
        num_tokens, hidden_size = a1.size()
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"W8A8 DeepEP LL expects BF16 input, got {a1.dtype}")
        if (
            hidden_size not in SUPPORTED_HIDDEN_SIZES
            or hidden_size % DEEPEP_QUANT_BLOCK_SIZE != 0
        ):
            raise ValueError(f"DeepEP LL does not support hidden_size={hidden_size}")
        if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != num_tokens:
            raise ValueError(
                f"topk ids/weights shapes {topk_ids.shape}/{topk_weights.shape} "
                f"do not match token count {num_tokens}"
            )
        if topk_ids.shape[1] != self._num_topk:
            raise ValueError(
                f"topk width {topk_ids.shape[1]} != DeepEP topk {self._num_topk}"
            )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        if tp_num_tokens > self._num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"tp_num_tokens {tp_num_tokens} exceeds LL capacity "
                f"{self._num_max_dispatch_tokens_per_rank}"
            )
        if self._handle is not None:
            raise RuntimeError("DeepEP LL dispatch handle was not finalized")

        tp_input, tp_topk_ids, tp_topk_weights = self._prepare_pre_tp_slice(
            a1, topk_ids, topk_weights
        )
        dispatch_args = {
            "x": tp_input,
            "topk_idx": tp_topk_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": False,
            "use_int8": self._use_int8_dispatch,
            "quant_size": hidden_size,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        return self._dispatch_send(dispatch_args, tp_topk_weights)


def deepep_fp8_dispatch_enabled() -> bool:
    """Whether the all-to-all carries FP8 instead of BF16.

    On by default: this is sglang-for-sail's production configuration on the
    same hardware (``use_fp8=True, quant_size=128``), and it halves the
    all-to-all payload. Measured end-to-end on this build the difference is
    neutral to slightly negative, so set ``RTP_GLM_DEEPEP_FP8=0`` to A/B it
    against the BF16 dispatch; the executor accepts either side.
    """
    return os.environ.get("RTP_GLM_DEEPEP_FP8", "1") == "1"


class PpuDeepEpLowLatencyRouterFp8PerBlock(DeepEpLowLatencyRouter):
    """DeepEP low-latency dispatch for FP8 per-block checkpoints.

    Two modes, picked by RTP_GLM_DEEPEP_FP8:

    * BF16 (default): dispatch BF16 activations and let the FP8 masked executor
      quantize per-128-group on the receiving side.
    * FP8: dispatch already-quantized FP8 with ``quant_size=128``, halving the
      all-to-all payload, and hand the dispatched scales straight to the masked
      GEMM. This is sglang-for-sail's production configuration on this same
      hardware -- ``use_fp8=True, quant_size=128``.

    PPU is not Blackwell, so sglang leaves ``round_scale``/``use_ue8m0`` off and
    DeepEP returns FP32 scales; the executor validates that rather than assuming.
    """

    QUANT_METHOD = "FP8_PER_BLOCK"

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(config, quant_config)
        self._use_fp8_dispatch = deepep_fp8_dispatch_enabled()

    def prepare_send(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> LowLatencyDispatchState:
        if a1_scale is not None or a2_scale is not None:
            raise ValueError("FP8 DeepEP LL expects an unquantized BF16 input")
        num_tokens, hidden_size = a1.size()
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"FP8 DeepEP LL expects BF16 input, got {a1.dtype}")
        if (
            hidden_size not in SUPPORTED_HIDDEN_SIZES
            or hidden_size % DEEPEP_QUANT_BLOCK_SIZE != 0
        ):
            raise ValueError(f"DeepEP LL does not support hidden_size={hidden_size}")
        if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != num_tokens:
            raise ValueError(
                f"topk ids/weights shapes {topk_ids.shape}/{topk_weights.shape} "
                f"do not match token count {num_tokens}"
            )
        if topk_ids.shape[1] != self._num_topk:
            raise ValueError(
                f"topk width {topk_ids.shape[1]} != DeepEP topk {self._num_topk}"
            )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        if tp_num_tokens > self._num_max_dispatch_tokens_per_rank:
            raise ValueError(
                f"tp_num_tokens {tp_num_tokens} exceeds LL capacity "
                f"{self._num_max_dispatch_tokens_per_rank}"
            )
        if self._handle is not None:
            raise RuntimeError("DeepEP LL dispatch handle was not finalized")

        tp_input, tp_topk_ids, tp_topk_weights = self._prepare_pre_tp_slice(
            a1, topk_ids, topk_weights
        )
        dispatch_args = {
            "x": tp_input,
            "topk_idx": tp_topk_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": self._use_fp8_dispatch,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        if self._use_fp8_dispatch:
            # Only meaningful alongside use_fp8, and passing it unconditionally
            # would make the switch-off path differ from the BF16 dispatch we
            # already measured -- the point of the switch is a clean A/B.
            dispatch_args["quant_size"] = DEEPEP_QUANT_BLOCK_SIZE
        return self._dispatch_send(dispatch_args, tp_topk_weights)
