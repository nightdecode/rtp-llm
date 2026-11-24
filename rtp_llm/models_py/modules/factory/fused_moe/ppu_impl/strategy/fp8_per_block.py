"""PPU fused-MoE strategies for FP8 per-block [128, 128] checkpoints."""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class PpuFp8PerBlockDpNormalDeepGemmStrategy(MoeStrategy):
    """FP8 DeepEP Normal communication with FP8 expert GEMMs."""

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_dp_normal_deepgemm"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # CP prefill exposes one attention partition per rank and runs eagerly.
        checker.check(config.tp_size == 1)
        checker.check(not config.enable_cuda_graph)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmFp8HybridExecutor
        from ..routers.deepep_normal_router import PpuDeepepNormalRouterFp8PerBlock

        return StrategyAttributes(
            router_class=PpuDeepepNormalRouterFp8PerBlock,
            executor_class=DeepGemmFp8HybridExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class PpuFp8PerBlockEpLowLatencyDeepGemmStrategy(MoeStrategy):
    """FP8 DeepEP low-latency communication with FP8 masked expert GEMMs."""

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_ep_low_latency_deepgemm"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy in ("auto", cls.STRATEGY))
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        # DP decode exposes one attention partition per rank.
        checker.check(config.tp_size == 1)
        # DeepEP strategies require expert-parallel dispatch; exclude pure-TP mode.
        checker.check(not resolver.use_all_gather(config))

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_masked_executor import DeepGemmFp8MaskedExecutor
        from ..routers.deepep_low_latency_router import (
            PpuDeepEpLowLatencyRouterFp8PerBlock,
        )

        return StrategyAttributes(
            router_class=PpuDeepEpLowLatencyRouterFp8PerBlock,
            executor_class=DeepGemmFp8MaskedExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )


class PpuFp8PerBlockPureCpDeepGemmStrategy(MoeStrategy):
    """CP prefill without all-to-all: all-gather tokens over the CP group, run
    the local experts with FP8 expert GEMMs, reduce-scatter the partial sums.

    Opt-in only (``--moe_strategy fp8_per_block_pure_cp``), as on CUDA: the
    name is what server_config_setup keys on to keep use_all_gather on under
    CP, and ``auto`` keeps choosing DeepEP normal.
    """

    QUANT_METHOD = "FP8_PER_BLOCK"
    STRATEGY = "fp8_per_block_pure_cp"
    strategy_name = STRATEGY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        checker.check(config.moe_strategy == cls.STRATEGY)
        checker.check(resolver.get_quant_method(config) == cls.QUANT_METHOD)
        checker.check(config.dp_size == 1)
        checker.check(resolver.is_cp_equal_ep(config))
        checker.check(config.ep_size > 1)
        checker.check(config.parallelism_config.prefill_cp_config.is_enabled())
        checker.check(resolver.use_all_gather(config))
        checker.check(not config.enable_cuda_graph)

    def get_attributes(self) -> StrategyAttributes:
        from ..executors.deepgemm_hybrid_executor import DeepGemmFp8HybridExecutor
        from ..routers.pure_cp_router import PpuPureCpRouterFp8PerBlock

        return StrategyAttributes(
            router_class=PpuPureCpRouterFp8PerBlock,
            executor_class=DeepGemmFp8HybridExecutor,
            quant_config=FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
        )
