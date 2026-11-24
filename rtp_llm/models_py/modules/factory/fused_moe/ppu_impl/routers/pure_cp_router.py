"""PPU pure-CP MoE router for FP8 per-block [128, 128] checkpoints.

Under prefill context parallelism each rank holds an equal slice of the
sequence. The MoE needs no all-to-all: ``prepare`` all-gathers the tokens over
the TP (= CP) group, each rank runs its own ``expert_num / ep_size`` experts
over the whole batch, and ``finalize`` reduce-scatters the partial outputs back
to the sequence slices. This matches what sglang-for-sail's GLM-5.3-Flash CP
prefill does without all-to-all (there the experts are TP-sharded instead of
EP-sharded; both sum partials over the group).

The CUDA router already implements that contract; PPU differs only in the
activation quantizer and in how the executor is told about expert ids.
"""

from typing import Optional, Tuple

import torch

from rtp_llm.models_py.kernels.ppu.fp8_quant import fp8_quantize_128
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_cp_router import (
    PureCpRouterFp8PerBlock,
)


class PpuPureCpRouterFp8PerBlock(PureCpRouterFp8PerBlock):
    """Pure-CP FP8 per-block router using the PPU activation quantizer."""

    def _do_quant(
        self, a1: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if a1.dtype != torch.bfloat16:
            raise ValueError(f"FP8 pure-CP router expects BF16 input, got {a1.dtype}")
        return fp8_quantize_128(a1)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        payload = super().prepare(a1, a1_scale, a2_scale, topk_weights, topk_ids)
        # recompute_topk_ids_sum_expert_count already subtracted this rank's
        # first expert id and wrote -1 for experts owned by other ranks. The
        # PPU executor would otherwise subtract the offset a second time and
        # drop every token on ranks other than 0.
        payload.expert_ids_are_local = True
        return payload
