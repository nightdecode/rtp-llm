import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch

from rtp_llm.models_py.distributed.collective_torch import Group, all_gather
from rtp_llm.models_py.distributed.deepep_wrapper import (
    DeepEPMode,
    DeepEPWrapper,
    DeepepWrapperConfig,
)
from rtp_llm.models_py.kernels.cuda.deepgemm_wrapper import is_deep_gemm_e8m0_used
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    ExpertTokensMetadata,
    FinalizeArgs,
    FusedMoeDataRouter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import RouterType
from rtp_llm.models_py.utils.arch import get_sm

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
# DeepEP Low-Latency supports hidden sizes
SUPPORTED_HIDDEN_SIZES = [1536, 2048, 2560, 3072, 4096, 5120, 6144, 7168, 8192]

_RECV_HOOK_ENV = "DEEPEP_LL_RECV_HOOK"


def deepep_ll_recv_hook_enabled() -> bool:
    """Whether to run the low-latency all-to-all as send now, receive on hook.

    With ``return_recv_hook=True`` ``low_latency_dispatch`` only issues the send
    and hands back a ``hook`` that issues the receive and waits for it. Anything
    the caller runs between the two no longer waits behind the transfer. With it
    off the call is synchronous and the whole transfer sits on the critical path.
    """

    return os.environ.get(_RECV_HOOK_ENV, "1") != "0"


@dataclass
class LowLatencyDispatchState:
    """A low-latency dispatch whose receive side has not been waited on yet.

    Produced by :meth:`DeepEpLowLatencyRouter.prepare_send` and consumed by
    :meth:`DeepEpLowLatencyRouter.prepare_recv`.
    """

    # (expert_x, expert_x_scale) when the dispatch quantizes, else a tensor.
    expert_x: Any
    expert_num_tokens: torch.Tensor
    event: Any
    hook: Optional[Callable[[], None]]
    expected_m: int
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    origin_dtype: torch.dtype


class DeepEpLowLatencyRouter(FusedMoeDataRouter):
    """
    A data router for Mixture-of-Experts that utilizes deep_ep's low-latency communication primitives.

    This router dispatches tokens to experts and receives results from experts across all ep ranks.
    """

    @classmethod
    def router_type(cls):
        return RouterType.DEEPEP_LOW_LATENCY

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        """Check if DeepEpLowLatencyRouter can handle the configuration"""
        from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
            MoeConfigResolver,
        )

        resolver = MoeConfigResolver()
        checker.check(get_sm()[0] >= 9)
        checker.check(resolver.is_ep_enabled(config))
        checker.check(resolver.use_low_latency(config))
        checker.check(DeepEPWrapper.supported())

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(config, quant_config)

        # Determine use_fp8_dispatch based on quant_config
        use_fp8_dispatch = (
            quant_config.is_quantized
            and quant_config.quant_dtype == torch.float8_e4m3fn
        )

        # DeepEpLowLatency-specific initialization
        self._num_experts = config.expert_num
        self._ll_num_max_token_per_rank = (
            DeepepWrapperConfig.calc_low_latency_max_token_per_rank(
                config.ll_num_max_token, config.tp_size, config.quant_config
            )
        )
        deepep_config = DeepepWrapperConfig.from_config_adapter(
            self.config, self._ll_num_max_token_per_rank
        )
        wrapper = DeepEPWrapper.get_instance(deepep_config)
        assert (
            wrapper.mode == DeepEPMode.LOW_LATENCY
        ), "DeepEP mode should be LOW_LATENCY"
        self._buffer = wrapper.buffer
        self._num_topk = wrapper.num_topk
        self._num_max_dispatch_tokens_per_rank = wrapper.ll_num_max_token_per_rank
        self._use_fp8_dispatch = use_fp8_dispatch
        self._zero_copy = False
        # Kept False in both call styles: with the receive hook the send is
        # already asynchronous, and without it the call must stay synchronous.
        self._async_finish = False
        self._return_recv_hook = deepep_ll_recv_hook_enabled()
        self._handle: Optional[Tuple[Any, ...]] = None

    @property
    def handle(self) -> Optional[Tuple[Any, ...]]:
        return self._handle

    @property
    def supports_dispatch_overlap(self) -> bool:
        # Splitting the call is only worth anything when the receive is deferred
        # to the hook, so one switch selects both halves of the change and an
        # A/B run compares the two call styles exactly.
        return self._return_recv_hook

    @property
    def supports_combine_overlap(self) -> bool:
        # Same switch as the dispatch window: without the receive hook the
        # combine call is synchronous and there is no window to fill.
        return self._return_recv_hook

    def _prepare_pre_tp_slice(
        self,
        a1: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Slice the dispatch activation, topk_ids and topk_weights by tp.
        Args:
            a1 (torch.Tensor): The dispatch activation tensor.
            topk_ids (torch.Tensor): The topk ids tensor.
            topk_weights (torch.Tensor): The topk weights tensor.
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The sliced dispatch activation, topk_ids and topk_weights.
        """
        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)
        # Slice by tp
        tp_size = self.config.tp_size
        tp_rank = self.config.tp_rank
        token_num = a1.size(0)
        tp_token_size = (token_num + tp_size - 1) // tp_size
        slice_begin = min(tp_token_size * tp_rank, token_num)
        slice_size = min(token_num - slice_begin, tp_token_size)
        tp_dispatch_input = torch.narrow(a1, 0, slice_begin, slice_size)
        tp_topk_ids = torch.narrow(topk_ids, 0, slice_begin, slice_size)
        tp_topk_weights = torch.narrow(topk_weights, 0, slice_begin, slice_size)

        return tp_dispatch_input, tp_topk_ids, tp_topk_weights

    def _dispatch_send(
        self, dispatch_args: dict[str, Any], tp_topk_weights: torch.Tensor
    ) -> LowLatencyDispatchState:
        """Issue the low-latency dispatch without waiting for the receive side.

        Args:
            dispatch_args (dict[str, Any]): Arguments for dispatching tokens to experts.
            tp_topk_weights (torch.Tensor): Topk weights tensor for this tp rank.
        """
        # Calculate expected_m
        tp_num_tokens = dispatch_args["x"].size(0)
        expected_m = max(
            1,
            int(
                tp_num_tokens
                * self.config.ep_size
                * self._num_topk
                // self._num_experts
            ),
        )

        # Dispatch tokens
        expert_x, expert_num_tokens, self._handle, event, hook = (
            self._buffer.low_latency_dispatch(**dispatch_args)
        )
        return LowLatencyDispatchState(
            expert_x=expert_x,
            expert_num_tokens=expert_num_tokens,
            event=event,
            hook=hook,
            expected_m=expected_m,
            topk_ids=dispatch_args["topk_idx"],
            topk_weights=tp_topk_weights,
            origin_dtype=dispatch_args["x"].dtype,
        )

    def _wait_transfer(self, event: Any, hook: Optional[Callable[[], None]]) -> None:
        """Make the transfer's result visible to the current stream."""
        if self._return_recv_hook:
            if hook is None:
                raise RuntimeError(
                    "DeepEP low-latency transfer did not return a receive hook; "
                    f"set {_RECV_HOOK_ENV}=0 if this deep_ep build lacks the mode"
                )
            hook()
        elif self._async_finish:
            event.current_stream_wait()

    def _split_dispatch_output(
        self, expert_x: Any
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self._use_fp8_dispatch:
            assert isinstance(expert_x, tuple), "expert_x should be a tuple"
            return expert_x[0], expert_x[1]
        assert isinstance(expert_x, torch.Tensor), "expert_x should be a tensor"
        return expert_x, None

    def _dispatch_recv(self, state: LowLatencyDispatchState) -> ExpertForwardPayload:
        """Wait for the dispatched tokens and build the expert payload."""
        self._wait_transfer(state.event, state.hook)
        expert_x, expert_x_scale = self._split_dispatch_output(state.expert_x)

        # Return expert forward payload
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

    def prepare_recv(self, state: Any) -> ExpertForwardPayload:
        return self._dispatch_recv(state)

    def prepare(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> ExpertForwardPayload:
        return self.prepare_recv(
            self.prepare_send(a1, a1_scale, a2_scale, topk_weights, topk_ids)
        )

    def prepare_send(
        self,
        a1: torch.Tensor,
        a1_scale: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> LowLatencyDispatchState:
        """
        Dispatches tokens to experts across all ep ranks.
        """
        # Check payload data
        num_tokens, hidden_size = a1.size()
        assert (
            hidden_size in SUPPORTED_HIDDEN_SIZES
            and hidden_size % DEEPEP_QUANT_BLOCK_SIZE == 0
        )
        tp_num_tokens = (num_tokens + self.config.tp_size - 1) // self.config.tp_size
        assert (
            tp_num_tokens <= self._num_max_dispatch_tokens_per_rank
        ), f"tp_num_tokens {tp_num_tokens} > self._num_max_dispatch_tokens_per_rank {self._num_max_dispatch_tokens_per_rank}"
        assert topk_ids.size(0) == num_tokens and topk_weights.size(0) == num_tokens
        assert (
            topk_ids.size(1) == self._num_topk
            and topk_weights.size(1) == self._num_topk
        )
        # Check quantization
        if self._use_fp8_dispatch:
            assert (
                self.quant_config.is_block_quantized
            ), "upstream DeepEP Low-Latency only supports FP8 block quantization"
        elif self.quant_config.is_per_group_fp4:
            pass
        else:
            assert not self.quant_config.is_quantized
        # Check handle
        assert self._handle is None

        # Slice dispatch activation, topk_ids and topk_weights by tp
        tp_dispatch_input, tp_topk_ids, tp_topk_weights = self._prepare_pre_tp_slice(
            a1, topk_ids, topk_weights
        )

        # Prepare dispatch basic arguments
        dispatch_args = {
            "x": tp_dispatch_input,
            "topk_idx": tp_topk_ids,
            "num_max_dispatch_tokens_per_rank": self._num_max_dispatch_tokens_per_rank,
            "num_experts": self._num_experts,
            "use_fp8": self._use_fp8_dispatch,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        # Set quantization config for DeepEP low latency dispatch
        if self.quant_config.is_block_quantized and is_deep_gemm_e8m0_used():
            dispatch_args.update({"round_scale": True, "use_ue8m0": True})

        return self._dispatch_send(dispatch_args, tp_topk_weights)

    def _normal_finalize(
        self,
        combine_args: dict[str, Any],
        overlap_fn: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Normal finalize for DeepEP Low-Latency.
        Args:
            combine_args (dict[str, Any]): Arguments for combining expert outputs.
            overlap_fn: work to run while the combine transfer is in flight. It
                must not read the combined output.
        """
        # Normal finalize
        combined_x, event, hook = self._buffer.low_latency_combine(**combine_args)
        if overlap_fn is not None:
            overlap_fn()
        self._wait_transfer(event, hook)

        return combined_x

    def _finalize_post_tp_gather(
        self,
        combined_x: torch.Tensor,
        extra_finalize_args: Optional[FinalizeArgs],
    ) -> torch.Tensor:
        """Finalize post tp gather for DeepEP Low-Latency.
        Args:
            combined_x (torch.Tensor): Combined output from all tp ranks.
            extra_finalize_args (Optional[FinalizeArgs]): Extra finalize arguments.
        """
        # Check input data
        assert combined_x.dim() == 2
        assert extra_finalize_args is not None
        assert "original_num_tokens" in extra_finalize_args
        # Get original number of tokens
        tp_size = self.config.tp_size
        original_num_tokens = extra_finalize_args["original_num_tokens"]
        tp_token_size = (original_num_tokens + tp_size - 1) // tp_size
        if tp_size > 1:
            # combine_x.size(0) might be 0
            if combined_x.size(0) < tp_token_size:
                # Pad combined output if needed
                padding_combined_x = torch.empty(
                    size=(tp_token_size - combined_x.size(0), combined_x.size(1)),
                    device=combined_x.device,
                    dtype=combined_x.dtype,
                )
                combined_x = torch.cat([combined_x, padding_combined_x], dim=0)
            # Gather combined output from all tp ranks
            gatherd_output = all_gather(combined_x, group=Group.TP).reshape(
                tp_size * tp_token_size, -1
            )
            combined_x = gatherd_output[:original_num_tokens, :]
        return combined_x

    def finalize(
        self,
        payload: CombineForwardPayload,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        extra_finalize_args: Optional[FinalizeArgs],
        combine_overlap_fn: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """
        Combines expert outputs back to all original ranks.
        Weight application and reduction happens in the combine kernel.
        """
        # Check handle
        assert (
            self._handle is not None
        ), "DeepEP Low-latency combine handle is missing for finalize()."

        # Convert topk_ids to int64
        topk_ids = topk_ids.to(torch.int64)

        # Prepare combine basic arguments
        combine_args: Dict[str, Any] = {
            "x": payload.fused_expert_output,
            "topk_idx": topk_ids,
            "topk_weights": topk_weights,
            "handle": self._handle,
            "zero_copy": self._zero_copy,
            "async_finish": self._async_finish,
            "return_recv_hook": self._return_recv_hook,
        }
        # Normal finalize
        combined_x = self._normal_finalize(combine_args, combine_overlap_fn)

        # Finalize post tp gather
        combined_x = self._finalize_post_tp_gather(combined_x, extra_finalize_args)
        # reset handle
        self._handle = None

        return combined_x
