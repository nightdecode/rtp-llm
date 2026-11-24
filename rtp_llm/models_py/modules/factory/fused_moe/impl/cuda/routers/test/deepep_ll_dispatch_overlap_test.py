"""DeepEP low-latency split-dispatch (send now, receive on hook) contract tests."""

import os
from types import SimpleNamespace
from unittest import TestCase, main

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
    DeepEpLowLatencyRouter,
    deepep_ll_recv_hook_enabled,
)


class _RecordingBuffer:
    """Stands in for a deep_ep low-latency buffer and records the call order."""

    def __init__(self, trace: list, with_hook: bool):
        self.trace = trace
        self.with_hook = with_hook

    def _transfer(self, phase: str, result):
        self.trace.append(f"{phase}_send")
        if not self.with_hook:
            self.trace.append(f"{phase}_recv")
            return result + (None, None)

        def hook():
            self.trace.append(f"{phase}_recv")

        return result + (None, hook)

    def low_latency_dispatch(self, **kwargs):
        x = kwargs["x"]
        expert_x = torch.zeros((2, x.size(0), x.size(1)), dtype=x.dtype)
        expert_num_tokens = torch.zeros((2,), dtype=torch.int32)
        return self._transfer("dispatch", (expert_x, expert_num_tokens, ("handle",)))

    def low_latency_combine(self, **kwargs):
        combined = kwargs["x"].sum(dim=0)
        return self._transfer("combine", (combined,))


def _new_router(trace: list, with_hook: bool) -> DeepEpLowLatencyRouter:
    router = object.__new__(DeepEpLowLatencyRouter)
    router.config = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2)
    router.quant_config = FusedMoEQuantConfig()
    router._buffer = _RecordingBuffer(trace, with_hook)
    router._num_experts = 4
    router._num_topk = 2
    router._num_max_dispatch_tokens_per_rank = 8
    router._use_fp8_dispatch = False
    router._zero_copy = False
    router._async_finish = False
    router._return_recv_hook = with_hook
    router._handle = None
    return router


def _dispatch_args(router: DeepEpLowLatencyRouter, tokens: int = 3) -> dict:
    return {
        "x": torch.randn((tokens, 128), dtype=torch.bfloat16),
        "topk_idx": torch.zeros((tokens, router._num_topk), dtype=torch.int64),
        "num_max_dispatch_tokens_per_rank": router._num_max_dispatch_tokens_per_rank,
        "num_experts": router._num_experts,
        "use_fp8": False,
        "async_finish": router._async_finish,
        "return_recv_hook": router._return_recv_hook,
    }


class DeepEpLlDispatchOverlapTest(TestCase):
    def test_receive_is_deferred_until_prepare_recv(self):
        trace: list = []
        router = _new_router(trace, with_hook=True)
        args = _dispatch_args(router)
        weights = torch.ones((args["x"].size(0), router._num_topk))

        state = router._dispatch_send(args, weights)
        self.assertEqual(trace, ["dispatch_send"])

        trace.append("overlapped_work")
        payload = router._dispatch_recv(state)

        self.assertEqual(
            trace, ["dispatch_send", "overlapped_work", "dispatch_recv"]
        )
        self.assertEqual(payload.expert_tokens_meta.expected_m, 3)
        self.assertIs(payload.expert_topk_weights, weights)
        self.assertEqual(router.handle, ("handle",))

    def test_without_hook_the_transfer_completes_inside_the_call(self):
        trace: list = []
        router = _new_router(trace, with_hook=False)
        args = _dispatch_args(router)
        weights = torch.ones((args["x"].size(0), router._num_topk))

        router._dispatch_send(args, weights)
        self.assertEqual(trace, ["dispatch_send", "dispatch_recv"])
        self.assertFalse(router.supports_dispatch_overlap)

    def test_overlap_is_advertised_only_with_the_hook(self):
        self.assertTrue(_new_router([], with_hook=True).supports_dispatch_overlap)
        self.assertFalse(_new_router([], with_hook=False).supports_dispatch_overlap)

    def test_missing_hook_is_reported_instead_of_silently_dropping_data(self):
        router = _new_router([], with_hook=True)
        router._buffer.with_hook = False
        args = _dispatch_args(router)
        weights = torch.ones((args["x"].size(0), router._num_topk))

        state = router._dispatch_send(args, weights)
        with self.assertRaisesRegex(RuntimeError, "receive hook"):
            router._dispatch_recv(state)

    def test_env_switch(self):
        previous = os.environ.get("DEEPEP_LL_RECV_HOOK")
        try:
            os.environ["DEEPEP_LL_RECV_HOOK"] = "0"
            self.assertFalse(deepep_ll_recv_hook_enabled())
            os.environ["DEEPEP_LL_RECV_HOOK"] = "1"
            self.assertTrue(deepep_ll_recv_hook_enabled())
            del os.environ["DEEPEP_LL_RECV_HOOK"]
            self.assertTrue(deepep_ll_recv_hook_enabled())
        finally:
            if previous is None:
                os.environ.pop("DEEPEP_LL_RECV_HOOK", None)
            else:
                os.environ["DEEPEP_LL_RECV_HOOK"] = previous


if __name__ == "__main__":
    main()
