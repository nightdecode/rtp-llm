"""DeepEP low-latency routed experts with W4A8 (E4M3 x MXFP4) masked GEMMs.

The engine's DeepEP LL group dispatches E4M3 rows with one FP32 scale per
token; both projections run DeepGEMM's FP8 x MXFP4 masked grouped GEMM on the
checkpoint's MXFP4 experts. Activation scales are per token, so this path is
W4A8 but not bit-exact with Route A's block-128 UE8M0 activation contract.
The masked W4A8 GEMM is compiled into the isolated Route A extension
(``deep_gemm_route_a._C``), which coexists with RTP's ``deep_gemm``.
"""

import logging
import os

import torch

from rtp_llm.models_py.modules.dsv4.ppu.moe.ppu_deepep_fp4 import (
    PpuDeepEPFP4Strategy,
    pad_topk,
    take_raw_routed_mxfp4_weights,
)

# (num_sms, block_m, block_n, block_k, warp_m, warp_n, warp_k, stages, n_expand).
# One PPU card, 32 experts, G1 (4096 x 4096): with 4-20 valid rows per expert
# the default selection takes 174 us and this tile 165 us; with 0-40 rows
# (skewed routing) BlockM16 tiles revisit and re-decode the weights for a
# second row block (258-269 us) while this BlockM32 tile takes 190 us.
_DECODE_TILE = (32, 128, 128, 32, 32, 128, 2, 1)


class _SharedExpertResources:
    """One GEMM stream and one workspace for every routed-expert layer.

    Route A caches each W4FA8 launch plan per caller stream and refuses to
    build one during capture. The engine warms on its compute stream but
    captures on a separate stream, so the expert GEMMs run on a private stream
    that both the warmup and the captured forward use. Its workspace is
    allocated once in eager mode: blocks freed on a side stream during capture
    are not reused, so per-call buffers would grow every captured layer. MoE
    layers already run serially (they share the engine's one DeepEP buffer).
    """

    def __init__(self):
        self.stream = None
        self.workspace = None

    def bind(self, device, experts, rows, inter):
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=device)
        shape = (experts, rows, inter)
        if self.workspace is None or self.workspace[1].shape != shape:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DeepEP W4A8 workspace must be allocated before graph capture")
            self.workspace = (
                torch.empty((experts, rows, 2 * inter), dtype=torch.bfloat16, device=device),
                torch.empty(shape, dtype=torch.float8_e4m3fn, device=device),
                torch.empty((experts, rows, 1), dtype=torch.float32, device=device),
            )
        return self.stream, self.workspace


_RESOURCES = _SharedExpertResources()


class PpuDeepEPW4A8Strategy(PpuDeepEPFP4Strategy):
    name = "ppu_deepep_w4a8"

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        from deep_gemm_route_a import _C as route_a

        if not all(
            hasattr(route_a, name)
            for name in ("m_grouped_gemm_fp8_fp4_bf16_nt_masked", "preprocess_weights_and_scales")
        ):
            raise RuntimeError("DeepEP W4A8 requires Route A's masked FP8 x MXFP4 GEMM")
        self._deep_gemm = route_a
        tile = os.environ.get("DSV4_PPU_W4A8_MOE_TILE", "decode")
        if tile not in ("decode", "auto"):
            raise ValueError("DSV4_PPU_W4A8_MOE_TILE must be decode or auto")
        num_sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
        self._configs = (num_sms, *_DECODE_TILE) if tile == "decode" else None
        self._logged_layout = False

    def setup_weights(self, layer_weights):
        w13, s13, w2, s2 = take_raw_routed_mxfp4_weights(self.cfg, layer_weights)
        mode = self._deep_gemm.PreprocessMode.W4A8_MXFP4_MMA
        self._w13 = self._deep_gemm.preprocess_weights_and_scales(
            mode, False, w13, s13.view(torch.uint8)
        )
        self._w2 = self._deep_gemm.preprocess_weights_and_scales(
            mode, False, w2, s2.view(torch.uint8)
        )

    def forward(self, x, weights, indices, *, active_token_mask=None):
        from rtp_llm.models_py.kernels.ppu.moe.ppu_w4a8_masked import (
            w4a8_experts_masked,
        )

        wrapper = self._wrapper if self._wrapper is not None else self._bind_wrapper()
        buffer = wrapper.buffer
        if indices.dtype != torch.int64:
            indices = indices.to(torch.int64)
        indices, weights = pad_topk(indices, weights, active_token_mask)
        indices, weights = indices.contiguous(), weights.contiguous()
        (payload, scale), counts, handle, _, _ = buffer.low_latency_dispatch(
            x=x.contiguous(),
            topk_idx=indices,
            num_max_dispatch_tokens_per_rank=wrapper.ll_num_max_token_per_rank,
            num_experts=self.cfg.n_routed_experts,
            use_fp8=True,
            round_scale=False,
            use_ue8m0=False,
            quant_size=x.shape[1],
            async_finish=False,
            return_recv_hook=False,
        )
        if scale.dim() == 3 and scale.shape[1] == 1 and scale.shape[2] != 1:
            scale = scale.transpose(1, 2).contiguous()
        if not self._logged_layout:
            self._logged_layout = True
            logging.info(
                "[ppu-deepep-w4a8] dispatch payload %s %s, scale %s %s, counts %s, tile %s",
                tuple(payload.shape), payload.dtype, tuple(scale.shape), scale.dtype,
                tuple(counts.shape), self._configs,
            )
        combine_slot = buffer.get_next_low_latency_combine_buffer(handle)
        stream, workspace = _RESOURCES.bind(
            x.device, payload.shape[0], payload.shape[1], self.cfg.moe_inter_dim
        )
        caller = torch.cuda.current_stream()
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            w4a8_experts_masked(
                (payload, scale),
                self._w13,
                self._w2,
                counts,
                inter=self.cfg.moe_inter_dim,
                expected_m=self.expected_rows(x.shape[0]),
                swiglu_limit=self.cfg.swiglu_limit if self.cfg.swiglu_limit > 0 else None,
                out=combine_slot,
                configs=self._configs,
                workspace=workspace,
            )
        caller.wait_stream(stream)
        combined, _, _ = buffer.low_latency_combine(
            x=combine_slot,
            topk_idx=indices,
            topk_weights=weights,
            handle=handle,
            zero_copy=True,
            async_finish=False,
            return_recv_hook=False,
        )
        return combined.to(self.output_dtype)
