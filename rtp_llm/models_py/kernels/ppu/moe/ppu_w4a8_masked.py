"""Masked W4A8 routed experts for DeepEP low-latency decode on PPU.

DeepEP delivers E4M3 rows with one FP32 scale per token. Both projections use
DeepGEMM's FP8 x MXFP4 masked grouped GEMM (W4A8_MXFP4_MMA weights), and the
SwiGLU between them requantizes each valid row to E4M3 with one FP32 scale.
The GEMM comes from the isolated Route A DeepGEMM extension
(``deep_gemm_route_a._C``), which already coexists with RTP's ``deep_gemm``.
"""

import triton
import triton.language as tl
import torch


@triton.jit
def _swiglu_masked_token_fp8(
    gate_up,
    payload,
    scales,
    counts,
    in_e,
    in_m,
    out_e,
    out_m,
    sc_e,
    H: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert = tl.program_id(1)
    rows = tl.load(counts + expert)
    col = tl.arange(0, BLOCK)
    for row in tl.range(tl.program_id(0), rows, tl.num_programs(0)):
        base = gate_up + expert.to(tl.int64) * in_e + row.to(tl.int64) * in_m
        gate = tl.load(base + col, col < H, 0).to(tl.float32)
        up = tl.load(base + H + col, col < H, 0).to(tl.float32)
        if LIMIT > 0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
        # Round the activation once at the BF16 boundary, as the other
        # producers do, before the per-token E4M3 quantization.
        value = ((gate * tl.sigmoid(gate)) * up).to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(tl.max(tl.abs(value), 0), 1.0e-4)
        quant = tl.minimum(tl.maximum(value * tl.div_rn(448.0, amax), -448.0), 448.0)
        out = payload + expert.to(tl.int64) * out_e + row.to(tl.int64) * out_m
        tl.store(out + col, quant, col < H)
        tl.store(scales + expert.to(tl.int64) * sc_e + row, amax / 448.0)


# Programs per expert. Each program walks rows ``pid, pid + programs, ...``;
# decode experts receive ~4-40 rows. One card, 32 experts with 4-20 rows:
# 4 programs take 14.4 us, 32 take 8.2 us (the engine timeline showed 32 us
# per layer with 4).
_SWIGLU_PROGRAMS = 32


def swiglu_masked_token_fp8(gate_up, counts, swiglu_limit=None, programs=_SWIGLU_PROGRAMS, out=None):
    """[E, M, 2H] BF16 gate|up -> ([E, M, H] E4M3, [E, M, 1] FP32); valid rows only."""
    e, m, two_h = gate_up.shape
    h = two_h // 2
    if out is None:
        payload = torch.empty((e, m, h), dtype=torch.float8_e4m3fn, device=gate_up.device)
        scales = torch.empty((e, m, 1), dtype=torch.float32, device=gate_up.device)
    else:
        payload, scales = out
    _swiglu_masked_token_fp8[(programs, e)](
        gate_up,
        payload,
        scales,
        counts,
        gate_up.stride(0),
        gate_up.stride(1),
        payload.stride(0),
        payload.stride(1),
        scales.stride(0),
        H=h,
        LIMIT=float(swiglu_limit or 0.0),
        BLOCK=triton.next_power_of_2(h),
        num_warps=4,
    )
    return payload, scales


def w4a8_experts_masked(
    expert_x,
    weight13,
    weight2,
    counts,
    *,
    inter,
    expected_m,
    swiglu_limit=None,
    out=None,
    configs=None,
    workspace=None,
):
    """Two masked W4A8 GEMMs around SwiGLU; rows beyond ``counts`` are undefined.

    ``expert_x`` is (E4M3 [E, M, D], FP32 [E, M, 1]); each weight is the
    (weights, scales, inv_scale) triple from W4A8_MXFP4_MMA preprocessing.
    ``workspace`` optionally supplies (gate_up [E, M, 2I] BF16, hidden
    [E, M, I] E4M3, hidden_scale [E, M, 1] FP32) so no per-call allocation is
    needed, e.g. on a side stream inside CUDA Graph capture.
    """
    from deep_gemm_route_a import _C as route_a

    x, scale = expert_x
    w13, s13, inv13 = weight13
    w2, s2, inv2 = weight2
    e, m, d = x.shape
    if scale.shape != (e, m, 1) or counts.shape != (e,) or counts.dtype != torch.int32:
        raise ValueError("W4A8 experts require [E,M,D] E4M3 rows, [E,M,1] scales and int32 counts")
    kwargs = {"configs": configs} if configs is not None else {}
    if workspace is None:
        gate_up = torch.empty((e, m, 2 * inter), dtype=torch.bfloat16, device=x.device)
        hidden_out = None
    else:
        gate_up, hidden_payload, hidden_scale = workspace
        if gate_up.shape != (e, m, 2 * inter) or hidden_payload.shape != (e, m, inter):
            raise ValueError("W4A8 workspace does not match the dispatched expert rows")
        hidden_out = (hidden_payload, hidden_scale)
    route_a.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
        (x, scale), (w13, s13), gate_up, counts, expected_m, inv_scale=inv13, **kwargs
    )
    hidden = swiglu_masked_token_fp8(gate_up, counts, swiglu_limit, out=hidden_out)
    if out is None:
        out = torch.empty((e, m, d), dtype=torch.bfloat16, device=x.device)
    route_a.m_grouped_gemm_fp8_fp4_bf16_nt_masked(
        hidden, (w2, s2), out, counts, expected_m, inv_scale=inv2, **kwargs
    )
    return out
