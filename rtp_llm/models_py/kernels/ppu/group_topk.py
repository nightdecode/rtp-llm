"""Exact small-batch GLM one-group routing on PPU."""

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.jit
def _group_topk_kernel(LOGITS, BIAS, IDS, WEIGHTS, N: tl.constexpr, N_PAD: tl.constexpr):
    """N 个专家，lane 数补到 2 的幂 N_PAD（Triton 的块大小必须是 2 的幂）。

    N == N_PAD 时每一步都与补位之前逐字相同（打包常数 2*N_PAD、N_PAD-1 在 N_PAD=256
    时就是原来的 512、255），所以 256 专家的模型结果不变。
    """
    row = tl.program_id(0)
    index = tl.arange(0, N_PAD)
    live = index < N
    logits = tl.load(LOGITS + row * N + index, mask=live, other=0.0)
    # Match Torch's FP32 sigmoid, including correctly rounded division. The
    # approximate tl.sigmoid/reciprocal can change both routes and their weights.
    score = tl.div_rn(1.0, 1.0 + libdevice.exp(-logits))
    # 补位 lane 必须判为「不可选」。给它们 -inf 而不是 0：0 加上 bias 之后是个有限值，
    # 会跟真实专家一起参与取最大值和 top-k。
    corrected = tl.where(live, score + tl.load(BIAS + index, mask=live, other=0.0), -float("inf"))
    finite = (corrected == corrected) & (tl.abs(corrected) != float("inf"))
    maximum = tl.max(tl.where(finite, corrected, -float("inf")), 0)
    count = tl.sum((finite & (corrected == maximum)).to(tl.int32), 0)
    second = tl.where(
        count > 1,
        maximum,
        tl.max(tl.where(finite & (corrected < maximum), corrected, -float("inf")), 0),
    )
    group_score = maximum + second
    # Native GroupTopK rejects nonfinite groups and uses lowest<float> as its
    # invalid-group sentinel, even when that sentinel is a finite group sum.
    lowest = -3.4028234663852886e38
    valid_group = (
        (group_score == group_score)
        & (tl.abs(group_score) != float("inf"))
        & (group_score != lowest)
    )
    choice = tl.where(finite, corrected, lowest)
    bits = choice.to(tl.int32, bitcast=True)
    bits = tl.where((bits & 0x7FFFFFFF) == 0, 0, bits)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(tl.int64)
    # Native WarpSelect is stable: smaller expert IDs win equal FP32 scores.
    # 低位放 N_PAD-1-index 当稳定排序的次级键，所以要留 log2(N_PAD)+1 位：乘 2*N_PAD，
    # 取回时按 2*N_PAD-1 掩码。ordered 最大 2^31，乘 1024 仍在 int64 内。
    keys = tl.sort(ordered * (2 * N_PAD) + (N_PAD - 1 - index), descending=True)
    lane = tl.arange(0, 8)
    selected = ((N_PAD - 1) - (tl.gather(keys, lane, 0) & (2 * N_PAD - 1))).to(tl.int32)
    value = tl.gather(score, selected, 0)
    valid_value = (value == value) & (tl.abs(value) != float("inf"))
    valid = valid_group & (tl.sum(valid_value.to(tl.int32), 0) == 8)
    # This reduction matches the native 32-lane sum of eight selected scores
    # followed by zero lanes. Keep division and scaling as separate operations.
    total = tl.sum(value, 0) + 1e-20
    weight = tl.div_rn(value, total) * 2.5
    tl.store(IDS + row * 8 + lane, tl.where(valid, selected, lane))
    tl.store(WEIGHTS + row * 8 + lane, tl.where(valid, weight, 0.125))


def try_fused_group_topk(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    scores: torch.Tensor,
    correction_bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
) -> bool:
    """Run only the qualified layout/configuration; otherwise keep native routing."""
    n_experts = int(scores.shape[1]) if scores.ndim == 2 else 0
    # 补到 2 的幂：256 → 256（行为与补位前逐字相同），288 → 512。
    # 上限 1024 只是 tl.sort 的块大小上限，不是算法限制。
    n_pad = 1 << max(0, (n_experts - 1)).bit_length() if n_experts else 0
    if (
        n_group != 1
        or topk_group != 1
        or topk != 8
        or not renormalize
        or routed_scaling_factor != 2.5
        or scores.ndim != 2
        or n_experts < topk
        or n_pad > 1024
        or scores.shape[0] > 1024
        or scores.dtype != torch.float32
        or not scores.is_cuda
        or not scores.is_contiguous()
        or correction_bias.shape != (n_experts,)
        or correction_bias.dtype != torch.float32
        or not correction_bias.is_contiguous()
        or topk_weights.shape != (scores.shape[0], 8)
        or topk_weights.dtype != torch.float32
        or not topk_weights.is_contiguous()
        or topk_ids.shape != topk_weights.shape
        or topk_ids.dtype not in (torch.int32, torch.int64)
        or not topk_ids.is_contiguous()
        or correction_bias.device != scores.device
        or topk_weights.device != scores.device
        or topk_ids.device != scores.device
    ):
        return False
    if scores.shape[0]:
        _group_topk_kernel[(scores.shape[0],)](
            scores,
            correction_bias,
            topk_ids,
            topk_weights,
            N=n_experts,
            N_PAD=n_pad,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return True
