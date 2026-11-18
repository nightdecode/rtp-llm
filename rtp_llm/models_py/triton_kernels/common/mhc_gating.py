"""Fused gating for multi-stream hyper-connections (MHC).

MHC mixes ``hc_mult`` residual streams before and after every sublayer. The
mixing weights come from one projection of the streams, after which the
combination matrix is made doubly stochastic by a Sinkhorn loop. Written in
torch that loop is ``2 * (sinkhorn_iters - 1)`` normalizations, each a reduce
plus two elementwise ops on a tensor of shape ``[tokens, n, n]``::

    comb = comb.softmax(-1) + hc_eps
    comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)

With ``n = 4`` and ``sinkhorn_iters = 20`` that is ~114 kernels for 16 numbers
per token, twice per layer. On a 45-layer model decoding one token at a time it
dominates the step: measured 10991 reduce/elementwise launches and 41.5 ms out
of a 94.8 ms step on PPU, versus 16.1 ms for every matmul in the same step.

This module runs the whole tail — scale/bias, softmax, the Sinkhorn loop, and
the two sigmoid gates that are sliced out of the same projection — in one
Triton kernel.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _mhc_gating_kernel(
    mixes_ptr,  # [T, n * (n + 2)] float32, contiguous
    base_ptr,  # [n * (n + 2)]    float32
    scale_ptr,  # [3]              float32
    pre_ptr,  # [T, n]           float32 out
    post_ptr,  # [T, n]           float32 out
    comb_ptr,  # [T, n, n]        float32 out
    T,
    hc_eps,
    post_mult,
    N: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_t = offs_t < T
    offs_n = tl.arange(0, N)
    row = N * (N + 2)

    scale_pre = tl.load(scale_ptr + 0)
    scale_post = tl.load(scale_ptr + 1)
    scale_comb = tl.load(scale_ptr + 2)

    # ---- pre / post gates: both are [T, n] slices of the same projection ----
    base_pre = tl.load(base_ptr + offs_n)
    base_post = tl.load(base_ptr + N + offs_n)
    idx_in = offs_t[:, None] * row + offs_n[None, :]
    idx_gate_out = offs_t[:, None] * N + offs_n[None, :]
    pre_raw = tl.load(mixes_ptr + idx_in, mask=mask_t[:, None], other=0.0)
    post_raw = tl.load(mixes_ptr + idx_in + N, mask=mask_t[:, None], other=0.0)

    pre = 1.0 / (1.0 + tl.exp(-(pre_raw * scale_pre + base_pre[None, :]))) + hc_eps
    post = post_mult / (1.0 + tl.exp(-(post_raw * scale_post + base_post[None, :])))
    tl.store(pre_ptr + idx_gate_out, pre, mask=mask_t[:, None])
    tl.store(post_ptr + idx_gate_out, post, mask=mask_t[:, None])

    # ---- combination matrix: [T, n, n] ----
    offs_r = tl.arange(0, N)
    offs_c = tl.arange(0, N)
    # The projection packs [pre | post | comb] into one row of width N*(N+2),
    # so the comb block starts at 2*N; the output is its own [T, N, N] tensor.
    inner = offs_r[:, None] * N + offs_c[None, :]
    idx_comb = offs_t[:, None, None] * row + (2 * N + inner)[None, :, :]
    idx_out = offs_t[:, None, None] * (N * N) + inner[None, :, :]
    comb = tl.load(mixes_ptr + idx_comb, mask=mask_t[:, None, None], other=0.0)
    comb_base = tl.load(base_ptr + 2 * N + inner)
    comb = comb * scale_comb + comb_base[None, :, :]

    # softmax over the last dim, then alternate column / row normalization.
    # Padding lanes hold zeros, so softmax gives them 1/N and every later
    # division has `+ hc_eps` in the denominator: no NaN, and their stores are
    # masked off anyway.
    m = tl.max(comb, axis=2)
    e = tl.exp(comb - m[:, :, None])
    comb = e / tl.sum(e, axis=2)[:, :, None] + hc_eps
    comb = comb / (tl.sum(comb, axis=1)[:, None, :] + hc_eps)
    # Plain `range`, not `tl.static_range`: 19 unrolled iterations over a
    # [BLOCK_T, N, N] tile would bloat the kernel and risk register spills for
    # the prefill block size, and the loop body is identical every time.
    for _ in range(SINKHORN_ITERS - 1):
        comb = comb / (tl.sum(comb, axis=2)[:, :, None] + hc_eps)
        comb = comb / (tl.sum(comb, axis=1)[:, None, :] + hc_eps)

    tl.store(comb_ptr + idx_out, comb, mask=mask_t[:, None, None])


def mhc_gating(
    mixes: torch.Tensor,
    hc_base: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_mult: int,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
):
    """Fused tail of the MHC pre-sublayer mixing.

    Args:
        mixes: ``[T, hc_mult * (hc_mult + 2)]`` projection output, float32.
        hc_base: ``[hc_mult * (hc_mult + 2)]`` biases.
        hc_scale: ``[3]`` scales for (pre, post, comb).
        hc_mult: number of residual streams, ``n``. Must be a power of two.
        hc_eps / sinkhorn_iters / post_mult: as in the torch reference.

    Returns:
        ``(pre [T, n], post [T, n], comb [T, n, n])``, all float32.
    """
    n = int(hc_mult)
    assert n & (n - 1) == 0, f"hc_mult must be a power of two, got {n}"
    assert mixes.shape[-1] == n * (n + 2), (
        f"mixes last dim {mixes.shape[-1]} != hc_mult*(hc_mult+2) = {n * (n + 2)}"
    )
    mixes = mixes.contiguous().float()
    base = hc_base.contiguous().float()
    scale = hc_scale.contiguous().float()
    tokens = mixes.shape[0]

    pre = torch.empty((tokens, n), dtype=torch.float32, device=mixes.device)
    post = torch.empty((tokens, n), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((tokens, n, n), dtype=torch.float32, device=mixes.device)
    if tokens == 0:
        return pre, post, comb

    # One program per BLOCK_T tokens. Decode runs a single token, so cap the
    # block at the actual token count to avoid launching empty lanes; prefill
    # wants enough programs to fill the device.
    block_t = min(64, max(1, triton.next_power_of_2(tokens)))
    grid = (triton.cdiv(tokens, block_t),)
    _mhc_gating_kernel[grid](
        mixes,
        base,
        scale,
        pre,
        post,
        comb,
        tokens,
        float(hc_eps),
        float(post_mult),
        N=n,
        SINKHORN_ITERS=int(sinkhorn_iters),
        BLOCK_T=block_t,
    )
    return pre, post, comb
