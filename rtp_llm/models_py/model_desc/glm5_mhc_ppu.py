"""PPU fused MHC for GLM-5.3-Flash, reusing the DSV4 HC kernels.

The reference MHC in ``glm5_next`` is a chain of torch ops: an FP32 RMS
statistic, an FP32 mixing GEMM, the fused gating kernel, a weighted stream
reduction, and in ``hc_post`` a [T, hc, hc, dim] product reduced over the
stream axis. That is about 11 launches for ``hc_pre`` plus 5 for ``hc_post``,
four times per layer, which dominates the small-kernel count of a decode step.

DSV4 already ships PPU kernels for exactly this math
(``rtp_llm/models_py/kernels/ppu/hc``), and GLM-5.3-Flash shares the shape the
deterministic prenorm GEMM is specialized for: hc_mult 4, hidden 4096, so
K = hc_mult * hidden = 16384 and N = hc_mult * (hc_mult + 2) = 24. The post
multiplier is 2.0 on both models. This module adapts the GLM calling
convention onto those kernels and refuses anything else, so an unsupported
shape falls back to the torch reference instead of silently miscomputing.

Note on numerics: the fused prenorm GEMM accumulates in TF32 where the torch
reference uses FP32, and the fused RMSNorm takes its denominator from the FP32
stream mix while the numerator keeps the BF16 activation boundary, where the
torch path rounds to BF16 before flashinfer's rmsnorm sees it. Both match the
sglang reference, so the fused path tracks the comparison baseline rather than
the torch chain.
"""

import logging
from typing import Dict, Optional, Tuple

import torch

# Geometry the deterministic PPU prenorm GEMM is specialized for; see
# rtp_llm/models_py/modules/dsv4/ppu/hc/ppu_hc_prenorm.py:launch_geometry.
_PRENORM_K = 16384
_PRENORM_N = 24


def fused_hc_supported(
    hc_mult: int,
    hidden_size: int,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    norm_eps: float,
) -> bool:
    """Whether the PPU HC kernels accept this model's shapes and weights.

    Every rejection is logged. These checks used to return False silently, which
    made "the fused path is off" indistinguishable from "the fused path is on and
    did not help" -- a whole benchmark round was spent on that ambiguity.
    """
    mix_width = hc_mult * (hc_mult + 2)

    def reject(reason: str) -> bool:
        logging.info("GLM-5.3 fused MHC off, using the torch path: %s", reason)
        return False

    if hc_mult * hidden_size != _PRENORM_K or mix_width != _PRENORM_N:
        return reject(
            f"kernel is specialised for K={_PRENORM_K}, N={_PRENORM_N}; "
            f"this model needs K={hc_mult * hidden_size}, N={mix_width}"
        )
    if tuple(hc_fn.shape) != (mix_width, hc_mult * hidden_size):
        return reject(
            f"hc_fn shape {tuple(hc_fn.shape)} != "
            f"{(mix_width, hc_mult * hidden_size)}"
        )
    # 只看元素个数，不要求形状逐项相等：权重加载器会给这两个一维权重补一个长度 1
    # 的尾维度（ckpt 里 hc_attn_scale 是 [3]，加载进来是 [3, 1]）。原来按 (3,) 严格
    # 比形状，45 层全部静默回退到 torch 路，实测白跑了一轮基准。
    # 真正要紧的是元素个数与内存连续，kernel 收的是一段平坦缓冲。
    if hc_scale.numel() != 3 or hc_base.numel() != mix_width:
        return reject(
            f"hc_scale 有 {hc_scale.numel()} 个元素（要 3 个）/ "
            f"hc_base 有 {hc_base.numel()} 个元素（要 {mix_width} 个）"
        )
    for name, t in (("hc_fn", hc_fn), ("hc_scale", hc_scale), ("hc_base", hc_base)):
        if t.dtype != torch.float32 or not t.is_cuda or not t.is_contiguous():
            return reject(
                f"{name} must be contiguous FP32 on the device, got dtype="
                f"{t.dtype}, is_cuda={t.is_cuda}, contiguous={t.is_contiguous()}"
            )
    # The pre kernel folds the sublayer RMSNorm in only for dense BF16 weights.
    if (
        norm_weight.dtype != torch.bfloat16
        or tuple(norm_weight.shape) != (hidden_size,)
        or not norm_weight.is_contiguous()
    ):
        return reject(
            "the sublayer RMSNorm weight must be a contiguous BF16 "
            f"[{hidden_size}], got dtype={norm_weight.dtype}, "
            f"shape={tuple(norm_weight.shape)}, "
            f"contiguous={norm_weight.is_contiguous()}"
        )
    return _probe_once(
        hc_mult, hidden_size, hc_fn.device,
        rms_eps, hc_eps, sinkhorn_iters, post_mult, norm_eps,
    )


_PROBE: Dict[Tuple, bool] = {}


def _probe_once(
    hc_mult: int,
    hidden_size: int,
    device: torch.device,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    norm_eps: float,
) -> bool:
    """Import the kernels and actually launch them once, memoized per shape.

    An import check alone is not enough: TileLang and DeepGEMM compile on
    first call, so a toolchain that imports cleanly but cannot build for this
    device would set the fused path on every layer and then raise inside the
    first real forward, where there is no fallback left. Running one tiny
    launch here moves that failure into the probe -- and warms the JIT, which
    has to happen outside CUDA graph capture anyway.

    The epsilons, the Sinkhorn count and the post multiplier are compile-time
    keys of the TileLang kernel, so the probe passes the real ones: a probe
    with different values would build a different kernel and prove nothing
    about the one the model runs. The split count is keyed on the token count
    instead, which the engine's per-capture warmup covers.
    """
    key = (
        int(hc_mult), int(hidden_size), str(device),
        float(rms_eps), float(hc_eps), int(sinkhorn_iters),
        float(post_mult), float(norm_eps),
    )
    cached = _PROBE.get(key)
    if cached is not None:
        return cached
    ok = True
    try:
        mhc_pre_big_fuse, mhc_post_fwd, tf32_hc_prenorm_gemm = _kernels()
        mix_width = hc_mult * (hc_mult + 2)
        residual = torch.zeros(
            1, hc_mult, hidden_size, dtype=torch.bfloat16, device=device
        )
        fn = torch.zeros(
            mix_width, hc_mult * hidden_size, dtype=torch.float32, device=device
        )
        scale = torch.ones(3, dtype=torch.float32, device=device)
        base = torch.zeros(mix_width, dtype=torch.float32, device=device)
        norm = torch.ones(hidden_size, dtype=torch.bfloat16, device=device)
        post, comb, normed = mhc_pre_big_fuse(
            residual, fn, scale, base,
            rms_eps=rms_eps, mhc_pre_eps=hc_eps, mhc_sinkhorn_eps=hc_eps,
            mhc_post_mult_value=post_mult, sinkhorn_repeat=sinkhorn_iters,
            backend="deepgemm_deterministic",
            prenorm_gemm=tf32_hc_prenorm_gemm,
            prenorm_output="reduced",
            norm_weight=norm, norm_eps=norm_eps,
        )
        mhc_post_fwd(
            normed.view(1, 1, hidden_size),
            residual.view(1, 1, hc_mult, hidden_size),
            post.view(1, 1, hc_mult, 1),
            comb.view(1, 1, hc_mult, hc_mult),
            enable_pdl=False,
        )
    except Exception as exc:  # capability probe: fall back, never fail startup
        logging.info("GLM-5.3 fused MHC unavailable, using the torch path: %s", exc)
        ok = False
    _PROBE[key] = ok
    return ok


def _kernels():
    # Prepare the vendored TileLang runtime before importing its JIT kernels.
    # tilelang_kernels preloads libz3 and fixes up the TVM tmpdir at import
    # time; both HC kernel modules do a module-level ``import tilelang``, and
    # without the prep the dlopen happens later inside TileLang's rewriter --
    # that is, during the first real forward rather than here, where the
    # capability probe could still fall back. Every other entry point into
    # these kernels does the same (see dsv4/ppu/hc/ppu_hc.py and
    # dsv4/hc/mhc_tilelang.py, and the import-order regression test in
    # dsv4/test/test_dsv4_kernel_jit_warmup.py).
    from rtp_llm.models_py.modules.dsv4 import tilelang_kernels  # noqa: F401
    from rtp_llm.models_py.kernels.ppu.hc.post_kernel import mhc_post_fwd
    from rtp_llm.models_py.kernels.ppu.hc.pre_big_fuse import mhc_pre_big_fuse
    from rtp_llm.models_py.modules.dsv4.ppu.hc.ppu_hc_prenorm import (
        tf32_hc_prenorm_gemm,
    )

    return mhc_pre_big_fuse, mhc_post_fwd, tf32_hc_prenorm_gemm


def hc_pre_norm_fused(
    streams: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    post_mult: float,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``hc_pre`` plus the sublayer RMSNorm in one kernel pair.

    Returns (normed [T, hidden] BF16, comb [T, hc, hc] FP32,
    post [T, hc, 1] FP32); the latter two feed the matching ``hc_post``.
    """
    mhc_pre_big_fuse, _, tf32_hc_prenorm_gemm = _kernels()
    num_tokens = streams.shape[0]
    hidden = streams.shape[-1] // hc_mult
    if num_tokens == 0:
        # The deterministic prenorm GEMM requires M > 0; mirror PpuHCUnit's
        # empty guard rather than launching it.
        return (
            streams.new_empty((0, hidden)),
            streams.new_empty((0, hc_mult, hc_mult), dtype=torch.float32),
            streams.new_empty((0, hc_mult, 1), dtype=torch.float32),
        )
    residual = streams.view(num_tokens, hc_mult, hidden)
    post, comb, normed = mhc_pre_big_fuse(
        residual,
        hc_fn,
        # 加载器给这两个一维权重补了长度 1 的尾维度（[3] → [3, 1]），kernel 要平坦的。
        # 两个都是连续张量，view 不复制。
        hc_scale.view(-1),
        hc_base.view(-1),
        rms_eps=rms_eps,
        mhc_pre_eps=hc_eps,
        mhc_sinkhorn_eps=hc_eps,
        mhc_post_mult_value=post_mult,
        sinkhorn_repeat=sinkhorn_iters,
        backend="deepgemm_deterministic",
        prenorm_gemm=tf32_hc_prenorm_gemm,
        prenorm_output="reduced",
        norm_weight=norm_weight,
        norm_eps=norm_eps,
    )
    return normed, comb, post


def hc_post_fused(
    x: torch.Tensor,
    streams: torch.Tensor,
    h_post: torch.Tensor,
    h_res: torch.Tensor,
    hc_mult: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """``hc_post`` as one kernel. Accepts h_post as [T, hc] or [T, hc, 1]."""
    _, mhc_post_fwd, _ = _kernels()
    num_tokens = x.shape[0]
    hidden = x.shape[-1]
    if num_tokens == 0:
        return streams
    # The kernel asserts a dense layout. Attention and MoE outputs are normally
    # already contiguous; make it explicit rather than fail the assert.
    x = x.contiguous()
    written = mhc_post_fwd(
        x.view(1, num_tokens, hidden),
        streams.view(1, num_tokens, hc_mult, hidden),
        h_post.reshape(1, num_tokens, hc_mult, 1),
        h_res.reshape(1, num_tokens, hc_mult, hc_mult),
        out=None if out is None else out.view(1, num_tokens, hc_mult, hidden),
        # The PPU HC post kernel runs without programmatic dependent launch;
        # see PpuHCUnit, which refuses DSV4_MHC_POST_PDL != "0".
        enable_pdl=False,
    )
    return written.view(num_tokens, hc_mult * hidden)
