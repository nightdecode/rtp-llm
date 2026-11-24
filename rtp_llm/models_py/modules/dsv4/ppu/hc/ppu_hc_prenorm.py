"""Deterministic M890P HC PRE adapter over the existing DeepGEMM kernel.

HcPrenormGemm already supports disjoint split-K output planes. Use that
specialization and PyTorch's fixed-axis reduction instead of global FP32
atomicAdd. No GEMM, quantization, or HC-mix kernel is reimplemented here.
Explicit opt-in via DSV4_MHC_PRE_GEMM_BACKEND=deepgemm_deterministic.
"""

import logging
import os
from functools import lru_cache
from pathlib import Path

import torch
import triton
import triton.language as tl

_TEMPLATE = """
using gemm_t = deep_gemm::HcPrenormGemm<
    {N}, {K}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {NUM_SPLITS},
    {FAST_BF16_TO_TF32}, false>;
gemm_t::run(d, sqr_sum, m, a, b, stream, num_sms, smem_size);
"""

_WORKSPACE_TEMPLATE = """
using gemm_t = deep_gemm::HcPrenormGemm<
    {N}, {K}, {BLOCK_M}, {BLOCK_N}, {BLOCK_K}, {NUM_SPLITS},
    {FAST_BF16_TO_TF32}, {BLOCK_M} * 2, 2, (1u << 22)>;
gemm_t::run(reduced, reduced_squares, m, a, b, stream, num_sms, smem_size,
            d, sqr_sum, counter);
"""


@triton.jit
def _split_reduce_kernel(
    PARTIALS, SQUARES, OUT, SQRSUM,
    M: tl.constexpr, N: tl.constexpr, S: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """把 split-K 的 S 个分片一次加完：同时产出 [M, N] 的和与 [M] 的平方和。

    两件事放一个 kernel 是因为它们读的是同一次 GEMM 的产物、形状又都极小
    （S=64、N=24，batch 1 时总共 6 KB）。分成两次 torch.sum 的话，
    代价几乎全是启动与归约往返：实测每次 7.4 和 11.6 微秒，而每个子层都要做一次，
    45 层两个子层就是每步 180 个 kernel、约 1.9 毫秒。
    """
    row = tl.program_id(0)
    lane = tl.arange(0, BLOCK_N)
    keep = lane < N
    plane = tl.arange(0, S)
    # [S, N]：第 s 个分片第 row 个 token 的 N 个输出
    offs = plane[:, None] * (M * N) + row * N + lane[None, :]
    vals = tl.load(PARTIALS + offs, mask=keep[None, :], other=0.0)
    tl.store(OUT + row * N + lane, tl.sum(vals, 0), mask=keep)
    sq = tl.load(SQUARES + plane * M + row)
    tl.store(SQRSUM + row, tl.sum(sq, 0))


def _fused_split_reduce(partials, squares, out, sqrsum) -> bool:
    """一个 kernel 做完两次 split-K 归约；形状不合适时返回 False 让调用方退回 torch。"""
    if os.environ.get("RTP_HC_FUSED_SPLIT_REDUCE", "1") == "0":
        return False
    splits, m, n = partials.shape
    # tl.arange 要求 2 的幂；S 由 launch_geometry 固定给 64，这里只做防御。
    if splits & (splits - 1) or splits > 1024 or n > 128:
        return False
    if squares.shape != (splits, m) or not partials.is_contiguous() or not squares.is_contiguous():
        return False
    _split_reduce_kernel[(m,)](
        partials, squares, out[0], sqrsum[0],
        M=m, N=n, S=splits,
        BLOCK_N=triton.next_power_of_2(n),
        num_warps=4,
    )
    return True


def launch_geometry(m: int, k: int, n: int, *, is_ppu1v5: bool = True) -> dict:
    """Match the vendor's M890P AIU and 810E CuTe tile alignment."""
    if m <= 0 or k != 16384 or n != 24:
        raise ValueError(
            "deterministic PPU HC requires M>0, K=16384, N=24 (DSV4 Flash)"
        )
    # The fused_890p implementation uses a 64x64 AIU load and two
    # 4096-element BF16 stages. The generic template accepts larger tiles,
    # but BLOCK_M=128 reads beyond a stage and BLOCK_K=128 skips half of K.
    block_k = 64
    n_alignment = 8 if is_ppu1v5 else 16
    splits = 64 if m <= 256 else (32 if m <= 512 or m >= 8192 else 16)
    return {
        "N": n,
        "K": k,
        "BLOCK_M": 64,
        "BLOCK_N": min(((n + n_alignment - 1) // n_alignment) * n_alignment, 32),
        "BLOCK_K": block_k,
        "NUM_SPLITS": max(1, min(splits, k // block_k)),
        # BF16 is exactly representable in TF32. The redundant rounding path
        # emits an invalid sreg instruction on the SDK 2.2 810E backend.
        "FAST_BF16_TO_TF32": "true",
    }


@lru_cache(maxsize=None)
def _require_device(index: int) -> None:
    logging.info(
        "DSV4_HC_BACKEND deepgemm_deterministic: existing DeepGEMM partials + torch.sum"
    )


def tf32_hc_prenorm_gemm(x, weight, out, sqrsum, requested_splits=None):
    """Fill the existing single-plane FP32 output ABI on the current stream.

    Like the vendor PPU function, actual split count is selected from M/K, not
    requested_splits. Scratch is call-owned (no cross-stream singleton); graph
    capture retains stable allocation addresses through the normal graph pool.
    """
    config = _check_inputs(x, weight)
    m, n = x.shape[0], weight.shape[0]
    tensors = (x, weight, out, sqrsum)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("HC tensors must be contiguous and on the same PPU")
    if any(t.dtype != torch.float32 for t in (out, sqrsum)):
        raise TypeError("HC requires FP32 outputs")
    if tuple(out.shape) != (1, m, n) or tuple(sqrsum.shape) != (1, m):
        raise ValueError("HC outputs must retain the single-plane ABI")
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if len(set(storage)) != len(storage):
        raise ValueError("HC input, weight and outputs must not alias")
    partials, squares = _launch_partials(x, weight, config)
    if not _fused_split_reduce(partials, squares, out, sqrsum):
        torch.sum(partials, dim=0, out=out[0])
        torch.sum(squares, dim=0, out=sqrsum[0])
    return out


def _check_inputs(x, weight):
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError("HC input and weight must be matrices")
    m, k = x.shape
    n, weight_k = weight.shape
    if k != weight_k:
        raise ValueError("HC input/weight K mismatch")
    tensors = (x, weight)
    if any(
        t.device != x.device or not t.is_cuda or not t.is_contiguous() for t in tensors
    ):
        raise ValueError("HC tensors must be contiguous and on the same PPU")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.float32:
        raise TypeError("HC requires BF16 input and FP32 weight")
    storage = [t.untyped_storage().data_ptr() for t in tensors]
    if len(set(storage)) != len(storage):
        raise ValueError("HC input and weight must not alias")
    _require_device(x.device.index)
    return launch_geometry(
        m, k, n, is_ppu1v5=torch.cuda.get_device_capability(x.device) == (8, 9)
    )


def tf32_hc_prenorm_partials(x, weight):
    """Expose existing split planes for consumers with their own reduction."""
    return _launch_partials(x, weight, _check_inputs(x, weight))


def _launch_partials(x, weight, config):

    import deep_gemm
    from deep_gemm.jit_kernels.tuner import jit_tuner
    from deep_gemm.jit_kernels.utils import get_num_sms

    splits = config["NUM_SPLITS"]
    m, n = x.shape[0], weight.shape[0]
    partials = torch.empty((splits, m, n), dtype=torch.float32, device=x.device)
    squares = torch.empty((splits, m), dtype=torch.float32, device=x.device)
    smem = 2 * (
        config["BLOCK_M"] * (config["BLOCK_K"] + 8) * 2
        + config["BLOCK_N"] * (config["BLOCK_K"] + 4) * 4
    )
    args = (
        x,
        weight,
        partials,
        squares,
        m,
        torch.cuda.current_stream(x.device),
        get_num_sms(),
        smem,
    )
    workspace_abi = (
        Path(deep_gemm.__file__).parent
        / "include/deep_gemm/impls/tf32_hc_prenorm_gemm.cuh"
    ).is_file()
    extra_defs = ()
    if workspace_abi:
        # The new kernel writes split-major workspace before its own reduction.
        # Keep returning those planes to the existing HC reduction consumers.
        reduced = torch.empty((m, n), dtype=torch.float32, device=x.device)
        reduced_squares = torch.empty((m,), dtype=torch.float32, device=x.device)
        counter = torch.zeros(
            ((m + config["BLOCK_M"] - 1) // config["BLOCK_M"],),
            dtype=torch.int32,
            device=x.device,
        )
        args += (reduced, reduced_squares, counter)
        extra_defs = (
            ("reduced", torch.float32),
            ("reduced_squares", torch.float32),
            ("counter", torch.int32),
        )
    runtime = jit_tuner.compile_and_tune(
        # Vendor's in-memory tuner omits template from its key. A distinct
        # name is mandatory: otherwise it can return the cached atomic kernel.
        name=("rtp_dsv4_hc_prenorm_workspace" if workspace_abi
              else "rtp_dsv4_hc_prenorm_partials"),
        keys=config,
        space=(),
        includes=(
            "<deep_gemm/impls/tf32_hc_prenorm_gemm.cuh>"
            if workspace_abi
            else '\"../deep_gemm/tf32_hc_prenorm_gemm.cuh\"',
        ),
        arg_defs=(
            ("a", torch.bfloat16),
            ("b", torch.float32),
            ("d", torch.float32),
            ("sqr_sum", torch.float32),
            ("m", int),
            ("stream", torch.cuda.Stream),
            ("num_sms", int),
            ("smem_size", int),
        ) + extra_defs,
        template=_WORKSPACE_TEMPLATE if workspace_abi else _TEMPLATE,
        args=args,
        jit_include_dir="actlize_v1.0.0",
    )
    runtime(*args)
    return partials, squares
