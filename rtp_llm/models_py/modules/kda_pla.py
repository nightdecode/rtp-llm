"""T-Head PLA decode kernels for KDA, bridged onto rtp-llm's block-mapped state.

PLA is the PPU vendor's linear-attention library (hand-written C++ plus PPU
inline assembly). sglang-for-sail runs KDA on PPU through it; we ran Triton.
On the vendor's own same-device sweep for our exact shape (GLM-5.3-Flash,
TP1, 64 heads, head_dim 128) the hand-written decode kernel beats FLA-Triton
by 2.28x at batch 1 and 1.76x at batch 15.

The one real impedance mismatch is where the recurrent state lives.

  PLA (and sglang, and vLLM): a state pool plus one slot index per request.
  ``initial_state_source[initial_state_indices[n]]`` is read, updated and
  written back in place.

  rtp-llm: the state is checkpointed per cache block, so a step reads one
  slot and writes a *different* one. ``fused_recurrent.py`` resolves both
  offsets inside the Triton kernel from ``sequence_lengths``:

      load  = block_map[n, (seq_len - 2) // seq_size_per_block]
      store = block_map[n, (seq_len - 1) // seq_size_per_block]

With seq_size_per_block 256 those two differ on one decode step in 256. We
keep rtp-llm's semantics by rotating the state (load slot -> store slot)
before handing PLA the store slot; the rotation kernel exits without touching
memory when the two agree, which is the overwhelmingly common case.

``intermediate_states_buffer`` is *not* a shortcut for the rotation: passing
it flips PLA into MTP target-verify mode, which is a different contract.

Everything here is opt-in via RTP_GLM_KDA_PLA and every rejection is logged.
A silently disabled fast path cost us a full benchmark round once already.
"""

import logging
import os
import sys
from typing import Optional, Tuple

import torch

from rtp_llm.models_py.triton_kernels.fla.block import kda_pool_v_first

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - local CPU checkouts have no triton
    triton = None
    tl = None

# Head counts the vendor's decode kernels are known-good on. The whitelist in
# fused_kda_decode_mega_host.cpp also lists 6/12/24/48/96, but reports/README.md
# records those same counts crashing out of bounds in the k_last path, and on
# 890P the crash leaves the device needing `ppu-smi -r`. Only accept powers of
# two, which is what both of our parallel splits produce (decode TP1 -> 64,
# prefill TP8 -> 8).
_SUPPORTED_NUM_HEADS = (4, 8, 16, 64)
_SUPPORTED_HEAD_DIM = 128

_IMPORT_ERROR: Optional[str] = None
_k_last = None


def _load_pla() -> bool:
    """Import PLA once, remembering why it failed if it did.

    PLA is built into the vendor's own virtualenv, not ours. RTP_GLM_PLA_PATH
    names that site-packages directory; it is appended (never prepended) so a
    second copy of torch or triton over there can never shadow ours. The
    vendor extension binds through pybind11 rather than registering torch
    custom ops, so it loads under our torch even though it was compiled
    against a different minor version -- verified by scripts/pla/probe_pla_abi.py.
    """
    global _k_last, _IMPORT_ERROR
    if _k_last is not None:
        return True
    if _IMPORT_ERROR is not None:
        return False
    extra = os.environ.get("RTP_GLM_PLA_PATH")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        from pla.decode import fused_sigmoid_gating_delta_rule_forward_k_last

        _k_last = fused_sigmoid_gating_delta_rule_forward_k_last
        return True
    except Exception as exc:  # noqa: BLE001 - any import failure means fall back
        _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def pla_kda_enabled() -> bool:
    return os.environ.get("RTP_GLM_KDA_PLA", "0") == "1"


def pla_kda_min_batch() -> int:
    """Smallest per-replica batch at which the vendor kernel is worth using.

    On an idle card PLA beats our Triton recurrence everywhere -- 3.1x at
    batch 1 down to 1.4x at batch 16. End to end it does not: measured against
    the same build with PLA off, a decode step costs

        concurrency    1      4     16     64    128
        delta ms    +1.43  +1.81  +0.94  -0.91  -0.01

    (positive = slower with PLA; per-replica batch is concurrency / 8). The
    isolated and end-to-end numbers disagree and the reason is not yet
    established -- pool size and our own slot/rotation plumbing were both
    measured and ruled out (the rotation kernel is 1.6 us x 34 layers =
    0.055 ms/step). Until it is understood, take the kernel only where it
    measurably wins. 0 enables it unconditionally.
    """
    try:
        return int(os.environ.get("RTP_GLM_KDA_PLA_MIN_BATCH", "8"))
    except ValueError:
        return 8


def pla_kda_supported(
    num_v_heads: int,
    head_dim: int,
    state: Optional[torch.Tensor],
    block_map: Optional[torch.Tensor],
    sequence_lengths: Optional[torch.Tensor],
) -> bool:
    """Capability probe. Logs the reason for every rejection."""

    def reject(reason: str) -> bool:
        logger.info("PLA KDA decode off: %s", reason)
        return False

    if not pla_kda_enabled():
        return reject("RTP_GLM_KDA_PLA is not 1")
    if triton is None:
        return reject("triton is unavailable, so the state rotation has no kernel")
    if not _load_pla():
        return reject(f"importing pla.decode failed ({_IMPORT_ERROR})")
    if num_v_heads not in _SUPPORTED_NUM_HEADS:
        return reject(
            f"num_v_heads {num_v_heads} is outside the vetted set {_SUPPORTED_NUM_HEADS}"
        )
    if head_dim != _SUPPORTED_HEAD_DIM:
        return reject(f"head_dim {head_dim} != {_SUPPORTED_HEAD_DIM} (compiled in)")
    if state is None:
        return reject("no SSM state tensor")
    # The vendor kernel reads the state through hard-typed `float const*`; a
    # BF16 state raises "expected scalar type Float but found BFloat16" from
    # inside graph capture and the server never comes up. sglang gates on the
    # same condition (fused_sigmoid_gating_recurrent.py, aone 87036227).
    if state.dtype != torch.float32:
        return reject(f"SSM state dtype is {state.dtype}, the kernel is FP32-only")
    # The vendor kernel addresses one slot as
    #   h0_idx * stride(0) + hv * V * K + v * K + k
    # -- only stride(0) is honoured, the inner block is assumed contiguous
    # [HV, V, K]. rtp-llm's default pool is [HV, K, V], so without this switch
    # PLA reads every state transposed. Measured on an idle card: output
    # relative error 1.0e-1 and state 1.0e+0 with the default layout, versus
    # 1.9e-3 and 7.9e-5 with it on (scripts/pla/check_kda_pla.py).
    if not kda_pool_v_first():
        return reject(
            "the SSM state pool is [HV, K, V]; PLA needs [HV, V, K], so set "
            "RTP_GLM_KDA_STATE_V_FIRST=1 on BOTH the prefill and decode roles"
        )
    if block_map is None or sequence_lengths is None:
        return reject("no block_map/sequence_lengths, cannot resolve state slots")
    return True


if triton is not None:

    @triton.jit
    def _rotate_state_kernel(
        STATE,
        LOAD_SLOT,
        STORE_SLOT,
        stride_slot,
        HV: tl.constexpr,
        KV: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Copy one head's state from its load slot to its store slot.

        Exits before touching memory when the slots agree (the common case) or
        when either resolves to the null/dummy block, matching the Triton
        recurrent kernel's `if read_block_id <= 0: return`.
        """
        i_n = tl.program_id(0)
        i_hv = tl.program_id(1)
        src = tl.load(LOAD_SLOT + i_n).to(tl.int64)
        dst = tl.load(STORE_SLOT + i_n).to(tl.int64)
        if src == dst or src <= 0 or dst <= 0:
            return
        p_src = STATE + src * stride_slot + i_hv * KV
        p_dst = STATE + dst * stride_slot + i_hv * KV
        for off in tl.range(0, KV, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            mask = idx < KV
            tl.store(p_dst + idx, tl.load(p_src + idx, mask=mask, other=0.0), mask=mask)

else:  # pragma: no cover - mirrors the import guard above
    _rotate_state_kernel = None


def resolve_state_slots(
    block_map: torch.Tensor,
    sequence_lengths: torch.Tensor,
    seq_size_per_block: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Physical state slots this step reads from and writes to, per request.

    Mirrors ``cal_block_idx(x, s) = (x - 1) // s`` from the Triton kernel:
    the load offset uses ``seq_len - 1`` and the store offset uses ``seq_len``.
    Pure tensor ops so a captured graph re-resolves them on every replay.

    block_map is [batch, max_blocks] int32/int64; sequence_lengths is [batch]
    and is already the "+1" form the caller passes to the Triton kernel.
    """
    max_blocks = block_map.shape[1]
    lengths = sequence_lengths.to(torch.int64)
    load_offset = torch.div(lengths - 2, seq_size_per_block, rounding_mode="floor")
    store_offset = torch.div(lengths - 1, seq_size_per_block, rounding_mode="floor")
    load_offset = load_offset.clamp_(0, max_blocks - 1)
    store_offset = store_offset.clamp_(0, max_blocks - 1)
    load_slot = block_map.gather(1, load_offset.unsqueeze(1)).squeeze(1)
    store_slot = block_map.gather(1, store_offset.unsqueeze(1)).squeeze(1)
    return load_slot.to(torch.int32), store_slot.to(torch.int32)


def fused_recurrent_kda_pla(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: Optional[torch.Tensor],
    initial_state: torch.Tensor,
    load_slot: torch.Tensor,
    store_slot: torch.Tensor,
    scale: Optional[float] = None,
    lower_bound: Optional[float] = None,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    """One decode step of KDA through PLA, on rtp-llm's block-mapped state.

    q/k/v are [batch, 1, heads, head_dim]; ``g`` is the raw forget gate and
    ``beta_raw`` the *pre-sigmoid* beta -- PLA applies ``sigmoid(b)`` itself,
    unlike our Triton call site which sigmoids beforehand.

    Returns the attention output; the state pool is updated in place at
    ``store_slot``.

    Padded graph rows can carry slot 0, the engine's reserved dummy block, and
    PLA has no per-row skip -- it writes there. That is harmless: nothing ever
    reads block 0's contents. The Triton recurrent kernel returns early on
    ``read_block_id <= 0`` and the block-map loader masks that row to zero, so
    a fresh request starts from a zero state regardless of what is in block 0.
    """
    if not _load_pla():
        raise RuntimeError(f"pla.decode is unavailable ({_IMPORT_ERROR})")

    batch = q.shape[0]
    num_v_heads = v.shape[2]
    head_dim = v.shape[3]
    kv = initial_state.shape[-1] * initial_state.shape[-2]

    # Check the contract before touching the device. The caller turns a raised
    # exception into a permanent fallback for this layer; raising here, ahead of
    # the rotation launch, keeps that fallback usable even mid graph capture --
    # nothing has been recorded yet, so the Triton path is what gets captured.
    # PLA itself validates host-side too, but by then the rotation has run.
    for name, tensor in (("q", q), ("k", k), ("v", v), ("g", g)):
        if tensor.dim() != 4:
            raise ValueError(
                f"{name} must be [batch, 1, heads, dim], got {tuple(tensor.shape)}"
            )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"PLA takes BF16 or FP16 activations, got {q.dtype}")
    if beta_raw.shape[-1] != num_v_heads:
        raise ValueError(
            f"beta must be [.., {num_v_heads}], got {tuple(beta_raw.shape)}"
        )
    # PLA requires the gate tensors to share q's dtype. Cast rather than
    # reject: the projections could land in FP32 and that is not a contract
    # violation on our side, just a conversion PLA will not do itself. The
    # cast is a no-op when they already match, which is the expected case.
    if g.dtype != q.dtype:
        g = g.to(q.dtype)
    if beta_raw.dtype != q.dtype:
        beta_raw = beta_raw.to(q.dtype)
    if A_log.numel() != num_v_heads:
        raise ValueError(f"A_log has {A_log.numel()} entries, want {num_v_heads}")
    if dt_bias is not None and dt_bias.numel() != num_v_heads * head_dim:
        raise ValueError(
            f"dt_bias has {dt_bias.numel()} entries, want {num_v_heads * head_dim}"
        )
    if initial_state.shape[-1] != head_dim or initial_state.shape[-2] != head_dim:
        raise ValueError(
            f"state slot must be [.., {head_dim}, {head_dim}], got "
            f"{tuple(initial_state.shape)}"
        )

    # Rotate the checkpoint before PLA updates it in place.
    _rotate_state_kernel[(batch, num_v_heads)](
        initial_state,
        load_slot,
        store_slot,
        initial_state.stride(0),
        HV=num_v_heads,
        KV=kv,
        BLOCK=1024,
    )

    # PLA reads A_log and dt_bias as q's dtype on every path except KDA MTP
    # verify; sglang casts them at the call site for the same reason.
    a_log = A_log.to(q.dtype)
    bias = dt_bias.to(q.dtype) if dt_bias is not None else None
    if scale is None:
        scale = k.shape[-1] ** -0.5

    return _k_last(
        a_log,
        g,
        bias,
        softplus_beta,
        softplus_threshold,
        q,
        k,
        v,
        beta_raw,
        initial_state,
        store_slot,
        scale,
        True,  # use_qk_l2norm_in_kernel
        None,  # cu_seqlens: decode is one token per request
        True,  # is_kda
        False,  # disable_state_update
        None,  # intermediate_states_buffer: MTP target-verify only
        None,  # intermediate_state_indices
        None,  # cache_steps_or_num_accept
        None,  # retrieve_parent_token
        lower_bound,
    )


# ---------------------------------------------------------------------------
# Prefill: pla.prefill.flashkdapro.flashkda_fwd
# ---------------------------------------------------------------------------
#
# sglang-for-sail runs KDA prefill on PPU through this kernel (kda_flashkda.py,
# on by default there); we ran Triton chunk_kda. It fuses the q/k L2 norm, the
# beta sigmoid and the KDA gate in-kernel, so it takes RAW q/k/g/beta plus
# A_log [H] fp32 and dt_bias [H, K] fp32, exactly like the Triton call with
# use_qk_l2norm_in_kernel / use_gate_in_kernel.
#
# It only emits the final state. rtp-llm's prefill normally also writes the
# state at every cache-block boundary inside the forward (store_ssm_state_to_
# block_map consumes chunk_kda's per-chunk states), but that store skips every
# position whose block id is 0, and with prefix reuse off the LINEAR group
# keeps only its tail block (active_tail_blocks == 1). So in that configuration
# only the final state is ever written, and a final-state-only kernel is a
# drop-in. Both conditions are checked: reuse via REUSE_CACHE, and the block
# map itself is audited on the first calls (one host sync each) -- any live
# intermediate slot switches the vendor path off for good.

_flashkda_fwd = None
_PREFILL_IMPORT_ERROR: Optional[str] = None
_PREFILL_AUDIT_LEFT = int(os.environ.get("RTP_GLM_KDA_PLA_PREFILL_AUDIT", "16"))
_prefill_disabled_reason: Optional[str] = None
_prefill_logged = False


def pla_kda_prefill_enabled() -> bool:
    return os.environ.get("RTP_GLM_KDA_PLA_PREFILL", "0") == "1"


def _load_pla_prefill() -> bool:
    global _flashkda_fwd, _PREFILL_IMPORT_ERROR
    if _flashkda_fwd is not None:
        return True
    if _PREFILL_IMPORT_ERROR is not None:
        return False
    extra = os.environ.get("RTP_GLM_PLA_PATH")
    if extra and extra not in sys.path:
        sys.path.append(extra)
    try:
        from pla.prefill.flashkdapro import flashkda_fwd

        _flashkda_fwd = flashkda_fwd
        return True
    except Exception as exc:  # noqa: BLE001 - any import failure means fall back
        _PREFILL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"
        return False


def pla_kda_prefill_supported(
    num_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    state: Optional[torch.Tensor],
    lower_bound: Optional[float],
) -> bool:
    """Static capability probe for the prefill kernel. Logs every rejection."""

    def reject(reason: str) -> bool:
        logger.info("PLA KDA prefill off: %s", reason)
        return False

    if not pla_kda_prefill_enabled():
        return reject("RTP_GLM_KDA_PLA_PREFILL is not 1")
    if _prefill_disabled_reason is not None:
        return reject(_prefill_disabled_reason)
    if not _load_pla_prefill():
        return reject(f"importing pla.prefill.flashkdapro failed ({_PREFILL_IMPORT_ERROR})")
    if num_heads not in _SUPPORTED_NUM_HEADS:
        return reject(f"num_heads {num_heads} is outside the vetted set {_SUPPORTED_NUM_HEADS}")
    if head_dim != _SUPPORTED_HEAD_DIM:
        return reject(f"head_dim {head_dim} != {_SUPPORTED_HEAD_DIM}")
    if q_dtype != torch.bfloat16:
        return reject(f"activations are {q_dtype}, the kernel is BF16-only")
    # sglang falls back on the unbounded gate too (_should_fall_back).
    if lower_bound is None:
        return reject("the kernel needs the bounded (lower_bound) KDA gate")
    if state is not None and state.dtype != torch.float32:
        return reject(f"SSM state dtype is {state.dtype}, the kernel is FP32-only")
    if os.environ.get("REUSE_CACHE", "0").lower() in ("1", "true"):
        return reject("prefix reuse is on; intermediate block states would be skipped")
    return True


def _audit_tail_only(
    block_map: torch.Tensor,
    prefix_lengths: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seq_size_per_block: int,
) -> bool:
    """True if no block strictly before each request's final block, within this
    forward's token span, has a live slot. One host sync."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    prefix = prefix_lengths.to(torch.int64)
    first = torch.div(prefix, seq_size_per_block, rounding_mode="floor")
    last = torch.div(prefix + lengths - 1, seq_size_per_block, rounding_mode="floor")
    pos = torch.arange(block_map.shape[1], device=block_map.device).unsqueeze(0)
    inner = (pos >= first.unsqueeze(1)) & (pos < last.unsqueeze(1))
    return not bool(((block_map.to(torch.int64) > 0) & inner).any().item())


def flashkda_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta_raw: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    cu_seqlens: torch.Tensor,
    lower_bound: float,
    block_map: Optional[torch.Tensor] = None,
    prefix_lengths: Optional[torch.Tensor] = None,
    seq_size_per_block: int = 0,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """KDA prefill through PLA. Returns (out [T, H, V], final_state [N, H, K, V]
    in chunk_kda's layout), or None if the caller must take the Triton path.

    q/k/v/g are [T, H, D] (packed varlen), beta_raw [T, H] pre-sigmoid,
    initial_state [N, H, K, V] fp32 as load_initial_state_from_block_map fills
    it for chunk_kda. PLA keeps the state as [N, H, V, K] -- sglang's pool
    layout -- so it is transposed on the way in and out.
    """
    global _PREFILL_AUDIT_LEFT, _prefill_disabled_reason
    if _PREFILL_AUDIT_LEFT > 0 and block_map is not None and prefix_lengths is not None:
        _PREFILL_AUDIT_LEFT -= 1
        if not _audit_tail_only(block_map, prefix_lengths, cu_seqlens, seq_size_per_block):
            _prefill_disabled_reason = (
                "the block map has live intermediate LINEAR slots in a prefill "
                "forward; the final-state-only kernel would leave them stale"
            )
            logger.warning("PLA KDA prefill off from now on: %s", _prefill_disabled_reason)
            return None
    T, H, D = q.shape
    n = cu_seqlens.numel() - 1
    out = torch.empty_like(v)
    if initial_state is None:
        init = torch.zeros(n, H, D, D, device=q.device, dtype=torch.float32)
    else:
        init = initial_state.transpose(-1, -2).contiguous()
    final = torch.empty(n, H, D, D, device=q.device, dtype=torch.float32)
    _flashkda_fwd(
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
        g=g.contiguous(),
        beta=beta_raw.to(torch.bfloat16).contiguous(),
        A_log=A_log.reshape(-1).float().contiguous(),
        dt_bias=dt_bias.reshape(H, -1).float().contiguous(),
        out=out,
        scale=D**-0.5,
        lower_bound=lower_bound,
        initial_state=init,
        final_state=final,
        cu_seqlens=cu_seqlens.to(torch.int64),
        raw_cu_seqlens=None,
    )
    global _prefill_logged
    if not _prefill_logged:
        _prefill_logged = True
        logger.info("PLA KDA prefill on: first call done (%d tokens, %d heads, %d requests)", T, H, n)
    return out, final.transpose(-1, -2).contiguous()
