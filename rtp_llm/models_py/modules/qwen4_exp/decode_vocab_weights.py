"""Decode vocabulary conversion, shared ownership and rank-local BF16 updates."""

import logging
import weakref
from typing import NamedTuple

import torch
import torch.distributed as dist

from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX
from rtp_llm.utils.model_weight import W


class _VocabLayout(NamedTuple):
    group: object
    rank: int
    vocab_size: int
    scales: object
    source: object
    can_update: bool


class PreparedVocabShard:
    def __init__(self, weight, layout):
        self.weight = weight
        self.scales = layout.scales
        self.vocab_size = layout.vocab_size
        self.start = layout.rank * weight.shape[0]
        self.layout = layout
        # The converted owner carries enough metadata for a later draft even
        # if its previous Python consumer has been destroyed. Neither the
        # metadata nor this weak cache retains the hidden-sharded source.
        weight._rtp_decode_vocab_layout = layout
        weight._rtp_embedding_vocab_size = layout.vocab_size
        weight._rtp_decode_vocab_shard = weakref.ref(self)

    @torch.inference_mode()
    def update(self, prepare):
        """Stage a local vocabulary shard from the complete IPC tensor.

        The caller pauses engines, serializes updates and resumes only after
        every rank has succeeded. This handler performs no collectives and
        provides no cross-rank transaction, like the ordinary weight updater.
        """
        if not self.layout.can_update or torch.cuda.is_current_stream_capturing():
            raise PermissionError("vocabulary is not updateable during this execution")
        # Pausing stops submission; drain reads already enqueued on other streams.
        torch.cuda.synchronize(self.weight.device)
        full, source = prepare()
        rows, width = self.weight.shape
        size = dist.get_world_size(self.layout.group)
        if not (
            isinstance(full, torch.Tensor)
            and full.dtype == torch.bfloat16
            and tuple(full.shape) == (self.vocab_size, width)
            and isinstance(source, torch.Tensor)
            and source.dtype == torch.bfloat16
            and source.device == self.weight.device
            and tuple(source.shape) == (self.vocab_size, width // size)
            and source.is_contiguous()
        ):
            raise ValueError(
                "vocabulary update requires full BF16 weights and the original hidden shard"
            )
        original = self.layout.source()
        if original is not None and (
            original.shape != source.shape
            or original.dtype != source.dtype
            or original.device != source.device
        ):
            raise ValueError(
                "original vocabulary owner no longer matches its hidden shard"
            )

        # Stage both loader and vocabulary layouts before touching any live
        # tensor. Copy directly from CPU or CUDA input, padding the final rank.
        candidate = torch.empty_like(self.weight)
        count = min(rows, self.vocab_size - self.start)
        candidate[:count].copy_(full[self.start : self.start + count])
        candidate[count:].zero_()
        torch.cuda.current_stream(self.weight.device).synchronize()

        # Preserve captured pointers and every shared target/draft consumer.
        self.weight.copy_(candidate)
        if original is not None:
            original.copy_(source)
        torch.cuda.current_stream(self.weight.device).synchronize()


@torch.no_grad()
def _repartition_vocab(weight, group, weight_scale=None):
    """Exchange hidden shards once at startup; never requantize FP8 weights.

    A source FP8 hidden shard has its own scalar scale. The vocabulary shard
    retains those scales per hidden slice, reproducing the original lookup's
    BF16 conversions. Shared target/draft weights share the converted storage.
    Converted owners are shared with the draft. All ranks agree on the
    layout before any rank skips the exchange.
    """
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("repartition vocabulary before Graph capture")
    rank, size = dist.get_rank(group), dist.get_world_size(group)
    fp8 = weight.dtype == torch.float8_e4m3fn
    layout = getattr(weight, "_rtp_decode_vocab_layout", None)
    can_update = (
        layout.can_update
        if layout is not None
        else weight.dtype == torch.bfloat16
        and not getattr(weight, "_rtp_requires_restart_for_update", False)
    )
    valid = (
        size == 2
        and weight.is_cuda
        and weight.ndim == 2
        and weight.is_contiguous()
        and min(weight.shape) > 0
        and weight.dtype in (torch.bfloat16, torch.float8_e4m3fn)
        and (
            (
                layout is not None
                and layout.group is group
                and layout.rank == rank
                and weight.shape[0] == (layout.vocab_size + size - 1) // size
            )
            or (
                layout is None
                and (
                    not fp8 or (weight_scale is not None and weight_scale.numel() == 1)
                )
            )
        )
    )
    peers = [None] * size
    policy = (
        valid,
        tuple(weight.shape),
        str(weight.dtype),
        layout is not None,
        None if layout is None else layout.vocab_size,
        can_update,
    )
    dist.all_gather_object(peers, policy, group=group)
    if not all(p[0] and p[1:] == policy[1:] for p in peers):
        raise ValueError(
            "decode vocabulary requires matching contiguous BF16/FP8 TP2 layouts"
        )
    if layout is not None:
        logging.info(
            "Reusing Qwen4 decode vocabulary shard for shared target/draft weights"
        )
        cached = getattr(weight, "_rtp_decode_vocab_shard", lambda: None)()
        return cached if cached is not None else PreparedVocabShard(weight, layout)

    vocab_size, local_width = weight.shape
    rows = (vocab_size + size - 1) // size
    raw = weight.view(torch.uint8) if fp8 else weight
    if rows * size != vocab_size:
        raw = torch.nn.functional.pad(raw, (0, 0, 0, rows * size - vocab_size))
    received = torch.empty_like(raw)
    dist.all_to_all_single(received, raw, group=group)
    converted = received.reshape(size, rows, local_width).transpose(0, 1).contiguous()
    converted = converted.reshape(rows, size * local_width)
    if fp8:
        converted = converted.view(torch.float8_e4m3fn)
        scale = weight_scale.to(device=weight.device, dtype=torch.float32).reshape(1)
        scales = torch.empty(size, device=weight.device, dtype=torch.float32)
        dist.all_gather_into_tensor(scales, scale, group=group)
    else:
        scales = None
    resource = PreparedVocabShard(
        converted,
        _VocabLayout(
            group,
            rank,
            vocab_size,
            scales,
            weakref.ref(weight),
            can_update,
        ),
    )
    weight._rtp_requires_restart_for_update = True
    converted._rtp_requires_restart_for_update = True
    if weight_scale is not None:
        weight_scale._rtp_requires_restart_for_update = True
    logging.info(
        "Qwen4 decode vocabulary repartitioned: hidden_shard=%s vocab_shard=%s dtype=%s",
        tuple(weight.shape),
        tuple(converted.shape),
        weight.dtype,
    )
    return resource


@torch.no_grad()
def prepare_decode_vocab(
    *, source_weights, compute_weights, group, update_handlers=None
) -> PreparedVocabShard:
    """Prepare one shared vocabulary owner before constructing its consumers.

    Both loaded and compute dictionaries are required, including when they are
    the same object. Publication replaces only embedding entries, preserving
    other consumers such as a tied output head. Source storage is not retained.
    """
    shard = _repartition_vocab(
        source_weights[W.embedding],
        group,
        source_weights.get(W.embedding + FP8_STORAGE_SCALE_SUFFIX),
    )
    source_weights[W.embedding] = shard.weight
    compute_weights[W.embedding] = shard.weight
    if update_handlers is not None and shard.layout.can_update:
        update_handlers[W.embedding] = shard.update
    # Protect the loader's scalar FP8 scale owner as well as the converted
    # weight; wrappers may otherwise only mark a reshaped view of the scale.
    for name in (W.embedding, W.embedding + FP8_STORAGE_SCALE_SUFFIX):
        owner = source_weights.get(name)
        if owner is not None:
            owner._rtp_requires_restart_for_update = True
    return shard
