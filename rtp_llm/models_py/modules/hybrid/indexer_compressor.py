"""GLM-5.3-Flash KPool reference math.

Taken from ``origin/feat/glm53-flash-main-port`` (HEAD ``2c38b8bd61``), file of
the same name. Only the two functions below are kept: the normalized Hadamard
rotation that pairs with the query-side rotation, and the per-channel pooling
reference. The typed-cache views and ring-buffer helpers that came with the
original file describe that branch's compressed indexer cache layout, which
this port does not use (see ``kpool.py`` for the layout it does use), so they
are dropped rather than carried as dead code.

Both functions are plain torch with no device-specific code, so they also serve
as the CPU reference the KPool tests compare against.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _hadamard_rotate_reference(x: torch.Tensor) -> torch.Tensor:
    width = int(x.size(-1))
    if width <= 0 or width & (width - 1):
        raise ValueError("Hadamard width must be a positive power of two")
    result = x
    stride = 1
    while stride < width:
        pairs = result.reshape(*result.shape[:-1], -1, 2, stride)
        left, right = pairs.unbind(dim=-2)
        result = torch.stack((left + right, left - right), dim=-2).reshape_as(result)
        stride *= 2
    return result * (width**-0.5)


def compress_indexer_projection_reference(
    kv_projection: torch.Tensor,
    score_projection: torch.Tensor,
    ape: torch.Tensor,
    *,
    compress_ratio: int = 4,
    overlap: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference: per-channel four-way softmax, pool, normalized Hadamard."""

    if kv_projection.dim() != 2 or score_projection.shape != kv_projection.shape:
        raise ValueError("kv_projection and score_projection must share shape [T, C]")
    if ape.dim() != 2 or tuple(ape.shape) != (
        compress_ratio,
        kv_projection.size(1),
    ):
        raise ValueError(
            f"ape must have shape [{compress_ratio}, {kv_projection.size(1)}]"
        )
    if overlap != 0:
        raise ValueError("GLM-5.3-Flash KPool uses non-overlapping groups")

    width = int(kv_projection.size(1))
    if width <= 0 or width & (width - 1):
        raise ValueError("head_dim must be a power of two")
    boundaries = torch.arange(
        compress_ratio - 1,
        int(kv_projection.size(0)),
        compress_ratio,
        dtype=torch.int64,
        device=kv_projection.device,
    )
    if boundaries.numel() == 0:
        return kv_projection.new_empty((0, width)), boundaries

    groups = []
    for boundary_tensor in boundaries:
        boundary = int(boundary_tensor.item())
        key = kv_projection[boundary - compress_ratio + 1 : boundary + 1]
        score = score_projection[boundary - compress_ratio + 1 : boundary + 1]
        weights = torch.softmax((score + ape).float(), dim=0).to(key.dtype)
        pooled = torch.sum(key * weights, dim=0).to(torch.bfloat16).float()
        groups.append(_hadamard_rotate_reference(pooled).to(torch.bfloat16).float())
    return torch.stack(groups, dim=0), boundaries
