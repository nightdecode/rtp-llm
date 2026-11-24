"""Pack several projections that read the same input into one GEMM.

Kept free of engine imports so it can be unit tested on CPU: it needs only
``torch`` and ``torch.nn``.
"""

from typing import Optional

import torch
from torch import nn

# Columns of packed projection weight to accept. The packed matrix duplicates
# the parts it copies, so the cap bounds that duplication: at 4096 columns of
# BF16 with a 4096-wide hidden size it is 32 MB per layer in the worst case,
# against the ~200 MB a single QKV projection would add. GLM-5.3-Flash's three
# narrow projections come to a few hundred columns, well inside it.
MAX_PACKED_PROJECTION_WIDTH = 4096


def packed_projection_weight(projection: nn.Module) -> Optional[torch.Tensor]:
    """The projection's weight as a contiguous [in, out] matrix, or None.

    Only unquantized BF16 projections without bias can share one GEMM.
    ``CudaF16Linear`` stores ``weight`` already transposed for ``F.linear``,
    so its ``.T`` is the original contiguous [in, out] checkpoint layout.
    """
    weight = getattr(projection, "weight", None)
    if weight is None or getattr(projection, "bias", None) is not None:
        return None
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        return None
    weight = weight.T
    return weight if weight.is_contiguous() else None
