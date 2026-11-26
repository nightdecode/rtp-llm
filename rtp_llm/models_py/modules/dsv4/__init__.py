"""DeepSeek-V4 specific module implementations.

Currently scaffolded for milestone-based incremental development:
  - mhc: M1 - Hyper-Connection residual stream
  - hca_attention: M2 - Heavy Compress Attention with grouped output proj
  - csa_attention: M3 - Compressed Sparse Attention with lightning indexer

See `/home/wangyin.yx/workspace/work_memory/develop/dsv4/` for design docs.
"""

# Must run before any ``deep_gemm`` import on the DSV4 path (the top-level
# ones live in dsv4/fp8/attention.py and dsv4/ppu/attention/fp8/attention.py,
# both children of this package).  No-op unless DSV4_PPU_DEEPGEMM_PACKAGE is
# set, so decode roles and CUDA deployments are unaffected.
from rtp_llm.models_py.modules.dsv4.deep_gemm_override import (
    apply_deep_gemm_package_override,
)

apply_deep_gemm_package_override()
