"""Prefill-role DeepGEMM package override for DSV4 on PPU.

The DSV4 PPU prefill path (W4A8 grouped / MegaMoE MoE, UE8M0 scale repacking)
requires the exact4 DeepGEMM build, which ships as the separate distribution
``deep_gemm_exact4`` so it can coexist with the decode-side ``deep_gemm``
1.0.0 in one image (same pattern as ``deep_gemm_mega`` / ``deep_gemm_route_a``).

On the development machines this was arranged by overlaying a directory onto
``PYTHONPATH`` so that ``import deep_gemm`` resolved to the exact4 build.
Images cannot do that per role, so the prefill role instead sets

    DSV4_PPU_DEEPGEMM_PACKAGE=deep_gemm_exact4

and this module loads that package *under the name* ``deep_gemm`` (the module
spec is created with name ``deep_gemm`` pointing at the selected package's
location, exactly like the PYTHONPATH overlay shadowing the stock install).
Every later ``import deep_gemm`` / ``from deep_gemm...`` — including lazy
imports inside functions and submodule imports such as
``deep_gemm.utils.layout`` — then resolves to the exact4 build, in a single
namespace: the ``_C`` extension loads once as ``deep_gemm._C``.

The override runs when the ``rtp_llm.models_py.modules.dsv4`` package is
imported, which precedes any ``deep_gemm`` import on the DSV4 model path (the
only top-level ``import deep_gemm`` statements in the runtime tree live in the
dsv4/dsv41 model families).  Processes without the environment variable —
decode roles, CUDA deployments, unit tests — are unaffected.

Known constraint: ``deep_gemm_exact4`` and ``deep_gemm_route_a`` cannot be
loaded in the same process (their ``_C`` extensions register the same C++
types, which collide via GNU unique symbol interposition).  This is safe for
the prefill role because every ``deep_gemm_route_a`` import site in the tree
is decode-side (ppu_deepep_w4a8 / ppu_w4a8_masked / ppu_decode_provider);
prefill must therefore not enable the DeepEP W4A8 backend while this override
is active.  ``deep_gemm_mega`` (2.1.0 lineage) coexists fine.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import sys

_ENV_VAR = "DSV4_PPU_DEEPGEMM_PACKAGE"

_logger = logging.getLogger(__name__)


def apply_deep_gemm_package_override(environ=None) -> None:
    """Load the package named by ``DSV4_PPU_DEEPGEMM_PACKAGE`` as ``deep_gemm``.

    No-op when the variable is unset, empty, or ``deep_gemm`` itself.
    """
    environ = os.environ if environ is None else environ
    package = environ.get(_ENV_VAR, "").strip()
    if not package or package == "deep_gemm":
        return
    if package != "deep_gemm_exact4":
        raise ValueError(
            f"{_ENV_VAR} only supports 'deep_gemm_exact4', got {package!r}"
        )
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise ImportError(
            f"{_ENV_VAR}={package} but the {package} distribution is not installed"
        )
    # Drop anything already imported under the old name so no stale submodule
    # survives the override.
    for name in [k for k in sys.modules if k == "deep_gemm" or k.startswith("deep_gemm.")]:
        del sys.modules[name]
    alias_spec = importlib.util.spec_from_file_location(
        "deep_gemm",
        spec.origin,
        submodule_search_locations=list(spec.submodule_search_locations or []),
    )
    module = importlib.util.module_from_spec(alias_spec)
    sys.modules["deep_gemm"] = module
    alias_spec.loader.exec_module(module)
    _logger.info(
        "DSV4 PPU: deep_gemm resolved to %s (%s)",
        package,
        getattr(module, "__file__", None),
    )
