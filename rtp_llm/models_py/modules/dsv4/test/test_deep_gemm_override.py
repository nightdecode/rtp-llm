"""Tests for the DSV4 PPU prefill-role deep_gemm package override."""

import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path

from rtp_llm.models_py.modules.dsv4.deep_gemm_override import (
    apply_deep_gemm_package_override,
)


class DeepGemmOverrideTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            k: v for k, v in sys.modules.items() if k.split(".")[0] == "deep_gemm"
        }

    def tearDown(self):
        for k in [k for k in sys.modules if k.split(".")[0] == "deep_gemm"]:
            del sys.modules[k]
        sys.modules.update(self._saved)

    def test_unset_is_noop(self):
        apply_deep_gemm_package_override({})
        self.assertNotIn("deep_gemm", sys.modules)

    def test_default_package_is_noop(self):
        apply_deep_gemm_package_override({"DSV4_PPU_DEEPGEMM_PACKAGE": "deep_gemm"})
        self.assertNotIn("deep_gemm", sys.modules)

    def test_unknown_package_rejected(self):
        with self.assertRaises(ValueError):
            apply_deep_gemm_package_override({"DSV4_PPU_DEEPGEMM_PACKAGE": "whatever"})

    def test_missing_package_raises(self):
        with self.assertRaises(ImportError):
            apply_deep_gemm_package_override(
                {"DSV4_PPU_DEEPGEMM_PACKAGE": "deep_gemm_exact4"}
            )

    def test_alias_loads_under_single_namespace(self):
        """A real temp package: after the override, `import deep_gemm` and
        `deep_gemm.sub` come from the selected package in one namespace."""
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "deep_gemm_exact4"
            (pkg / "sub").mkdir(parents=True)
            (pkg / "__init__.py").write_text(
                "from . import sub\nMARKER = 'exact4'\n"
            )
            (pkg / "sub").joinpath("__init__.py").write_text("VALUE = 7\n")
            sys.path.insert(0, td)
            try:
                apply_deep_gemm_package_override(
                    {"DSV4_PPU_DEEPGEMM_PACKAGE": "deep_gemm_exact4"}
                )
                import deep_gemm

                self.assertEqual(deep_gemm.MARKER, "exact4")
                self.assertIn("deep_gemm_exact4", deep_gemm.__file__)
                self.assertEqual(deep_gemm.sub.VALUE, 7)
                # submodule was loaded under the alias name, single namespace
                self.assertIs(sys.modules["deep_gemm.sub"], deep_gemm.sub)
                self.assertNotIn("deep_gemm_exact4", sys.modules)

                import importlib

                self.assertIs(importlib.import_module("deep_gemm"), deep_gemm)
                spec = importlib.util.find_spec("deep_gemm")
                self.assertIn("deep_gemm_exact4", spec.origin)
            finally:
                sys.path.remove(td)

    def test_stale_entries_purged(self):
        stale = types.ModuleType("deep_gemm")
        sys.modules["deep_gemm"] = stale
        sys.modules["deep_gemm.jit_kernels"] = types.ModuleType("deep_gemm.jit_kernels")
        with tempfile.TemporaryDirectory() as td:
            pkg = Path(td) / "deep_gemm_exact4"
            pkg.mkdir()
            (pkg / "__init__.py").write_text("MARKER = 'exact4'\n")
            sys.path.insert(0, td)
            try:
                apply_deep_gemm_package_override(
                    {"DSV4_PPU_DEEPGEMM_PACKAGE": "deep_gemm_exact4"}
                )
                self.assertIsNot(sys.modules["deep_gemm"], stale)
                self.assertNotIn("deep_gemm.jit_kernels", sys.modules)
            finally:
                sys.path.remove(td)


if __name__ == "__main__":
    unittest.main()
