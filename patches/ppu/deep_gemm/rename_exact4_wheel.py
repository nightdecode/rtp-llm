#!/usr/bin/env python3
"""Repackage the exact4 deep_gemm wheel as the ``deep_gemm_exact4`` distribution.

The exact4 runtime wheel (built by ``package_exact4_runtime.py``) ships its
Python package under the name ``deep_gemm``, so it cannot coexist with the
decode-side ``deep_gemm`` 1.0.0 in one image.  This script renames the
distribution — same approach as ``deep_gemm_mega`` / ``deep_gemm_route_a`` —
so both can be installed side by side and selected per role via
``DSV4_PPU_DEEPGEMM_PACKAGE`` (see rtp_llm/models_py/modules/dsv4/deep_gemm_override.py).

The rename is mechanical and safe for this wheel because:

- the compiled extension is ``_C``; extension init symbols depend only on the
  last module component, so ``deep_gemm_exact4._C`` imports fine;
- Python code inside the wheel uses relative imports, except four files under
  ``deep_gemm_tuner/`` whose absolute self-imports are rewritten to relative
  form (``from deep_gemm.deep_gemm_tuner.x import`` -> ``from .x import``,
  ``from deep_gemm[.utils[.math]] import`` -> ``from ..[utils[.math]] import``,
  and the bare ``import deep_gemm`` becomes a ``sys.modules`` lookup of the
  top-level package).  Relative imports make the package name-agnostic, so it
  works both under its real name and aliased as ``deep_gemm`` — with no second
  ``_C`` extension load (an absolute self-import would re-execute the package
  under its real name and crash on duplicate pybind11 registrations);
- JIT-compiled kernels reference headers as ``"deep_gemm/xxx.cuh"`` and the
  include root is computed from ``__file__`` (``jit/compiler.py``), so the
  ``<pkg>/include/deep_gemm/`` layout keeps resolving after the rename;
- JIT cache keys are content hashes, so sharing ``~/.deep_gemm`` with the
  stock ``deep_gemm`` install cannot cross-contaminate.

Usage:
    python rename_exact4_wheel.py INPUT_WHEEL [-o OUTPUT_DIR]

Prints the output wheel path and its sha256 for the requirements lock entry.
"""

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import zipfile

OLD_PKG = "deep_gemm"
NEW_PKG = "deep_gemm_exact4"
NEW_DIST_NAME = "deep-gemm-exact4"

# absolute self-imports -> relative / sys.modules forms (name-agnostic)
_PAT_FROM_TUNER = re.compile(r"^(\s*)from deep_gemm\.deep_gemm_tuner\.", re.M)
_PAT_FROM_UTILS_SUB = re.compile(r"^(\s*)from deep_gemm\.utils\.", re.M)
_PAT_FROM_UTILS = re.compile(r"^(\s*)from deep_gemm\.utils import", re.M)
_PAT_FROM_TOP = re.compile(r"^(\s*)from deep_gemm import", re.M)
_PAT_IMPORT_TOP = re.compile(r"^(\s*)import deep_gemm\s*$", re.M)


def _junk(arcname: str) -> bool:
    # The source wheel was zipped on macOS and carries AppleDouble litter.
    return any(p.startswith("._") or p == ".DS_Store" for p in arcname.split("/"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_wheel")
    parser.add_argument("-o", "--output-dir", default=".")
    args = parser.parse_args()

    src = zipfile.ZipFile(args.input_wheel)
    names = [n for n in src.namelist() if not _junk(n)]
    tops = {n.split("/")[0] for n in names}
    dist_infos = [t for t in tops if t.endswith(".dist-info")]
    assert tops == {OLD_PKG, dist_infos[0]} and len(dist_infos) == 1, tops
    old_dist = dist_infos[0]
    version = old_dist[: -len(".dist-info")].split("-", 1)[1]
    new_dist = f"{NEW_PKG}-{version}.dist-info"

    work = os.path.join(args.output_dir, f".{NEW_PKG}_build")
    if os.path.exists(work):
        shutil.rmtree(work)
    os.makedirs(work)
    for n in names:
        src.extract(n, work)
    os.rename(os.path.join(work, OLD_PKG), os.path.join(work, NEW_PKG))
    os.rename(os.path.join(work, old_dist), os.path.join(work, new_dist))

    changed = []
    for root, _dirs, files in os.walk(os.path.join(work, NEW_PKG)):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            with open(p, encoding="utf-8", errors="surrogateescape") as f:
                text = f.read()
            orig = text
            text = _PAT_FROM_TUNER.sub(r"\1from .", text)
            text = _PAT_FROM_UTILS_SUB.sub(r"\1from ..utils.", text)
            text = _PAT_FROM_UTILS.sub(r"\1from ..utils import", text)
            text = _PAT_FROM_TOP.sub(r"\1from .. import", text)
            text = _PAT_IMPORT_TOP.sub(
                r'\1import sys as _sys' + "\n" + r'\1deep_gemm = _sys.modules[__name__.split(".")[0]]',
                text,
            )
            if text != orig:
                changed.append(os.path.relpath(p, work))
                with open(p, "w", encoding="utf-8", errors="surrogateescape") as f:
                    f.write(text)

    meta_path = os.path.join(work, new_dist, "METADATA")
    with open(meta_path, encoding="utf-8") as f:
        meta = f.read()
    assert "\nName: deep-gemm\n" in "\n" + meta
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(meta.replace("Name: deep-gemm\n", f"Name: {NEW_DIST_NAME}\n", 1))

    def record_line(arcname, data):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
        return (arcname, f"sha256={digest.decode()}", str(len(data)))

    entries = []
    for root, _dirs, files in os.walk(work):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            arc = os.path.relpath(p, work).replace(os.sep, "/")
            if arc == f"{new_dist}/RECORD":
                continue
            with open(p, "rb") as f:
                entries.append(record_line(arc, f.read()))
    entries.append((f"{new_dist}/RECORD", "", ""))
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(entries)
    with open(os.path.join(work, new_dist, "RECORD"), "w", encoding="utf-8") as f:
        f.write(buf.getvalue())

    base = os.path.basename(args.input_wheel)
    assert base.startswith(f"{OLD_PKG}-") and base.endswith(".whl"), base
    tags = base[len(f"{OLD_PKG}-{version}-") : -len(".whl")]
    out = os.path.join(args.output_dir, f"{NEW_PKG}-{version}-{tags}.whl")
    all_files = []
    for root, _dirs, files in os.walk(work):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            all_files.append((p, os.path.relpath(p, work).replace(os.sep, "/")))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in sorted(all_files, key=lambda t: t[1]):
            # fixed timestamps -> byte-identical output across runs
            info = zipfile.ZipInfo(arc, date_time=(2026, 11, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(p, "rb") as f:
                z.writestr(info, f.read())
    shutil.rmtree(work)

    print("rewritten self-imports:")
    for p in changed:
        print("  ", p)
    with open(out, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    print(f"wheel: {out}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()
