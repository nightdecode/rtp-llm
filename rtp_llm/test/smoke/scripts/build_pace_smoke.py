#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def commit_sha(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("expected a full 40-character Git commit SHA")
    return value.lower()


def run(*command, cwd, capture=False):
    return subprocess.run(
        [str(arg) for arg in command],
        cwd=cwd,
        check=True,
        universal_newlines=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def stage_file(source, destination):
    with tempfile.TemporaryDirectory(dir=destination.parent) as staging:
        pending = Path(staging) / destination.name
        shutil.copy2(source, pending)
        pending.replace(destination)


def checkout(path, repo, sha):
    if not (path / ".git").is_dir():
        path.mkdir(exist_ok=True)
        if any(path.iterdir()):
            raise RuntimeError(f"cannot initialize checkout in nonempty directory {path}")
        run("git", "init", "--quiet", cwd=path)
        run("git", "remote", "add", "origin", repo, cwd=path)
    try:
        try:
            actual = run(
                "git", "rev-parse", "--verify", "--quiet", "HEAD", cwd=path, capture=True
            ).strip()
        except subprocess.CalledProcessError:
            run("git", "fetch", "--no-tags", "--depth=1", "origin", sha, cwd=path)
            run("git", "checkout", "--quiet", "--detach", "FETCH_HEAD", cwd=path)
            actual = run("git", "rev-parse", "HEAD", cwd=path, capture=True).strip()
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"failed to check out {path.name} at {sha}; check Git access and the commit"
        ) from error
    if actual != sha:
        raise RuntimeError(f"{path} is at {actual}, expected {sha}")


def build_pace(args, source, output):
    run("git", "submodule", "update", "--init", "--recursive", "--depth=1", cwd=source)
    build_dir = output / "pace-build"
    run(
        "cmake", "-S", source, "-B", build_dir, *args.cmake_arg,
        "-DCMAKE_BUILD_TYPE=Release", "-DENABLE_CUDA=OFF", "-DBUILD_CLIENT=OFF",
        "-DENABLE_MPI=OFF", "-DENABLE_OTEL=OFF", "-DTENT_ENABLE_BAREX=OFF",
        "-DTENT_ENABLE_ETCD_META=OFF", "-DTENT_ENABLE_REDIS_META=OFF",
        "-DTENT_ENABLE_RDMA=ON", "-DTENT_BUILD_SHARED=ON",
        cwd=source,
    )
    cache = (build_dir / "CMakeCache.txt").read_text()
    for variable in ("IBVERBS_LIB:FILEPATH", "IBVERBS_INCLUDE:PATH"):
        if not re.search(rf"^{variable}=(?!.*-NOTFOUND$).+$", cache, re.MULTILINE):
            raise RuntimeError(f"PACE native RDMA requires {variable}; install libibverbs development files")
    run(
        "cmake", "--build", build_dir, "--parallel", args.jobs,
        "--target", "tair_mempool_meta_service", "tair_mempool_server", "tent_shared",
        cwd=source,
    )
    bin_dir = output / "pace" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    for binary in ("tair_mempool_meta_service", "tair_mempool_server"):
        stage_file(build_dir / "bin" / binary, bin_dir / binary)
    libraries = sorted((build_dir / "lib").glob("libtent_shared.so*"))
    if not libraries:
        raise RuntimeError("PACE build did not produce libtent_shared")
    for library in libraries:
        stage_file(library, bin_dir / library.name)


def build_kvcm(args, public, mempool, output):
    stub = public / "stub_source"
    if not stub.is_symlink():
        raise RuntimeError(f"expected the KVCM stub_source symlink at {stub}")
    if os.readlink(stub) != "../internal_source":
        stub.unlink()
        stub.symlink_to("../internal_source")

    override = output / "tair_mempool"
    override.mkdir(exist_ok=True)
    for entry in mempool.iterdir():
        if entry.name not in (".git", "BUILD", "WORKSPACE") and not (override / entry.name).is_symlink():
            (override / entry.name).symlink_to(entry)
    shutil.copy2(public / "3rdparty/tair_mempool/tair_mempool.BUILD", override / "BUILD")
    (override / "WORKSPACE").write_text('workspace(name = "tair_mempool")\n')

    bazel = [
        args.bazel, f"--output_user_root={output / 'bazel'}", "build",
        f"--jobs={args.jobs}", *args.bazel_arg, f"--override_repository=tair_mempool={override}",
    ]
    sdk = output / "kvcm-sdk" / "file"
    sdk.mkdir(parents=True, exist_ok=True)
    run(
        *bazel, "--config=client_with_cuda", "--config=client_only_header",
        f"--@rules_cuda//cuda:archs={args.cuda_archs}",
        "//package/rpm/client:kv-cache-manager-client-rpm",
        cwd=public,
    )
    stage_file(public / "bazel-bin/package/rpm/client/kv-cache-manager-client.rpm", sdk / "sdk.rpm")
    (sdk / "BUILD.bazel").write_text(
        'filegroup(name = "file", srcs = ["sdk.rpm"], visibility = ["//visibility:public"])\n'
    )
    (sdk.parent / "WORKSPACE").write_text("")

    manager = output / "kvcm-manager"
    run(*bazel, "--action_env=PATH", "//kv_cache_manager:kv_cache_manager_bin", cwd=public)
    (manager / "bin").mkdir(parents=True, exist_ok=True)
    stage_file(
        public / "bazel-bin/kv_cache_manager/kv_cache_manager_bin",
        manager / "bin/kv_cache_manager_bin",
    )
    (manager / "etc").mkdir(exist_ok=True)
    for config in (public / "package/etc").iterdir():
        if config.suffix in (".conf", ".json"):
            shutil.copy2(config, manager / "etc" / config.name)
    if not (manager / "etc/default_startup_config.json").is_file():
        raise RuntimeError("missing KVCM default_startup_config.json")
    (manager / "BUILD.bazel").write_text('exports_files(["bin/kv_cache_manager_bin"])\n')
    (manager / "WORKSPACE").write_text("")
    is_arm = platform.machine() == "aarch64"
    sdk_repo = "remote_kv_cache_manager_client_rpm_cuda130_" + ("arm" if is_arm else "x86")
    manager_repo = "remote_kv_cache_manager_server_" + ("aarch64" if is_arm else "cuda130")
    (output / "rtp-smoke.bazelrc").write_text(
        f"build:pace_source {shlex.quote(f'--override_repository={sdk_repo}={sdk.parent}')}\n"
        f"build:pace_source {shlex.quote(f'--override_repository={manager_repo}={manager}')}\n"
        "test:pace_source --test_env=KVCM_PACE_FIXTURE\n"
    )


def main():
    parser = argparse.ArgumentParser(description="Build the PACE smoke dependencies from three exact commits.")
    parser.add_argument("--kvcm-public-commit", required=True, type=commit_sha)
    parser.add_argument("--kvcm-internal-commit", required=True, type=commit_sha)
    parser.add_argument("--mempool-commit", required=True, type=commit_sha)
    parser.add_argument("--kvcm-public-repo", default="https://github.com/alibaba/tair-kvcache.git")
    parser.add_argument("--kvcm-internal-repo", help="Git URL or path for the KVCM internal overlay")
    parser.add_argument("--mempool-repo", required=True, help="Git URL or path for the PACE source")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cuda-archs", required=True, help="rules_cuda architecture list for the KVCM SDK")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--bazel", default="bazelisk")
    parser.add_argument("--cmake-arg", action="append", default=[])
    parser.add_argument("--bazel-arg", action="append", default=[])
    parser.add_argument("--component", choices=("all", "pace", "kvcm"), default="all")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.component != "pace":
        if not args.kvcm_internal_repo:
            parser.error("--kvcm-internal-repo is required when building KVCM")
        if not shutil.which("rpmbuild"):
            parser.error("install rpm-build in the KVCM build environment first")
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "kvcm_public": args.kvcm_public_commit,
        "kvcm_internal": args.kvcm_internal_commit,
        "mempool": args.mempool_commit,
        "machine": platform.machine(),
        "cuda_archs": args.cuda_archs,
        "cmake_args": args.cmake_arg,
        "bazel_args": args.bazel_arg,
    }
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = output / "sources.json"
        if manifest.exists() and json.loads(manifest.read_text()) != spec:
            raise RuntimeError(f"{output} already belongs to a different build")
        manifest.write_text(json.dumps(spec, indent=2) + "\n")
        mempool = output / "mempool-src"
        checkout(mempool, args.mempool_repo, args.mempool_commit)
        if args.component in ("all", "kvcm"):
            internal = output / "kvcm-src"
            public = internal / "github-opensource"
            checkout(internal, args.kvcm_internal_repo, args.kvcm_internal_commit)
            checkout(public, args.kvcm_public_repo, args.kvcm_public_commit)
        if args.component in ("all", "pace"):
            build_pace(args, mempool, output)
        if args.component in ("all", "kvcm"):
            build_kvcm(args, public, mempool, output)
    print(f"Build artifacts: {output}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"PACE smoke build failed: {error}", file=sys.stderr)
        sys.exit(1)
