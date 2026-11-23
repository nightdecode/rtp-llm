# Local PACE smoke

`scripts/start_pace_smoke_stack.sh` starts a memory-backed Meta, a Provider and
a Consumer once for the test command. Each PACE smoke case receives the same
`KVCM_PACE_FIXTURE`, starts its own KVCM Manager and configures PACE storage.
The launcher stops PACE when the whole command exits.

## Runtime and artifacts

The prebuilt smoke configuration pins its artifacts in
`3rdparty/pace_test_runtime/repositories.bzl`.

- PACE Meta and CSSDV Provider use `452ba5b` from 2026-11-18, CI runs
  `91125106` and `91125037`. These are the RPMs from the corresponding
  `v3.1.21_2026_11_18_16_14_452ba5b` image builds.
- PACE Consumer (`f14290c5f767`) and the CUDA13 KVCM SDK
  (`2026_11_20_16_05`) match RTP internal branch
  `feature/dsv41-dev-main-202161013` at `8cf717a43c`, for x86_64 and aarch64.
- `--config=pace_prebuilt` selects that SDK and the Manager archive from
  `isearch/kv_cache_manager_prod_dashllm:2026_11_07_00_14_a48396c_13e2046`
  on x86_64 or the matching `_arm64` tag on aarch64.

`//3rdparty/pace_test_runtime:runtime` extracts Meta and Provider under
`bazel-bin/3rdparty/pace_test_runtime` and Consumer in its `consumer/`
subdirectory. Provider and Consumer each use the `libtent_shared.so.5` from
their own package. A binary directory from the source build contains one
server binary used for both roles.

For authenticated artifacts, use `--distdir` with the pinned files. Save each
RPM using its URL's filename. Save the Manager image's `COPY` layer under its
`sha256:…` blob filename as declared in `repositories.bzl`. Bazel checks the
hashes and extracts the Manager archive inside that layer. This command uses
the x86_64 digest. For aarch64, use the ARM digest from `repositories.bzl`.

```bash
layer=sha256:a8332a4e6e44110cd4d299071c3c5320a90f736b141454733777f900062c6277
crane blob "hub.docker.alibaba-inc.com/isearch/kv_cache_manager_prod_dashllm@$layer" \
  > "$distfiles/$layer"
```

The test environment must provide compatible ELF dependencies (including
glibc, libstdc++, libibverbs, librdmacm, Boost, Prometheus, glog, OpenSSL, hiredis,
MySQL client and jemalloc), `rpm2cpio`, and `cpio`. The CUDA13 SDK initializes a
CUDA device even for the CPU-buffer contract.

## Run a group of tests

The launcher isolates Provider networking and shared memory on the Linux test
machine. Run it as root with permission to create mount and network namespaces.
It runs the binaries and tests as the supplied non-root `UID:GID`. In a container,
grant `SYS_ADMIN` and `NET_ADMIN` and set `--ulimit memlock=-1`. The launcher needs
ports 24910–24917 and subnet `198.18.78.0/30`.

Use `rtp_config=cuda13` on x86_64 or `rtp_config=cuda13_arm` on aarch64.
Set `distfiles` to the directory containing the authenticated downloads for
that architecture.

```bash
rtp_config=cuda13
bazel build --config="$rtp_config" --distdir="$distfiles" \
  //3rdparty/pace_test_runtime:runtime
```

Set `test_user` to the test user's non-root `UID:GID`. Omit the transport
arguments to use TENT TCP. For RDMA or GDR, supply the mode and active HCA,
optionally followed by its port and GID index (defaults 1 and 0). GDR requires
a GPUDirect-capable GPU and HCA. Pass all test targets that should share
PACE to the same command. The example includes one SDK contract and one
model smoke. The model smoke requires its configured model files and GPU.

```bash
scripts="$PWD/rtp_llm/test/smoke/scripts"
bin="$PWD/bazel-bin/3rdparty/pace_test_runtime"
transport=()  # For RDMA or GDR, use (rdma mlx5_2) or (gdr mlx5_2).
bazel_rc=()
artifact_config=(--config=pace_prebuilt --distdir="$distfiles")
run="/tmp/pace-smoke-$(date +%s)-$$"

bash "$scripts/start_pace_smoke_stack.sh" "$test_user" "$bin" "$run" \
  "${transport[@]}" -- \
  bazel --batch "${bazel_rc[@]}" test \
  --config="$rtp_config" --config=remote_kv_cache "${artifact_config[@]}" \
  --test_strategy=standalone \
  --run_under=//rtp_llm/test/utils:gpu_lock \
  --test_env=KVCM_PACE_FIXTURE --test_env=LD_LIBRARY_PATH \
  --test_env=NO_PROXY --test_env=no_proxy --nocache_test_results --test_output=all \
  //rtp_llm/test/smoke:remote_cache_pace_contract \
  //rtp_llm/test/smoke:remote_cache_pace_basic
```

The cases run serially and PACE state persists between them. The existing
`kill_remote` fault case stops its own KVCM Manager. The launcher keeps
`meta.log`, `provider.log` and `consumer.log` under `run`. KVCM logs are saved
under `kvcm_logs` in each test's undeclared outputs.

To attach an already running PACE environment, pass
`--test_env=KVCM_PACE_FIXTURE=/path/to/fixture.json` to `bazel test`. The caller
owns the PACE processes.

The launcher records the selected `transport` in the Consumer fixture. The
contract checks payload contents and uses that mode to select the transport
counters. For GDR, it uses GPU buffers and requires matching native RDMA and
GDR byte deltas in both directions. An explicit `--test_arg=--transport=...`
must agree with the fixture. Use this argument to set the mode for older
fixtures without the field.

### Build from three commits

Provide Git URLs or local paths for the private repositories and full commit
SHAs. Git uses the developer's existing repository access. Build PACE in a
CMake environment with libibverbs headers and KVCM in its CUDA build
environment with Bazelisk and `rpmbuild`.

```bash
pins=(
  --kvcm-public-commit "$KVCM_PUBLIC_SHA"
  --kvcm-internal-repo "$KVCM_INTERNAL_REPO"
  --kvcm-internal-commit "$KVCM_INTERNAL_SHA"
  --mempool-repo "$MEMPOOL_REPO"
  --mempool-commit "$MEMPOOL_SHA"
  --cuda-archs "$CUDA_ARCHS"
)
python3 "$scripts/build_pace_smoke.py" "${pins[@]}" --component=pace \
  --cmake-arg=-DTENT_ENABLE_AFT=OFF --output "$artifacts/pace"
python3 "$scripts/build_pace_smoke.py" "${pins[@]}" --component=kvcm \
  --bazel-arg=--config=py310 --output "$artifacts/kvcm"
```

Use `bin="$artifacts/pace/pace/bin"`,
`bazel_rc=(--bazelrc="$artifacts/kvcm/rtp-smoke.bazelrc")`, and
`artifact_config=(--config=pace_source)` in the launcher command. Keep the
artifacts at the same paths when running the test so Bazel can resolve the
KVCM Manager and SDK overrides.
