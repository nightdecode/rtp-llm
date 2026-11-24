# Local PACE smoke

## One Linux environment (no Docker needed by the launcher)

From the RTP checkout, on an x86_64 Linux machine or in an existing CUDA13 RTP
development environment:

```bash
scripts=rtp_llm/test/smoke/scripts
bash "$scripts/run_pace_smoke.sh" --namespace
```

This runs the TENT TCP byte-round-trip case with the pinned binaries and SDK.
It gives Provider its own IP and `/dev/shm`. Consumer and the RTP test share
the other IP and `/dev/shm`; Meta and the test's KVCM Manager run there too.
The launcher uses `sudo -n` for namespace setup and TCP fault control only;
the binaries and the test run as the original user, with the usual NVIDIA
libraries and GPU lock. When the test exits, its processes and namespaces go
away. Service logs and counters are saved under `pace/` in the Bazel test's
undeclared outputs. Set `BAZEL_OUTPUT_USER_ROOT` to reuse a worktree's Bazel
cache, and pass any common Bazel options after `--namespace` or the source pins.

Use a non-root account with noninteractive `sudo` and permission to create
network, mount, IPC and PID namespaces, mount `tmpfs` and create a `veth` pair.
The environment needs `unshare`, `nsenter`, `setpriv`, `ip`, `mount`, `curl`,
Python 3, Bazel, a CUDA device (also for the current TCP SDK), and the runtime
libraries below. If starting from an existing container, its security policy
must already allow these operations. The launcher does not install host
packages or change host networking or mounts.

Set the mode once on the launcher; the test reads it from the fixture:

```bash
PACE_SMOKE_FAULTS=1 bash "$scripts/run_pace_smoke.sh" --namespace
PACE_SMOKE_TRANSPORT=aft-gdr PACE_SMOKE_RDMA_DEVICE="$hca" \
  bash "$scripts/run_pace_smoke.sh" --namespace
```

For the model recovery case use `PACE_SMOKE_TRANSPORT=aft`. RDMA modes need a
shared RDMA namespace, an active HCA and unlimited `memlock` in the test.
The AFT modes expose only the selected HCA to the test stack. For TENT RDMA
or GDR, `PACE_SMOKE_RDMA_PORT` and `PACE_SMOKE_GID_INDEX` default to `1` and `0`.
TCP faults also need `tc` and kernel `prio`, `flower` and `netem` support.

To use three commits rather than the pinned releases:

```bash
PACE_SMOKE_KVCM_INTERNAL_REPO="$KVCM_INTERNAL_REPO" PACE_SMOKE_MEMPOOL_REPO="$MEMPOOL_REPO" \
  bash "$scripts/run_pace_smoke.sh" --namespace \
  --source "$KVCM_PUBLIC_SHA" "$KVCM_INTERNAL_SHA" "$MEMPOOL_SHA"
```

Use Git URLs or absolute paths and a CUDA13 environment with access to all
three repos and the PACE/KVCM build dependencies (including CMake, the C++ and
libibverbs development libraries, and `rpmbuild`). Complete builds are reused
from the `source/` subdirectory of `PACE_SMOKE_CACHE_DIR` per commit tuple and
CUDA architecture list; use a separate cache when changing toolchains. Manager
and the SDK come from the same source build. To reuse an existing build, set
`PACE_SMOKE_BIN_DIR` to its `pace/bin` directory and pass its
`--bazelrc=/path/to/rtp-smoke.bazelrc` instead of `--source`.

## Two containers

`scripts/start_pace_smoke_stack.sh` starts two containers on the same Docker
host: one for Meta and Provider, and one for Consumer and the RTP tests. The
containers get different IPs on a fresh bridge and separate `/dev/shm` mounts.
Each PACE smoke case uses the same `KVCM_PACE_FIXTURE` and starts its own KVCM
Manager binary. The launcher removes both containers and the bridge when the
test command exits. It returns the test exit code and keeps component logs,
transport counters and Bazel test outputs in the new run directory.

## Choose a test

| Launcher mode | Test |
| --- | --- |
| `aft-rdma RDMA_DEVICE` | Native AFT read/write smoke with CPU buffers. Checks payload bytes and requires a completed Provider RDMA read. |
| `aft-gdr RDMA_DEVICE` | Native AFT read/write smoke with GPU buffers. Requires successful GPUDirect read bytes; writes use staging. |
| `aft RDMA_DEVICE` | Primary RDMA test: one Qwen Prefill with native AFT and GDR reads. Takes the Provider offline and verifies safe release of the original RTP frozen blocks. |
| `tcp` (default) | TENT TCP SDK smoke with byte-for-byte readback. Add `--faults` for slow reads, slow writes, and packet loss. No RTP thaw assertion. |
| `rdma RDMA_DEVICE` or `gdr RDMA_DEVICE` | Existing TENT transport diagnostics. Checks payload and transport counters only. |

Use an x86_64 Linux CUDA13 RTP development environment. All AFT modes need a native
AFT-compatible HCA with port 1 active; `aft-gdr` and `aft` also need a
GPUDirect-capable GPU.
The model build needs `internal_source` and compiles the late KVCM read-safety
recovery from this checkout; no source overlay is needed. Keep the KVCM SDK
and PACE Consumer compatible with late read-safety queries when building them
from custom commits. The AFT profiles use DRAM Push without SSD, and both
GPU-read cases require actual GPUDirect reads. In those modes the Consumer
waits for GPU memory registration before advertising its connections.

Run from the public checkout (`repo`). Set `test_user` to the launcher user's
non-root `UID:GID`, except that a root launcher can select a non-root test user.
The user must exist in the image or a mounted passwd file. `checkout_root`
covers both the public checkout and its `internal_source` path; `run` must
not already exist. Set `git_options` as shown below before running the command.

```bash
scripts="$repo/rtp_llm/test/smoke/scripts"
transport=(tcp)                   # Or (aft-rdma "$hca"), (aft-gdr "$hca"), (aft "$hca").
faults=()                         # Use (--faults) for TCP fault cases.
source_args=()                    # Set the three pins below to build from source.
runtime=-                        # Use the pinned prebuilt runtime by default.
bash "$scripts/start_pace_smoke_stack.sh" "${faults[@]}" "${source_args[@]}" \
  "$test_user" "$image" "$runtime" "$run" \
  "${transport[@]}" -- \
  --gpus "device=$gpu" \
  --volume "$checkout_root:$checkout_root" \
  --volume /etc/passwd:/etc/passwd:ro --volume /etc/group:/etc/group:ro \
  "${git_options[@]}" \
  -- bash "$scripts/run_pace_smoke.sh"
```

Set `git_options=()` if the image already has access to the Git dependencies.
Otherwise, give the build and test containers your normal Git credentials.
For SSH, mount a socket from an already authorized SSH agent and a trusted
`known_hosts` file rather than baking credentials into the image:

```bash
git_options=(
  --mount "type=bind,source=$SSH_AUTH_SOCK,target=/run/git-agent"
  --mount "type=bind,source=$known_hosts,target=/run/git-known-hosts,readonly"
  --env SSH_AUTH_SOCK=/run/git-agent
  --env 'GIT_SSH_COMMAND=ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o UserKnownHostsFile=/run/git-known-hosts'
)
```

Pass additional Docker options between the two `--` separators. They apply
to the Consumer/test container and, in source mode, to the temporary PACE
build container, never to the Provider. The launcher owns networking and IPC.

The launcher installs missing runtime RPMs from the image's configured package
repositories, including `iproute-tc` in the Consumer when using TCP faults.
With `BIN_DIR=-`, `run_pace_smoke.sh` prepares the pinned PACE and KVCM
artifacts before the stack starts and picks the test from the launcher mode.
Only the AFT model case needs the Qwen checkpoint. Downloads use
the test account's existing registry and artifact access. Images that allow
anonymous pull are downloaded directly; otherwise the script uses `crane` with
the caller's Docker credentials and supplies a pinned build if needed. If
GitHub downloads are slow or blocked, pass
`--env PACE_SMOKE_CRANE_URL=<mirror-archive-url>`; the same SHA256 is checked.

The model is Qwen2.5-0.5B-Instruct at revision `7ae557604adf67be50417f59c2c2f167def9a775`.
The entry verifies a mounted copy or downloads the pinned files from Hugging
Face. Pass `--env CHECKPOINT_PATH` to use a mounted copy, `--env HF_ENDPOINT`
to choose a download mirror, or `--env PACE_SMOKE_CACHE_DIR` to choose a mounted
persistent download cache. To reuse the Bazel output cache, mount it and pass
`--env BAZEL_OUTPUT_USER_ROOT=/path/to/cache` as a Docker option. Extra
arguments to `run_pace_smoke.sh` are common Bazel build and test options.

### What counts as recovery?

`remote_cache_pace_provider_offline` first requires a real remote hit and a
positive Consumer GDR read byte delta. The launcher pauses only its test
Provider, allows a cold-prefix read to enter the data path, and kills that
Provider process while leaving Meta running. The test requires an SDK timeout
that freezes RTP DEVICE blocks, Meta's eviction and notification, and late
buffer-safety evidence from Consumer. The original read's blocks must all
release naturally within 10 seconds of Provider pause, with forced release
disabled. The faulted request must recompute the same output, and the same RTP
engine must serve another request with the Provider still down. The time limit
applies to this profile (one-second Meta polling, eviction after three failed
probes), not to arbitrary production configurations.

With TCP faults enabled, `remote_cache_pace_contract` applies `tc`/`netem` only to
the test TCP data connection. It verifies a fault hit and SDK timeout, then
healthy payload I/O after fault removal. It doesn't run a model, require a
freeze, or wait for a thaw.

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

## Run selected targets

Use a Docker daemon that can start both containers, mount the checkout and
artifacts, and expose the required devices. The launcher does not need host
networking, `SYS_ADMIN`, or an AppArmor override. On Aone, run the same command
on a GPU runner with that Docker access and use paths visible to the daemon
(for DinD, bind mounts must be under the job workspace). The `services` field
alone is not sufficient because containers in the same job Pod share an IP.
Use a separate run directory per mode and publish it as the job's artifacts
even on failure. Keep the source, download, and Bazel caches outside that
directory so later jobs can mount them again.
The test UID must exist in the image or in a passwd file mounted by the daemon.
Without a writable mounted home, the launcher uses a temporary home inside the
containers. Pass cache mounts and credential locations explicitly to reuse them.

Use `rtp_config=cuda13` on x86_64 or `rtp_config=cuda13_arm` on aarch64.
Set `distfiles` to the directory containing the authenticated downloads for
that architecture.

```bash
rtp_config=cuda13
bazel build --config="$rtp_config" --distdir="$distfiles" \
  //3rdparty/pace_test_runtime:runtime
```

Pass all compatible targets that should share PACE to the same command.
Run the AFT Provider-offline test on its own stack, since it kills that
stack's Provider. For the optional TENT RDMA/GDR diagnostics, the HCA can be
followed by a port and GID index (defaults 1 and 0). Bazel's default test
wrapper locates the NVIDIA wheel libraries and locks the GPU.

```bash
scripts="$PWD/rtp_llm/test/smoke/scripts"
bin="$PWD/bazel-bin/3rdparty/pace_test_runtime"
bash "$scripts/start_pace_smoke_stack.sh" "$test_user" "$image" "$bin" "$run" \
  tcp -- \
  --gpus "device=$gpu" \
  --volume "$checkout_root:$checkout_root" --volume "$distfiles:$distfiles:ro" \
  --volume /etc/passwd:/etc/passwd:ro --volume /etc/group:/etc/group:ro \
  -- bazel --batch test \
  --config="$rtp_config" --config=remote_kv_cache --config=pace_prebuilt --distdir="$distfiles" \
  --test_strategy=standalone \
  --test_env=KVCM_PACE_FIXTURE --test_env=LD_LIBRARY_PATH \
  --test_env=NO_PROXY --test_env=no_proxy --nocache_test_results --test_output=all \
  //rtp_llm/test/smoke:remote_cache_pace_contract
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
counters. TENT `gdr` requires matching RDMA and GDR byte deltas in both
directions. The two native AFT smoke modes use the same contract target and
reject TENT traffic. They require Provider RDMA read bytes in `aft-rdma` and
Consumer GPUDirect read bytes in `aft-gdr`, respectively. The Provider-offline
mode `aft` remains a separate model case. An explicit `--test_arg=--transport=...`
must agree with the fixture. Use this argument to set the mode for older
fixtures without the field.

### Build from three commits in containers

Set these variables before running the same launcher command above. No
separate build, service startup, or SDK override command is needed.

```bash
export PACE_SMOKE_KVCM_INTERNAL_REPO="$KVCM_INTERNAL_REPO"
export PACE_SMOKE_MEMPOOL_REPO="$MEMPOOL_REPO"
source_args=(--source "$KVCM_PUBLIC_SHA" "$KVCM_INTERNAL_SHA" "$MEMPOOL_SHA")
runtime="$source_cache"
```

Use full 40-character SHAs and Git URLs or paths mounted in the build
containers. `PACE_SMOKE_KVCM_PUBLIC_REPO` can override the default GitHub URL.
Use a mirror containing the same commit if the runner cannot reach GitHub.
For the known legacy Poco and Havenask pins, the builder uses the upstream
public repositories without changing the pinned revisions or Bazel patches.
The source cache must be writable by `test_user`.
The launcher builds Meta and both PACE roles from the mempool pin in the PACE
build image, and builds the Manager and SDK together in the RTP CUDA13 image.
The test links that SDK, not the prebuilt SDK. The cache reuses complete
artifacts for the same three pins, visible CUDA architectures and build images.
The run directory records the pins in `sources.json`.
`PACE_SMOKE_PACE_BUILD_IMAGE` and `PACE_SMOKE_BUILD_JOBS` can override the PACE
build image and source build parallelism.

For AFT, the pinned Consumer and SDK must support late read-safety evidence,
and the Provider must support DRAM Push and GPU memory-registration exchange.
No fault-enabled PACE binary is required. TENT TCP faults use `--faults`;
real RDMA/GDR Provider-loss recovery uses `aft`, not a `tc` rule on the HCA.
