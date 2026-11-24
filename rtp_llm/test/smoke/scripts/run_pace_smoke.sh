#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' 'Usage: run_pace_smoke.sh [--namespace] [--bazelrc=SOURCE_BAZELRC] [COMMON_BAZEL_ARGS...]' \
        '       run_pace_smoke.sh --namespace --source PUBLIC_SHA INTERNAL_SHA MEMPOOL_SHA [COMMON_BAZEL_ARGS...]' \
        'Without --namespace, run as the command of start_pace_smoke_stack.sh.' \
        'With --namespace, set PACE_SMOKE_TRANSPORT (default tcp) and PACE_SMOKE_RDMA_DEVICE for RDMA.' \
        'To use an existing source build, pair --bazelrc=SOURCE_BAZELRC with its PACE_SMOKE_BIN_DIR.'
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
namespace=0
if [[ ${1:-} == --namespace ]]; then
    namespace=1
    shift
    export PACE_SMOKE_TRANSPORT=${PACE_SMOKE_TRANSPORT:-tcp}
fi
if [[ $(uname -sm) != "Linux x86_64" ]]; then
    printf 'PACE smoke requires an x86_64 Linux CUDA13 environment.\n' >&2
    exit 2
fi
if (( ! namespace )) && [[ -z ${PACE_SMOKE_STATE:-} ]]; then
    usage >&2
    exit 2
fi
if (( namespace )) && { (( EUID == 0 )) || ! command -v sudo >/dev/null; }; then
    printf 'Run --namespace as a non-root user with noninteractive sudo.\n' >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd -- "$script_dir/../../../.."
model_args=()
case ${PACE_SMOKE_TRANSPORT:-} in
    aft)
        if [[ ! -r internal_source/deps/git.bzl ]]; then
            printf 'The AFT model case requires an RTP checkout with internal_source.\n' >&2
            exit 1
        fi
        model_args=(--model)
        target=//rtp_llm/test/smoke:remote_cache_pace_provider_offline
        ;;
    tcp|rdma|gdr|aft-rdma|aft-gdr) target=//rtp_llm/test/smoke:remote_cache_pace_contract ;;
    *) usage >&2; exit 2 ;;
esac

cache_dir=$(realpath -m -- "${PACE_SMOKE_CACHE_DIR:-$HOME/.cache/rtp-llm/pace-slow-read}")
bazel=${BAZEL:-bazelisk}
if [[ -z ${BAZEL:-} ]] && ! command -v "$bazel" >/dev/null; then
    bazel=bazel
fi
bazel_startup=(--batch)
if [[ -n ${BAZEL_OUTPUT_USER_ROOT:-} ]]; then
    bazel_startup+=("--output_user_root=$BAZEL_OUTPUT_USER_ROOT")
fi
bazel_args=(--config=cuda13 --config=remote_kv_cache)
if ! capabilities=${PACE_SMOKE_CUDA_CAPABILITIES:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -Vu | paste -sd,)} ||
    [[ ! $capabilities =~ ^[0-9]+\.[0-9]+(,[0-9]+\.[0-9]+)*$ ]]; then
    printf 'Expose a CUDA GPU to the RTP test environment, including for the TCP contract.\n' >&2
    exit 2
fi
for env_flag in repo_env action_env host_action_env; do
    bazel_args+=("--$env_flag=TF_CUDA_COMPUTE_CAPABILITIES=$capabilities")
done
source_rc=${PACE_SMOKE_SOURCE_BAZELRC:-}
if (( namespace )) && [[ ${1:-} == --source ]]; then
    if (( $# < 4 )) || [[ -n $source_rc || -n ${PACE_SMOKE_BIN_DIR:-} ||
        ! $2 =~ ^[[:xdigit:]]{40}$ || ! $3 =~ ^[[:xdigit:]]{40}$ || ! $4 =~ ^[[:xdigit:]]{40}$ ]]; then
        printf 'Pass three full Git SHAs to --source without source artifact overrides.\n' >&2
        exit 2
    fi
    : "${PACE_SMOKE_KVCM_INTERNAL_REPO:?set the KVCM internal Git URL or path}"
    : "${PACE_SMOKE_MEMPOOL_REPO:?set the mempool Git URL or path}"
    source_pins=("--kvcm-public-commit=${2,,}" "--kvcm-internal-commit=${3,,}" "--mempool-commit=${4,,}")
    shift 4
    if [[ ${1:-} == --bazelrc=* ]]; then
        printf 'Use either --source or --bazelrc, not both.\n' >&2
        exit 2
    fi
    IFS=, read -r -a architectures <<< "$capabilities"
    cuda_archs=()
    for arch in "${architectures[@]}"; do
        sm=${arch/./}
        cuda_archs+=("compute_$sm:sm_$sm")
    done
    build_key=$(printf '%s\n' "${source_pins[@]}" "$capabilities" | sha256sum | cut -d' ' -f1)
    source_output="$cache_dir/source/$build_key"
    build_args=("${source_pins[@]}" "--kvcm-internal-repo=$PACE_SMOKE_KVCM_INTERNAL_REPO"
                "--mempool-repo=$PACE_SMOKE_MEMPOOL_REPO" "--cuda-archs=$(IFS=';'; echo "${cuda_archs[*]}")"
                "--jobs=${PACE_SMOKE_BUILD_JOBS:-8}" "--bazel=$bazel" --cmake-arg=-DTENT_ENABLE_AFT=OFF
                --bazel-arg=--config=py310)
    if [[ -n ${PACE_SMOKE_KVCM_PUBLIC_REPO:-} ]]; then
        build_args+=("--kvcm-public-repo=$PACE_SMOKE_KVCM_PUBLIC_REPO")
    fi
    mkdir -p -- "$source_output"
    printf 'Preparing pinned PACE/KVCM binaries and SDK. Log: %s/build.log\n' "$source_output" >&2
    if ! python3 "$script_dir/build_pace_smoke.py" "${build_args[@]}" --output "$source_output" \
        >> "$source_output/build.log" 2>&1; then
        tail -n 30 -- "$source_output/build.log" >&2
        exit 1
    fi
    export PACE_SMOKE_BIN_DIR="$source_output/pace/bin"
    source_rc="$source_output/rtp-smoke.bazelrc"
fi
if [[ ${1:-} == --bazelrc=* ]]; then
    if [[ ${PACE_SMOKE_PREPARE:-0} == 1 || -n $source_rc ]]; then
        printf 'Use either launcher --source, or manual --bazelrc with an explicit BIN_DIR.\n' >&2
        exit 2
    fi
    source_rc=${1#--bazelrc=}
    shift
fi
if [[ -n $source_rc ]]; then
    if [[ ! -r $source_rc ]]; then
        printf 'Cannot read source Bazel configuration: %s\n' "$source_rc" >&2
        exit 2
    fi
    bazel_startup+=("--bazelrc=$source_rc")
    bazel_args+=(--config=pace_source)
else
    bazel_args+=(--config=pace_prebuilt "--distdir=$cache_dir/distfiles")
fi
bazel_args+=("$@")
prepare="$script_dir/prepare_pace_smoke.py"
test_wrapper=//tools:nvidia_wheel_env
namespace_env=()
if (( namespace )); then
    if [[ -z $source_rc ]]; then
        if [[ -n ${PACE_SMOKE_BIN_DIR:-} ]]; then
            printf 'Pair PACE_SMOKE_BIN_DIR with the Bazel configuration from the same source build.\n' >&2
            exit 2
        fi
        model_args+=(--artifacts)
    else
        : "${PACE_SMOKE_BIN_DIR:?set the PACE runtime directory from the same source build}"
        PACE_SMOKE_BIN_DIR=$(realpath -e -- "$PACE_SMOKE_BIN_DIR")
        export PACE_SMOKE_BIN_DIR
    fi
    if (( ${#model_args[@]} )); then
        checkpoint=$(python3 "$prepare" "$cache_dir" "${model_args[@]}")
        if [[ -n $checkpoint ]]; then export CHECKPOINT_PATH="$checkpoint"; fi
    fi
    test_wrapper=//rtp_llm/test/smoke:pace_namespace_env
    namespace_env=(--test_env=PACE_SMOKE_TRANSPORT --test_env=PACE_SMOKE_BIN_DIR --test_env=PACE_SMOKE_FAULTS
                   --test_env=PACE_SMOKE_RDMA_DEVICE --test_env=PACE_SMOKE_RDMA_PORT --test_env=PACE_SMOKE_GID_INDEX
                   --test_env=CUDA_VISIBLE_DEVICES --test_env=REMOTE_JIT_DIR)
elif [[ ${PACE_SMOKE_PREPARE:-0} == 1 ]]; then
    runtime=()
    if [[ -z $source_rc ]]; then
        model_args+=(--artifacts)
        runtime=(//3rdparty/pace_test_runtime:runtime)
    fi
    if (( ${#model_args[@]} )); then
        python3 "$prepare" "$cache_dir" "${model_args[@]}" > "$PACE_SMOKE_STATE/checkpoint"
    fi
    "$bazel" "${bazel_startup[@]}" build "${bazel_args[@]}" "${runtime[@]}" "$target" "$test_wrapper"
    if [[ -z $source_rc ]]; then
        bin_dir=$("$bazel" "${bazel_startup[@]}" info "${bazel_args[@]}" bazel-bin)
        printf '%s\n' "$bin_dir/3rdparty/pace_test_runtime" > "$PACE_SMOKE_STATE/bin-dir"
    fi
    exit 0
fi

if (( ! namespace )); then
    : "${KVCM_PACE_FIXTURE:?start the PACE stack before testing}"
    if [[ ${PACE_SMOKE_TRANSPORT} == aft ]]; then
        export CHECKPOINT_PATH
        if [[ -s $PACE_SMOKE_STATE/checkpoint ]]; then
            CHECKPOINT_PATH=$(< "$PACE_SMOKE_STATE/checkpoint")
        else
            CHECKPOINT_PATH=$(python3 "$prepare" "$cache_dir" --model)
        fi
    fi
    touch "$PACE_SMOKE_STATE/test-start"
fi
result=0
"$bazel" "${bazel_startup[@]}" test "${bazel_args[@]}" \
    --test_strategy=standalone --strategy=TestRunner=local \
    --run_under="$test_wrapper" \
    --test_env=KVCM_PACE_FIXTURE --test_env=CHECKPOINT_PATH --test_env=LD_LIBRARY_PATH \
    --test_env=NO_PROXY --test_env=no_proxy --test_env=RTP_GPU_ACQUIRE_TIMEOUT \
    "${namespace_env[@]}" --nocache_test_results --test_output=all --test_timeout=600 "$target" || result=$?
if (( namespace )); then exit "$result"; fi
testlogs="bazel-testlogs/${target#//}"
testlogs=${testlogs/:/\/}
if [[ $testlogs/test.log -nt $PACE_SMOKE_STATE/test-start ]]; then
    realpath -e -- "$testlogs" > "$PACE_SMOKE_STATE/test-logs-dir"
fi
exit "$result"
