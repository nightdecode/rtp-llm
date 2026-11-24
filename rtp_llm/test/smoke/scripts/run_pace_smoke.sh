#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' 'Usage: run_pace_smoke.sh [--bazelrc=SOURCE_BAZELRC] [COMMON_BAZEL_ARGS...]' \
        'Run as the command of start_pace_smoke_stack.sh. Its transport selects the test.' \
        'The launcher selects the source SDK/Manager in --source mode.' \
        'For manual --bazelrc=SOURCE_BAZELRC, provide BIN_DIR from the same build.'
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if [[ -z ${PACE_SMOKE_STATE:-} || $(uname -sm) != "Linux x86_64" ]]; then
    usage >&2
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
    printf 'Expose a CUDA GPU to the RTP test container (for example, --gpus device=GPU_UUID).\n' >&2
    exit 2
fi
for env_flag in repo_env action_env host_action_env; do
    bazel_args+=("--$env_flag=TF_CUDA_COMPUTE_CAPABILITIES=$capabilities")
done
source_rc=${PACE_SMOKE_SOURCE_BAZELRC:-}
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
if [[ ${PACE_SMOKE_PREPARE:-0} == 1 ]]; then
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
result=0
"$bazel" "${bazel_startup[@]}" test "${bazel_args[@]}" \
    --test_strategy=standalone --strategy=TestRunner=local \
    --run_under="$test_wrapper" \
    --test_env=KVCM_PACE_FIXTURE --test_env=CHECKPOINT_PATH --test_env=LD_LIBRARY_PATH \
    --test_env=NO_PROXY --test_env=no_proxy --test_env=RTP_GPU_ACQUIRE_TIMEOUT \
    --nocache_test_results --test_output=all --test_timeout=600 "$target" || result=$?
testlogs="bazel-testlogs/${target#//}"
testlogs=${testlogs/:/\/}
if [[ $testlogs/test.log -nt $PACE_SMOKE_STATE/test-start ]]; then
    realpath -e -- "$testlogs" > "$PACE_SMOKE_STATE/test-logs-dir"
fi
exit "$result"
