#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: start_pace_smoke_stack.sh [--faults] [--source PUBLIC_SHA INTERNAL_SHA MEMPOOL_SHA]' \
        '       UID:GID IMAGE BIN_DIR|- NEW_WORK_DIR' \
        '       [tcp | aft RDMA_DEVICE | rdma|gdr RDMA_DEVICE [RDMA_PORT [GID_INDEX]]]' \
        '       -- [CONSUMER_DOCKER_OPTIONS...] -- COMMAND [ARG ...]' \
        'Use aft for the Provider-offline GDR test. --faults enables TCP fault cases.' \
        'Use - to let COMMAND prepare the prebuilt runtime before the stack starts.' \
        'With --source, BIN_DIR is a persistent build cache and COMMAND is run_pace_smoke.sh.'
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
enable_faults=0
if [[ ${1:-} == --faults ]]; then
    enable_faults=1
    shift
fi
source_pins=()
if [[ ${1:-} == --source ]]; then
    if (( $# < 4 )) || [[ ! $2 =~ ^[[:xdigit:]]{40}$ || ! $3 =~ ^[[:xdigit:]]{40}$ || ! $4 =~ ^[[:xdigit:]]{40}$ ]]; then
        printf 'Pass three full Git commit SHAs to --source.\n' >&2
        exit 2
    fi
    source_pins=("--kvcm-public-commit=$2" "--kvcm-internal-commit=$3" "--mempool-commit=$4")
    shift 4
fi
options=()
while (( $# > 0 )) && [[ $1 != -- ]]; do
    options+=("$1")
    shift
done
if (( $# < 3 || ${#options[@]} < 4 || ${#options[@]} > 8 )) ||
    [[ ! ${options[0]} =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
    usage >&2
    exit 2
fi
shift
consumer_options=()
while (( $# > 0 )) && [[ $1 != -- ]]; do
    consumer_options+=("$1")
    shift
done
if (( $# < 2 )); then usage >&2; exit 2; fi
shift
test_command=("$@")

test_user=${options[0]}
image=${options[1]}
bin_dir=${options[2]}
work_dir=$(realpath -m -- "${options[3]}")
transport=${options[4]:-tcp}
devices=()
transport_args=(--transport "$transport")
case "$transport" in
    tcp)
        if (( ${#options[@]} > 5 )); then usage >&2; exit 2; fi
        ;;
    aft)
        if (( ${#options[@]} != 6 )); then usage >&2; exit 2; fi
        transport_args+=(--rdma-device "${options[5]}")
        for node in /sys/class/infiniband_verbs/uverbs*; do
            if [[ -r $node/ibdev && $(< "$node/ibdev") == "${options[5]}" ]] &&
                [[ -c /dev/infiniband/${node##*/} ]]; then
                devices=(--device="/dev/infiniband/${node##*/}")
                break
            fi
        done
        if (( ${#devices[@]} == 0 )); then
            printf 'No uverbs device found for AFT HCA %s.\n' "${options[5]}" >&2
            exit 2
        fi
        ;;
    rdma|gdr)
        if [[ -z ${options[5]:-} ]]; then usage >&2; exit 2; fi
        if (( enable_faults )); then
            printf 'Use aft for RDMA fault recovery or --faults tcp for TCP faults.\n' >&2
            exit 2
        fi
        transport_args+=(--rdma-device "${options[5]}" --rdma-port "${options[6]:-1}"
                         --gid-index "${options[7]:-0}")
        devices=(--device=/dev/infiniband)
        ;;
    *) usage >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd -- "$script_dir/../../../.." && pwd)
state=/tmp/pace-smoke
provider_bin_dir=$state/runtime
runtime_mount=()
source_mount=()
if (( ${#source_pins[@]} )); then
    if [[ $bin_dir == - ]]; then
        printf 'Pass a persistent build cache directory instead of - with --source.\n' >&2
        exit 2
    fi
    : "${PACE_SMOKE_KVCM_INTERNAL_REPO:?set the KVCM internal Git URL or mounted path}"
    : "${PACE_SMOKE_MEMPOOL_REPO:?set the mempool Git URL or mounted path}"
    mkdir -p -- "$bin_dir"
    source_cache=$(realpath -e -- "$bin_dir")
    source_mount=(--mount "type=bind,source=$source_cache,target=$source_cache")
    bin_dir=-
elif [[ $bin_dir != - ]]; then
    bin_dir=$(realpath -e -- "$bin_dir")
    runtime_mount=(--mount "type=bind,source=$bin_dir,target=$provider_bin_dir,readonly")
    bin_dir=$provider_bin_dir
fi

mkdir -m 0755 -- "$work_dir"
printf 'PACE smoke logs: %s\n' "$work_dir"
name="pace-smoke-$$-$RANDOM"
network_id= consumer= provider= builder= fault_pid=
cleanup() {
    trap '' HUP INT TERM
    if [[ -n $fault_pid ]]; then kill "$fault_pid" 2>/dev/null || true; fi
    if [[ -n $consumer ]]; then docker rm -f "$consumer" >/dev/null || true; fi
    if [[ -n $provider ]]; then docker rm -f "$provider" >/dev/null || true; fi
    if [[ -n $builder ]]; then docker rm -f "$builder" >/dev/null || true; fi
    if [[ -n $network_id ]]; then docker network rm "$network_id" >/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

network_id=$(docker network create --driver bridge --label "rtp.pace-smoke=$name" "$name")
common=(--init --user 0:0 --network "$name" --ipc private --ulimit memlock=-1
        --label "rtp.pace-smoke=$name" --entrypoint sleep
        --mount "type=bind,source=$script_dir,target=/pace-smoke-scripts,readonly"
        "${devices[@]}" "${runtime_mount[@]}")
packages=(boost-system boost-thread glog hiredis jemalloc libprometheus-cpp liburing mysql-libs iproute)
consumer_packages=("${packages[@]}" cpio)
build_options=("${consumer_options[@]}")
if (( ${#source_pins[@]} )); then
    consumer_packages+=(rpm-build)
fi
if (( enable_faults )) && [[ $transport == tcp ]]; then
    consumer_options+=(--cap-add NET_ADMIN)
    consumer_packages+=(iproute-tc)
fi
if [[ $transport == aft ]]; then
    mkdir -m 0700 -- "$work_dir/offline"
    consumer_options+=(--mount "type=bind,source=$work_dir,target=$state/run,readonly"
                       --mount "type=bind,source=$work_dir/offline,target=$state/run/offline")
fi
consumer=$(docker create "${consumer_options[@]}" "${common[@]}" "${source_mount[@]}" --name "$name-consumer" \
    --shm-size=6g --workdir "$repo" --mount "type=bind,source=$repo,target=$repo" "$image" infinity)
docker start "$consumer" >/dev/null
if ! user_home=$(docker exec "$consumer" getent passwd "${test_user%:*}" | cut -d: -f6) || [[ -z $user_home ]]; then
    printf 'Test UID %s must have a passwd entry in the Consumer container.\n' "${test_user%:*}" >&2
    exit 2
fi
prepare_container() {
    local container=$1 log=$2
    shift 2
    if ! docker exec "$container" env -u LD_LIBRARY_PATH -u LD_PRELOAD -u PYTHONPATH -u PYTHONHOME bash -ec '
        install -d -m 0755 -o "${1%:*}" -g "${1#*:}" "$2"
        install -d -m 0700 -o "${1%:*}" -g "${1#*:}" "$2/home"
        shift 2
        if (( $# )) && ! rpm -q "$@" >/dev/null; then
            dnf -y install "$@"
        fi
    ' bash "$test_user" "$state" "$@" > "$log" 2>&1; then
        printf 'PACE runtime dependency setup failed. See %s\n' "$log" >&2
        return 1
    fi
}

prepare_container "$consumer" "$work_dir/consumer-packages.log" "${consumer_packages[@]}"
if [[ $transport == aft ]]; then
    docker exec "$consumer" chown "$test_user" "$state/run/offline"
    if [[ ! -w $work_dir/offline ]]; then
        printf 'The launcher and test user must both be able to write the fault handshake directory.\n' >&2
        exit 2
    fi
fi
if ! docker exec --user "$test_user" "$consumer" test -w "$user_home"; then
    user_home="$state/home"
fi
as_user=(docker exec --user "$test_user" --env "HOME=$user_home" --env "PACE_SMOKE_STATE=$state"
         --env "PACE_SMOKE_TRANSPORT=$transport" --env GIT_TERMINAL_PROMPT=0)
if [[ $transport != tcp ]]; then
    "${as_user[@]}" "$consumer" python3 /pace-smoke-scripts/start_pace_smoke.py check-rdma \
        --rdma-device "${options[5]}" --rdma-port "${options[6]:-1}"
fi
if (( ${#source_pins[@]} )); then
    if ! "${as_user[@]}" "$consumer" test -w "$source_cache"; then
        printf 'Source cache %s must be writable by test user %s.\n' "$source_cache" "$test_user" >&2
        exit 2
    fi
    if ! capabilities=$("${as_user[@]}" "$consumer" nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -Vu | paste -sd,) ||
        [[ ! $capabilities =~ ^[0-9]+\.[0-9]+(,[0-9]+\.[0-9]+)*$ ]]; then
        printf 'Expose a CUDA GPU to the RTP test container before building from source.\n' >&2
        exit 2
    fi
    IFS=, read -r -a architectures <<< "$capabilities"
    cuda_archs=()
    for arch in "${architectures[@]}"; do
        sm=${arch/./}
        cuda_archs+=("compute_$sm:sm_$sm")
    done
    pace_image=${PACE_SMOKE_PACE_BUILD_IMAGE:-hub.docker.alibaba-inc.com/system-analysis/pace-mp-dev:deps-20260818}
    if ! docker image inspect "$pace_image" >/dev/null 2>&1; then
        printf 'Pulling PACE build image. See %s/pace-image.log\n' "$work_dir"
        docker pull "$pace_image" > "$work_dir/pace-image.log" 2>&1
    fi
    build_args=("${source_pins[@]}" "--kvcm-internal-repo=$PACE_SMOKE_KVCM_INTERNAL_REPO"
                "--mempool-repo=$PACE_SMOKE_MEMPOOL_REPO" "--cuda-archs=$(IFS=';'; echo "${cuda_archs[*]}")"
                "--jobs=${PACE_SMOKE_BUILD_JOBS:-8}" --cmake-arg=-DTENT_ENABLE_AFT=OFF --bazel-arg=--config=py310)
    if [[ -n ${PACE_SMOKE_KVCM_PUBLIC_REPO:-} ]]; then
        build_args+=("--kvcm-public-repo=$PACE_SMOKE_KVCM_PUBLIC_REPO")
    fi
    build_key=$(printf '%s\n' "${source_pins[@],,}" "$capabilities" \
        "$(docker inspect --format '{{.Image}}' "$consumer")" "$(docker image inspect --format '{{.Id}}' "$pace_image")" |
        sha256sum | cut -d' ' -f1)
    output="$source_cache/$build_key"
    if [[ ! -f $output/pace.ready || ! -x $output/pace/bin/tair_mempool_meta_service ||
          ! -x $output/pace/bin/tair_mempool_server || ! -f $output/pace/bin/libtent_shared.so.5 ]]; then
        printf 'Building PACE from source. See %s/pace-build.log\n' "$work_dir"
        builder=$(docker create "${build_options[@]}" "${common[@]}" "${source_mount[@]}" \
            --name "$name-build" "$pace_image" infinity)
        docker start "$builder" >/dev/null
        prepare_container "$builder" "$work_dir/pace-build-container.log"
        "${as_user[@]}" "$builder" python3 /pace-smoke-scripts/build_pace_smoke.py \
            "${build_args[@]}" --component=pace --output "$output" > "$work_dir/pace-build.log" 2>&1 &
        wait "$!"
        docker rm -f "$builder" >/dev/null
        builder=
    else
        printf 'Reusing PACE source build in %s\n' "$output"
    fi
    printf 'Preparing KVCM from source. See %s/kvcm-build.log\n' "$work_dir"
    "${as_user[@]}" "$consumer" python3 /pace-smoke-scripts/build_pace_smoke.py \
        "${build_args[@]}" --component=kvcm --output "$output" > "$work_dir/kvcm-build.log" 2>&1 &
    wait "$!"
    cp -- "$output/sources.json" "$work_dir/sources.json"
    as_user+=(--env "PACE_SMOKE_SOURCE_BAZELRC=$output/rtp-smoke.bazelrc"
              --env "PACE_SMOKE_CUDA_CAPABILITIES=$capabilities")
fi
if [[ $bin_dir == - ]]; then
    printf 'Preparing PACE smoke target and test inputs. See %s/prepare.log\n' "$work_dir"
    "${as_user[@]}" --env PACE_SMOKE_PREPARE=1 "$consumer" "${test_command[@]}" > "$work_dir/prepare.log" 2>&1 &
    wait "$!"
    if (( ${#source_pins[@]} )); then
        bin_dir="$output/pace/bin"
    else
        bin_dir=$(docker exec "$consumer" cat "$state/bin-dir")
    fi
fi

provider=$(docker create "${common[@]}" --name "$name-provider" --shm-size=4g \
    --tmpfs /var/log:rw,nosuid,nodev,mode=1777 "$image" infinity)
docker start "$provider" >/dev/null
prepare_container "$provider" "$work_dir/provider-packages.log" "${packages[@]}"
if (( ${#runtime_mount[@]} == 0 )); then
    "${as_user[@]}" "$consumer" tar -C "$bin_dir" -chf - \
        tair_mempool_meta_service tair_mempool_server libtent_shared.so.5 |
        "${as_user[@]}" -i "$provider" bash -ec 'mkdir -- "$1"; tar -xf - -C "$1"' bash "$provider_bin_dir"
fi

ip_format="{{(index .NetworkSettings.Networks \"$name\").IPAddress}}"
consumer_ip=$(docker inspect --format "$ip_format" "$consumer")
provider_ip=$(docker inspect --format "$ip_format" "$provider")
if [[ -z $consumer_ip || -z $provider_ip || $consumer_ip == "$provider_ip" ]]; then
    printf 'PACE needs two distinct container IPs on the test bridge.\n' >&2
    exit 1
fi
route_interface() {
    docker exec "$1" ip -4 -o route get "$2" |
        awk -v source="$3" '{for (field = 1; field < NF; ++field) {
            if ($field == "dev") device = $(field + 1)
            if ($field == "src" && $(field + 1) == source) print device
        }}'
}
consumer_interface=$(route_interface "$consumer" "$provider_ip" "$consumer_ip")
provider_interface=$(route_interface "$provider" "$consumer_ip" "$provider_ip")
if [[ -z $consumer_interface || -z $provider_interface ]]; then
    printf 'PACE container routes do not use the test bridge.\n' >&2
    exit 1
fi
printf 'PACE %s: Consumer %s (%s), Provider %s (%s)\n' \
    "$transport" "$consumer_ip" "$consumer_interface" "$provider_ip" "$provider_interface"
for variable in NO_PROXY no_proxy; do
    proxy=$(docker exec "$consumer" printenv "$variable" || true)
    as_user+=(--env "$variable=${proxy:+$proxy,}127.0.0.1,$consumer_ip,$provider_ip")
done

wait_http() {
    local pid=$1 url=$2 log=$3 attempt
    for attempt in {1..60}; do
        kill -0 "$pid" 2>/dev/null || break
        if docker exec "$consumer" curl --noproxy '*' -fsS --max-time 1 "$url" > "$log" 2>/dev/null; then
            return 0
        fi
        sleep 0.5
    done
    printf 'PACE endpoint did not become ready: %s. See %s/*.log\n' "$url" "$work_dir" >&2
    return 1
}

launcher=/pace-smoke-scripts/start_pace_smoke.py
data_port_base=24914
meta_url="http://$provider_ip:24910"
provider_status_url="http://$provider_ip:$((data_port_base + 2))/api/status"
metrics_url="http://127.0.0.1:$((data_port_base + 3))/pace/metrics"
"${as_user[@]}" "$provider" python3 "$launcher" meta \
    --bin-dir "$provider_bin_dir" --work-dir "$state/meta" --port-base 24910 \
    --transport "$transport" > "$work_dir/meta.log" 2>&1 &
wait_http "$!" "$meta_url/v1/api/healthy" "$work_dir/meta-health.json"
"${as_user[@]}" "$provider" python3 "$launcher" provider --bin-dir "$provider_bin_dir" \
    --work-dir "$state/provider" --port-base "$data_port_base" --meta-url "$meta_url" \
    --interface "$provider_interface" "${transport_args[@]}" > "$work_dir/provider.log" 2>&1 &
wait_http "$!" "$provider_status_url" "$work_dir/provider-status.json"

consumer_args=()
if [[ $transport == aft ]]; then
    consumer_args=(--provider-offline-dir "$state/run/offline")
elif (( enable_faults )); then
    fault_socket="$state/fault.sock"
    consumer_args=(--fault-socket "$fault_socket")
fi
"${as_user[@]}" "$consumer" bash -ec '
    bin_dir=$1
    shift
    if [[ -d "$bin_dir/consumer" ]]; then bin_dir="$bin_dir/consumer"; fi
    exec python3 "$1" consumer --bin-dir "$bin_dir" "${@:2}"
' bash "$bin_dir" "$launcher" --work-dir "$state/consumer" --port-base "$data_port_base" \
    --meta-url "$meta_url" --provider-status-url "$provider_status_url" \
    --interface "$consumer_interface" "${transport_args[@]}" "${consumer_args[@]}" > "$work_dir/consumer.log" 2>&1 &
wait_http "$!" "$metrics_url" "$work_dir/metrics-before.txt"

if [[ $transport == tcp ]] && (( enable_faults )); then
    docker exec "$consumer" python3 -u /pace-smoke-scripts/pace_smoke_faults.py "$fault_socket" \
        "${test_user%:*}" "${test_user#*:}" "$consumer_interface" "$consumer_ip" "$provider_ip" \
        "$((data_port_base + 1))" > "$work_dir/faults.log" 2>&1 &
    fault_pid=$!
    for attempt in {1..60}; do
        docker exec "$consumer" test -S "$fault_socket" && break
        kill -0 "$fault_pid" 2>/dev/null || break
        sleep 0.1
    done
    if ! kill -0 "$fault_pid" 2>/dev/null || ! docker exec "$consumer" test -S "$fault_socket"; then
        printf 'PACE fault control did not start. See %s/faults.log\n' "$work_dir" >&2
        exit 1
    fi
fi

signal_provider() {
    docker exec --user "$test_user" "$provider" bash -ec '
        read -r pid < "$1"
        [[ $pid =~ ^[1-9][0-9]*$ && $pid -gt 1 ]]
        [[ $(readlink "/proc/$pid/exe") == "$2" ]]
        kill -s "$3" "$pid"
    ' bash "$state/provider/pid" "$provider_bin_dir/tair_mempool_server" "$1"
}

if [[ $transport == aft ]]; then
    (
        control_dir=$work_dir/offline
        until [[ -f $control_dir/pause.request ]]; do sleep 0.025; done
        offline_us=$(date +%s%6N)
        signal_provider STOP
        printf 'Provider paused epoch_us=%s\n' "$offline_us"
        printf '%s\n' "$offline_us" > "$control_dir/paused.next"
        mv "$control_dir/paused.next" "$control_dir/paused"
        deadline=$((SECONDS + 2))
        until [[ -f $control_dir/kill.request ]] || (( SECONDS >= deadline )); do sleep 0.025; done
        signal_provider KILL
        killed_us=$(date +%s%6N)
        printf 'Provider killed epoch_us=%s\n' "$killed_us"
        printf '%s\n' "$killed_us" > "$control_dir/killed.next"
        mv "$control_dir/killed.next" "$control_dir/killed"
    ) > "$work_dir/faults.log" 2>&1 &
    fault_pid=$!
fi

fixture="$state/consumer/fixture.json"
docker exec "$consumer" cat "$fixture" > "$work_dir/fixture.json"
printf 'PACE %s fixture ready at %s inside Consumer\n' "$transport" "$fixture"
("${as_user[@]}" --env "KVCM_PACE_FIXTURE=$fixture" "$consumer" "${test_command[@]}" 2>&1 |
    tee "$work_dir/test.log") &
result=0
wait "$!" || result=$?
if test_logs=$(docker exec "$consumer" cat "$state/test-logs-dir" 2>/dev/null); then
    docker cp "$consumer:$test_logs/." "$work_dir/bazel-testlogs" || \
        printf 'Could not collect Bazel test outputs. See %s/test.log\n' "$work_dir" >&2
fi
docker exec "$consumer" curl --noproxy '*' -fsS --max-time 5 "$metrics_url" > "$work_dir/metrics-after.txt" || true
exit "$result"
