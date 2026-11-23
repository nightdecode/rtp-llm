#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf '%s\n' \
        'Usage: start_pace_smoke_stack.sh UID:GID BIN_DIR NEW_WORK_DIR [tcp|rdma|gdr [RDMA_DEVICE [RDMA_PORT [GID_INDEX]]]] -- COMMAND [ARG ...]'
}

if [[ ${1:-} == --help ]]; then
    usage
    exit 0
fi
if (( EUID != 0 )); then
    printf 'Run the namespace launcher as root. Services and the test run as UID:GID.\n' >&2
    exit 2
fi
if [[ ${1:-} != --private-mounts ]]; then
    current_mount_namespace=$(readlink /proc/self/ns/mnt)
    exec unshare --mount --propagation private -- bash "$0" \
        --private-mounts "$current_mount_namespace" "$@"
fi
if [[ ${2:-} != mnt:\[*\] || $2 == "$(readlink /proc/self/ns/mnt)" ]]; then
    printf 'Could not create a private mount namespace.\n' >&2
    exit 2
fi
shift 2
options=()
while (( $# > 0 )) && [[ $1 != -- ]]; do
    options+=("$1")
    shift
done
if (( $# < 2 || ${#options[@]} < 3 || ${#options[@]} > 7 )) ||
    [[ ! ${options[0]} =~ ^[1-9][0-9]*:[0-9]+$ ]]; then
    usage >&2
    exit 2
fi
uid=${options[0]%:*}
gid=${options[0]#*:}
shift
test_command=("$@")
transport=${options[3]:-tcp}
case "$transport" in
    tcp)
        if (( ${#options[@]} > 4 )); then usage >&2; exit 2; fi
        transport_args=(--transport tcp)
        ;;
    rdma|gdr)
        if [[ -z ${options[4]:-} ]]; then usage >&2; exit 2; fi
        transport_args=(--transport "$transport" --rdma-device "${options[4]}"
                        --rdma-port "${options[5]:-1}" --gid-index "${options[6]:-0}")
        ;;
    *) usage >&2; exit 2 ;;
esac

bin_dir=$(realpath -e -- "${options[1]}")
consumer_bin_dir="$bin_dir"
if [[ -d "$bin_dir/consumer" ]]; then
    consumer_bin_dir="$bin_dir/consumer"
fi
work_dir=$(realpath -m -- "${options[2]}")
launcher="$(dirname "$(realpath -e -- "$0")")/start_pace_smoke.py"
if ! user_home=$(getent passwd "$uid" | cut -d: -f6) || [[ -z "$user_home" ]]; then
    printf 'UID %s must have a passwd entry.\n' "$uid" >&2
    exit 2
fi
as_user=(setpriv --reuid "$uid" --regid "$gid" --clear-groups --no-new-privs env "HOME=$user_home")
parent_ip=198.18.78.1
provider_ip=198.18.78.2
meta_url="http://$parent_ip:24910"
provider_status_url="http://$provider_ip:24916/api/status"
export no_proxy="${no_proxy:+$no_proxy,}127.0.0.1,$parent_ip,$provider_ip"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,$parent_ip,$provider_ip"

mkdir -m 0755 -- "$work_dir"
install -d -m 0755 -o "$uid" -g "$gid" "$work_dir/services" "$work_dir/meta-state"
mount -t tmpfs -o size=6g,mode=1777,nosuid,nodev tmpfs /dev/shm
mount --bind "$work_dir/meta-state" /var/log
listeners=$(ss -H -ltn '( sport >= :24910 and sport <= :24917 )')
routes=$(ip route show match 198.18.78.0/30)
if [[ -n "$listeners" ]] ||
    [[ -n $(sed '/^default /d' <<<"$routes") ]]; then
    printf 'PACE smoke ports or subnet already in use in this network namespace.\n' >&2
    exit 1
fi

link_created=0
cleanup() {
    trap '' INT TERM
    local -a running
    mapfile -t running < <(jobs -pr)
    if (( ${#running[@]} )); then
        kill -TERM "${running[@]}" 2>/dev/null || true
        for attempt in {1..40}; do
            mapfile -t running < <(jobs -pr)
            (( ${#running[@]} )) || break
            sleep 0.25
        done
        if (( ${#running[@]} )); then
            kill -KILL "${running[@]}" 2>/dev/null || true
        fi
        wait 2>/dev/null || true
    fi
    if (( link_created )) && ip link show dev pace-consumer >/dev/null 2>&1; then
        ip link del pace-consumer
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

wait_http() {
    local pid=$1 url=$2
    for attempt in {1..60}; do
        kill -0 "$pid" 2>/dev/null || break
        if curl --noproxy '*' -fsS --max-time 1 "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.5
    done
    printf 'PACE endpoint did not become ready: %s. See %s/*.log\n' "$url" "$work_dir" >&2
    return 1
}

unshare --net --mount --propagation private -- bash -c '
    set -euo pipefail
    work_dir=$1
    shift
    mount -t tmpfs -o size=4g,mode=1777,nosuid,nodev tmpfs /dev/shm
    touch "$work_dir/provider.namespace"
    for attempt in {1..120}; do
        [[ ! -f "$work_dir/provider.start" ]] || exec "$@"
        sleep 1
    done
    exit 1
' bash "$work_dir" "${as_user[@]}" python3 "$launcher" provider \
    --bin-dir "$bin_dir" --work-dir "$work_dir/services/provider" --port-base 24914 \
    --meta-url "$meta_url" --interface pace-provider "${transport_args[@]}" \
    > "$work_dir/provider.log" 2>&1 &
provider_pid=$!
for attempt in {1..60}; do
    [[ ! -f "$work_dir/provider.namespace" ]] || break
    kill -0 "$provider_pid" 2>/dev/null || break
    sleep 0.1
done
if [[ ! -f "$work_dir/provider.namespace" ]]; then
    printf 'Provider namespace setup failed. See %s/provider.log\n' "$work_dir" >&2
    exit 1
fi

ip link add pace-consumer type veth peer name pace-provider
link_created=1
ip link set pace-provider netns "$provider_pid"
ip addr add "$parent_ip/30" dev pace-consumer
ip link set pace-consumer up
nsenter --target "$provider_pid" --net ip addr add "$provider_ip/30" dev pace-provider
nsenter --target "$provider_pid" --net ip link set lo up
nsenter --target "$provider_pid" --net ip link set pace-provider up

"${as_user[@]}" python3 "$launcher" meta \
    --bin-dir "$bin_dir" --work-dir "$work_dir/services/meta" --port-base 24910 \
    > "$work_dir/meta.log" 2>&1 &
meta_pid=$!
wait_http "$meta_pid" "$meta_url/v1/api/healthy"
touch "$work_dir/provider.start"
wait_http "$provider_pid" "$provider_status_url"
"${as_user[@]}" python3 "$launcher" consumer \
    --bin-dir "$consumer_bin_dir" --work-dir "$work_dir/services/consumer" --port-base 24914 \
    --meta-url "$meta_url" --provider-status-url "$provider_status_url" \
    --interface pace-consumer "${transport_args[@]}" \
    > "$work_dir/consumer.log" 2>&1 &
consumer_pid=$!
wait_http "$consumer_pid" "http://127.0.0.1:24917/pace/metrics"

export KVCM_PACE_FIXTURE="$work_dir/services/consumer/fixture.json"
printf 'PACE %s fixture ready at %s\n' "$transport" "$KVCM_PACE_FIXTURE"
"${as_user[@]}" "${test_command[@]}" &
wait "$!"
