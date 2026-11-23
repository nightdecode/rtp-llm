"""Real SDK and PACE contracts without importing the model/GPU smoke runner."""

import argparse
import json
import os
import subprocess
import tempfile
import time
from decimal import Decimal
from pathlib import Path
from urllib.request import urlopen

from rtp_llm.test.smoke.pace_fixture import runfile
from rtp_llm.test.smoke.remote_kvcm_server import RemoteKVCMServer


_TRANSPORT_COUNTERS = {
    "tcp": (
        "tairmempool_mp_server_tent_tcp_read_succ_bytes",
        "tairmempool_mp_server_tent_tcp_write_succ_bytes",
    ),
    "rdma": (
        "tairmempool_mp_server_tent_rdma_read_succ_bytes",
        "tairmempool_mp_server_tent_rdma_write_succ_bytes",
    ),
    "gdr": (
        "tairmempool_mp_server_gdr_read_success_bytes",
        "tairmempool_mp_server_gdr_write_success_bytes",
    ),
}
_COUNTERS = {name for pair in _TRANSPORT_COUNTERS.values() for name in pair}


def read_counters(url):
    counters = {}
    with urlopen(url, timeout=5) as response:
        for line in response.read().decode().splitlines():
            fields = line.split()
            if len(fields) >= 2 and fields[0] in _COUNTERS:
                counters[fields[0]] = Decimal(fields[1])
    if counters.keys() != _COUNTERS:
        missing = sorted(_COUNTERS - counters.keys())
        raise RuntimeError(f"Missing PACE transport counters at {url}: {missing}")
    return counters


def check_transport(url, before, transport):
    deadline = time.monotonic() + 5
    expected = {"rdma", "gdr"} if transport == "gdr" else {transport}
    while True:
        after = read_counters(url)
        delta = {
            mode: tuple(after[name] - before[name] for name in names)
            for mode, names in _TRANSPORT_COUNTERS.items()
        }
        active = {mode for mode, counts in delta.items() if counts != (0, 0)}
        read, written = delta[transport]
        gdr_matches_rdma = transport != "gdr" or delta["gdr"] == delta["rdma"]
        if read > 0 and written > 0 and active == expected and gdr_matches_rdma:
            print(f"PASS: {transport} payload bytes read={read} write={written}")
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Expected {transport} payload I/O in both directions, got {delta}")
        time.sleep(0.25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk", required=True)
    transports = ("local", *_TRANSPORT_COUNTERS)
    parser.add_argument("--transport", choices=transports)
    args = parser.parse_args()
    server_path = runfile(
        os.environ["TEST_WORKSPACE"],
        "3rdparty/remote_kv_cache_manager/bin/kv_cache_manager_bin",
    ).parent.parent
    output = Path(os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR", os.getcwd()))
    with tempfile.TemporaryDirectory(prefix="pace-smoke-", dir=os.environ.get("TEST_TMPDIR")) as directory:
        server = RemoteKVCMServer(str(server_path), {
            "PACE_REQUIRED": "true",
        }, str(Path(directory) / "logs"), str(output / "kvcm_logs"))
        try:
            fixture_config = server.pace_fixture.config
            transport = fixture_config.get("transport", args.transport or "local")
            if transport not in transports:
                parser.error(f"invalid PACE fixture transport {transport!r}")
            if args.transport is not None and args.transport != transport:
                parser.error(
                    f"--transport={args.transport} conflicts with PACE fixture transport {transport!r}"
                )
            metrics_url = fixture_config.get("consumer_metrics_url") if transport != "local" else None
            if transport != "local" and not metrics_url:
                parser.error("the PACE fixture needs consumer_metrics_url for transport verification")

            if not server.start_server():
                raise RuntimeError("KVCM manager did not start")
            env = os.environ.copy()
            env.update(server.client_env())
            instance_id = f"pace_contract_{server.rpc_port}"
            config = server.pace_fixture.sdk_config(server, instance_id)
            config_path = Path(directory) / "client.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            command = [args.sdk, str(config_path), str(server.pace_fixture.storage_type)]
            if transport == "gdr":
                command.append("--gdr")
            before = read_counters(metrics_url) if metrics_url else None
            subprocess.run(command, env=env, check=True, timeout=120)
            if before is not None:
                check_transport(metrics_url, before, transport)
        finally:
            try:
                server.close()
            finally:
                server.copy_logs()


if __name__ == "__main__":
    main()
