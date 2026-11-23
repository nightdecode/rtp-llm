#!/usr/bin/env python3

import argparse
import fcntl
import json
import os
import re
import sys
from pathlib import Path


def check_rdma(device, port):
    verbs = Path("/dev/infiniband")
    hca = Path("/sys/class/infiniband") / device
    userspace = [
        entry
        for entry in Path("/sys/class/infiniband_verbs").glob("uverbs*")
        if (entry / "ibdev").is_file() and (entry / "ibdev").read_text().strip() == device
    ]
    if not any(os.access(verbs / entry.name, os.R_OK | os.W_OK) for entry in userspace):
        raise RuntimeError(f"{device}: no accessible RDMA verbs device under {verbs}")
    state = hca / "ports" / str(port) / "state"
    if not state.is_file() or state.read_text().split(":", 1)[0] != "4":
        raise RuntimeError(f"{device} port {port} is not ACTIVE")


def start(args, parser):
    if args.role == "check-rdma":
        if not args.rdma_device:
            parser.error("check-rdma requires --rdma-device")
        check_rdma(args.rdma_device, args.rdma_port)
        print(f"{args.rdma_device} port {args.rdma_port} is accessible and ACTIVE")
        return
    if args.bin_dir is None or args.work_dir is None or args.port_base is None:
        parser.error("starting a service requires --bin-dir, --work-dir, and --port-base")
    if not 1024 <= args.port_base <= 65532:
        parser.error("--port-base must be between 1024 and 65532")
    data_role = args.role in ("consumer", "provider")
    tent_transport = "rdma" if args.transport == "gdr" else args.transport
    if data_role:
        if not all((args.meta_url, args.interface)):
            parser.error("a data service requires --meta-url and --interface")
        if args.role == "consumer" and not args.provider_status_url:
            parser.error("consumer requires --provider-status-url for the smoke fixture")
        if tent_transport == "rdma":
            if not args.rdma_device:
                parser.error("RDMA requires --rdma-device")
            check_rdma(args.rdma_device, args.rdma_port)
        lock = Path("/dev/shm/rtp_llm_pace_test.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if any((Path("/dev/shm") / name).exists() for name in ("shm_mps", "shm_mps_meta")):
            raise RuntimeError("PACE shared memory already exists. Give each data service its own /dev/shm.")
        os.set_inheritable(lock.fileno(), True)
    work_dir = args.work_dir.resolve()
    bin_dir = args.bin_dir.resolve()
    binary = bin_dir / ("tair_mempool_server" if data_role else "tair_mempool_meta_service")
    if not os.access(binary, os.X_OK):
        raise RuntimeError(f"{binary} is not executable")
    if data_role and len(os.fsencode(work_dir / "mp_data.sock")) >= 108:
        raise RuntimeError("use a shorter --work-dir for the Unix sockets")
    work_dir.mkdir(parents=True)
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, (str(bin_dir), env.get("LD_LIBRARY_PATH"))))
    env["ENABLE_TAIR_MEMPOOL_HUGEPAGES"] = "OFF"
    if not data_role:
        command = [
            str(binary), "-b", "memory", "-l", "no", "-p", str(args.port_base),
            "-M", str(args.port_base + 1), "-P", "2",
        ]
    else:
        env.setdefault("TAIR_MEMPOOL_PROVIDER_HTTP_THREADS", "4")
        tent_config = work_dir / "tent.json"
        transports = {
            name: {"enable": name == tent_transport}
            for name in ("rdma", "aft", "barex", "tcp", "uring", "shm", "gds", "mnnvl")
        }
        profile = {
            "transports": transports,
            "policy": [{"name": tent_transport, "segment_type": "memory", "transports": [tent_transport]}],
        }
        if tent_transport == "rdma":
            transports["rdma"]["device"] = {"port": args.rdma_port, "gid_index": args.gid_index}
            profile["topology"] = {"rdma_whitelist": [args.rdma_device]}
        tent_config.write_text(json.dumps(profile, indent=2) + "\n")

        zero_copy = "1" if tent_transport == "rdma" else "0"
        client_env = {
            "MC_TENT_CONF": str(tent_config),
            "TAIR_MEMPOOL_SOCK": str(work_dir / "mp.sock"),
            "TAIR_MEMPOOL_DATA_SOCK": str(work_dir / "mp_data.sock"),
            "TAIR_MEMPOOL_ENABLE_TENT": "1",
            "TAIR_MEMPOOL_ENABLE_EXTERNAL_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_READ_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_ENABLE_GDR_WRITE_ZERO_COPY": zero_copy,
            "TAIR_MEMPOOL_GDR_TRANSFER_MODE": "direct",
        }
        env.update(client_env)
        if args.role == "consumer":
            (work_dir / "fixture.json").write_text(json.dumps({
                "domain": args.meta_url,
                "transport": args.transport,
                "provider_status_urls": [args.provider_status_url],
                "consumer_metrics_url": f"http://127.0.0.1:{args.port_base + 3}/pace/metrics",
                "client_env": client_env,
            }, indent=2) + "\n")
        cache_args = ["-R", "10"] if args.role == "consumer" else ["-R", "0", "-c"]
        command = [
            str(binary), "--role", args.role, "--use_tent", "--tiering_mode", "off",
            "--network_interface", args.interface, "--metaservice", args.meta_url,
            "--metric_type", "prometheus", "-s", "2048", "--shm_meta_size_mb", "1024",
            *cache_args, "-p", "2", "--ctrl_threads", "2", "-a", str(args.port_base),
            "-r", str(args.port_base + 1), "-t", str(args.port_base + 2), "-m", str(args.port_base + 3),
        ]
    os.chdir(work_dir)
    os.execve(str(binary), command, env)


def main():
    parser = argparse.ArgumentParser(description="Start one PACE DRAM role for TENT smoke.")
    parser.add_argument("role", choices=("check-rdma", "meta", "provider", "consumer"))
    parser.add_argument("--bin-dir", type=Path)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--port-base", type=int)
    parser.add_argument("--meta-url")
    parser.add_argument("--provider-status-url")
    parser.add_argument("--interface")
    parser.add_argument("--transport", choices=("rdma", "tcp", "gdr"), default="rdma")
    parser.add_argument("--rdma-device")
    parser.add_argument("--rdma-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=0)
    args = parser.parse_args()
    if args.rdma_device and not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.rdma_device):
        parser.error("invalid --rdma-device")
    if args.rdma_port < 1 or args.gid_index < 0:
        parser.error("RDMA port must be positive and GID index must be nonnegative")
    start(args, parser)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(f"PACE smoke startup failed: {error}", file=sys.stderr)
        sys.exit(1)
