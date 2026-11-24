"""Pure PD-disaggregated Bazel server launcher for Qwen 4.5-Flash (``qwen4_5``).

Launches dedicated ``ROLE_TYPE=PREFILL`` and ``ROLE_TYPE=DECODE`` RTP-LLM instances
via ``MagaServerManager`` + ``ServiceRoute`` (no unified / PDFUSION serving mode).
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

DEFAULT_3FS_CKPT = "/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8"


def _parse_gpu_list(raw: str) -> List[str]:
    gpus = [item.strip() for item in raw.split(",") if item.strip()]
    if not gpus:
        raise ValueError("GPU list must be non-empty")
    return gpus


def _wait_tcp_port_ready(host: str, port: int, timeout_s: float, label: str) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1.0):
                logging.info("TCP ready: %s (%s:%d)", label, host, port)
                return
        except OSError as exc:
            last_err = exc
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {label} ({host}:{port}): {last_err}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch pure PD-disaggregated Qwen 4.5-Flash RTP-LLM servers (PREFILL + DECODE)."
    )
    parser.add_argument(
        "--ckpt-path",
        default=DEFAULT_3FS_CKPT,
        help="Path to Qwen 4.5-Flash checkpoint directory (read-only).",
    )
    parser.add_argument(
        "--prefill-gpus",
        default="0,1,2,3",
        help="Comma-separated GPU IDs assigned to ROLE_TYPE=PREFILL.",
    )
    parser.add_argument(
        "--decode-gpus",
        default="4,5,6,7",
        help="Comma-separated GPU IDs assigned to ROLE_TYPE=DECODE.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=None,
        help="TP size per role (defaults to len(--prefill-gpus)).",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=None,
        help="EP size per role (defaults to --tp-size).",
    )
    parser.add_argument("--prefill-port", type=int, default=28000)
    parser.add_argument("--decode-port", type=int, default=29000)
    parser.add_argument("--seq-size-per-block", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--reserver-runtime-mem-mb", type=int, default=16384)
    parser.add_argument("--load-workers", type=int, default=8)
    parser.add_argument("--startup-timeout", type=int, default=1800)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args = build_parser().parse_args(argv)
    if not Path(args.ckpt_path).is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {args.ckpt_path}")

    prefill_gpus = _parse_gpu_list(args.prefill_gpus)
    decode_gpus = _parse_gpu_list(args.decode_gpus)
    if set(prefill_gpus) & set(decode_gpus):
        raise ValueError(
            f"Prefill GPUs {prefill_gpus} and Decode GPUs {decode_gpus} must be disjoint"
        )
    if len(prefill_gpus) != len(decode_gpus):
        raise ValueError(
            f"Prefill ({len(prefill_gpus)}) and Decode ({len(decode_gpus)}) GPU counts must match"
        )

    role_size = len(prefill_gpus)
    tp_size = args.tp_size if args.tp_size is not None else role_size
    ep_size = args.ep_size if args.ep_size is not None else tp_size
    if tp_size != role_size:
        raise ValueError(f"--tp-size ({tp_size}) must match role GPU count ({role_size})")

    route = ServiceRoute(
        service_id="qwen4_5_pd_server",
        use_local=True,
        role_endpoints=[
            GroupEndPoint(
                group="default",
                prefill_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{args.prefill_port}",
                    protocol="http",
                    path="/",
                ),
                decode_endpoint=EndPoint(
                    type="Vipserver",
                    address=f"127.0.0.1:{args.decode_port}",
                    protocol="http",
                    path="/",
                ),
            )
        ],
    )

    common_env = {
        "LOAD_PYTHON_MODEL": "1",
        "WORLD_SIZE": str(role_size),
        "REMOTE_RPC_SERVER_IP": "127.0.0.1",
        "QWEN4_5_LOAD_WORKERS": str(args.load_workers),
    }
    common_args = (
        f"--tp_size {tp_size} "
        f"--ep_size {ep_size} "
        f"--dp_size 1 "
        f"--world_size {role_size} "
        f"--act_type BF16 "
        f"--seq_size_per_block {args.seq_size_per_block} "
        f"--max_seq_len {args.max_seq_len} "
        f"--reuse_cache 1 "
        f"--enable_cuda_graph 0 "
        f"--cache_store_rdma_mode 0 "
        f"--use_local 1 "
        f"--load_cache_timeout_ms 120000 "
        f"--reserver_runtime_mem_mb {args.reserver_runtime_mem_mb} "
        f"--warm_up 0"
    )

    prefill_mgr = MagaServerManager(
        env_args={
            **common_env,
            "REMOTE_SERVER_PORT": args.decode_port,
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
        },
        device_ids=prefill_gpus,
        port=args.prefill_port,
        role_name="qwen4_5_prefill",
        smoke_args_str=f"{common_args} --role_type PREFILL",
    )
    decode_mgr = MagaServerManager(
        env_args={
            **common_env,
            "REMOTE_SERVER_PORT": args.prefill_port,
            "MODEL_SERVICE_CONFIG": route.model_dump_json(),
        },
        device_ids=decode_gpus,
        port=args.decode_port,
        role_name="qwen4_5_decode",
        smoke_args_str=f"{common_args} --role_type DECODE",
    )

    stop_requested = False

    def _handle_signal(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        logging.info("Received signal %d; shutting down pure-PD servers...", signum)
        stop_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futs = [
                executor.submit(
                    mgr.start_server,
                    model_path=args.ckpt_path,
                    model_type="qwen4_5",
                    timeout=args.startup_timeout,
                )
                for mgr in (prefill_mgr, decode_mgr)
            ]
            for fut in futs:
                if not fut.result():
                    raise RuntimeError("Pure-PD server startup failed /health check")

        for role_name, base_port in (("prefill", args.prefill_port), ("decode", args.decode_port)):
            for rank in range(role_size):
                _wait_tcp_port_ready(
                    "127.0.0.1",
                    int(base_port) + rank * MIN_WORKER_INFO_PORT_NUM + 1,
                    timeout_s=120.0,
                    label=f"{role_name} rank {rank} model rpc",
                )

        print(
            f"\n[Qwen 4.5-Flash Pure PD Ready]\n"
            f"  Prefill HTTP : http://127.0.0.1:{args.prefill_port} (GPUs={prefill_gpus})\n"
            f"  Decode  HTTP : http://127.0.0.1:{args.decode_port} (GPUs={decode_gpus})\n"
            f"  Sample curl  :\n"
            f"    curl -s http://127.0.0.1:{args.prefill_port}/ -H \"Content-Type: application/json\" "
            f"-d '{{\"prompt\": \"中国的首都是\", \"generate_config\": {{\"max_new_tokens\": 24, \"top_k\": 1, \"temperature\": 0.0, \"can_use_pd_separation\": true}}}}'\n",
            flush=True,
        )
        while not stop_requested:
            time.sleep(1.0)
    finally:
        decode_mgr.stop_server()
        prefill_mgr.stop_server()

    return 0


if __name__ == "__main__":
    sys.exit(main())
