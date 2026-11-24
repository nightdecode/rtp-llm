import logging
import multiprocessing
import os
import sys
import time
import traceback
from typing import Iterator

# Apply before torch/native imports can allocate memory or start worker threads.
if __name__ == "__main__":
    from rtp_llm.utils.numa import configure_numa_interleave

    configure_numa_interleave()

import requests
import torch

from rtp_llm.utils.time_util import timer_wrapper

CUR_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(str(CUR_PATH), ".."))

from rtp_llm.config.log_config import setup_logging
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.config.server_config_setup import (
    load_gpu_nic_affinity,
    setup_and_configure_server,
)
from rtp_llm.distribute.distributed_server import get_local_world_info
from rtp_llm.models_py.distributed.rank_layout import RankLayout
from rtp_llm.ops import RoleType, VitSeparation
from rtp_llm.server.server_args.server_args import setup_args
from rtp_llm.utils.concurrency_controller import init_controller
from rtp_llm.utils.process_manager import (
    DEFER_FIRST_SIGTERM_ENV,
    DEFER_FIRST_SIGTERM_SECONDS_ENV,
    DEFER_FIRST_SIGTERM_VALUE,
    ProcessManager,
)
from rtp_llm.utils.warmup import configure_warmup

setup_logging()


def _install_hot_hook_runtime(role: str, py_env_configs) -> None:
    """Install the hot hook runtime from the parsed config.

    The single switch is the hot-hook config path (--hot_hook_config /
    RTP_HOT_HOOK_CONFIG): a non-empty path pointing at a loadable JSON enables
    the runtime; no path means no install.
    """
    try:
        from rtp_llm.utils.hot_hook_runtime import install_from_config

        config_path = py_env_configs.hot_hook_config.config_path
        if config_path and install_from_config(config_path):
            logging.info(
                "RTP hot hook runtime installed for %s (config=%s)", role, config_path
            )
    except Exception as e:
        logging.error("failed to install RTP hot hook runtime for %s: %s", role, e)


def check_server_health(server_port, path="/health"):
    try:
        import requests

        response = requests.get(f"http://localhost:{server_port}{path}", timeout=60)
        health_ok = False
        if response.status_code == 200:
            try:
                health_body = response.json()
                if isinstance(health_body, dict):
                    health_ok = health_body.get("status", "") == "ok"
                elif isinstance(health_body, str):
                    health_ok = health_body == "ok"
            except ValueError:
                health_ok = response.text.strip() == "ok"
        if health_ok:
            logging.info(
                f"{server_port}{path}, response status_code = {response.status_code}, text = {response.text}, len = {len(response.text)}"
            )
            return True
        else:
            return False
    except requests.RequestException:
        return False


def _backend_deferred_sigterm_seconds(py_env_configs: PyEnvConfigs) -> str:
    timeout = ProcessManager.normalize_shutdown_timeout_seconds(
        py_env_configs.server_config.shutdown_timeout
    )
    deferred = ProcessManager.deferred_group_shutdown_timeout_seconds(timeout)
    # None must serialize as "-1": the backend parses this env with float().
    return "-1" if deferred is None else str(deferred)


def _sync_server_shutdown_timeout(py_env_configs: PyEnvConfigs):
    py_env_configs.server_config.shutdown_timeout = (
        ProcessManager.sync_shutdown_timeout_env(
            py_env_configs.server_config.shutdown_timeout
        )
    )


@timer_wrapper(description="start backend server")
def start_backend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
    service_draining=None,
):
    from rtp_llm.start_backend_server import start_backend_server

    # only for debug
    if py_env_configs.profiling_debug_logging_config.debug_load_server:
        start_backend_server(
            global_controller, py_env_configs, None, service_draining=service_draining
        )
        os._exit(-1)

    # Create pipe for subprocess startup status communication
    pipe_reader, pipe_writer = torch.multiprocessing.Pipe(duplex=False)
    logging.info(f"[PROCESS_SPAWN]Start backend server process outer")

    old_defer = os.environ.get(DEFER_FIRST_SIGTERM_ENV)
    old_defer_seconds = os.environ.get(DEFER_FIRST_SIGTERM_SECONDS_ENV)
    _sync_server_shutdown_timeout(py_env_configs)
    os.environ[DEFER_FIRST_SIGTERM_ENV] = DEFER_FIRST_SIGTERM_VALUE
    os.environ[DEFER_FIRST_SIGTERM_SECONDS_ENV] = _backend_deferred_sigterm_seconds(
        py_env_configs
    )
    try:
        backend_process = torch.multiprocessing.Process(
            target=start_backend_server,
            args=(
                global_controller,
                py_env_configs,
                pipe_writer,
                os.getpid(),
                service_draining,
            ),
            name="backend_manager",
        )
        backend_process.start()
    finally:
        if old_defer is None:
            os.environ.pop(DEFER_FIRST_SIGTERM_ENV, None)
        else:
            os.environ[DEFER_FIRST_SIGTERM_ENV] = old_defer
        if old_defer_seconds is None:
            os.environ.pop(DEFER_FIRST_SIGTERM_SECONDS_ENV, None)
        else:
            os.environ[DEFER_FIRST_SIGTERM_SECONDS_ENV] = old_defer_seconds
        # Close parent's write end on both paths; on spawn failure leaving it open masks EOF as a hang.
        pipe_writer.close()

    if process_manager:
        # Register right after start so later failures in start_server() still clean this process up.
        process_manager.add_process(backend_process, shutdown_group="backend")

    # Create check_ready_fn for pipe-based health check
    startup_status = {"ready": False, "error": None}

    def check_backend_ready():
        """Check if backend server is ready via pipe communication"""
        if startup_status["ready"]:
            return True

        if startup_status["error"]:
            raise Exception(startup_status["error"])

        # Non-blocking check if data is available
        if pipe_reader.poll(timeout=0):
            try:
                status_msg = pipe_reader.recv()
                if status_msg.get("status") == "success":
                    logging.info(
                        f"Backend server started successfully: {status_msg.get('message', '')}"
                    )
                    startup_status["ready"] = True
                    pipe_reader.close()
                    return True
                else:
                    # Startup failed
                    error_msg = status_msg.get("message", "Unknown error")
                    traceback_info = status_msg.get("traceback", "")
                    if traceback_info:
                        logging.error(f"Traceback: {traceback_info}")

                    error = f"Backend server start failed: {error_msg}"
                    startup_status["error"] = error
                    pipe_reader.close()
                    raise Exception(error)
            except EOFError:
                error = "Backend server pipe closed unexpectedly"
                startup_status["error"] = error
                pipe_reader.close()
                raise Exception(error)

        return False

    # Register health check with ProcessManager using custom check_ready_fn
    if process_manager:
        process_manager.register_health_check(
            processes=[backend_process],
            process_name="backend_server",
            check_ready_fn=check_backend_ready,
            retry_interval_seconds=1,
        )

    return backend_process


def _iter_serving_world_ranks(py_env_configs: PyEnvConfigs) -> Iterator[int]:
    """Yield global ranks of local workers that should host frontend/dash_sc.

    Only TP roots on the first PP stage admit inference requests. Local rank 0
    on every node must also start for Kubernetes heartbeat checks. DashSc uses
    the same serving ranks as the HTTP frontend.
    """
    pc = py_env_configs.parallelism_config
    local_world_info = get_local_world_info(
        py_env_configs.server_config,
        py_env_configs.distribute_config,
        pc,
    )
    layout = RankLayout.from_parallelism_config(pc)
    for member in local_world_info.members:
        coord = layout.coord_of(member.world_rank)
        if member.local_rank == 0 or (coord.tp == 0 and coord.pp == 0):
            yield member.world_rank
        else:
            logging.info(
                f"world rank {member.world_rank} skipping frontend/dash_sc startup"
            )


@timer_wrapper(description="start dash_sc server")
def start_dash_sc_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager=None,
    service_draining=None,
):
    from rtp_llm.start_dash_sc_server import start_dash_sc_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
    pc = py_env_configs.parallelism_config
    dash_sc_processes = []
    dash_sc_pipe_readers = []

    worker_specs = [
        (world_rank, server_id)
        for world_rank in _iter_serving_world_ranks(py_env_configs)
        for server_id in range(frontend_server_count)
    ]
    if not worker_specs:
        return dash_sc_processes

    bind_barrier = multiprocessing.Barrier(len(worker_specs))
    for world_rank, server_id in worker_specs:
        local_rank = world_rank % pc.local_world_size
        pipe_reader, pipe_writer = multiprocessing.Pipe(duplex=False)
        logging.info(
            f"[PROCESS_SPAWN]Start dash_sc server process "
            f"rank_{local_rank}_server_{server_id} outer"
        )
        process = multiprocessing.Process(
            target=start_dash_sc_server,
            args=(
                world_rank,
                server_id,
                global_controller,
                py_env_configs,
                pipe_writer,
                bind_barrier,
                service_draining,
            ),
            name=f"dash_sc_server_{local_rank}_{server_id}",
        )
        dash_sc_processes.append(process)
        process.start()
        if process_manager:
            # Register right after start: a failure in a later loop iteration must not leak running servers.
            process_manager.add_process(process, shutdown_group="frontend")
        pipe_writer.close()
        dash_sc_pipe_readers.append(pipe_reader)

    startup_status = {"remaining": set(range(len(dash_sc_pipe_readers)))}

    # Process.start() drops its args while spawned children may still be rebuilding
    # named semaphores. The default keeps the Barrier alive with the health check.
    def check_dash_sc_ready(_bind_barrier_keepalive=bind_barrier):
        if not startup_status["remaining"]:
            return True

        done_now = []
        for idx in list(startup_status["remaining"]):
            reader = dash_sc_pipe_readers[idx]
            try:
                if not reader.poll(timeout=0):
                    continue
                status_msg = reader.recv()
            except EOFError:
                raise Exception(
                    f"DashSc server rank_idx={idx} pipe closed unexpectedly"
                )
            if status_msg.get("status") == "success":
                logging.info(
                    f"DashSc server rank_idx={idx} ready: {status_msg.get('message', '')}"
                )
                done_now.append(idx)
                reader.close()
            else:
                error_msg = status_msg.get("message", "Unknown error")
                traceback_info = status_msg.get("traceback", "")
                if traceback_info:
                    logging.error(f"DashSc server traceback: {traceback_info}")
                raise Exception(
                    f"DashSc server rank_idx={idx} start failed: {error_msg}"
                )

        for idx in done_now:
            startup_status["remaining"].discard(idx)
        return not startup_status["remaining"]

    if process_manager and dash_sc_processes:
        process_manager.register_health_check(
            processes=dash_sc_processes,
            process_name="dash_sc_server",
            check_ready_fn=check_dash_sc_ready,
            retry_interval_seconds=1,
        )

    return dash_sc_processes


@timer_wrapper(description="start vit server")
def start_vit_server_impl(
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager = None,
    service_draining=None,
):
    """Start VIT children and ensure each successful child has an owner."""
    started_processes = []

    try:
        return _start_vit_server_impl(
            py_env_configs, process_manager, started_processes, service_draining
        )
    except Exception:
        if process_manager is None:
            _cleanup_unmanaged_vit_processes(started_processes)
        raise


def _cleanup_unmanaged_vit_processes(processes):
    """Best-effort bounded cleanup that never replaces the startup exception.

    Two bounded phases: up to 5s to terminate (SIGTERM) and join, then up to
    5s to kill (SIGKILL) and reap stragglers, so the worst case is ~10s.
    """
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            logging.exception(
                "Failed to terminate partially started VIT child %s", process
            )
    for process in processes:
        try:
            process.join(timeout=max(0, deadline - time.monotonic()))
        except Exception:
            logging.exception("Failed to join partially started VIT child %s", process)
    for process in processes:
        try:
            if process.is_alive():
                process.kill()
        except Exception:
            logging.exception("Failed to kill partially started VIT child %s", process)
    deadline = time.monotonic() + 5
    for process in processes:
        try:
            process.join(timeout=max(0, deadline - time.monotonic()))
        except Exception:
            logging.exception("Failed to reap partially started VIT child %s", process)


def _start_vit_server_impl(
    py_env_configs: PyEnvConfigs,
    process_manager: ProcessManager,
    started_processes: list,
    service_draining=None,
):
    """
    启动 VIT 服务器

    Args:
        py_env_configs: 配置对象
        process_manager: 进程管理器
        started_processes: successfully started children for failure cleanup
        service_draining: shared draining Event forwarded to each VIT child
    """
    from rtp_llm.multimodal.vit_proxy_start_server import vit_proxy_start_server
    from rtp_llm.multimodal.vit_start_server import vit_start_server
    from rtp_llm.server.vit_proxy_server import resolve_min_healthy_workers

    def start_vit_process(process):
        process.start()
        started_processes.append(process)
        if process_manager is not None:
            try:
                process_manager.add_process(process)
            except Exception:
                # Never leak a started child when registration fails: the
                # manager does not own it yet, so reap it locally before
                # propagating the failure.
                try:
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                except Exception:
                    logging.exception(
                        "Failed to reap unregistered VIT child %s", process
                    )
                raise

    server_config = py_env_configs.server_config
    start_port = server_config.start_port
    vit_server_count = server_config.vit_server_count
    resolve_min_healthy_workers(
        vit_server_count, py_env_configs.vit_config.vit_proxy_min_healthy_workers
    )

    # Load once before spawning workers. Each worker consumes the inherited
    # mapping with the existing process-local rank semantics.
    load_gpu_nic_affinity()

    if vit_server_count > 1:
        logging.info(
            f"[VIT_SERVER] Starting in PROXY mode: 1 proxy + {vit_server_count} workers "
            f"(role_type=VIT)"
        )

        worker_processes = []
        worker_addresses = []
        worker_http_addresses = []

        base_grpc_port = py_env_configs.server_config.rpc_server_port
        base_rdma_port = py_env_configs.vit_config.output_transport.rdma.port

        # Per-worker we consume two ports (grpc + http). Fail fast if the
        # range would walk past the 16-bit TCP port ceiling — otherwise
        # bind() errors only surface inside the spawned worker and look like
        # health-check timeouts.
        max_port = base_grpc_port + vit_server_count * 2
        if max_port >= 65536:
            raise ValueError(
                f"VIT worker port range exceeds 65535: base={base_grpc_port}, "
                f"vit_server_count={vit_server_count}, max={max_port}"
            )

        if base_rdma_port > 0:
            max_rdma_port = base_rdma_port + vit_server_count - 1
            if max_rdma_port >= 65536:
                raise ValueError(
                    f"VIT worker RDMA port range exceeds 65535: "
                    f"base={base_rdma_port}, vit_server_count={vit_server_count}, "
                    f"max={max_rdma_port}"
                )

        for i in range(vit_server_count):
            internal_grpc_port = base_grpc_port + i * 2 + 1
            internal_http_port = base_grpc_port + i * 2 + 2
            worker_rdma_port = base_rdma_port + i if base_rdma_port > 0 else None
            worker_addresses.append(f"127.0.0.1:{internal_grpc_port}")
            worker_http_addresses.append(f"127.0.0.1:{internal_http_port}")

            logging.info(
                f"[PROCESS_SPAWN] Start vit worker process worker_{i} "
                f"(internal grpc_port={internal_grpc_port}, "
                f"internal http_port={internal_http_port})"
            )
            process = torch.multiprocessing.Process(
                target=vit_start_server,
                args=(
                    i,
                    py_env_configs,
                    internal_grpc_port,  # grpc_port
                    internal_http_port,  # http_port
                    True,  # is_proxy_mode (proxy 模式下的 worker 进程)
                    worker_rdma_port,
                ),
                kwargs={"service_draining": service_draining},
                name=f"vit_worker_{i}",
            )
            start_vit_process(process)
            worker_processes.append(process)

        external_grpc_port = base_grpc_port  # 主进程使用基础 gRPC 端口
        external_http_port = py_env_configs.server_config.server_port

        # 2. 启动主进程（代理服务器）
        logging.info(
            f"[PROCESS_SPAWN] Start vit proxy process "
            f"(external grpc_port={external_grpc_port}, http_port={external_http_port})"
        )
        proxy_process = torch.multiprocessing.Process(
            target=vit_proxy_start_server,
            args=(
                py_env_configs,
                worker_addresses,
                external_grpc_port,  # grpc_port
                external_http_port,  # http_port
                worker_http_addresses,  # worker_http_addresses
            ),
            kwargs={"service_draining": service_draining},
            name="vit_proxy",
        )
        start_vit_process(proxy_process)

        vit_processes = [proxy_process] + worker_processes

        # 健康检查：检查代理服务器的端口（代理模式只在 VIT 角色时启用）
        vit_server_port = py_env_configs.server_config.server_port
        # 保存 worker 地址列表，用于健康检查

    else:
        grpc_port = py_env_configs.server_config.rpc_server_port
        http_port = py_env_configs.server_config.server_port
        vit_server_port = http_port
        logging.info(
            f"[PROCESS_SPAWN] Start vit server process "
            f"(grpc_port={grpc_port}, http_port={http_port})"
        )
        process = torch.multiprocessing.Process(
            target=vit_start_server,
            args=(
                0,
                py_env_configs,
                grpc_port,  # grpc_port
                http_port,  # http_port
                False,  # is_proxy_mode (standalone 模式，需要记录 QPS)
            ),
            kwargs={"service_draining": service_draining},
            name="vit_server",
        )
        start_vit_process(process)
        vit_processes = [process]

    if process_manager is not None and vit_processes:
        logging.info(
            f"[VIT_SERVER] Registering health check for {len(vit_processes)} VIT processes, "
            f"current_managed_processes={len(process_manager.processes)}"
        )

        def check_vit_ready():
            return check_server_health(vit_server_port)

        process_manager.register_health_check(
            processes=vit_processes,
            process_name="vit_server",
            check_ready_fn=check_vit_ready,
            retry_interval_seconds=1,
        )
        logging.info(
            f"[VIT_SERVER] Health check registered, after_registration: "
            f"managed_processes={len(process_manager.processes)}, "
            f"health_check_processes={len(process_manager.health_check_processes)}"
        )

    return vit_processes


@timer_wrapper(description="start frontend server")
def start_frontend_server_impl(
    global_controller,
    py_env_configs: PyEnvConfigs,
    process_manager=None,
    service_draining=None,
):
    from rtp_llm.start_frontend_server import start_frontend_server

    frontend_server_count = py_env_configs.server_config.frontend_server_count
    if frontend_server_count < 1:
        logging.info(
            "frontend server's count is {frontend_server_count}, this may be a mistake"
        )

    frontend_processes = []

    pc = py_env_configs.parallelism_config
    # The shared iterator applies TP/PP filtering and the per-node heartbeat exception.
    for world_rank in _iter_serving_world_ranks(py_env_configs):
        local_rank = world_rank % pc.local_world_size
        for i in range(frontend_server_count):
            logging.info(
                f"[PROCESS_SPAWN]Start frontend server process rank_{local_rank}_server_{i} outer"
            )
            process = multiprocessing.Process(
                target=start_frontend_server,
                args=(
                    world_rank,
                    i,
                    global_controller,
                    py_env_configs,
                    service_draining,
                ),
                name=f"frontend_server_{i}",
            )
            frontend_processes.append(process)
            process.start()
            if process_manager:
                process_manager.add_process(process, shutdown_group="frontend")

    if process_manager and frontend_processes:

        def check_frontend_ready():
            return check_server_health(
                py_env_configs.server_config.start_port, path="/frontend_health"
            )

        process_manager.register_health_check(
            processes=frontend_processes,
            process_name="frontend_server",
            check_ready_fn=check_frontend_ready,
            retry_interval_seconds=1,
        )

    return frontend_processes


def main():
    py_env_configs: PyEnvConfigs = setup_args()
    _install_hot_hook_runtime("main", py_env_configs)
    setup_and_configure_server(py_env_configs)
    start_server(py_env_configs)


def start_server(py_env_configs: PyEnvConfigs):
    logging.info(f"[PROCESS_START]Start server")
    configure_warmup(
        py_env_configs.runtime_config.warm_up,
        py_env_configs.runtime_config.model_warm_up,
    )
    start_time = time.time()
    try:
        multiprocessing.set_start_method("spawn", force=True)
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as e:
        logging.warning(str(e))

    # 仅补全进程身份；各子进程独立解析继承的 JSON Trace 配置。
    from rtp_llm.telemetry import resolve_pod_ip

    resolve_pod_ip()

    logging.info(
        f"dp_size:  parallelism_config={py_env_configs.parallelism_config.dp_size}"
    )
    global_controller = init_controller(
        py_env_configs.concurrency_config,
        dp_size=py_env_configs.parallelism_config.dp_size,
    )
    _sync_server_shutdown_timeout(py_env_configs)

    service_draining = multiprocessing.Event()
    # Create process manager with config values
    process_manager = ProcessManager(
        service_draining=service_draining,
        shutdown_timeout=py_env_configs.server_config.shutdown_timeout,
        monitor_interval=py_env_configs.server_config.monitor_interval,
        frontend_pre_stop_drain_seconds=(
            py_env_configs.server_config.frontend_pre_stop_drain_seconds
        ),
        dash_sc_grpc_pre_stop_drain_seconds=(
            py_env_configs.server_config.dash_sc_grpc_pre_stop_drain_seconds
        ),
        pre_stop_drain_headroom_seconds=(
            py_env_configs.server_config.pre_stop_drain_headroom_seconds
        ),
        pre_stop_drain_signal=py_env_configs.server_config.pre_stop_drain_signal,
        backend_post_frontend_drain_seconds=(
            py_env_configs.server_config.backend_post_frontend_drain_seconds
        ),
    )
    # Backward compat: VIT_SEPARATION=ROLE without ROLE_TYPE=VIT
    if (
        py_env_configs.vit_config.vit_separation == VitSeparation.VIT_SEPARATION_ROLE
        and py_env_configs.role_config.role_type == RoleType.PDFUSION
    ):
        logging.warning(
            "VIT_SEPARATION=ROLE detected without ROLE_TYPE=VIT. "
            "Auto-setting ROLE_TYPE=VIT for backward compatibility. "
            "Please migrate to ROLE_TYPE=VIT explicitly."
        )
        py_env_configs.role_config.role_type = RoleType.VIT

    dash_sc_enabled = py_env_configs.role_config.role_type != RoleType.VIT
    py_env_configs.server_config.validate_port_layout(dash_sc_enabled=dash_sc_enabled)

    # Impls register spawned processes right after start, so later failures still reach them in the finally cleanup.

    try:
        if py_env_configs.role_config.role_type == RoleType.VIT:
            logging.info("start vit server")
            start_vit_server_impl(py_env_configs, process_manager, service_draining)

        if (
            py_env_configs.role_config.role_type != RoleType.FRONTEND
            and py_env_configs.role_config.role_type != RoleType.VIT
        ):
            # For backend server, vit_process_engine is None when vit is separated
            logging.info("start backend server")
            start_backend_server_impl(
                global_controller, py_env_configs, process_manager, service_draining
            )

        if py_env_configs.role_config.role_type != RoleType.VIT:
            # vit has its own frontend server
            logging.info("start frontend server")
            start_frontend_server_impl(
                global_controller, py_env_configs, process_manager, service_draining
            )

            logging.info("start dash_sc server")
            start_dash_sc_server_impl(
                global_controller, py_env_configs, process_manager, service_draining
            )

        # Start parallel health checks and wait; the bound is total (shared deadline across checks).
        # Only -1 is the unbounded opt-out; other non-positive values fall back to 3600.
        startup_timeout = py_env_configs.server_config.startup_health_check_timeout
        if startup_timeout == -1:
            startup_timeout = None
        elif startup_timeout <= 0:
            logging.warning(
                "invalid startup_health_check_timeout=%s; using 3600", startup_timeout
            )
            startup_timeout = 3600
        if not process_manager.run_health_checks(timeout=startup_timeout):
            logging.error("[START_SERVER] Health checks failed")
            raise Exception("Health checks failed")


        logging.info(
            f"Backend RPC service is listening on 0.0.0.0, IP/IP range can be customized as needed"
        )
        consume_s = time.time() - start_time
        logging.info(f"start server took {consume_s:.2f}s")
    except Exception as e:
        logging.error(f"start failed, trace: {traceback.format_exc()}")
        # If a SIGTERM/SIGINT already triggered shutdown before this exception,
        # the exception is a side-effect of the signal (health check tripped on
        # shutdown_requested), not a real failure — preserve graceful exit
        # semantics. Otherwise mark failure so the process manager uses bounded
        # timeouts and the parent exits non-zero.
        if not process_manager.shutdown_requested:
            process_manager.request_failure_shutdown()
    finally:
        process_manager.monitor_and_release_processes()


if __name__ == "__main__":
    main()
