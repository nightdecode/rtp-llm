"""Pure PD-disaggregated (PREFILL -> CacheStore -> DECODE) Bazel smoke test for Qwen 4.5-Flash.

Supports two modes via ``QWEN4_5_PD_SMOKE_MODE``:
  - ``synthetic_8layer`` (default, 2 GPUs: 1P + 1D, TP=1): builds a valid 8-layer
    Qwen 4.5-Flash safetensors checkpoint + HF Qwen2Tokenizer in a TemporaryDirectory,
    starts native RTP-LLM ``ROLE_TYPE=PREFILL`` and ``ROLE_TYPE=DECODE`` servers via
    ``MagaServerManager``, and verifies HTTP ``/`` generation, determinism, and prefix reuse.
  - ``real_3fs_40layer`` (8 GPUs: 4P + 4D, TP=4/EP=4): reads the real 40-layer 3FS
    checkpoint strictly read-only, launches 4-GPU Prefill + 4-GPU Decode via
    ``MagaServerManager``, and verifies coherent Chinese/English HTTP ``/`` generation.

Pure PD only: unified / PDFUSION serving mode is intentionally not included.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
import torch

from rtp_llm.config.py_config_modules import MIN_WORKER_INFO_PORT_NUM
from rtp_llm.server.host_service import EndPoint, GroupEndPoint, ServiceRoute
from rtp_llm.test.utils.device_resource import get_gpu_ids
from rtp_llm.test.utils.maga_server_manager import MagaServerManager

DEFAULT_3FS_CKPT = "/mnt/hf3fs/3fs/models/Qwen4.5-Flash-1109-bf16-re-mxfp4-ple-fp8"


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


def _write_synthetic_qwen2_tokenizer(ckpt_dir: Path, vocab_size: int) -> None:
    """Write a minimal valid HF Qwen2Tokenizer directory matching ``vocab_size``."""
    vocab: Dict[str, int] = {
        "<|pad|>": 0,
        "<|im_start|>": 1,
        "<|endoftext|>": 2,
        "<|im_end|>": 3,
    }
    # Fill ASCII printable characters first so ordinary English prompts tokenize cleanly.
    next_id = 4
    for cp in range(32, 127):
        ch = chr(cp)
        tok = "\u0120" if ch == " " else ch
        if tok not in vocab and next_id < vocab_size:
            vocab[tok] = next_id
            next_id += 1
    while next_id < vocab_size:
        vocab[f"tok_{next_id}"] = next_id
        next_id += 1

    (ckpt_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False), encoding="utf-8"
    )
    (ckpt_dir / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (ckpt_dir / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
                "model_max_length": 4096,
                "clean_up_tokenization_spaces": False,
                "added_tokens_decoder": {
                    "0": {"content": "<|pad|>", "special": True},
                    "1": {"content": "<|im_start|>", "special": True},
                    "2": {"content": "<|endoftext|>", "special": True},
                    "3": {"content": "<|im_end|>", "special": True},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ckpt_dir / "special_tokens_map.json").write_text(
        json.dumps(
            {
                "bos_token": "<|im_start|>",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|pad|>",
                "unk_token": "<|endoftext|>",
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _build_synthetic_8layer_checkpoint(ckpt_dir: Path) -> Path:
    """Build a deterministic 8-layer Qwen 4.5-Flash (MXFP4 + FP8 PLE + YoCo) checkpoint + tokenizer."""
    from rtp_llm.models_py.model_desc.qwen4_5 import Qwen4_5ForCausalLM
    from rtp_llm.models_py.modules.qwen4_5.checkpoint_loader import (
        save_qwen4_5_safetensors_checkpoint,
    )
    from rtp_llm.models_py.modules.qwen4_5.contracts import Qwen4_5ArchConfig

    cfg = Qwen4_5ArchConfig(
        vocab_size=2048,
        hidden_size=512,
        num_hidden_layers=8,
        intermediate_size=1024,
        first_k_dense_replace=1,
        rms_norm_eps=1e-6,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        max_position_embeddings=256,
        full_attention_interval=4,
        num_attention_heads=8,
        num_key_value_heads=1,
        head_dim=256,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        rope_theta=10000.0,
        use_full_attn_kv_sconv=True,
        full_attn_kv_sconv_residual=True,
        full_attn_kv_sconv_use_activation=False,
        linear_conv_kernel_dim=4,
        linear_num_key_heads=4,
        linear_key_head_dim=64,
        linear_num_value_heads=16,
        linear_value_head_dim=64,
        gdn_conv_residual=False,
        gdn_no_q_conv=False,
        norm_before_gate=True,
        output_gate_type="sigmoid",
        use_linear_attn_output_sconv=True,
        use_full_attn_output_sconv=True,
        use_ffn_output_sconv=True,
        output_sconv_use_activation=False,
        hc_count=4,
        hc_lowrank=32,
        hc_prev_mix=True,
        hc_per_branch_norm=True,
        hc_rmsnorm_bias=True,
        num_experts=32,
        num_experts_per_tok=4,
        routed_expert_hidden_size=256,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=128,
        shared_expert_gate=True,
        router_precision="fp32_logits",
        router_score_function="sigmoid",
        moe_router_pre_softmax=True,
        router_normalize_topk_scores=True,
        moe_router_enable_expert_bias=True,
        hidden_act="situ_glu",
        hidden_act_gate_beta=4.0,
        hidden_act_up_beta=25.0,
        clip_mlp_output=1000.0,
        ple_layer_ids=(2,),
        ple_embed_dim=512,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=4,
        ngram_vocab_size_base=512,
        make_ngram_vocab_size_divisible_by=64,
        split_ngram_parts=8,
        ple_quant_scheme="fp8_pertensor",
        share_map="4->8,5->6,5->7",
        enable_yoco_decode=True,
        quant_method="mxfp4",
    )
    with torch.no_grad():
        model = Qwen4_5ForCausalLM(
            cfg,
            use_triton_mxfp4=False,
            compute_mode="bf16",
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
        )
        model.init_synthetic_weights(seed=2027, ple_mode="gpu_full")
        # Suppress special tokens 0..3 so greedy decode emits normal text tokens.
        model.lm_head.weight[:4].mul_(0.05)

    save_qwen4_5_safetensors_checkpoint(model, ckpt_dir)
    _write_synthetic_qwen2_tokenizer(ckpt_dir, cfg.vocab_size)
    return ckpt_dir


def _dump_manager_logs(managers: Sequence[MagaServerManager]) -> None:
    for mgr in managers:
        log_path = getattr(mgr, "log_file_path", None)
        if log_path and Path(log_path).exists():
            tail = Path(log_path).read_text(errors="replace")[-6000:]
            logging.error("=== Tail of %s ===\n%s", log_path, tail)


class Qwen4_5PurePDSmokeTest(unittest.TestCase):
    """Pure PD-disaggregated smoke test (ROLE_TYPE=PREFILL + ROLE_TYPE=DECODE only)."""

    def setUp(self) -> None:
        self.mode = os.environ.get("QWEN4_5_PD_SMOKE_MODE", "synthetic_8layer").strip()
        self.session = requests.Session()
        self.session.trust_env = False
        self.addCleanup(self.session.close)
        self._tmpdir: Optional[tempfile.TemporaryDirectory[str]] = None

        if self.mode == "synthetic_8layer":
            self.role_size = 1
            self.tp_size = 1
            self.ep_size = 1
            self.seq_size_per_block = 16
            self.max_seq_len = 256
            self.reserve_mem_mb = 1024
            self.startup_timeout_s = 600
            self._tmpdir = tempfile.TemporaryDirectory(prefix="qwen45_pd_8l_")
            self.addCleanup(self._tmpdir.cleanup)
            self.ckpt_path = str(_build_synthetic_8layer_checkpoint(Path(self._tmpdir.name)))
        elif self.mode == "real_3fs_40layer":
            self.role_size = 4
            self.tp_size = 4
            self.ep_size = 4
            self.seq_size_per_block = 64
            self.max_seq_len = 2048
            self.reserve_mem_mb = 16384
            self.startup_timeout_s = 1800
            self.ckpt_path = os.environ.get("QWEN4_5_CKPT_PATH", DEFAULT_3FS_CKPT)
            if not Path(self.ckpt_path).is_dir():
                self.fail(f"Real 3FS checkpoint not found: {self.ckpt_path}")
        else:
            self.fail(f"Unsupported QWEN4_5_PD_SMOKE_MODE={self.mode!r}")

        self.prefill_mgr, self.decode_mgr = self._start_pd_servers()

    def _start_pd_servers(self) -> tuple[MagaServerManager, MagaServerManager]:
        cuda_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cuda_env:
            gpu_ids = [g.strip() for g in cuda_env.split(",") if g.strip()]
        else:
            gpu_ids = list(get_gpu_ids())
        need_gpus = 2 * self.role_size
        self.assertGreaterEqual(
            len(gpu_ids),
            need_gpus,
            f"Pure PD mode {self.mode!r} needs {need_gpus} GPUs, found {gpu_ids}",
        )

        prefill_port = MagaServerManager.get_free_port()
        decode_port = MagaServerManager.get_free_port()
        route = ServiceRoute(
            service_id=f"qwen4_5_pd_{self.mode}",
            use_local=True,
            role_endpoints=[
                GroupEndPoint(
                    group="default",
                    prefill_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{prefill_port}",
                        protocol="http",
                        path="/",
                    ),
                    decode_endpoint=EndPoint(
                        type="Vipserver",
                        address=f"127.0.0.1:{decode_port}",
                        protocol="http",
                        path="/",
                    ),
                )
            ],
        )

        common_env = {
            "LOAD_PYTHON_MODEL": "1",
            "FT_SERVER_TEST": "1",
            "WORLD_SIZE": str(self.role_size),
            "REMOTE_RPC_SERVER_IP": "127.0.0.1",
            "QWEN4_5_LOAD_WORKERS": os.environ.get("QWEN4_5_LOAD_WORKERS", "8"),
        }
        common_args = (
            f"--tp_size {self.tp_size} "
            f"--ep_size {self.ep_size} "
            f"--dp_size 1 "
            f"--world_size {self.role_size} "
            f"--act_type BF16 "
            f"--seq_size_per_block {self.seq_size_per_block} "
            f"--max_seq_len {self.max_seq_len} "
            f"--reuse_cache 1 "
            f"--enable_cuda_graph 0 "
            f"--cache_store_rdma_mode 0 "
            f"--use_local 1 "
            f"--load_cache_timeout_ms 120000 "
            f"--reserver_runtime_mem_mb {self.reserve_mem_mb} "
            f"--warm_up 0"
        )

        prefill_mgr = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": decode_port,
                "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            },
            device_ids=gpu_ids[: self.role_size],
            port=prefill_port,
            role_name=f"qwen4_5_prefill_{self.mode}",
            smoke_args_str=f"{common_args} --role_type PREFILL",
        )
        decode_mgr = MagaServerManager(
            env_args={
                **common_env,
                "REMOTE_SERVER_PORT": prefill_port,
                "MODEL_SERVICE_CONFIG": route.model_dump_json(),
            },
            device_ids=gpu_ids[self.role_size : 2 * self.role_size],
            port=decode_port,
            role_name=f"qwen4_5_decode_{self.mode}",
            smoke_args_str=f"{common_args} --role_type DECODE",
        )
        self.addCleanup(prefill_mgr.stop_server)
        self.addCleanup(decode_mgr.stop_server)

        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futs = [
                    executor.submit(
                        mgr.start_server,
                        model_path=self.ckpt_path,
                        model_type="qwen4_5",
                        timeout=self.startup_timeout_s,
                    )
                    for mgr in (prefill_mgr, decode_mgr)
                ]
                for fut in futs:
                    self.assertTrue(fut.result(), "PD server failed to pass /health")

            for role_name, base_port in (("prefill", prefill_port), ("decode", decode_port)):
                for rank in range(self.role_size):
                    _wait_tcp_port_ready(
                        "127.0.0.1",
                        int(base_port) + rank * MIN_WORKER_INFO_PORT_NUM + 1,
                        timeout_s=120.0,
                        label=f"{role_name} rank {rank} model rpc",
                    )
        except Exception:
            _dump_manager_logs((prefill_mgr, decode_mgr))
            raise

        return prefill_mgr, decode_mgr

    def _post_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        reuse_cache: bool = True,
    ) -> Dict[str, Any]:
        payload = {
            "prompt": prompt,
            "generate_config": {
                "max_new_tokens": max_new_tokens,
                "top_k": 1,
                "top_p": 1.0,
                "temperature": 0.0,
                "random_seed": 42,
                "reuse_cache": reuse_cache,
                "can_use_pd_separation": True,
                "return_input_ids": True,
                "return_output_ids": True,
                "aux_info": True,
            },
        }
        try:
            resp = self.session.post(
                f"http://127.0.0.1:{self.prefill_mgr.port}/",
                json=payload,
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            _dump_manager_logs((self.prefill_mgr, self.decode_mgr))
            raise
        self.assertNotIn("error_code", data, f"Unexpected PD error response: {data}")
        self.assertIs(data.get("finished"), True, f"Generation did not finish: {data}")
        return data

    def test_pure_pd_generation(self) -> None:
        if self.mode == "synthetic_8layer":
            # Prompt longer than 1 block (seq_size_per_block=16) so shared-prefix hits block 0.
            shared_prefix = "Qwen4.5 Flash pure PD disaggregation verification prefix: "
            r1 = self._post_generate(shared_prefix + "alpha", max_new_tokens=8, reuse_cache=True)
            out1 = r1.get("response", "")
            self.assertIsInstance(out1, str)
            self.assertGreater(len(r1.get("output_ids", [[]])[0]), 0)
            logging.info("[8L PD req1] output_ids=%s aux=%s", r1.get("output_ids"), r1.get("aux_info"))

            # Determinism under greedy decoding across PD
            r1_repeat = self._post_generate(shared_prefix + "alpha", max_new_tokens=8, reuse_cache=True)
            self.assertEqual(r1.get("output_ids"), r1_repeat.get("output_ids"))
            reuse_len = int(r1_repeat.get("aux_info", {}).get("prefill_total_reuse_len", 0))
            logging.info("[8L PD repeat] reuse_len=%d", reuse_len)
            self.assertGreaterEqual(reuse_len, self.seq_size_per_block)

            # Different suffix sharing the cached prefix block
            r2 = self._post_generate(shared_prefix + "beta", max_new_tokens=8, reuse_cache=True)
            self.assertGreater(len(r2.get("output_ids", [[]])[0]), 0)
            reuse_len2 = int(r2.get("aux_info", {}).get("prefill_total_reuse_len", 0))
            logging.info("[8L PD shared-prefix req2] reuse_len=%d output_ids=%s", reuse_len2, r2.get("output_ids"))
            self.assertGreaterEqual(reuse_len2, self.seq_size_per_block)
            return

        # Real 40-layer 3FS pure PD (4P + 4D)
        cases = [
            ("中国的首都是", "北京"),
            ("The capital of China is", "Beijing"),
            ("17 * 24 = ", "408"),
        ]
        for prompt, expected_substr in cases:
            res = self._post_generate(prompt, max_new_tokens=24, reuse_cache=True)
            text = res.get("response", "")
            logging.info("[40L 3FS PD] prompt=%r -> response=%r aux=%s", prompt, text, res.get("aux_info"))
            self.assertTrue(text.strip(), f"Empty response for {prompt!r}")
            self.assertIn(
                expected_substr,
                text,
                f"Expected {expected_substr!r} in real 3FS PD response for {prompt!r}, got {text!r}",
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
