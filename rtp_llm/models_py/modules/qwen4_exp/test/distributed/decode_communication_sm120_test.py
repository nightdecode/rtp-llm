"""Real two-rank IPC, vocabulary and primitive Graph tests for all N=1..64.

These are primitive Graphs, not additional model capture buckets.
"""

import gc
import os
import socket
import sys
import unittest
import weakref
from contextlib import ExitStack
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def _embedding(owners, reducer, compute=None):
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        DecodeVocabEmbedding,
    )
    from rtp_llm.models_py.modules.qwen4_exp.decode_vocab_weights import (
        prepare_decode_vocab,
    )

    prepared = prepare_decode_vocab(
        source_weights=owners,
        compute_weights=owners if compute is None else compute,
        group=reducer.workspace.group,
    )
    return DecodeVocabEmbedding(prepared, reducer)


def _check_vocab_hot_updates(rank, reducer, device):
    from rtp_llm.model_loader import weight_manager as manager_module
    from rtp_llm.model_loader.model_weight_info import ModelWeights
    from rtp_llm.model_loader.weight_module import AtomicWeight
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        DecodeVocabEmbedding,
    )
    from rtp_llm.models_py.modules.qwen4_exp.decode_vocab_weights import (
        prepare_decode_vocab,
    )
    from rtp_llm.utils.model_weight import CkptWeightInfo, W

    vocab, width = 129, 2560
    full = (
        torch.arange(vocab * width, device=device).reshape(vocab, width) % 11
    ).bfloat16()
    # ModelLoader creates its weights under inference_mode.
    with torch.inference_mode():
        source = full[:, rank * 1280 : (rank + 1) * 1280].contiguous()
    weights = ModelWeights(0, str(device), torch.bfloat16)
    # A live tied consumer must update as well, without retaining retired sources.
    weights.global_weights = {W.embedding: source, W.lm_head: source}
    shard = prepare_decode_vocab(
        source_weights=weights.global_weights,
        compute_weights=weights.global_weights,
        group=reducer.workspace.group,
        update_handlers=weights.global_weight_update_handlers,
    )
    target = DecodeVocabEmbedding(shard, reducer)
    draft = _embedding(weights.global_weights, reducer)
    assert target.shard is draft.shard
    pointer = target.weight.data_ptr()
    load_config = SimpleNamespace(
        tp_size=2,
        tp_rank=rank,
        ep_size=1,
        ep_rank=0,
        dp_size=1,
        dp_rank=0,
        ffn_tp_size=1,
        ffn_tp_rank=0,
        hidden_size=width,
        head_num=1,
        head_num_kv=1,
        size_per_head=width,
        moe_pure_tp_mode=False,
        bit=16,
        use_swizzleA=False,
        exported_device=SimpleNamespace(
            maybe_rewrite_weight_by_key=lambda name, value, **kw: value
        ),
    )
    descriptor = AtomicWeight(
        W.embedding, [CkptWeightInfo("model.embed_tokens.weight")]
    )
    loader = SimpleNamespace(
        _model_weights_info=SimpleNamespace(weights=[descriptor]),
        get_load_config=lambda: load_config,
        is_global_weight_local=lambda name: True,
    )
    manager = manager_module.WeightManager(device, weights, loader)
    request = dict(name="model.embed_tokens.weight", method="cuda_ipc", desc="00")
    ids = torch.tensor([0, 64, 128, 3], device=device)
    target(ids)
    draft(ids)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        target_output = target(ids)
        draft_output = draft(ids)
    reducer.finish_graph_capture()
    try:

        def update_local(request, value):
            # RPC delivery is rank-local: no peer is executing update here.
            # Fail fast instead of hanging a regression test on a new collective.
            with ExitStack() as stack:
                for name in (
                    "all_reduce",
                    "all_to_all_single",
                    "all_gather_object",
                    "all_gather",
                    "all_gather_into_tensor",
                    "barrier",
                ):
                    stack.enter_context(
                        patch.object(
                            dist,
                            name,
                            side_effect=AssertionError(
                                "weight update must not wait for another rank"
                            ),
                        )
                    )
                stack.enter_context(
                    patch.object(manager, "_load_update_tensor", return_value=value)
                )
                manager.update(request)

        # Update rank 0 completely before rank 1 enters its handler. Both CUDA
        # and reconstructed CPU inputs must preserve pointers and odd-vocab pad.
        for gain in (2, 3):
            updated = full * gain
            incoming = updated if gain == 2 else updated.cpu().t().contiguous().t()
            for owner in (0, 1):
                if rank == owner:
                    update_local(request, incoming)
                    expected = torch.zeros_like(target.weight)
                    begin = owner * target.weight.shape[0]
                    count = min(target.weight.shape[0], vocab - begin)
                    expected[:count] = updated[begin : begin + count]
                    torch.testing.assert_close(target.weight, expected, rtol=0, atol=0)
                dist.barrier()
            graph.replay()
            torch.testing.assert_close(target_output, updated[ids], rtol=0, atol=0)
            torch.testing.assert_close(draft_output, updated[ids], rtol=0, atol=0)
            assert pointer == target.weight.data_ptr() == draft.weight.data_ptr()
            torch.testing.assert_close(
                source, updated[:, rank * 1280 : (rank + 1) * 1280], rtol=0, atol=0
            )
            if rank == 1:
                assert not bool(target.weight[-1].count_nonzero())

        before = target.weight.clone()
        before_alias = source.clone()
        # A failed local preparation changes neither live destination. The
        # caller keeps engines paused until all intended updates have succeeded.
        for failure in ("shape", "width", "dtype", "decode", "allocation", "pack"):
            candidate = full + 9
            if failure == "shape":
                candidate = candidate[:-1]
            if failure == "dtype":
                candidate = candidate.float()
            if failure == "width":
                candidate = torch.nn.functional.pad(candidate, (0, 1))

            def decode(*_):
                if failure == "decode":
                    raise ValueError("injected IPC decode failure")
                return candidate

            empty_like, copy = torch.empty_like, torch.Tensor.copy_

            def allocate(value, **kwargs):
                if failure == "allocation" and value is target.weight:
                    raise torch.OutOfMemoryError("injected staging allocation failure")
                return empty_like(value, **kwargs)

            def pack(value, other, *args, **kwargs):
                if (
                    failure == "pack"
                    and value.ndim == 2
                    and value.shape[-1] == width
                    and value.data_ptr() != pointer
                ):
                    raise RuntimeError("injected layout preparation failure")
                return copy(value, other, *args, **kwargs)

            for owner in (0, 1):
                if rank == owner:
                    with (
                        patch.object(
                            manager, "_load_update_tensor", side_effect=decode
                        ),
                        patch.object(torch, "empty_like", side_effect=allocate),
                        patch.object(torch.Tensor, "copy_", pack),
                        patch.object(
                            dist,
                            "all_reduce",
                            side_effect=AssertionError(
                                "local failure must not wait for a peer"
                            ),
                        ),
                    ):
                        try:
                            manager.update(request)
                        except (RuntimeError, ValueError):
                            pass
                        else:
                            raise AssertionError("invalid preparation must be rejected")
                    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)
                    torch.testing.assert_close(source, before_alias, rtol=0, atol=0)
                    update_local(request, full * 3)
                dist.barrier()
            graph.replay()
            torch.testing.assert_close(target_output, (full * 3)[ids], rtol=0, atol=0)

        # A shared draft's non-owning manager must not update its target owner.
        alias_manager = manager_module.WeightManager(
            device, weights, loader, (W.embedding,)
        )
        with patch.object(
            manager_module.CudaIpcHelper, "build_from_meta", return_value=full
        ):
            try:
                alias_manager.update(request)
            except PermissionError as error:
                assert "non-owning" in str(error)
            else:
                raise AssertionError("alias update must be rejected")
            manager.update(request)
        graph.replay()
        torch.testing.assert_close(target_output, full[ids], rtol=0, atol=0)
        torch.testing.assert_close(draft_output, full[ids], rtol=0, atol=0)
        try:
            weights.update_global_weight(W.embedding, torch.zeros_like(target.weight))
        except PermissionError:
            pass
        else:
            raise AssertionError(
                "copies bypassing the prepared layout must remain protected"
            )
    finally:
        torch.cuda.synchronize()
        graph.reset()

    # Existing quantized/restart-only ownership must not acquire an updater.
    from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

    fp8_weights = ModelWeights(0, str(device), torch.bfloat16)
    fp8_weights.global_weights = {
        W.embedding: source.to(torch.float8_e4m3fn),
        W.embedding + FP8_STORAGE_SCALE_SUFFIX: torch.ones(1, device=device),
    }
    prepare_decode_vocab(
        source_weights=fp8_weights.global_weights,
        compute_weights=fp8_weights.global_weights,
        group=reducer.workspace.group,
        update_handlers=fp8_weights.global_weight_update_handlers,
    )
    assert W.embedding not in fp8_weights.global_weight_update_handlers
    fp8_manager = manager_module.WeightManager(device, fp8_weights, loader)
    with patch.object(
        manager_module.CudaIpcHelper,
        "build_from_meta",
        side_effect=AssertionError("must reject before decoding"),
    ):
        try:
            fp8_manager.update(request)
        except PermissionError:
            pass
        else:
            raise AssertionError("quantized update must remain guarded")


def _worker(rank, port, hot_update_only=False):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=180),
    )
    from rtp_llm.models_py.distributed import collective_torch
    from rtp_llm.models_py.modules.qwen4_exp.decode_communication import (
        Qwen4DecodeReduce,
    )

    collective_torch._group_map[collective_torch.Group.TP] = dist.group.WORLD
    collective_torch._group_map[collective_torch.Group.WORLD] = dist.group.WORLD
    collective_torch._group_map[collective_torch.Group.STAGE] = dist.group.WORLD
    collective_torch._initialized = True
    collective_torch._parallelism_config = SimpleNamespace(
        tp_size=2, dp_size=1, pp_size=1, world_size=2
    )
    group = collective_torch.Group.TP
    device = torch.device("cuda", rank)
    from flashinfer.comm import cuda_ipc, vllm_ar

    from rtp_llm.models_py.modules.qwen4_exp import decode_communication as module

    communicator = SimpleNamespace(
        group=dist.group.WORLD, device=device, disabled=False, cpp_small_enabled=True
    )
    parallelism = SimpleNamespace(
        role_type="DECODE", tp_size=2, dp_size=1, local_world_size=2
    )
    # Admission must reject incompatible peer policies before entering IPC setup.
    for field, local in (
        (
            "_qwen4_decode_communication_mode",
            "default" if rank == 0 else "vocab_parallel_ipc",
        ),
        ("_qwen4_target_decode_reduce", None),
    ):
        config = SimpleNamespace(hidden_size=2560)
        if field.endswith("mode") or rank == 0:
            setattr(config, field, local)
        with patch.object(module, "get_all_reduce", return_value=communicator):
            try:
                module.prepare_decode_reduce(config, parallelism)
            except ValueError:
                pass
            else:
                raise AssertionError("inconsistent rank admission must be rejected")
    # Fail allocation on just one rank, inside the real coordinated constructor.
    # Both ranks must retain the original layout and remain able to use NCCL.
    malloc = cuda_ipc.cudart.cudaMalloc

    def allocate(size):
        if rank == 0:
            raise RuntimeError("injected rank-local CUDA allocation failure")
        return malloc(size)

    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(cuda_ipc.cudart, "cudaMalloc", side_effect=allocate),
    ):
        assert (
            module.prepare_decode_reduce(SimpleNamespace(hidden_size=2560), parallelism)
            is None
        )
    register = vllm_ar.register_buffer

    def register_buffer(*args):
        if rank == 1:
            raise RuntimeError("injected rank-local IPC registration failure")
        return register(*args)

    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(vllm_ar, "register_buffer", side_effect=register_buffer),
    ):
        assert (
            module.prepare_decode_reduce(SimpleNamespace(hidden_size=2560), parallelism)
            is None
        )
    value = torch.tensor([rank + 1.0], device=device)
    dist.all_reduce(value)
    torch.testing.assert_close(value, torch.tensor([3.0], device=device))
    draft_config = SimpleNamespace(hidden_size=2560, _qwen4_target_decode_reduce=None)
    with (
        patch.object(module, "get_all_reduce", return_value=communicator),
        patch.object(
            Qwen4DecodeReduce,
            "__init__",
            side_effect=AssertionError("draft retried failed IPC"),
        ),
    ):
        assert module.prepare_decode_reduce(draft_config, parallelism) is None

    reducer = Qwen4DecodeReduce(communicator)
    graphs = []
    try:
        if hot_update_only:
            _check_vocab_hot_updates(rank, reducer, device)
            if rank == 0:
                print(
                    "PASS: rank-local BF16 hot update, target/draft Graph replay, stable pointers, local failure and retry",
                    flush=True,
                )
            return
        # Converted owners must release hidden shards, even while the model
        # and its shared draft embedding remain alive. Reserved allocator
        # memory is intentionally not used as evidence of tensor lifetime.
        from rtp_llm.models_py.modules.qwen4_exp.weight_views import PreparedWeightDict
        from rtp_llm.utils.model_weight import W

        owners = {
            W.embedding: torch.ones(
                4096 * 1280, device=device, dtype=torch.bfloat16
            ).view(4096, 1280)
        }
        compute = PreparedWeightDict(owners)
        base_ref = weakref.ref(owners[W.embedding]._base)
        source_ref = weakref.ref(owners[W.embedding])
        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        retired = _embedding(owners, reducer, compute)
        gc.collect()
        torch.cuda.synchronize()
        assert source_ref() is None, "old hidden-sharded embedding must be released"
        assert base_ref() is None, "view-backed hidden storage must also be released"
        assert owners[W.embedding] is compute[W.embedding] is retired.weight
        assert (
            torch.cuda.memory_allocated() - allocated_before < 1024 * 1024
        ), "repartition must not retain a second 10 MiB vocabulary shard"
        converted_ptr = retired.weight.data_ptr()
        if rank == 1:
            del retired
            gc.collect()
        shared_owner = _embedding(owners, reducer)
        assert shared_owner.weight.data_ptr() == converted_ptr
        torch.testing.assert_close(
            shared_owner(torch.tensor([0, 2048, 4095], device=device)),
            torch.ones(3, 2560, device=device, dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )
        inconsistent = owners[W.embedding] if rank == 0 else owners[W.embedding].clone()
        try:
            _embedding({W.embedding: inconsistent}, reducer)
        except ValueError as exc:
            assert "matching" in str(exc)
        else:
            raise AssertionError(
                "mixed hidden/vocabulary layouts must fail on every rank"
            )
        del inconsistent, shared_owner, owners, compute
        if rank == 0:
            del retired
        gc.collect()
        for rows in range(1, 65):
            value = torch.full(
                (rows, 2560), rank + 1, device=device, dtype=torch.bfloat16
            )
            original = value.clone()
            eager = reducer(value, group)
            torch.testing.assert_close(eager, torch.full_like(value, 3), rtol=0, atol=0)
            torch.testing.assert_close(value, original, rtol=0, atol=0)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = reducer(value, group)
            reducer.finish_graph_capture()
            graphs.append((graph, value, out))
        # Different sizes repeatedly reuse one workspace; inputs stay live and
        # are changed between replays to detect stale registration/staging.
        for sequence in (range(64), reversed(range(64)), range(0, 64, 3)):
            for index in sequence:
                graph, value, out = graphs[index]
                value.fill_(rank + index % 7)
                graph.replay()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2 * (index % 7) + 1), rtol=0, atol=0
                )
        # Target and draft capture/register separate graphs on one ordered IPC
        # workspace. Closing either wrapper must leave the other graph usable.
        draft_config._qwen4_target_decode_reduce = reducer
        with patch.object(module, "get_all_reduce", return_value=communicator):
            draft = module.prepare_decode_reduce(draft_config, parallelism)
        assert draft is not reducer and draft.workspace is reducer.workspace
        for rows in (2, 8, 16, 32, 64):
            value = torch.full(
                (rows, 2560), rank + 1, device=device, dtype=torch.bfloat16
            )
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = draft(value, group)
            draft.finish_graph_capture()
            graphs.append((graph, value, out))
            for shift in range(3):
                graphs[rows - 1][0].replay()
                value.fill_(rank + shift)
                graph.replay()
                torch.testing.assert_close(
                    out, torch.full_like(out, 2 * shift + 1), rtol=0, atol=0
                )
        draft.close()
        draft.close()
        assert not reducer.closed
        graphs[0][0].replay()
        torch.testing.assert_close(
            graphs[0][2], torch.ones_like(graphs[0][2]), rtol=0, atol=0
        )

        for noncontiguous in (False, True):
            storage = torch.full(
                (8 * 2560 * 2 + 8,), rank + 1, device=device, dtype=torch.bfloat16
            )
            value = (
                storage[rank : rank + 8 * 2560 * 2 : 2].view(8, 2560)
                if noncontiguous
                else storage[rank : rank + 8 * 2560].view(8, 2560)
            )
            with patch.object(
                collective_torch,
                "all_reduce",
                side_effect=AssertionError("rank-local layout changed backend"),
            ):
                actual = reducer(value, group)
                torch.testing.assert_close(
                    actual, torch.full_like(actual, 3), rtol=0, atol=0
                )
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    actual = reducer(value, group)
                reducer.finish_graph_capture()
                value.fill_(rank + 3)
                graph.replay()
                torch.testing.assert_close(
                    actual, torch.full_like(actual, 7), rtol=0, atol=0
                )
                graphs.append((graph, value, actual))

        # Both oversized and wrong-dtype calls preserve the NCCL fallback.
        for rows, dtype in ((65, torch.bfloat16), (8, torch.float32)):
            value = torch.full((rows, 2560), rank + 1, device=device, dtype=dtype)
            torch.testing.assert_close(
                reducer(value, group), torch.full_like(value, 3), rtol=0, atol=0
            )

        for fp8 in (False, True):
            vocab, half_width = 129, 1280
            full = (
                (
                    torch.arange(vocab * half_width * 2, device=device).reshape(
                        vocab, half_width * 2
                    )
                    % 15
                )
                - 7
            ).to(torch.bfloat16)
            weight = full[:, rank * half_width : (rank + 1) * half_width].contiguous()
            scale = None
            if fp8:
                weight = weight.to(torch.float8_e4m3fn)
                scale = torch.tensor([0.7 if rank == 0 else 0.3], device=device)
                expected_weight = torch.cat(
                    [
                        (
                            full[:, :half_width]
                            * torch.tensor(0.7, device=device, dtype=torch.bfloat16)
                        ).to(torch.bfloat16),
                        (
                            full[:, half_width:]
                            * torch.tensor(0.3, device=device, dtype=torch.bfloat16)
                        ).to(torch.bfloat16),
                    ],
                    dim=1,
                )
            else:
                expected_weight = full
            from rtp_llm.utils.fp8_constants import FP8_STORAGE_SCALE_SUFFIX

            owners = {W.embedding: weight}
            if scale is not None:
                owners[W.embedding + FP8_STORAGE_SCALE_SUFFIX] = scale
            embedding = _embedding(owners, reducer)
            shared = _embedding(owners, reducer)
            assert (
                shared.shard is embedding.shard
            ), "target/draft must share vocabulary storage"
            assert weight._rtp_requires_restart_for_update
            for rows in range(1, 65):
                ids = (
                    torch.arange(rows, device=device, dtype=torch.int64)
                    * 17
                    % (vocab + 3)
                    - 1
                )
                mask = torch.arange(rows, device=device) % 5 != 0
                wanted = expected_weight[ids.clamp(0, vocab - 1)].clone()
                wanted[(ids < 0) | (ids >= vocab) | ~mask] = 0
                actual = embedding(ids, text_tokens_mask=mask)
                torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = embedding(ids, text_tokens_mask=mask)
                reducer.finish_graph_capture()
                for shift in (0, 1, 63, 64, 128):
                    ids.fill_(shift)
                    graph.replay()
                    wanted = expected_weight[ids].clone()
                    wanted[~mask] = 0
                    torch.testing.assert_close(output, wanted, rtol=0, atol=0)
                graphs.append((graph, ids, mask, output, embedding))
            empty = embedding(torch.empty(0, device=device, dtype=torch.int64))
            assert empty.shape == (0, 2560)
        torch.cuda.synchronize()
        dist.barrier()
        if rank == 0:
            print(
                "PASS: N=1..64 BF16 reduction, BF16/FP8 embedding, repeated Graph replay, boundary/fallback/shared-weight cases",
                flush=True,
            )
    finally:
        # On failure, let mp.spawn report the exception and terminate peers;
        # a cleanup collective would hide the original error behind a timeout.
        if sys.exc_info()[0] is None:
            # Graphs keep registered pointers alive until both ranks finish.
            torch.cuda.synchronize()
            dist.barrier()
            graphs.clear()
            reducer.workspace.destroy()
            dist.destroy_process_group()


@unittest.skipUnless(
    torch.cuda.device_count() >= 2 and torch.cuda.get_device_capability(0) == (12, 0),
    "requires two SM120 GPUs",
)
class DecodeCommunicationSm120Test(unittest.TestCase):
    def test_vocab_store_mask_does_not_overflow_large_logical_rows(self):
        from rtp_llm.models_py.triton_kernels.common.decode_vocab_lookup import (
            _vocab_rows,
        )

        tokens = torch.zeros(1, device="cuda", dtype=torch.int32)
        weight = torch.ones(1, 2560, device="cuda", dtype=torch.bfloat16)
        # Launch only the first CTA: every access is within these small buffers,
        # while the runtime row count exercises the int32 N*D boundary.
        for rows in (64, 838860, 838861):
            output = torch.full((1024,), -7, device="cuda", dtype=torch.bfloat16)
            _vocab_rows[(1,)](
                tokens,
                weight,
                tokens,
                tokens,
                output,
                rows,
                START=0,
                LOCAL_V=1,
                V=1,
                D=2560,
                FP8=False,
                TOKEN_STRIDE=1,
                MASK_STRIDE=0,
                HAS_MASK=False,
                BLOCK=1024,
                num_warps=4,
            )
            torch.testing.assert_close(output, torch.ones_like(output), rtol=0, atol=0)

    def test_two_rank_decode_communication(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port,), nprocs=2, join=True)

    def test_two_rank_vocab_hot_updates(self):
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        mp.spawn(_worker, args=(port, True), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
