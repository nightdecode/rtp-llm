import asyncio
import logging
import pickle
from dataclasses import replace
from typing import Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.multimodal.mm_process_engine import MMProcessEngine
from rtp_llm.multimodal.multimodal_mixin_factory import MultimodalMixinFactory
from rtp_llm.ops import RtpEmbeddingOp, VitSeparation


class EmbeddingCppEngine(BaseEngine):
    def __init__(
        self,
        model,
        engine_config: EngineConfig,
    ):
        super().__init__(model)
        logging.info("creating cpp embedding engine")
        self.model = model
        self.engine_config = engine_config
        from rtp_llm.ops import ensure_engine_ops_loaded

        ensure_engine_ops_loaded()
        from rtp_llm.ops import RtpEmbeddingOp

        self.cpp_engine = RtpEmbeddingOp()
        # Owned here: created in _start (local VIT), torn down in _stop. None until
        # started, so _stop before _start is safe.
        self.mm_process_engine: Optional[MMProcessEngine] = None

    def set_serving(self, serving: bool) -> None:
        self.cpp_engine.set_serving(serving)

    @override
    def _stop(self) -> None:
        # Stop the C++ engine first so it issues no more embedding calls, then
        # close the local MMProcessEngine (which stops its MMScheduler thread and
        # preprocessing executor).
        self.cpp_engine.stop()
        if self.mm_process_engine is not None:
            self.mm_process_engine.stop()
            self.mm_process_engine = None

    @override
    def _start(self):
        self.mm_process_engine = None
        if (
            self.model.is_multimodal()
            and self.model.vit_config.vit_separation
            == VitSeparation.VIT_SEPARATION_LOCAL
        ):
            self.mm_process_engine = (
                MultimodalMixinFactory.create_multimodal_process_engine(
                    model_config=self.model.model_config,
                    engine_config=self.engine_config,
                    vit_config=self.model.vit_config,
                    device=f"cuda:{self.engine_config.parallelism_config.local_rank}",
                )
            )
        self.cpp_engine.init(
            self.model,
            self._graph_gated_engine_config(),
            self.model.vit_config,
            self.mm_process_engine,
        )

    def _graph_gated_engine_config(self) -> EngineConfig:
        """Gate the cache-less embedding prefill CUDA graph on model support.

        The embedding prefill CUDA graph is opt-in per model family: only
        wrappers declaring ``support_embedding_prefill_graph`` may capture.
        Everything else keeps the eager path even when the server was started
        with ``--enable_cuda_graph`` (for example qwen2-gte, whose flashinfer
        paged-prefill attention cannot even instantiate at capture-warmup
        bucket shapes). The multimodal engine and any other consumer keep the
        original config; only the config handed to the embedding engine is a
        copy with graphs disabled.
        """
        hw_kernel_config = self.engine_config.hw_kernel_config
        if not hw_kernel_config.enable_cuda_graph:
            return self.engine_config
        if _declares_embedding_prefill_graph(self.model):
            return self.engine_config
        logging.warning(
            "model %s does not declare support_embedding_prefill_graph; "
            "embedding prefill CUDA graph disabled, using eager path",
            type(self.model).__name__,
        )
        gated = replace(self.engine_config)
        gated.hw_kernel_config = pickle.loads(pickle.dumps(hw_kernel_config))
        gated.hw_kernel_config.enable_cuda_graph = False
        return gated


def _declares_embedding_prefill_graph(model) -> bool:
    flag = getattr(model, "support_embedding_prefill_graph", None)
    if flag is None:
        return False
    return bool(flag() if callable(flag) else flag)
