"""Unified dense MLP implementation supporting multiple activation types."""

from dataclasses import dataclass
from typing import Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.distributed.collective_torch import Group, all_reduce
from rtp_llm.models_py.modules.base import FusedSiluAndMul
from rtp_llm.models_py.modules.factory import LinearFactory
from rtp_llm.ops import ActivationType, HWKernelConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W

# CUDA-only fused silu_and_mul + per-token-group fp8 quant. Activated when
# down_proj is CudaFp8GEMMLinear with UE8M0 scales — saves one launch
# (silu_and_mul) and one launch (per_token_group_quant_8bit) per forward.
_DEVICE_TYPE = get_device_type()
if _DEVICE_TYPE == DeviceType.Cuda:
    from rtp_llm.models_py.modules.factory.linear.impl.cuda.fp8_gemm_linear import (
        CudaFp8GEMMLinear,
    )
    from rtp_llm.models_py.triton_kernels.common.activation import (
        silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd,
    )
else:
    CudaFp8GEMMLinear = None  # type: ignore

if _DEVICE_TYPE == DeviceType.Ppu:
    from rtp_llm.models_py.kernels.ppu.fp8_quant import silu_mul_fp8_quantize_128
    from rtp_llm.models_py.modules.factory.linear.ppu_impl.fp8_deepgemm_linear import (
        PpuFp8DeepGemmLinear,
    )
else:
    PpuFp8DeepGemmLinear = None  # type: ignore

_ACTIVATION_FUNC_MAP: Dict[ActivationType, Type[nn.Module]] = {
    ActivationType.Swiglu: FusedSiluAndMul,
    ActivationType.Gelu: nn.GELU,
}

_GATED_ACTIVATION_TYPE_LIST = [ActivationType.Swiglu]


@dataclass
class DenseMlpActivation:
    """down_proj 还没跑时的中间结果，由 forward_activation 产出。

    拆成两段是为了让调用方能在两段之间插入别的工作。MoE 层用它把 shared expert
    的两个 GEMM 分别放进 DeepEP 的 dispatch 和 combine 两个传输窗口里。
    """

    # down_proj 的输入：量化路径下是 FP8 张量，其余路径下是激活后的 BF16 张量
    down_input: torch.Tensor
    # 量化路径下 down_input 的 per-group scale；非量化路径为 None
    down_scales: Optional[torch.Tensor] = None


class DenseMLP(nn.Module):
    """
    Unified DenseMLP implementation supporting both SiGLU and GELU activations.

    - For SiGLU (Swiglu): Uses gate_up_proj + fused silu_and_mul + down_proj
    - For GELU (Gelu): Uses intermediate_proj + GELU activation + output_proj
    """

    def __init__(
        self,
        activation_type: ActivationType,
        parallelism_config: ParallelismConfig,
        weights: Dict[str, torch.Tensor],
        quant_config: object,
        hw_kernel_config: Optional["HWKernelConfig"] = None,
        swiglu_limit: float = 0.0,
    ):
        super().__init__()

        self.activation_type = activation_type
        self.parallelism_config = parallelism_config
        # GLM-5.3-Flash clamps before SwiGLU: gate.clamp(max=limit) and
        # up.clamp(-limit, limit). 0 disables the clamp (every other model).
        self.swiglu_limit = float(swiglu_limit or 0.0)
        if self.activation_type not in _ACTIVATION_FUNC_MAP:
            raise ValueError(f"Unsupported activation type: {activation_type}")
        self.act_fn = _ACTIVATION_FUNC_MAP[activation_type]()
        self.is_gated = activation_type in _GATED_ACTIVATION_TYPE_LIST

        if self.is_gated:
            if W.ffn_w13 not in weights:
                self.up_proj = LinearFactory.create_merged_linear(
                    weights,
                    weight_keys=[W.ffn_w1, W.ffn_w3],
                    scale_keys=[W.ffn_s1, W.ffn_s3],
                    bias_keys=[W.ffn_b1, W.ffn_b3],
                    quant_config=quant_config,
                    dim=-1,
                    hw_kernel_config=hw_kernel_config,
                    scale2_keys=[W.ffn_w1_s2, W.ffn_w3_s2],
                    input_scale_keys=[W.ffn_w1_i_s, W.ffn_w3_i_s],
                )
            else:
                self.up_proj = LinearFactory.create_linear_from_weights(
                    weights,
                    W.ffn_w13,
                    W.ffn_s13,
                    W.ffn_b13,
                    quant_config=quant_config,
                    hw_kernel_config=hw_kernel_config,
                    weight_scale_2_key=W.ffn_w13_s2,
                    input_scale_key=W.ffn_w13_i_s,
                )

        else:
            self.up_proj = LinearFactory.create_linear_from_weights(
                weights,
                W.ffn_w3,
                W.ffn_s3,
                W.ffn_b3,
                quant_config=quant_config,
                hw_kernel_config=hw_kernel_config,
                weight_scale_2_key=W.ffn_w3_s2,
                input_scale_key=W.ffn_w3_i_s,
            )

        self.down_proj = LinearFactory.create_linear_from_weights(
            weights,
            W.ffn_w2,
            W.ffn_s2,
            W.ffn_b2,
            quant_config=quant_config,
            hw_kernel_config=hw_kernel_config,
            weight_scale_2_key=W.ffn_w2_s2,
            input_scale_key=W.ffn_w2_i_s,
        )

        from rtp_llm.models_py.utils.fuse_config import fuse_kernels_enabled

        self._fuse_kernels = fuse_kernels_enabled(hw_kernel_config)
        # Only the PPU fused kernel below takes a clamp limit. The CUDA fused
        # kernels do not, so a clamping model falls back to the generic path.
        self._fuse_silu_quant = (
            self._fuse_kernels
            and self.is_gated
            and self.swiglu_limit <= 0
            and CudaFp8GEMMLinear is not None
            and isinstance(self.down_proj, CudaFp8GEMMLinear)
            and (self.down_proj.K % 128 == 0)
        )
        if self._fuse_silu_quant and self.down_proj.scale_ue8m0:
            self._fuse_silu_quant = self.down_proj.K % 512 == 0
        self._fuse_ppu_silu_quant = (
            self._fuse_kernels
            and self.is_gated
            and PpuFp8DeepGemmLinear is not None
            and isinstance(self.down_proj, PpuFp8DeepGemmLinear)
            and self.down_proj.k % 128 == 0
        )

    @property
    def accepts_fp8_input(self) -> bool:
        return (
            CudaFp8GEMMLinear is not None
            and isinstance(self.up_proj, CudaFp8GEMMLinear)
        ) or (
            PpuFp8DeepGemmLinear is not None
            and isinstance(self.up_proj, PpuFp8DeepGemmLinear)
        )

    @property
    def _down_proj_fuses_activation(self) -> bool:
        """down_proj 是否把 silu、量化和 GEMM 融在一次调用里。

        融在一起时 activation 和 down_proj 之间没有缝，forward 不能拆成两段。
        """
        return bool(
            self.is_gated
            and self.swiglu_limit <= 0
            and getattr(self, "_fuse_kernels", True)
            and getattr(self.down_proj, "supports_fused_silu_quant", False)
        )

    @property
    def supports_split_down(self) -> bool:
        """forward 能否拆成 forward_activation + forward_down 两段。"""
        return not self._down_proj_fuses_activation

    def forward_activation(
        self,
        x: torch.Tensor,
        *,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> DenseMlpActivation:
        """跑到 down_proj 之前为止，返回 down_proj 还差的输入。"""
        if not self.is_gated and self.activation_type == ActivationType.Gelu:
            return DenseMlpActivation(self.up_proj.forward_with_bias_gelu(x))
        if x_fp8 is not None and x_scale is not None and self.accepts_fp8_input:
            up = self.up_proj(x_fp8, input_scales=x_scale)
        else:
            up = self.up_proj(x)
        if getattr(self, "_fuse_ppu_silu_quant", False) and up.dim() == 2:
            fp8_out, scale_out = silu_mul_fp8_quantize_128(
                up.contiguous(),
                gate_first=True,
                column_major_scales=self.down_proj.prefers_column_major_scales(
                    up.shape[0]
                ),
                swiglu_limit=self.swiglu_limit,
            )
            return DenseMlpActivation(fp8_out, scale_out)
        if self._fuse_silu_quant and up.dim() == 2:
            scale_ue8m0 = self.down_proj.scale_ue8m0
            fp8_out, scale_out = (
                silu_and_mul_per_token_group_fp8_quant_dense_packed_fwd(
                    up.contiguous(),
                    quant_group_size=128,
                    scale_ue8m0=scale_ue8m0,
                )
            )
            return DenseMlpActivation(fp8_out, scale_out)
        if self.is_gated and self.swiglu_limit > 0:
            # Unfused fallback: clamp the packed [gate, up] halves before
            # the activation, matching the reference's pre-SwiGLU clamp.
            d = up.shape[-1] // 2
            up = torch.cat(
                (
                    up[..., :d].clamp(max=self.swiglu_limit),
                    up[..., d:].clamp(-self.swiglu_limit, self.swiglu_limit),
                ),
                dim=-1,
            )
        return DenseMlpActivation(self.act_fn(up))

    def forward_down(
        self,
        activation: DenseMlpActivation,
        skip_allreduce: bool = False,
    ) -> torch.Tensor:
        """跑 down_proj 以及（需要时）TP all-reduce。"""
        if activation.down_scales is not None:
            output = self.down_proj(
                activation.down_input, input_scales=activation.down_scales
            )
        else:
            output = self.down_proj(activation.down_input)
        if not skip_allreduce and self.parallelism_config.get_ffn_tp_size() > 1:
            output = all_reduce(output, group=Group.TP)
        return output

    def forward(
        self,
        x: torch.Tensor,
        skip_allreduce: bool = False,
        *,
        x_fp8: Optional[torch.Tensor] = None,
        x_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._down_proj_fuses_activation:
            if x_fp8 is not None and x_scale is not None and self.accepts_fp8_input:
                up = self.up_proj(x_fp8, input_scales=x_scale)
            else:
                up = self.up_proj(x)
            output = self.down_proj.forward_silu_and_mul_quantized(up)
            if not skip_allreduce and self.parallelism_config.get_ffn_tp_size() > 1:
                output = all_reduce(output, group=Group.TP)
            return output
        return self.forward_down(
            self.forward_activation(x, x_fp8=x_fp8, x_scale=x_scale),
            skip_allreduce=skip_allreduce,
        )
