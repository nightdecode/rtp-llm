"""CPU-level tests for the glm5_next model desc: MHC layer plumbing.

The KDA/MLA/MoE compute modules are device-bound, so they are patched with
identity stubs; what is verified is the MHC residual-stream wiring of
Glm5NextDecoderLayer/Glm5NextModel (hc_pre -> norm -> attn -> hc_post ->
hc_pre -> norm -> ffn -> hc_post) against hand-computed hc_pre/hc_post.
"""

from types import SimpleNamespace  # noqa: F401
from unittest import TestCase, main
from unittest.mock import patch

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.model_desc import glm5_next
from rtp_llm.models_py.model_desc.glm5_next import Glm5NextDecoderLayer, Glm5NextModel
from rtp_llm.models_py.modules.base.common.norm import RMSNormTorch
from rtp_llm.ops import HybridAttentionType, MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


class _FakeModelWeights:
    """Duck-typed stand-in for ModelWeights used by construction tests."""

    def __init__(self, layer_weights, global_weights):
        self.weights = layer_weights
        self.global_weights = global_weights

    def get_global_weight(self, name):
        return self.global_weights[name]


class _IdentityAttn(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _IdentityMlp(nn.Module):
    def forward(self, hidden_states, *args, **kwargs):
        return hidden_states


def _make_config(hc_mult=4):
    config = ModelConfig()
    config.num_layers = 2
    config.hidden_size = 8
    config.vocab_size = 32
    config.max_seq_len = 64
    config.layernorm_eps = 1e-5
    config.activation_type = "SiGLU"
    config.attn_config.use_mla = True
    config.attn_config.head_num = 2
    config.hybrid_attention_config.enable_hybrid_attention = True
    config.hybrid_attention_config.hybrid_attention_types = [
        HybridAttentionType.LINEAR,
        HybridAttentionType.NONE,
    ]
    config.linear_attention_config.linear_key_head_dim = 4
    config.linear_attention_config.linear_value_head_dim = 4
    config.linear_attention_config.linear_num_key_heads = 2
    config.linear_attention_config.linear_num_value_heads = 2
    config.linear_attention_config.linear_conv_kernel_dim = 4
    config.linear_gate_lower_bound = -5.0
    # MoE on layer 1 only.
    config.expert_num = 4
    config.moe_k = 2
    config.moe_inter_size = 4
    config.n_shared_experts = 1
    config.inter_size = 4
    config.moe_style = 2
    config.moe_layer_index = [1]
    config.hc_mult = hc_mult
    config.hc_sinkhorn_iters = 20
    config.hc_eps = 1e-6
    return config


def _layer_weights(hidden=8, hc_mult=4):
    mix = (2 + hc_mult) * hc_mult
    hc_dim = hc_mult * hidden
    torch.manual_seed(0)
    return {
        W.pre_ln_gamma: torch.ones(hidden, dtype=torch.bfloat16),
        W.post_ln_gamma: torch.ones(hidden, dtype=torch.bfloat16),
        W.hc_attn_fn: torch.randn(mix, hc_dim, dtype=torch.float32) * 0.05,
        W.hc_attn_scale: torch.randn(3, dtype=torch.float32),
        W.hc_attn_base: torch.randn(mix, dtype=torch.float32) * 0.02,
        W.hc_ffn_fn: torch.randn(mix, hc_dim, dtype=torch.float32) * 0.02,
        W.hc_ffn_scale: torch.randn(3, dtype=torch.float32),
        W.hc_ffn_base: torch.randn(mix, dtype=torch.float32) * 0.02,
    }


def _expected_layer_streams(layer, streams):
    """Hand-computed MHC layer flow with identity attention/FFN."""
    n = layer.hc_mult
    hidden = streams.shape[-1] // n
    li, h_res, h_post = glm5_next.hc_pre(
        streams,
        layer.hc_attn_fn,
        layer.hc_attn_scale,
        layer.hc_attn_base,
        n,
        layer.rms_eps,
        layer.hc_eps,
        layer.sinkhorn_iters,
    )
    normed = layer.input_layernorm(li)
    attn_out = normed  # identity attention stub
    streams = glm5_next.hc_post(attn_out, streams, h_post, h_res, n)
    li, h_res, h_post = glm5_next.hc_pre(
        streams,
        layer.hc_ffn_fn,
        layer.hc_ffn_scale,
        layer.hc_ffn_base,
        n,
        layer.rms_eps,
        layer.hc_eps,
        layer.sinkhorn_iters,
    )
    normed = layer.post_attention_layernorm(li)
    ffn_out = normed  # identity ffn stub
    return glm5_next.hc_post(ffn_out, streams, h_post, h_res, n)


class Glm5NextDecoderLayerTest(TestCase):
    def _build_layer(self, layer_idx):
        config = _make_config()
        weights = _layer_weights()
        with (
            patch.object(
                glm5_next, "KimiLinearKDA", return_value=_IdentityAttn()
            ),
            patch.object(glm5_next, "MlaAttention", return_value=_IdentityAttn()),
            patch.object(glm5_next, "DenseMLP", return_value=_IdentityMlp()),
            patch.object(glm5_next, "GenericMoeLayer", return_value=_IdentityMlp()),
            patch.object(glm5_next, "Glm5MoeLayer", return_value=_IdentityMlp()),
            patch.object(glm5_next, "RMSNorm", RMSNormTorch),
        ):
            layer = Glm5NextDecoderLayer(
                config,
                ParallelismConfig(),
                weights,
                {},
                layer_idx,
                MoeConfig(),
            )
        return layer

    def test_kda_layer_mhc_flow(self):
        layer = self._build_layer(0)
        self.assertTrue(layer.use_mhc)
        streams = torch.randn(3, 4 * 8, dtype=torch.bfloat16)
        out = layer(streams)
        self.assertEqual(out.shape, streams.shape)
        expected = _expected_layer_streams(layer, streams)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)

    def test_mla_layer_mhc_flow(self):
        layer = self._build_layer(1)
        streams = torch.randn(5, 4 * 8, dtype=torch.bfloat16)
        out = layer(streams)
        self.assertEqual(out.shape, streams.shape)
        expected = _expected_layer_streams(layer, streams)
        torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)


class Glm5NextModelTest(TestCase):
    def test_model_constructs_and_contracts(self):
        config = _make_config()
        global_weights = {
            W.embedding: torch.randn(32, 8, dtype=torch.bfloat16),
            W.final_ln_gamma: torch.ones(8, dtype=torch.bfloat16),
        }
        per_layer = [_layer_weights(), _layer_weights()]
        weights = _FakeModelWeights(per_layer, global_weights)
        with (
            patch.object(glm5_next, "KimiLinearKDA", return_value=_IdentityAttn()),
            patch.object(glm5_next, "MlaAttention", return_value=_IdentityAttn()),
            patch.object(glm5_next, "DenseMLP", return_value=_IdentityMlp()),
            patch.object(glm5_next, "GenericMoeLayer", return_value=_IdentityMlp()),
            patch.object(glm5_next, "RMSNorm", RMSNormTorch),
        ):
            model = Glm5NextModel(
                config,
                ParallelismConfig(),
                weights,
                MoeConfig(),
                max_generate_batch_size=1,
            )
        self.assertEqual(len(model.layers), 2)
        self.assertEqual(model.hc_mult, 4)
        # embed -> hc_expand -> contract round trip on shapes
        ids = torch.tensor([1, 2, 3], dtype=torch.long)
        emb = model.embed_tokens.weight[ids]  # bypass device embedding op
        streams = glm5_next.hc_expand(emb.to(torch.bfloat16), model.hc_mult)
        self.assertEqual(streams.shape, (3, 4 * 8))
        back = glm5_next.hc_contract(streams, model.hc_mult)
        self.assertEqual(back.shape, (3, 8))
        torch.testing.assert_close(back, emb.to(torch.bfloat16))


if __name__ == "__main__":
    main()
