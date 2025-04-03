# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormContinuous, AdaLayerNormZero, RMSNorm, AdaLayerNormZeroSingle


try:
    # from .dcformer import DCMHAttention
    from .customer_attention_processor import Attention, CustomJointAttnProcessor2_0, CustomLiteLAProcessor2_0, CustomerAttnProcessor2_0, CustomLiteLAMMDiTProcessor2_0
except ImportError:
    # from dcformer import DCMHAttention
    from customer_attention_processor import Attention, CustomJointAttnProcessor2_0, CustomLiteLAProcessor2_0, CustomerAttnProcessor2_0, CustomLiteLAMMDiTProcessor2_0


logger = logging.get_logger(__name__)


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output


@maybe_allow_in_graph
class GatedSelfAttentionDense(nn.Module):
    r"""
    A gated self-attention dense layer that combines visual features and object features.

    Parameters:
        query_dim (`int`): The number of channels in the query.
        context_dim (`int`): The number of channels in the context.
        n_heads (`int`): The number of heads to use for attention.
        d_head (`int`): The number of channels in each head.
    """

    def __init__(self, query_dim: int, context_dim: int, n_heads: int, d_head: int):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = Attention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, activation_fn="geglu")

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter("alpha_attn", nn.Parameter(torch.tensor(0.0)))
        self.register_parameter("alpha_dense", nn.Parameter(torch.tensor(0.0)))

        self.enabled = True

    def forward(self, x: torch.Tensor, objs: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x

        n_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.alpha_attn.tanh() * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:, :n_visual, :]
        x = x + self.alpha_dense.tanh() * self.ff(self.norm2(x))

        return x


@maybe_allow_in_graph
class BasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_rms_norm: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_rms_norm = use_rms_norm

        # We keep these boolean flags for backward-compatibility.
        self.use_ada_layer_norm_zero = (num_embeds_ada_norm is not None) and norm_type == "ada_norm_zero"
        self.use_ada_layer_norm = (num_embeds_ada_norm is not None) and norm_type == "ada_norm"
        self.use_ada_layer_norm_single = norm_type == "ada_norm_single"
        self.use_layer_norm = norm_type == "layer_norm"
        self.use_ada_layer_norm_continuous = norm_type == "ada_norm_continuous"

        if norm_type in ("ada_norm", "ada_norm_zero") and num_embeds_ada_norm is None:
            raise ValueError(
                f"`norm_type` is set to {norm_type}, but `num_embeds_ada_norm` is not defined. Please make sure to"
                f" define `num_embeds_ada_norm` if setting `norm_type` to {norm_type}."
            )

        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        if positional_embeddings and (num_positional_embeddings is None):
            raise ValueError(
                "If `positional_embedding` type is defined, `num_positition_embeddings` must also be defined."
            )

        if positional_embeddings == "sinusoidal":
            self.pos_embed = SinusoidalPositionalEmbedding(dim, max_seq_length=num_positional_embeddings)
        else:
            self.pos_embed = None

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        if norm_type == "ada_norm":
            self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_zero":
            self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        elif norm_type == "ada_norm_continuous":
            self.norm1 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "rms_norm",
            )
        else:
            if use_rms_norm:
                self.norm1 = RMSNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            if norm_type == "ada_norm":
                self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm)
            elif norm_type == "ada_norm_continuous":
                self.norm2 = AdaLayerNormContinuous(
                    dim,
                    ada_norm_continous_conditioning_embedding_dim,
                    norm_elementwise_affine,
                    norm_eps,
                    ada_norm_bias,
                    "rms_norm",
                )
            else:
                if use_rms_norm:
                    self.norm2 = RMSNorm(dim, norm_eps, norm_elementwise_affine)
                else:
                    self.norm2 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        if norm_type == "ada_norm_continuous":
            self.norm3 = AdaLayerNormContinuous(
                dim,
                ada_norm_continous_conditioning_embedding_dim,
                norm_elementwise_affine,
                norm_eps,
                ada_norm_bias,
                "layer_norm",
            )

        elif norm_type in ["ada_norm_zero", "ada_norm", "layer_norm", "ada_norm_continuous"]:
            if use_rms_norm:
                self.norm3 = RMSNorm(dim, norm_eps, norm_elementwise_affine)
            else:
                self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        elif norm_type == "layer_norm_i2vgen":
            self.norm3 = None

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        # 4. Fuser
        if attention_type == "gated" or attention_type == "gated-text-image":
            self.fuser = GatedSelfAttentionDense(dim, cross_attention_dim, num_attention_heads, attention_head_dim)

        # 5. Scale-shift for PixArt-Alpha.
        if norm_type == "ada_norm_single":
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is depcrecated. `scale` will be ignored.")

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        if self.norm_type == "ada_norm":
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.norm_type == "ada_norm_zero":
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.norm_type in ["layer_norm", "layer_norm_i2vgen"]:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm1(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif self.norm_type == "ada_norm_single":
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
            ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
            norm_hidden_states = norm_hidden_states.squeeze(1)
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )
        if self.norm_type == "ada_norm_zero":
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.norm_type == "ada_norm_single":
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 1.2 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.norm_type == "ada_norm":
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.norm_type in ["ada_norm_zero", "layer_norm", "layer_norm_i2vgen"]:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.norm_type == "ada_norm_single":
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            elif self.norm_type == "ada_norm_continuous":
                norm_hidden_states = self.norm2(hidden_states, added_cond_kwargs["pooled_text_emb"])
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.norm_type != "ada_norm_single":
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        # i2vgen doesn't have this norm 🤷‍♂️
        if self.norm_type == "ada_norm_continuous":
            norm_hidden_states = self.norm3(hidden_states, added_cond_kwargs["pooled_text_emb"])
        elif not self.norm_type == "ada_norm_single":
            norm_hidden_states = self.norm3(hidden_states)

        if self.norm_type == "ada_norm_zero":
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.norm_type == "ada_norm_single" and self.norm2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.norm_type == "ada_norm_zero":
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.norm_type == "ada_norm_single":
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


@maybe_allow_in_graph
class TemporalBasicTransformerBlock(nn.Module):
    r"""
    A basic Transformer block for video like data.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        time_mix_inner_dim (`int`): The number of channels for temporal attention.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
    """

    def __init__(
        self,
        dim: int,
        time_mix_inner_dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        cross_attention_dim: Optional[int] = None,
    ):
        super().__init__()
        self.is_res = dim == time_mix_inner_dim

        self.norm_in = nn.LayerNorm(dim)

        # Define 3 blocks. Each block has its own normalization layer.
        # 1. Self-Attn
        self.ff_in = FeedForward(
            dim,
            dim_out=time_mix_inner_dim,
            activation_fn="geglu",
        )

        self.norm1 = nn.LayerNorm(time_mix_inner_dim)
        self.attn1 = Attention(
            query_dim=time_mix_inner_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            cross_attention_dim=None,
        )

        # 2. Cross-Attn
        if cross_attention_dim is not None:
            # We currently only use AdaLayerNormZero for self attention where there will only be one attention block.
            # I.e. the number of returned modulation chunks from AdaLayerZero would not make sense if returned during
            # the second cross attention block.
            self.norm2 = nn.LayerNorm(time_mix_inner_dim)
            self.attn2 = Attention(
                query_dim=time_mix_inner_dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
            )  # is self-attn if encoder_hidden_states is none
        else:
            self.norm2 = None
            self.attn2 = None

        # 3. Feed-forward
        self.norm3 = nn.LayerNorm(time_mix_inner_dim)
        self.ff = FeedForward(time_mix_inner_dim, activation_fn="geglu")

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = None

    def set_chunk_feed_forward(self, chunk_size: Optional[int], **kwargs):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        # chunk dim should be hardcoded to 1 to have better speed vs. memory trade-off
        self._chunk_dim = 1

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        num_frames: int,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        batch_size = hidden_states.shape[0]

        batch_frames, seq_length, channels = hidden_states.shape
        batch_size = batch_frames // num_frames

        hidden_states = hidden_states[None, :].reshape(batch_size, num_frames, seq_length, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * seq_length, num_frames, channels)

        residual = hidden_states
        hidden_states = self.norm_in(hidden_states)

        if self._chunk_size is not None:
            hidden_states = _chunked_feed_forward(self.ff_in, hidden_states, self._chunk_dim, self._chunk_size)
        else:
            hidden_states = self.ff_in(hidden_states)

        if self.is_res:
            hidden_states = hidden_states + residual

        norm_hidden_states = self.norm1(hidden_states)
        attn_output = self.attn1(norm_hidden_states, encoder_hidden_states=None)
        hidden_states = attn_output + hidden_states

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = self.norm2(hidden_states)
            attn_output = self.attn2(norm_hidden_states, encoder_hidden_states=encoder_hidden_states)
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = self.norm3(hidden_states)

        if self._chunk_size is not None:
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        if self.is_res:
            hidden_states = ff_output + hidden_states
        else:
            hidden_states = ff_output

        hidden_states = hidden_states[None, :].reshape(batch_size, seq_length, num_frames, channels)
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        hidden_states = hidden_states.reshape(batch_size * num_frames, seq_length, channels)

        return hidden_states


class SkipFFTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        kv_input_dim: int,
        kv_input_dim_proj_use_bias: bool,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        attention_out_bias: bool = True,
    ):
        super().__init__()
        if kv_input_dim != dim:
            self.kv_mapper = nn.Linear(kv_input_dim, dim, kv_input_dim_proj_use_bias)
        else:
            self.kv_mapper = None

        self.norm1 = RMSNorm(dim, 1e-06)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim,
            out_bias=attention_out_bias,
        )

        self.norm2 = RMSNorm(dim, 1e-06)

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            out_bias=attention_out_bias,
        )

    def forward(self, hidden_states, encoder_hidden_states, cross_attention_kwargs):
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        if self.kv_mapper is not None:
            encoder_hidden_states = self.kv_mapper(F.silu(encoder_hidden_states))

        norm_hidden_states = self.norm1(hidden_states)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        norm_hidden_states = self.norm2(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            **cross_attention_kwargs,
        )

        hidden_states = attn_output + hidden_states

        return hidden_states


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        linear_cls = nn.Linear

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(linear_cls(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AdaRMSNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(self, embedding_dim: int, num_embeddings: Optional[int] = None):
        super().__init__()
        self.emb = None
        self.silu = nn.SiLU()
        self.linear = nn.Linear(embedding_dim, 6 * embedding_dim, bias=True)
        self.norm = RMSNorm(embedding_dim, eps=1e-6, elementwise_affine=False)

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


@maybe_allow_in_graph
class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(self, dim, num_attention_heads, attention_head_dim, context_pre_only=False, use_mmdit=True, cross_attention_dim=None):
        super().__init__()

        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"

        self.norm1 = AdaRMSNormZero(dim)

        if context_norm_type == "ada_norm_continous" and use_mmdit:
            self.norm1_context = AdaLayerNormContinuous(
                dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="rms_norm"
            )
        elif context_norm_type == "ada_norm_zero" and use_mmdit:
            self.norm1_context = AdaRMSNormZero(dim)

        if hasattr(F, "scaled_dot_product_attention"):
            processor = CustomJointAttnProcessor2_0()
        else:
            raise ValueError(
                "The current PyTorch version does not support the `scaled_dot_product_attention` function."
            )
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=dim if use_mmdit else None,
            dim_head=attention_head_dim // num_attention_heads,
            heads=num_attention_heads,
            out_dim=attention_head_dim,
            context_pre_only=context_pre_only,
            bias=True,
            qk_norm="rms_norm",
            processor=processor,
        )

        self.norm2 = RMSNorm(dim, 1e-06, elementwise_affine=False)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = RMSNorm(dim, 1e-06, elementwise_affine=False)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        norm_encoder_hidden_states = None
        if encoder_hidden_states is not None:
            if self.context_pre_only:
                norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            else:
                norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                    encoder_hidden_states, emb=temb
                )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            rotary_freqs_cis=rotary_freqs_cis,
            rotary_freqs_cis_cross=rotary_freqs_cis_cross,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only or encoder_hidden_states is None:
            encoder_hidden_states = None
        if not self.context_pre_only and encoder_hidden_states is not None:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states


def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=None,
        act=None,
    ):
        super().__init__()
        if padding is None:
            padding = get_same_padding(kernel_size)
            padding *= dilation

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.padding = padding
        self.use_bias = use_bias

        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=use_bias,
        )
        if norm is not None:
            self.norm = RMSNorm(out_dim, elementwise_affine=False)
        else:
            self.norm = None
        if act is not None:
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_feature=None,
        kernel_size=3,
        stride=1,
        padding: Union[int, None] = None,
        use_bias=False,
        norm=(None, None, None),
        act=("silu", "silu", None),
        dilation=1,
    ):
        out_feature = out_feature or in_features
        super().__init__()
        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act = val2tuple(act, 3)

        self.glu_act = nn.SiLU(inplace=False)
        self.inverted_conv = ConvLayer(
            in_features,
            hidden_features * 2,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act=act[0],
        )
        self.depth_conv = ConvLayer(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size,
            stride=stride,
            groups=hidden_features * 2,
            padding=padding,
            use_bias=use_bias[1],
            norm=norm[1],
            act=None,
            dilation=dilation,
        )
        self.point_conv = ConvLayer(
            hidden_features,
            out_feature,
            1,
            use_bias=use_bias[2],
            norm=norm[2],
            act=act[2],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.inverted_conv(x)
        x = self.depth_conv(x)

        x, gate = torch.chunk(x, 2, dim=1)
        gate = self.glu_act(gate)
        x = x * gate

        x = self.point_conv(x)
        x = x.transpose(1, 2)

        return x


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class LinearTransformerBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        use_adaln_single=True,
        cross_attention_dim=None,
        added_kv_proj_dim=None,
        context_pre_only=False,
        mlp_ratio=4.0,
        add_cross_attention=False,
        add_cross_attention_dim=None,
        qk_norm=None,
    ):
        super().__init__()

        self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim,
            added_kv_proj_dim=added_kv_proj_dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm=qk_norm,
            processor=CustomLiteLAProcessor2_0(),
        )

        self.add_cross_attention = add_cross_attention
        self.context_pre_only = context_pre_only

        if add_cross_attention and add_cross_attention_dim is not None:
            self.cross_attn = Attention(
                query_dim=dim,
                cross_attention_dim=add_cross_attention_dim,
                added_kv_proj_dim=add_cross_attention_dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=context_pre_only,
                bias=True,
                qk_norm=qk_norm,
                processor=CustomerAttnProcessor2_0(),
            )

        self.norm2 = RMSNorm(dim, 1e-06, elementwise_affine=False)

        self.ff = GLUMBConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            use_bias=(True, True, False),
            norm=(None, None, None),
            act=("silu", "silu", None),
        )
        self.use_adaln_single = use_adaln_single
        if use_adaln_single:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: torch.FloatTensor = None,
        encoder_attention_mask: torch.FloatTensor = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        temb: torch.FloatTensor = None,
    ):

        N = hidden_states.shape[0]

        # step 1: AdaLN single
        if self.use_adaln_single:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

        norm_hidden_states = self.norm1(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

        # step 2: attention
        if not self.add_cross_attention:
            attn_output, encoder_hidden_states = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
        else:
            attn_output, _ = self.attn(
                hidden_states=norm_hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=None,
            )

        if self.use_adaln_single:
            attn_output = gate_msa * attn_output
        hidden_states = attn_output + hidden_states

        if self.add_cross_attention:
            attn_output = self.cross_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                rotary_freqs_cis=rotary_freqs_cis,
                rotary_freqs_cis_cross=rotary_freqs_cis_cross,
            )
            hidden_states = attn_output + hidden_states

        # step 3: add norm
        norm_hidden_states = self.norm2(hidden_states)
        if self.use_adaln_single:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        # step 4: feed forward
        ff_output = self.ff(norm_hidden_states)
        if self.use_adaln_single:
            ff_output = gate_mlp * ff_output

        hidden_states = hidden_states + ff_output

        return hidden_states
    

class LinearMMDiTBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """
    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        mlp_ratio=4.0,
        is_single=True,
    ):
        super().__init__()
        self.is_single = is_single

        if is_single:
            self.mlp_hidden_dim = int(dim * mlp_ratio)
            self.norm = AdaLayerNormZeroSingle(dim)
            self.proj_mlp = nn.Linear(dim, self.mlp_hidden_dim)
            self.act_mlp = nn.GELU(approximate="tanh")
            self.proj_out = nn.Linear(dim + self.mlp_hidden_dim, dim)
            processor = CustomLiteLAMMDiTProcessor2_0()
            self.attn = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                bias=True,
                processor=processor,
                qk_norm="rms_norm",
                eps=1e-6,
                pre_only=True,
            )
        else:
            self.scale_shift_table = nn.Parameter(torch.randn(6, dim) / dim**0.5)
            self.scale_shift_table_context = nn.Parameter(torch.randn(6, dim) / dim**0.5)
            self.norm1 = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
            self.norm1_context = RMSNorm(dim, elementwise_affine=False, eps=1e-6)
            processor = CustomLiteLAMMDiTProcessor2_0()
            self.attn = Attention(
                query_dim=dim,
                cross_attention_dim=None,
                added_kv_proj_dim=dim,
                dim_head=attention_head_dim,
                heads=num_attention_heads,
                out_dim=dim,
                context_pre_only=False,
                bias=True,
                processor=processor,
                qk_norm="rms_norm",
            )

            self.norm2 = RMSNorm(dim, 1e-06, elementwise_affine=False)
            self.norm2_context = RMSNorm(dim, 1e-06, elementwise_affine=False)
            self.ff = GLUMBConv(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=("silu", "silu", None),
            )
            self.ff_context = GLUMBConv(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                use_bias=(True, True, False),
                norm=(None, None, None),
                act=("silu", "silu", None),
            )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        temb: torch.FloatTensor = None,
        image_rotary_emb: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
    ):
        if self.is_single:
            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )
            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)
            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states
        else:
            N = hidden_states.shape[0]

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa

            shift_msa_context, scale_msa_context, gate_msa_context, shift_mlp_context, scale_mlp_context, gate_mlp_context = (
                self.scale_shift_table_context[None] + temb.reshape(N, 6, -1)
            ).chunk(6, dim=1)

            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_msa_context) + shift_msa_context

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
            )

            # Process attention outputs for the `hidden_states`.
            attn_output = gate_msa * attn_output
            hidden_states = attn_output + hidden_states

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp * ff_output
            hidden_states = hidden_states + ff_output

            # Process attention outputs for the `encoder_hidden_states`.
            context_attn_output = gate_msa_context * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + scale_mlp_context) + shift_mlp_context

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            context_ff_output = gate_mlp_context * context_ff_output
            encoder_hidden_states = encoder_hidden_states + context_ff_output 
            return encoder_hidden_states, hidden_states
