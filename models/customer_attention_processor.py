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
import inspect
from importlib import import_module
from typing import Callable, Optional, Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.utils import logging
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
import numbers
from diffusers.models.attention_processor import (
    AttentionProcessor,
    AttnProcessor,
    AttnProcessor2_0,
    AttnProcessorNPU,
    AttnAddedKVProcessor,
    AttnAddedKVProcessor2_0,
    SlicedAttnProcessor,
    SlicedAttnAddedKVProcessor,
    XFormersAttnProcessor,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
    LoRAXFormersAttnProcessor,
    CustomDiffusionAttnProcessor,
    CustomDiffusionAttnProcessor2_0,
    CustomDiffusionXFormersAttnProcessor,
    XFormersAttnAddedKVProcessor,
    LoRAAttnAddedKVProcessor,
    SpatialNorm,
    xformers
)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class RMSNorm(nn.Module):
    def __init__(self, dim, eps: float, elementwise_affine: bool = True):
        super().__init__()

        self.eps = eps

        if isinstance(dim, numbers.Integral):
            dim = (dim,)

        self.dim = torch.Size(dim)

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.weight = None

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


@maybe_allow_in_graph
class Attention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        only_cross_attention (`bool`, *optional*, defaults to `False`):
            Set to `True` to only use cross attention and not added_kv_proj_dim. Can only be set to `True` if
            `added_kv_proj_dim` is not `None`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
        processor (`AttnProcessor`, *optional*, defaults to `None`):
            The attention processor to use. If `None`, defaults to `AttnProcessor2_0` if `torch 2.x` is used and
            `AttnProcessor` otherwise.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: Optional[str] = None,
        cross_attention_norm_num_groups: int = 32,
        qk_norm: Optional[str] = None,  # [layer_norm, group_norm, rms_norm]
        added_kv_proj_dim: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        spatial_norm_dim: Optional[int] = None,
        out_bias: bool = True,
        scale_qk: bool = True,
        only_cross_attention: bool = False,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        processor: Optional["AttnProcessor"] = None,
        out_dim: int = None,
        context_pre_only=None,
    ):
        super().__init__()
        self.inner_dim = out_dim if out_dim is not None else dim_head * heads
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = cross_attention_dim if cross_attention_dim is not None else query_dim
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.context_pre_only = context_pre_only

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads
        # for slice_size > 0 the attention score computation
        # is split across the batch axis to save memory
        # You can set slice_size with `set_attention_slice`
        self.sliceable_head_dim = heads

        self.added_kv_proj_dim = added_kv_proj_dim
        self.only_cross_attention = only_cross_attention

        if self.added_kv_proj_dim is None and self.only_cross_attention:
            raise ValueError(
                "`only_cross_attention` can only be set to True if `added_kv_proj_dim` is not None. Make sure to set either `only_cross_attention=False` or define `added_kv_proj_dim`."
            )

        if norm_num_groups is not None:
            self.group_norm = nn.GroupNorm(num_channels=query_dim, num_groups=norm_num_groups, eps=eps, affine=True)
        else:
            self.group_norm = None

        if spatial_norm_dim is not None:
            self.spatial_norm = SpatialNorm(f_channels=query_dim, zq_channels=spatial_norm_dim)
        else:
            self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "layer_norm":
            self.norm_q = nn.LayerNorm(dim_head, eps=eps)
            self.norm_k = nn.LayerNorm(dim_head, eps=eps)
        elif qk_norm == "rms_norm":
            self.norm_q = RMSNorm(dim_head, eps=eps)
            self.norm_k = RMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None or 'layer_norm' or 'rsm_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        elif cross_attention_norm == "layer_norm":
            self.norm_cross = nn.LayerNorm(self.cross_attention_dim)
        elif cross_attention_norm == "group_norm":
            if self.added_kv_proj_dim is not None:
                # The given `encoder_hidden_states` are initially of shape
                # (batch_size, seq_len, added_kv_proj_dim) before being projected
                # to (batch_size, seq_len, cross_attention_dim). The norm is applied
                # before the projection, so we need to use `added_kv_proj_dim` as
                # the number of channels for the group norm.
                norm_cross_num_channels = added_kv_proj_dim
            else:
                norm_cross_num_channels = self.cross_attention_dim

            self.norm_cross = nn.GroupNorm(
                num_channels=norm_cross_num_channels, num_groups=cross_attention_norm_num_groups, eps=1e-5, affine=True
            )
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None, 'layer_norm' or 'group_norm'"
            )

        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=bias)

        if not self.only_cross_attention:
            # only relevant for the `AddedKVProcessor` classes
            self.to_k = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
            self.to_v = nn.Linear(self.cross_attention_dim, self.inner_dim, bias=bias)
        else:
            self.to_k = None
            self.to_v = None

        if self.added_kv_proj_dim is not None:
            self.add_k_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            self.add_v_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)
            if self.context_pre_only is not None:
                self.add_q_proj = nn.Linear(added_kv_proj_dim, self.inner_dim)

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, self.out_dim, bias=out_bias))
        self.to_out.append(nn.Dropout(dropout))

        if self.context_pre_only is not None and not self.context_pre_only and self.added_kv_proj_dim is not None:
            self.to_add_out = nn.Linear(self.inner_dim, self.out_dim, bias=out_bias)

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "fp32_layer_norm":
                self.norm_added_q = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
                self.norm_added_k = FP32LayerNorm(dim_head, elementwise_affine=False, bias=False, eps=eps)
            elif qk_norm == "rms_norm":
                self.norm_added_q = RMSNorm(dim_head, eps=eps)
                self.norm_added_k = RMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

        # set attention processor
        # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
        # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
        # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
        if processor is None:
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_npu_flash_attention(self, use_npu_flash_attention: bool) -> None:
        r"""
        Set whether to use npu flash attention from `torch_npu` or not.

        """
        if use_npu_flash_attention:
            processor = AttnProcessorNPU()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )
        self.set_processor(processor)

    def set_use_memory_efficient_attention_xformers(
        self, use_memory_efficient_attention_xformers: bool, attention_op: Optional[Callable] = None
    ) -> None:
        r"""
        Set whether to use memory efficient attention from `xformers` or not.

        Args:
            use_memory_efficient_attention_xformers (`bool`):
                Whether to use memory efficient attention from `xformers` or not.
            attention_op (`Callable`, *optional*):
                The attention operation to use. Defaults to `None` which uses the default attention operation from
                `xformers`.
        """
        # is_lora = hasattr(self, "processor") and isinstance(
        #     self.processor,
        #     LORA_ATTENTION_PROCESSORS,
        # )
        is_custom_diffusion = hasattr(self, "processor") and isinstance(
            self.processor,
            (CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0),
        )
        is_added_kv_processor = hasattr(self, "processor") and isinstance(
            self.processor,
            (
                AttnAddedKVProcessor,
                AttnAddedKVProcessor2_0,
                SlicedAttnAddedKVProcessor,
                XFormersAttnAddedKVProcessor,
                LoRAAttnAddedKVProcessor,
            ),
        )

        if use_memory_efficient_attention_xformers:
            if is_added_kv_processor and (is_lora or is_custom_diffusion):
                raise NotImplementedError(
                    f"Memory efficient attention is currently not supported for LoRA or custom diffusion for attention processor type {self.processor}"
                )
            if not is_xformers_available():
                raise ModuleNotFoundError(
                    (
                        "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                        " xformers"
                    ),
                    name="xformers",
                )
            elif not torch.cuda.is_available():
                raise ValueError(
                    "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is"
                    " only available for GPU "
                )
            else:
                try:
                    # Make sure we can run the memory efficient attention
                    _ = xformers.ops.memory_efficient_attention(
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                        torch.randn((1, 2, 40), device="cuda"),
                    )
                except Exception as e:
                    raise e

            if is_lora:
                # TODO (sayakpaul): should we throw a warning if someone wants to use the xformers
                # variant when using PT 2.0 now that we have LoRAAttnProcessor2_0?
                processor = LoRAXFormersAttnProcessor(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            elif is_custom_diffusion:
                processor = CustomDiffusionXFormersAttnProcessor(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    attention_op=attention_op,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            elif is_added_kv_processor:
                # TODO(Patrick, Suraj, William) - currently xformers doesn't work for UnCLIP
                # which uses this type of cross attention ONLY because the attention mask of format
                # [0, ..., -10.000, ..., 0, ...,] is not supported
                # throw warning
                logger.info(
                    "Memory efficient attention with `xformers` might currently not work correctly if an attention mask is required for the attention operation."
                )
                processor = XFormersAttnAddedKVProcessor(attention_op=attention_op)
            else:
                processor = XFormersAttnProcessor(attention_op=attention_op)
        else:
            if is_lora:
                attn_processor_class = (
                    LoRAAttnProcessor2_0 if hasattr(F, "scaled_dot_product_attention") else LoRAAttnProcessor
                )
                processor = attn_processor_class(
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                    rank=self.processor.rank,
                )
                processor.load_state_dict(self.processor.state_dict())
                processor.to(self.processor.to_q_lora.up.weight.device)
            elif is_custom_diffusion:
                attn_processor_class = (
                    CustomDiffusionAttnProcessor2_0
                    if hasattr(F, "scaled_dot_product_attention")
                    else CustomDiffusionAttnProcessor
                )
                processor = attn_processor_class(
                    train_kv=self.processor.train_kv,
                    train_q_out=self.processor.train_q_out,
                    hidden_size=self.processor.hidden_size,
                    cross_attention_dim=self.processor.cross_attention_dim,
                )
                processor.load_state_dict(self.processor.state_dict())
                if hasattr(self.processor, "to_k_custom_diffusion"):
                    processor.to(self.processor.to_k_custom_diffusion.weight.device)
            else:
                # set attention processor
                # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
                # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
                # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
                processor = (
                    AttnProcessor2_0()
                    if hasattr(F, "scaled_dot_product_attention") and self.scale_qk
                    else AttnProcessor()
                )

        self.set_processor(processor)

    def set_attention_slice(self, slice_size: int) -> None:
        r"""
        Set the slice size for attention computation.

        Args:
            slice_size (`int`):
                The slice size for attention computation.
        """
        if slice_size is not None and slice_size > self.sliceable_head_dim:
            raise ValueError(f"slice_size {slice_size} has to be smaller or equal to {self.sliceable_head_dim}.")

        if slice_size is not None and self.added_kv_proj_dim is not None:
            processor = SlicedAttnAddedKVProcessor(slice_size)
        elif slice_size is not None:
            processor = SlicedAttnProcessor(slice_size)
        elif self.added_kv_proj_dim is not None:
            processor = AttnAddedKVProcessor()
        else:
            # set attention processor
            # We use the AttnProcessor2_0 by default when torch 2.x is used which uses
            # torch.nn.functional.scaled_dot_product_attention for native Flash/memory_efficient_attention
            # but only if it has the default `scale` argument. TODO remove scale_qk check when we move to torch 2.1
            processor = (
                AttnProcessor2_0() if hasattr(F, "scaled_dot_product_attention") and self.scale_qk else AttnProcessor()
            )

        self.set_processor(processor)

    def set_processor(self, processor: "AttnProcessor") -> None:
        r"""
        Set the attention processor to use.

        Args:
            processor (`AttnProcessor`):
                The attention processor to use.
        """
        # if current processor is in `self._modules` and if passed `processor` is not, we need to
        # pop `processor` from `self._modules`
        if (
            hasattr(self, "processor")
            and isinstance(self.processor, torch.nn.Module)
            and not isinstance(processor, torch.nn.Module)
        ):
            logger.info(f"You are removing possibly trained weights of {self.processor} with {processor}")
            self._modules.pop("processor")

        self.processor = processor

    def get_processor(self, return_deprecated_lora: bool = False) -> "AttentionProcessor":
        r"""
        Get the attention processor in use.

        Args:
            return_deprecated_lora (`bool`, *optional*, defaults to `False`):
                Set to `True` to return the deprecated LoRA attention processor.

        Returns:
            "AttentionProcessor": The attention processor in use.
        """
        if not return_deprecated_lora:
            return self.processor

        # TODO(Sayak, Patrick). The rest of the function is needed to ensure backwards compatible
        # serialization format for LoRA Attention Processors. It should be deleted once the integration
        # with PEFT is completed.
        is_lora_activated = {
            name: module.lora_layer is not None
            for name, module in self.named_modules()
            if hasattr(module, "lora_layer")
        }

        # 1. if no layer has a LoRA activated we can return the processor as usual
        if not any(is_lora_activated.values()):
            return self.processor

        # If doesn't apply LoRA do `add_k_proj` or `add_v_proj`
        is_lora_activated.pop("add_k_proj", None)
        is_lora_activated.pop("add_v_proj", None)
        # 2. else it is not possible that only some layers have LoRA activated
        if not all(is_lora_activated.values()):
            raise ValueError(
                f"Make sure that either all layers or no layers have LoRA activated, but have {is_lora_activated}"
            )

        # 3. And we need to merge the current LoRA layers into the corresponding LoRA attention processor
        non_lora_processor_cls_name = self.processor.__class__.__name__
        lora_processor_cls = getattr(import_module(__name__), "LoRA" + non_lora_processor_cls_name)

        hidden_size = self.inner_dim

        # now create a LoRA attention processor from the LoRA layers
        if lora_processor_cls in [LoRAAttnProcessor, LoRAAttnProcessor2_0, LoRAXFormersAttnProcessor]:
            kwargs = {
                "cross_attention_dim": self.cross_attention_dim,
                "rank": self.to_q.lora_layer.rank,
                "network_alpha": self.to_q.lora_layer.network_alpha,
                "q_rank": self.to_q.lora_layer.rank,
                "q_hidden_size": self.to_q.lora_layer.out_features,
                "k_rank": self.to_k.lora_layer.rank,
                "k_hidden_size": self.to_k.lora_layer.out_features,
                "v_rank": self.to_v.lora_layer.rank,
                "v_hidden_size": self.to_v.lora_layer.out_features,
                "out_rank": self.to_out[0].lora_layer.rank,
                "out_hidden_size": self.to_out[0].lora_layer.out_features,
            }

            if hasattr(self.processor, "attention_op"):
                kwargs["attention_op"] = self.processor.attention_op

            lora_processor = lora_processor_cls(hidden_size, **kwargs)
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(self.to_out[0].lora_layer.state_dict())
        elif lora_processor_cls == LoRAAttnAddedKVProcessor:
            lora_processor = lora_processor_cls(
                hidden_size,
                cross_attention_dim=self.add_k_proj.weight.shape[0],
                rank=self.to_q.lora_layer.rank,
                network_alpha=self.to_q.lora_layer.network_alpha,
            )
            lora_processor.to_q_lora.load_state_dict(self.to_q.lora_layer.state_dict())
            lora_processor.to_k_lora.load_state_dict(self.to_k.lora_layer.state_dict())
            lora_processor.to_v_lora.load_state_dict(self.to_v.lora_layer.state_dict())
            lora_processor.to_out_lora.load_state_dict(self.to_out[0].lora_layer.state_dict())

            # only save if used
            if self.add_k_proj.lora_layer is not None:
                lora_processor.add_k_proj_lora.load_state_dict(self.add_k_proj.lora_layer.state_dict())
                lora_processor.add_v_proj_lora.load_state_dict(self.add_v_proj.lora_layer.state_dict())
            else:
                lora_processor.add_k_proj_lora = None
                lora_processor.add_v_proj_lora = None
        else:
            raise ValueError(f"{lora_processor_cls} does not exist.")

        return lora_processor

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        # The `Attention` class can call different attention processors / attention functions
        # here we simply pass along all tensors to the selected processor class
        # For standard processors that are defined here, `**cross_attention_kwargs` is empty

        attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
        quiet_attn_parameters = {"ip_adapter_masks"}
        unused_kwargs = [
            k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
        ]
        if len(unused_kwargs) > 0:
            logger.warning(
                f"cross_attention_kwargs {unused_kwargs} are not expected by {self.processor.__class__.__name__} and will be ignored."
            )
        cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}

        return self.processor(
            self,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            **cross_attention_kwargs,
        )

    def batch_to_head_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size // heads, seq_len, dim * heads]`. `heads`
        is the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        batch_size, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def head_to_batch_dim(self, tensor: torch.Tensor, out_dim: int = 3) -> torch.Tensor:
        r"""
        Reshape the tensor from `[batch_size, seq_len, dim]` to `[batch_size, seq_len, heads, dim // heads]` `heads` is
        the number of heads initialized while constructing the `Attention` class.

        Args:
            tensor (`torch.Tensor`): The tensor to reshape.
            out_dim (`int`, *optional*, defaults to `3`): The output dimension of the tensor. If `3`, the tensor is
                reshaped to `[batch_size * heads, seq_len, dim // heads]`.

        Returns:
            `torch.Tensor`: The reshaped tensor.
        """
        head_size = self.heads
        if tensor.ndim == 3:
            batch_size, seq_len, dim = tensor.shape
            extra_dim = 1
        else:
            batch_size, extra_dim, seq_len, dim = tensor.shape
        tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3)

        if out_dim == 3:
            tensor = tensor.reshape(batch_size * head_size, seq_len * extra_dim, dim // head_size)

        return tensor

    def get_attention_scores(
        self, query: torch.Tensor, key: torch.Tensor, attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        r"""
        Compute the attention scores.

        Args:
            query (`torch.Tensor`): The query tensor.
            key (`torch.Tensor`): The key tensor.
            attention_mask (`torch.Tensor`, *optional*): The attention mask to use. If `None`, no mask is applied.

        Returns:
            `torch.Tensor`: The attention probabilities/scores.
        """
        dtype = query.dtype
        if self.upcast_attention:
            query = query.float()
            key = key.float()

        if attention_mask is None:
            baddbmm_input = torch.empty(
                query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device
            )
            beta = 0
        else:
            baddbmm_input = attention_mask
            beta = 1

        attention_scores = torch.baddbmm(
            baddbmm_input,
            query,
            key.transpose(-1, -2),
            beta=beta,
            alpha=self.scale,
        )
        del baddbmm_input

        if self.upcast_softmax:
            attention_scores = attention_scores.float()

        attention_probs = attention_scores.softmax(dim=-1)
        del attention_scores

        attention_probs = attention_probs.to(dtype)

        return attention_probs

    def prepare_attention_mask(
        self, attention_mask: torch.Tensor, target_length: int, batch_size: int, out_dim: int = 3
    ) -> torch.Tensor:
        r"""
        Prepare the attention mask for the attention computation.

        Args:
            attention_mask (`torch.Tensor`):
                The attention mask to prepare.
            target_length (`int`):
                The target length of the attention mask. This is the length of the attention mask after padding.
            batch_size (`int`):
                The batch size, which is used to repeat the attention mask.
            out_dim (`int`, *optional*, defaults to `3`):
                The output dimension of the attention mask. Can be either `3` or `4`.

        Returns:
            `torch.Tensor`: The prepared attention mask.
        """
        head_size = self.heads
        if attention_mask is None:
            return attention_mask

        current_length: int = attention_mask.shape[-1]
        if current_length != target_length:
            if attention_mask.device.type == "mps":
                # HACK: MPS: Does not support padding by greater than dimension of input tensor.
                # Instead, we can manually construct the padding tensor.
                padding_shape = (attention_mask.shape[0], attention_mask.shape[1], target_length)
                padding = torch.zeros(padding_shape, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([attention_mask, padding], dim=2)
            else:
                # TODO: for pipelines such as stable-diffusion, padding cross-attn mask:
                #       we want to instead pad by (0, remaining_length), where remaining_length is:
                #       remaining_length: int = target_length - current_length
                # TODO: re-enable tests/models/test_models_unet_2d_condition.py#test_model_xattn_padding
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)

        if out_dim == 3:
            if attention_mask.shape[0] < batch_size * head_size:
                attention_mask = attention_mask.repeat_interleave(head_size, dim=0)
        elif out_dim == 4:
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.repeat_interleave(head_size, dim=1)

        return attention_mask

    def norm_encoder_hidden_states(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        r"""
        Normalize the encoder hidden states. Requires `self.norm_cross` to be specified when constructing the
        `Attention` class.

        Args:
            encoder_hidden_states (`torch.Tensor`): Hidden states of the encoder.

        Returns:
            `torch.Tensor`: The normalized encoder hidden states.
        """
        assert self.norm_cross is not None, "self.norm_cross must be defined to call self.norm_encoder_hidden_states"

        if isinstance(self.norm_cross, nn.LayerNorm):
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
        elif isinstance(self.norm_cross, nn.GroupNorm):
            # Group norm norms along the channels dimension and expects
            # input to be in the shape of (N, C, *). In this case, we want
            # to norm along the hidden dimension, so we need to move
            # (batch_size, sequence_length, hidden_size) ->
            # (batch_size, hidden_size, sequence_length)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
            encoder_hidden_states = self.norm_cross(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.transpose(1, 2)
        else:
            assert False

        return encoder_hidden_states

    @torch.no_grad()
    def fuse_projections(self, fuse=True):
        device = self.to_q.weight.data.device
        dtype = self.to_q.weight.data.dtype

        if not self.is_cross_attention:
            # fetch weight matrices.
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            # create a new single projection layer and copy over the weights.
            self.to_qkv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_qkv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
                self.to_qkv.bias.copy_(concatenated_bias)

        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            in_features = concatenated_weights.shape[1]
            out_features = concatenated_weights.shape[0]

            self.to_kv = nn.Linear(in_features, out_features, bias=self.use_bias, device=device, dtype=dtype)
            self.to_kv.weight.copy_(concatenated_weights)
            if self.use_bias:
                concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
                self.to_kv.bias.copy_(concatenated_bias)

        self.fused_projections = fuse


class CustomJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None and not attn.is_cross_attention:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None and not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states


class CustomLiteLAProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states_len = hidden_states.shape[1]

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        if encoder_hidden_states is not None:
            context_input_ndim = encoder_hidden_states.ndim
            if context_input_ndim == 4:
                batch_size, channel, height, width = encoder_hidden_states.shape
                encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # `context` projections.
        has_encoder_hidden_state_proj = hasattr(attn, "add_q_proj") and hasattr(attn, "add_k_proj") and hasattr(attn, "add_v_proj")
        if encoder_hidden_states is not None and has_encoder_hidden_state_proj:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # attention
            if not attn.is_cross_attention:
                query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
                key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
                value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)
            else:
                query = hidden_states
                key = encoder_hidden_states
                value = encoder_hidden_states

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)
        key = key.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1).transpose(-1, -2)
        value = value.transpose(-1, -2).reshape(batch_size, attn.heads, head_dim, -1)

        # RoPE需要 [B, H, S, D] 输入
        # 此时 query是 [B, H, D, S], 需要转成 [B, H, S, D] 才能应用RoPE
        query = query.permute(0, 1, 3, 2)  # [B, H, S, D]  (从 [B, H, D, S])

        # Apply query and key normalization if needed
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        # 此时 query是 [B, H, S, D]，需要还原成 [B, H, D, S]
        query = query.permute(0, 1, 3, 2)  # [B, H, D, S]

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, S, 1]
            attention_mask = attention_mask[:, None, :, None].to(key.dtype)  # [B, 1, S, 1]
            query = query * attention_mask.permute(0, 1, 3, 2)  # [B, H, S, D] * [B, 1, S, 1]
            if not attn.is_cross_attention:
                key = key * attention_mask  # key: [B, h, S, D] 与 mask [B, 1, S, 1] 相乘
                value = value * attention_mask.permute(0, 1, 3, 2)  # 如果 value 是 [B, h, D, S]，那么需调整mask以匹配S维度

        if attn.is_cross_attention and encoder_attention_mask is not None and has_encoder_hidden_state_proj:
            encoder_attention_mask = encoder_attention_mask[:, None, :, None].to(key.dtype)  # [B, 1, S_enc, 1]
            # 此时 key: [B, h, S_enc, D], value: [B, h, D, S_enc]
            key = key * encoder_attention_mask  # [B, h, S_enc, D] * [B, 1, S_enc, 1]
            value = value * encoder_attention_mask.permute(0, 1, 3, 2)  # [B, h, D, S_enc] * [B, 1, 1, S_enc]

        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        vk = torch.matmul(value, key)

        hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        hidden_states = hidden_states.view(batch_size, attn.heads * head_dim, -1).permute(0, 2, 1)

        hidden_states = hidden_states.to(dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(dtype)

        # Split the attention outputs.
        if encoder_hidden_states is not None and not attn.is_cross_attention and has_encoder_hidden_state_proj:
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : hidden_states_len],
                hidden_states[:, hidden_states_len:],
            )

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if encoder_hidden_states is not None and not attn.context_pre_only and not attn.is_cross_attention and hasattr(attn, "to_add_out"):
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if encoder_hidden_states is not None and context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return hidden_states, encoder_hidden_states



class CustomerAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        rotary_freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        rotary_freqs_cis_cross: Union[torch.Tensor, Tuple[torch.Tensor]] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:

        residual = hidden_states
        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        
        has_encoder_hidden_state_proj = hasattr(attn, "add_q_proj") and hasattr(attn, "add_k_proj") and hasattr(attn, "add_v_proj")

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Apply RoPE if needed
        if rotary_freqs_cis is not None:
            query = self.apply_rotary_emb(query, rotary_freqs_cis)
            if not attn.is_cross_attention:
                key = self.apply_rotary_emb(key, rotary_freqs_cis)
            elif rotary_freqs_cis_cross is not None and has_encoder_hidden_state_proj:
                key = self.apply_rotary_emb(key, rotary_freqs_cis_cross)

        if attn.is_cross_attention and encoder_attention_mask is not None and has_encoder_hidden_state_proj:
            # attention_mask: N x S1
            # encoder_attention_mask: N x S2
            # cross attention 整合attention_mask和encoder_attention_mask
            combined_mask = attention_mask[:, :, None] * encoder_attention_mask[:, None, :]
            attention_mask = torch.where(combined_mask == 1, 0.0, -torch.inf)
            attention_mask = attention_mask[:, None, :, :].expand(-1, attn.heads, -1, -1).to(query.dtype)

        elif not attn.is_cross_attention and attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class CustomLiteLAMMDiTProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections. add rms norm for query and key and apply RoPE"""

    def __init__(self):
        self.kernel_func = nn.ReLU(inplace=False)
        self.eps = 1e-15
        self.pad_val = 1.0

    def apply_linear_attention(self, query, key, value):
        key = key.permute(0, 1, 3, 2)
        query = self.kernel_func(query)
        key = self.kernel_func(key)

        query, key, value = query.float(), key.float(), value.float()

        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=self.pad_val)

        vk = torch.matmul(value, key)

        hidden_states = torch.matmul(vk, query)

        if hidden_states.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.float()

        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)

        return hidden_states

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
        to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
        reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
        tensors contain rotary embeddings and are returned as real tensors.

        Args:
            x (`torch.Tensor`):
                Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
            freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
        """
        cos, sin = freqs_cis  # [S, D] or [N, S, D]
        if cos.ndim == 2:
            cos = cos[None, None]
            sin = sin[None, None]
        elif cos.ndim == 3:
            cos = cos[:, None]
            sin = sin[:, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:

        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        dtype = hidden_states.dtype
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            key = self.apply_rotary_emb(key, image_rotary_emb)

        # hidden_states = F.scaled_dot_product_attention(
        #     query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        # )
        # apply linear attention
        hidden_states = self.apply_linear_attention(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class FluxAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def apply_rotary_emb(
        self,
        x: torch.Tensor,
        freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]]
    ):
        cos, sin = freqs_cis  # [S, D] or [N, S, D]
        if cos.ndim == 2:
            cos = cos[None, None]
            sin = sin[None, None]
        elif cos.ndim == 3:
            cos = cos[:, None]
            sin = sin[:, None]
        cos, sin = cos.to(x.device), sin.to(x.device)
        # Used for flux, cogvideox, hunyuan-dit
        x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
        x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            query = self.apply_rotary_emb(query, image_rotary_emb)
            key = self.apply_rotary_emb(key, image_rotary_emb)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            # dropout
            hidden_states = attn.to_out[1](hidden_states)

            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            return hidden_states

