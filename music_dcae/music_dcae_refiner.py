from typing import Tuple, Union, Optional, Dict, Any
import torch
import torch.nn as nn
from diffusers.models.autoencoders.autoencoder_dc import DCUpBlock2d, get_block, RMSNorm, Decoder
from diffusers.models.transformers.sana_transformer import SanaTransformerBlock
from diffusers.models.embeddings import get_2d_sincos_pos_embed
from diffusers.models.normalization import AdaLayerNormSingle, RMSNorm
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention_processor import AttentionProcessor
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.utils import is_torch_version
from diffusers.models.unets import UNet2DModel


class Encoder(nn.Module):

    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 8,
        attention_head_dim: int = 32,
        block_out_channels: Tuple[int] = (512, 1024, 2048),
        layers_per_block: Tuple[int] = (3, 3, 3),
        block_type: str = "EfficientViTBlock",
        norm_type: str = "rms_norm",
        act_fn: str = "silu",
        qkv_multiscales: tuple = (5,),
    ):
        super(Encoder, self).__init__()

        num_blocks = len(block_out_channels)
        
        self.dump_encoder = False
        if num_blocks == 0:
            self.dump_encoder = True
            return

        self.conv_in = nn.Conv2d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, padding=1)

        up_blocks = []
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            up_block_list = []

            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=True,
                    shortcut=True,
                )
                up_block_list.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type,
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type,
                    act_fn=act_fn,
                    qkv_mutliscales=qkv_multiscales,
                )
                up_block_list.append(block)

            up_blocks.insert(0, nn.Sequential(*up_block_list))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.norm_out = RMSNorm(block_out_channels[0], 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.dump_encoder:
            return hidden_states

        hidden_states = self.conv_in(hidden_states)
        i = 0
        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)
            i += 1

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding with support for SD3 cropping.

    Args:
        height (`int`, defaults to `224`): The height of the image.
        width (`int`, defaults to `224`): The width of the image.
        patch_size (`int`, defaults to `16`): The size of the patches.
        in_channels (`int`, defaults to `3`): The number of input channels.
        embed_dim (`int`, defaults to `768`): The output dimension of the embedding.
        layer_norm (`bool`, defaults to `False`): Whether or not to use layer normalization.
        flatten (`bool`, defaults to `True`): Whether or not to flatten the output.
        bias (`bool`, defaults to `True`): Whether or not to use bias.
        interpolation_scale (`float`, defaults to `1`): The scale of the interpolation.
        pos_embed_type (`str`, defaults to `"sincos"`): The type of positional embedding.
        pos_embed_max_size (`int`, defaults to `None`): The maximum size of the positional embedding.
    """

    def __init__(
        self,
        height=16,
        width=128,
        patch_size=(16,1),
        in_channels=16,
        embed_dim=768,
        layer_norm=False,
        flatten=True,
        bias=True,
        interpolation_scale=1,
        pos_embed_type="sincos",
        pos_embed_max_size=None,  # For SD3 cropping
    ):
        super().__init__()

        num_patches = (height // patch_size[0]) * (width // patch_size[1])
        self.flatten = flatten
        self.layer_norm = layer_norm
        self.pos_embed_max_size = pos_embed_max_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias
        )
        if layer_norm:
            self.norm = nn.LayerNorm(embed_dim, elementwise_affine=False, eps=1e-6)
        else:
            self.norm = None

        self.patch_size = patch_size
        self.height, self.width = height // patch_size[0], width // patch_size[1]
        self.base_size = height // patch_size[1]
        self.interpolation_scale = interpolation_scale

        # Calculate positional embeddings based on max size or default
        if pos_embed_max_size:
            grid_size = pos_embed_max_size
        else:
            grid_size = int(num_patches**0.5)

        if pos_embed_type is None:
            self.pos_embed = None
        elif pos_embed_type == "sincos":
            pos_embed = get_2d_sincos_pos_embed(
                embed_dim,
                grid_size,
                base_size=self.base_size,
                interpolation_scale=self.interpolation_scale,
                output_type="pt",
            )
            persistent = True if pos_embed_max_size else False
            self.register_buffer("pos_embed", pos_embed.float().unsqueeze(0), persistent=persistent)
        else:
            raise ValueError(f"Unsupported pos_embed_type: {pos_embed_type}")

    def cropped_pos_embed(self, height, width):
        """Crops positional embeddings for SD3 compatibility."""
        if self.pos_embed_max_size is None:
            raise ValueError("`pos_embed_max_size` must be set for cropping.")

        height = height // self.patch_size
        width = width // self.patch_size
        if height > self.pos_embed_max_size:
            raise ValueError(
                f"Height ({height}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )
        if width > self.pos_embed_max_size:
            raise ValueError(
                f"Width ({width}) cannot be greater than `pos_embed_max_size`: {self.pos_embed_max_size}."
            )

        top = (self.pos_embed_max_size - height) // 2
        left = (self.pos_embed_max_size - width) // 2
        spatial_pos_embed = self.pos_embed.reshape(1, self.pos_embed_max_size, self.pos_embed_max_size, -1)
        spatial_pos_embed = spatial_pos_embed[:, top : top + height, left : left + width, :]
        spatial_pos_embed = spatial_pos_embed.reshape(1, -1, spatial_pos_embed.shape[-1])
        return spatial_pos_embed

    def forward(self, latent):
        if self.pos_embed_max_size is not None:
            height, width = latent.shape[-2:]
        else:
            height, width = latent.shape[-2] // self.patch_size[0], latent.shape[-1] // self.patch_size[1]
        latent = self.proj(latent)
        if self.flatten:
            latent = latent.flatten(2).transpose(1, 2)  # BCHW -> BNC
        if self.layer_norm:
            latent = self.norm(latent)
        if self.pos_embed is None:
            return latent.to(latent.dtype)
        # Interpolate or crop positional embeddings as needed
        if self.pos_embed_max_size:
            pos_embed = self.cropped_pos_embed(height, width)
        else:
            if self.height != height or self.width != width:
                pos_embed = get_2d_sincos_pos_embed(
                    embed_dim=self.pos_embed.shape[-1],
                    grid_size=(height, width),
                    base_size=self.base_size,
                    interpolation_scale=self.interpolation_scale,
                    device=latent.device,
                    output_type="pt",
                )
                pos_embed = pos_embed.float().unsqueeze(0)
            else:
                pos_embed = self.pos_embed

        return (latent + pos_embed).to(latent.dtype)


class DiTDecoder(ModelMixin, ConfigMixin):

    _supports_gradient_checkpointing = True
    @register_to_config

    def __init__(
        self,
        sample_size: Tuple[int, int] = (16, 128),
        in_channels: int = 16,
        out_channels: int = 8,
        patch_size: Tuple[int, int] = (16, 1),
        inner_dim: int = 1152,
        num_attention_heads: int = 36,
        attention_head_dim: int = 32,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        num_cross_attention_heads: Optional[int] = None,
        cross_attention_head_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: int = 1,
        mlp_ratio: float = 2.5,
        num_layers: int = 12,
    ):
        super(DiTDecoder, self).__init__()
        interpolation_scale = interpolation_scale if interpolation_scale is not None else max(sample_size // 64, 1)
        self.interpolation_scale = interpolation_scale

        self.patch_embed = PatchEmbed(
            height=sample_size[0],
            width=sample_size[1],
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.time_embed = AdaLayerNormSingle(inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                SanaTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    num_cross_attention_heads=num_cross_attention_heads,
                    cross_attention_head_dim=cross_attention_head_dim,
                    cross_attention_dim=cross_attention_dim,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    mlp_ratio=mlp_ratio,
                )
                for _ in range(num_layers)
            ]
        )

        self.scale_shift_table = nn.Parameter(torch.randn(2, inner_dim) / inner_dim ** 0.5)
        self.norm_out = nn.LayerNorm(inner_dim, eps=1e-6, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, patch_size[0] * patch_size[1] * out_channels)
        self.gradient_checkpointing = False

    def _set_gradient_checkpointing(self, module, value=False):
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = value

    @property
    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.attn_processors
    def attn_processors(self) -> Dict[str, AttentionProcessor]:
        r"""
        Returns:
            `dict` of attention processors: A dictionary containing all attention processors used in the model with
            indexed by its weight name.
        """
        # set recursively
        processors = {}

        def fn_recursive_add_processors(name: str, module: torch.nn.Module, processors: Dict[str, AttentionProcessor]):
            if hasattr(module, "get_processor"):
                processors[f"{name}.processor"] = module.get_processor()

            for sub_name, child in module.named_children():
                fn_recursive_add_processors(f"{name}.{sub_name}", child, processors)

            return processors

        for name, module in self.named_children():
            fn_recursive_add_processors(name, module, processors)

        return processors

    # Copied from diffusers.models.unets.unet_2d_condition.UNet2DConditionModel.set_attn_processor
    def set_attn_processor(self, processor: Union[AttentionProcessor, Dict[str, AttentionProcessor]]):
        r"""
        Sets the attention processor to use to compute attention.

        Parameters:
            processor (`dict` of `AttentionProcessor` or only `AttentionProcessor`):
                The instantiated processor class or a dictionary of processor classes that will be set as the processor
                for **all** `Attention` layers.

                If `processor` is a dict, the key needs to define the path to the corresponding cross attention
                processor. This is strongly recommended when setting trainable attention processors.

        """
        count = len(self.attn_processors.keys())

        if isinstance(processor, dict) and len(processor) != count:
            raise ValueError(
                f"A dict of processors was passed, but the number of processors {len(processor)} does not match the"
                f" number of attention layers: {count}. Please make sure to pass {count} processor classes."
            )

        def fn_recursive_attn_processor(name: str, module: torch.nn.Module, processor):
            if hasattr(module, "set_processor"):
                if not isinstance(processor, dict):
                    module.set_processor(processor)
                else:
                    module.set_processor(processor.pop(f"{name}.processor"))

            for sub_name, child in module.named_children():
                fn_recursive_attn_processor(f"{name}.{sub_name}", child, processor)

        for name, module in self.named_children():
            fn_recursive_attn_processor(name, module, processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[int] = None,
        return_dict: bool = True,
    ):
        
        # 1. Input
        batch_size, num_channels, height, width = hidden_states.shape
        patch_size = self.config.patch_size

        post_patch_height, post_patch_width = height // patch_size[0], width // patch_size[1]
        
        hidden_states = self.patch_embed(hidden_states)

        timestep, embedded_timestep = self.time_embed(
            timestep, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        # 2. Transformer blocks
        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module, return_dict=None):
                def custom_forward(*inputs):
                    if return_dict is not None:
                        return module(*inputs, return_dict=return_dict)
                    else:
                        return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}

            for block in self.transformer_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                    **ckpt_kwargs,
                )

        else:
            for block in self.transformer_blocks:
                hidden_states = block(
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    post_patch_height,
                    post_patch_width,
                )

        # 3. Normalization
        shift, scale = (
            self.scale_shift_table[None] + embedded_timestep[:, None].to(self.scale_shift_table.device)
        ).chunk(2, dim=1)

        # 4. Modulation
        hidden_states = hidden_states * (1 + scale) + shift
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        hidden_states = hidden_states.reshape(
            batch_size, post_patch_height, post_patch_width, self.config.patch_size[0], self.config.patch_size[1], -1
        )
        hidden_states = hidden_states.permute(0, 5, 1, 3, 2, 4)
        output = hidden_states.reshape(batch_size, -1, post_patch_height * patch_size[0], post_patch_width * patch_size[1])

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class MusicDcaeRefiner(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        in_channels: int = 32,
        attention_head_dim: int = 32,
        block_out_channels: Tuple[int] = (512, 1024, 2048),
        layers_per_block: Tuple[int] = (3, 3, 3),
        conv_block_out_channels: Tuple[int] = (224, 448, 672, 896),
        out_channels: int = 8,
        block_type: str = "EfficientViTBlock",
        norm_type: str = "rms_norm",
        act_fn: str = "silu",
        qkv_multiscales: tuple = (5,),
        sample_size: Tuple[int, int] = (16, 128),
        patch_size: Tuple[int, int] = (16, 1),
        inner_dim: int = 1152,
        num_attention_heads: int = 36,
        dropout: float = 0.0,
        cross_attention_dim: Optional[int] = None,
        num_cross_attention_heads: Optional[int] = None,
        cross_attention_head_dim: Optional[int] = None,
        attention_bias: bool = False,
        norm_elementwise_affine: bool = False,
        norm_eps: float = 1e-6,
        interpolation_scale: int = 1,
        mlp_ratio: float = 2.5,
        num_layers: int = 12,
        decoder_type: str = "ConvDecoder",

    ):
        super(MusicDcaeRefiner, self).__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=out_channels,
            attention_head_dim=attention_head_dim,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            block_type=block_type,
            norm_type=norm_type,
            act_fn=act_fn,
            qkv_multiscales=qkv_multiscales,
        )
        if decoder_type == "DiTDecoder":
            self.decoder = DiTDecoder(
                sample_size=sample_size,
                in_channels=out_channels * 2,
                out_channels=out_channels,
                patch_size=patch_size,
                inner_dim=inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                cross_attention_dim=cross_attention_dim,
                num_cross_attention_heads=num_cross_attention_heads,
                cross_attention_head_dim=cross_attention_head_dim,
                attention_bias=attention_bias,
                norm_elementwise_affine=norm_elementwise_affine,
                norm_eps=norm_eps,
                interpolation_scale=interpolation_scale,
                mlp_ratio=mlp_ratio,
                num_layers=num_layers,
            )
        else:
            self.decoder = UNet2DModel(
                sample_size=sample_size,
                in_channels=out_channels * 2,
                out_channels=out_channels,
                block_out_channels=conv_block_out_channels,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: Optional[int] = None,
        return_dict: bool = True
    ):
        encoder_hidden_states = self.encoder(encoder_hidden_states)
        hidden_states = torch.cat([hidden_states, encoder_hidden_states], dim=1)
        output = self.decoder(hidden_states, timestep=timestep, return_dict=return_dict)
        return output



if __name__ == "__main__":
    # f32c32 -> f8c8
    # model = MusicDcaeRefiner()

    # x = torch.randn(1, 8, 16, 128)
    # encoder_x = torch.randn(1, 32, 4, 32)
    # timestep = 0
    # y = model(x, encoder_x, timestep=timestep)
    # print("y", y.sample.shape)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"模型参数总数: {total_params / 1e6:.2f}M")

    # # 分别计算encoder和decoder的参数量
    # encoder_params_count = sum(p.numel() for p in model.encoder.parameters())
    # decoder_params_count = sum(p.numel() for p in model.decoder.parameters())
    # print(f"encoder参数量: {encoder_params_count/1e6:.2f}M")
    # print(f"decoder参数量: {decoder_params_count/1e6:.2f}M")


    # f8c8 -> mel
    import json
    with open("music_dcae/config_f8c8_to_mel_refiner.json", "r") as f:
        config = json.load(f)
    model = MusicDcaeRefiner(**config)

    x = torch.randn(1, 2, 128, 1024)
    encoder_x = torch.randn(1, 2, 128, 1024)
    timestep = 0
    y = model(x, encoder_x, timestep=timestep)
    print("y", y.sample.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params / 1e6:.2f}M")

    # 分别计算encoder和decoder的参数量
    encoder_params_count = sum(p.numel() for p in model.encoder.parameters())
    decoder_params_count = sum(p.numel() for p in model.decoder.parameters())
    print(f"encoder参数量: {encoder_params_count/1e6:.2f}M")
    print(f"decoder参数量: {decoder_params_count/1e6:.2f}M")
