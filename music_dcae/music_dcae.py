import torch
import torch.nn as nn
from diffusers import AutoencoderDC
import json


DEFAULT_CONFIG_PATH = "/root/sag_train/music_dcae/config_f32c32_large.json"

class MusicDCAE(nn.Module):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH):
        super(MusicDCAE, self).__init__()
        with open(config_path) as f:
            config = json.load(f)
        self.dcae = AutoencoderDC(**config)

    def encode(self, x):
        return self.dcae.encode(x).latent

    def decode(self, latent):
        sample = self.dcae.decode(latent).sample
        return sample

    def forward(self, x):
        sample = self.dcae(x).sample
        return sample
    
    def return_middle_layers(self):
        last_down_block = self.dcae.encoder.down_blocks[-1]
        encoder_conv_out = self.dcae.encoder.conv_out
        decoder_conv_in = self.dcae.decoder.conv_in
        decoder_up_blocks = self.dcae.decoder.up_blocks[0]
        middle_layers = [last_down_block, encoder_conv_out, decoder_conv_in, decoder_up_blocks]
        return middle_layers
    
    def return_head_layers(self):
        decoder_up_blocks = self.dcae.decoder.up_blocks[-1]
        conv_out = self.dcae.decoder.conv_out
        head_layers = [decoder_up_blocks, conv_out]
        return head_layers


if __name__ == "__main__":
    model = MusicDCAE("/root/sag_train/music_dcae/config_f8c8_large.json")

    x = torch.randn(1, 2, 128, 1024)
    # mask = None
    # if mask is None:
    #     mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3]).to(x.device)
    # # N x 1024
    # elif len(mask.shape) == 2:
    #     mask = mask.unsqueeze(1).unsqueeze(1).float()
    #     mask = mask.repeat(1, 1, x.shape[2], 1)
    latent = model.encode(x)
    print("latent shape: ", latent.shape)
    y = model(x)
    print("y", y.shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params / 1e6:.2f}M")

    # middle_layers = model.return_middle_layers()
    # middle_params_count = 0
    # for layer in middle_layers:
    #     for name, param in layer.named_parameters():
    #         layer_param_count = param.numel()
    #         middle_params_count += layer_param_count
    #         print(f"{name}: {param.shape}, 参数量: {layer_param_count/1e6:.2f}M")
    
    # print(f"中间层总参数量: {middle_params_count/1e6:.2f}M")

    # head_layers = model.return_head_layers()
    # head_params_count = 0
    # for layer in head_layers:
    #     for name, param in layer.named_parameters():
    #         layer_param_count = param.numel()
    #         head_params_count += layer_param_count
    #         print(f"{name}: {param.shape}, 参数量: {layer_param_count/1e6:.2f}M")
    
    # print(f"头部层总参数量: {head_params_count/1e6:.2f}M")