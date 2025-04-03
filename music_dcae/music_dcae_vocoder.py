import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderDC
import json
import torchvision.transforms as transforms
import torchaudio

try:
    from .music_vocoder import ADaMoSHiFiGANV1
except ImportError:
    from music_vocoder import ADaMoSHiFiGANV1


DEFAULT_CONFIG_PATH = "/root/sag_train/music_dcae/config_f32c32_large.json"
DCAE_PRETRAINED_PATH = "/root/sag_train/checkpoints/music_dcae_f32c32"
VOCODER_PRETRAINED_PATH = "/root/sag_train/checkpoints/music_vocoder.pt"


class MusicDCAEVocoder(nn.Module):
    def __init__(self, config_path=DEFAULT_CONFIG_PATH, pretrained_path=DCAE_PRETRAINED_PATH):
        super(MusicDCAEVocoder, self).__init__()
        if pretrained_path is None:
            with open(config_path) as f:
                config = json.load(f)
            self.dcae = AutoencoderDC(**config)
        else:
            self.dcae = AutoencoderDC.from_pretrained(pretrained_path)
        self.vocoder = ADaMoSHiFiGANV1(VOCODER_PRETRAINED_PATH)
        self.freeze_vocoder()
        self.transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5),
        ])
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.target_sr = 44100

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        if audio.shape[0] == 1:
            audio = torch.cat([audio, audio], dim=0)
        return audio, sr

    def resample_audio(self, audio, sr=48000):
        resampler = torchaudio.transforms.Resample(sr, self.target_sr)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
        return audio

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels
    
    def norm_mel(self, mels):
        normed_mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        normed_mels = self.transform(normed_mels)
        return normed_mels
    
    def denorm_mel(self, normed_mels):
        mels = normed_mels * 0.5 + 0.5
        mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
        return mels

    def encode_latent(self, normed_mels):
        # N x 2 x 128 x W -> N x C x 128//F x W//F
        latent = self.dcae.encode(normed_mels).latent
        return latent
    
    def decode_mel(self, latent):
        # N x C x 128//F x W//F -> N x 2 x 128 x W
        normed_mels = self.dcae.decode(latent).sample
        return normed_mels
    
    def decode_audio(self, mels):
        # mels: N x 2 x 128 x W -> 2N x 128 x W
        bs = mels.shape[0]
        mono_mels = mels.reshape(-1, 128, mels.shape[-1])
        mono_audios = self.vocoder(mono_mels)
        audios = mono_audios.reshape(bs, 2, -1)
        return audios
    
    def encode(self, audios):
        mels = self.forward_mel(audios)
        normed_mels = self.norm_mel(mels)
        latent = self.encode_latent(normed_mels)
        return latent, mels

    def decode(self, latent):
        recon_normed_mels = self.decode_mel(latent)
        recon_mels = self.denorm_mel(recon_normed_mels)
        recon_audios = self.decode_audio(recon_mels)
        return recon_audios, recon_mels

    def forward(self, audios):
        audios_len = audios.shape[-1]
        latent, mels = self.encode(audios)
        recon_audios, recon_mels = self.decode(latent)
        if recon_audios.shape[-1] > audios_len:
            recon_audios = recon_audios[:, :, :audios_len]
        elif recon_audios.shape[-1] < audios_len:
            recon_audios = F.pad(recon_audios, (0, audios_len - recon_audios.shape[-1]))
        return recon_audios, mels, recon_mels, latent

    def freeze_vocoder(self):
        self.vocoder.eval()
        self.vocoder.requires_grad_(False)

    def unfreeze_vocoder(self):
        self.vocoder.train()
        self.vocoder.requires_grad_(True)

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
    model = MusicDCAEVocoder()

    audio_path = "/root/sag_train/orig2.wav"
    audio, sr = model.load_audio(audio_path)
    audio = model.resample_audio(audio, sr)

    model.eval()
    model = model.to("cuda:0")
    audio = audio.to("cuda:0")
    with torch.no_grad():
        audios_len = audio.shape[-1]
        min_frame = 512 * 32
        if audios_len % min_frame != 0:
            padding = torch.zeros(audio.shape[0], 2, min_frame - audios_len % min_frame).to(audios.device)
            audios = torch.cat([audio, padding], dim=-1)
        recon_audios, mels, recon_mels, latent = model(audio.unsqueeze(0))
        recon_audios = recon_audios[:, :, :audios_len]

    print("latent shape: ", latent.shape)
    print("recon_audios", recon_audios.shape)
    print("mels", mels.shape, "min:", mels.min(), "max:", mels.max(), "mean:", mels.mean(), "std:", mels.std())
    print("recon_mels", recon_mels.shape, "min:", recon_mels.min(), "max:", recon_mels.max(), "mean:", recon_mels.mean(), "std:", recon_mels.std())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params / 1e6:.2f}M")

    torchaudio.save("/root/sag_train/recon2.wav", recon_audios[0].cpu(), 44100)
