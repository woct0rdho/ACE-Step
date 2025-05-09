"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import torch
from diffusers import AutoencoderDC
import torchaudio
import torchvision.transforms as transforms
from diffusers.models.modeling_utils import ModelMixin
from diffusers.loaders import FromOriginalModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config
from tqdm import tqdm

try:
    from .music_vocoder import ADaMoSHiFiGANV1
except ImportError:
    from music_vocoder import ADaMoSHiFiGANV1


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PRETRAINED_PATH = os.path.join(root_dir, "checkpoints", "music_dcae_f8c8")
VOCODER_PRETRAINED_PATH = os.path.join(root_dir, "checkpoints", "music_vocoder")


class MusicDCAE(ModelMixin, ConfigMixin, FromOriginalModelMixin):
    @register_to_config
    def __init__(
        self,
        source_sample_rate=None,
        dcae_checkpoint_path=DEFAULT_PRETRAINED_PATH,
        vocoder_checkpoint_path=VOCODER_PRETRAINED_PATH,
    ):
        super(MusicDCAE, self).__init__()

        self.dcae = AutoencoderDC.from_pretrained(dcae_checkpoint_path)
        self.vocoder = ADaMoSHiFiGANV1.from_pretrained(vocoder_checkpoint_path)

        if source_sample_rate is None:
            source_sample_rate = 48000

        self.resampler = torchaudio.transforms.Resample(source_sample_rate, 44100)

        self.transform = transforms.Compose(
            [
                transforms.Normalize(0.5, 0.5),
            ]
        )
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def load_audio(self, audio_path):
        audio, sr = torchaudio.load(audio_path)
        return audio, sr

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        # audios: N x 2 x T, 48kHz
        device = audios.device
        dtype = audios.dtype

        if sr is None:
            sr = 48000
            resampler = self.resampler
        else:
            resampler = torchaudio.transforms.Resample(sr, 44100).to(device).to(dtype)

        audio = resampler(audios)

        max_audio_len = audio.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audio = torch.nn.functional.pad(
                audio, (0, 8 * 512 - max_audio_len % (8 * 512))
            )

        mels = self.forward_mel(audio)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latent_lengths = (
            audio_lengths / sr * 44100 / 512 / self.time_dimention_multiple
        ).long()
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents, latent_lengths

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            mels = self.dcae.decoder(latent.unsqueeze(0))
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value

            # wav = self.vocoder.decode(mels[0]).squeeze(1)
            # decode waveform for each channels to reduce vram footprint
            wav_ch1 = self.vocoder.decode(mels[:,0,:,:]).squeeze(1).cpu()
            wav_ch2 = self.vocoder.decode(mels[:,1,:,:]).squeeze(1).cpu()
            wav = torch.cat([wav_ch1, wav_ch2],dim=0)

            if sr is not None:
                resampler = (
                    torchaudio.transforms.Resample(44100, sr)
                )
                wav = resampler(wav.cpu().float())
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [
                wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)
            ]
        return sr, pred_wavs
    
    
    @torch.no_grad()
    def decode_overlap(self, latents, audio_lengths=None, sr=None):
        print("Using Overlapped DCAE and Vocoder ")
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            latent = latent.unsqueeze(0)

            dcae_win_len = 512
            dcae_mel_win_len = dcae_win_len * 8
            latent_len = latent.shape[3]
            
            mels = []
            for start in tqdm(range(dcae_win_len//4,latent_len-dcae_win_len//4,dcae_win_len//2),desc="DCAE Decoding"):
                latent_win = latent[:,:,:,start-dcae_win_len//4:start-dcae_win_len//4+dcae_win_len]
                mel_win = self.dcae.decoder(latent_win)
                if start == dcae_win_len//4:
                    mel_win = mel_win[:,:,:,:-dcae_mel_win_len//4]
                elif start+dcae_win_len//2>latent_len:
                    mel_win = mel_win[:,:,:,dcae_mel_win_len//4:]
                else:
                    mel_win = mel_win[:,:,:,dcae_mel_win_len//4:-dcae_mel_win_len//4]
                mels.append(mel_win)
            mels = torch.cat(mels,dim=3)

            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            
            # wav_ch1 = self.vocoder.decode(mels[:,0,:,:]).squeeze(1)
            # wav_ch2 = self.vocoder.decode(mels[:,1,:,:]).squeeze(1)
            # wav = torch.cat([wav_ch1, wav_ch2],dim=0)

            vocoder_win_len = 512 * 512
            vocoder_overlap_len = 1024
            vocoder_hop_len = vocoder_win_len - 2*vocoder_overlap_len
            mel_len = mels.shape[3]
            mel_hop_len = 512
                    
            crossfade_len = 128
            crossfade_win_tail = torch.linspace(1, 0, crossfade_len).unsqueeze(0).unsqueeze(1)
            crossfade_win_head = torch.linspace(0, 1, crossfade_len).unsqueeze(0).unsqueeze(1)

            with tqdm(total=int(1+mel_len*mel_hop_len/vocoder_hop_len),desc="Vocoder Decoding") as pbar:
                wav = self.vocoder.decode(mels[0, :, :, :vocoder_win_len//mel_hop_len]).cpu()
                wav = wav[:,:,:-vocoder_overlap_len]
                p = vocoder_hop_len
                pbar.update(1)

                while p < mel_len * mel_hop_len:
                    wav_win = self.vocoder.decode(mels[0, :, :, p//mel_hop_len:p//mel_hop_len+vocoder_win_len//mel_hop_len]).cpu()

                    wav[:,:,-crossfade_len:] = wav[:,:,-crossfade_len:] * crossfade_win_tail + wav_win[:,:,vocoder_overlap_len-crossfade_len:vocoder_overlap_len] * crossfade_win_head
                    if p + vocoder_hop_len < mel_len * mel_hop_len:
                        wav_win = wav_win[:,:,vocoder_overlap_len:-vocoder_overlap_len]
                    else:
                        wav_win = wav_win[:,:,vocoder_overlap_len:]
                    wav = torch.cat([wav,wav_win],axis=2)
                    p+=vocoder_hop_len
                    pbar.update(1)

            wav = wav.squeeze(1)


            if sr is not None:
                resampler = (
                    torchaudio.transforms.Resample(44100, sr)
                )
                wav = resampler(wav.cpu().float())
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [
                wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)
            ]
        return sr, pred_wavs

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(
            audios=audios, audio_lengths=audio_lengths, sr=sr
        )
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths


if __name__ == "__main__":

    audio, sr = torchaudio.load("test.wav")
    audio_lengths = torch.tensor([audio.shape[1]])
    audios = audio.unsqueeze(0)

    # test encode only
    model = MusicDCAE()
    # latents, latent_lengths = model.encode(audios, audio_lengths)
    # print("latents shape: ", latents.shape)
    # print("latent_lengths: ", latent_lengths)

    # test encode and decode
    sr, pred_wavs, latents, latent_lengths = model(audios, audio_lengths, sr)
    print("reconstructed wavs: ", pred_wavs[0].shape)
    print("latents shape: ", latents.shape)
    print("latent_lengths: ", latent_lengths)
    print("sr: ", sr)
    torchaudio.save("test_reconstructed.wav", pred_wavs[0], sr)
    print("test_reconstructed.wav")
