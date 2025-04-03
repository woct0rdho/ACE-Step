import json

import matplotlib
import torch
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.core import LightningModule
from torch.utils.data import DataLoader

# from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from models.transformer_sana_text2music_large_dcae_0319 import ACEFlowBaseModel
from loguru import logger
from transformers import AutoModel
from lyric_processor_v2 import LyricProcessor
from optimizers.cosine_wsd import configure_lr_scheduler
import traceback
import torchaudio
from transformers import Wav2Vec2FeatureExtractor
from music_dcae.music_dcae_pipeline import MusicDCAE
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from apg_guidance import apg_forward, MomentumBuffer
from tqdm import tqdm
import random
import os


matplotlib.use("Agg")
torch.backends.cudnn.benchmark = False
torch.set_float32_matmul_precision('high')

# Enable TF32 for faster training on Ampere GPUs,
# cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
# torch.backends.cuda.matmul.allow_tf32 = True


class Pipeline(LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        num_workers: int = 4,
        infer: bool = False,
        train: bool = True,
        T: int = 1000,
        minibatch_size: int = 32,
        batch_size: int = 1,
        snr_gamma: float = 0.5,
        prediction_type: str = "v_prediction",  # epsilon, sample, v_prediction
        beta_start: float = 0.0015,
        beta_end: float = 0.0195,
        noise_offset: float = 0.1,
        input_perturbation: float = 0.1,
        use_ema: bool = False,
        enable_xformers_memory_efficient_attention: bool = False,
        weight_decay: float = 1e-2,
        num_chunk: int = 2,
        beta_schedule: str = "scaled_linear",
        scheduler_type: str = "ddpm",
        every_plot_step: int = 2000,
        vocal_noise: float = 0,
        max_length: int = 6400,
        sample_size: int = None,
        target_orig: bool = True,
        csv_path: str = None,
        config_path: str = "./models/config_sana_text2music_dcae_0225_3.5B_simple.json",
        shift: float = 3.0,
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        timestep_densities_type: str = "logit_normal",
        ssl_coeff: float = 1.0,
        wav_max_seconds: float = 30.0,
        max_steps: int = -1,
        fix_cut_level: int = 3,
        ipa_max_length: int = 8192,
        text_max_length: int = 1024,
    ):
        super().__init__()

        self.save_hyperparameters()
        self.is_train = train
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.scheduler = self.get_scheduler()
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.transformers = ACEFlowBaseModel(**self.config)

        self.lyric_processor = LyricProcessor()
        self.lyric_processor.requires_grad_(False)

        if not infer and self.is_train:
            self.mert_model = AutoModel.from_pretrained("./checkpoints/MERT-v1-330M", trust_remote_code=True).eval()
            self.mert_model.requires_grad_(False)
            self.resampler_mert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=24000)
            self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained("./checkpoints/MERT-v1-330M", trust_remote_code=True)

            self.hubert_model = AutoModel.from_pretrained("checkpoints/mHuBERT-147", local_files_only=True).eval()
            self.hubert_model.requires_grad_(False)
            self.resampler_mhubert = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
            self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained("checkpoints/mHuBERT-147", local_files_only=True)

            self.ssl_coeff = ssl_coeff

            self.vae = MusicDCAE(encoder_only=False).eval()
            self.vae.requires_grad_(False)

            # self.mert_model = torch.compile(self.mert_model)
            # self.hubert_model = torch.compile(self.hubert_model)
            # self.vae = torch.compile(self.vae)
            # self.transformers = torch.compile(self.transformers)
        else:
            self.vae = MusicDCAE(encoder_only=False).eval()
            self.vae.requires_grad_(False)

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        # 输入为 N x 2 x T (48kHz)，转换为 N x T (24kHz)，单声道
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2  # 48kHz -> 24kHz

        # 对实际音频部分进行归一化
        means = torch.stack([mert_input_wavs_mono_24k[i, :actual_lengths_24k[i]].mean() for i in range(bsz)])
        vars = torch.stack([mert_input_wavs_mono_24k[i, :actual_lengths_24k[i]].var() for i in range(bsz)])
        mert_input_wavs_mono_24k = (mert_input_wavs_mono_24k - means.view(-1, 1)) / torch.sqrt(vars.view(-1, 1) + 1e-7)
        
        # MERT SSL 约束
        # 定义每个 chunk 的长度（5 秒的采样点数）
        chunk_size = 24000 * 5  # 5 秒，每秒 24000 个采样点
        total_length = mert_input_wavs_mono_24k.shape[1]

        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

        # 分块处理
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input_wavs_mono_24k[i]
            actual_length = actual_lengths_24k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))  # 不足部分用零填充
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # 堆叠所有块为 (total_chunks, chunk_size)
        all_chunks = torch.stack(all_chunks, dim=0)

        # 批量推理
        with torch.no_grad():
            # 输出形状: (total_chunks, seq_len, hidden_size)
            mert_ssl_hidden_states = self.mert_model(all_chunks).last_hidden_state

        # 计算每个块的特征数量
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # 裁剪每个块的隐藏状态
        chunk_hidden_states = [mert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))]

        # 按音频组织隐藏状态
        mert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[chunk_idx:chunk_idx + num_chunks_per_audio[i]]
            audio_hidden = torch.cat(audio_chunks, dim=0)  # 拼接同一音频的块
            mert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]

        return mert_ssl_hidden_states_list


    def infer_mhubert_ssl(self, target_wavs, wav_lengths):
        # Step 1: Preprocess audio
        # Input: N x 2 x T (48kHz, stereo) -> N x T (16kHz, mono)
        mhubert_input_wavs_mono_16k = self.resampler_mhubert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_16k = wav_lengths // 3  # Convert lengths from 48kHz to 16kHz

        # Step 2: Zero-mean unit-variance normalization (only on actual audio)
        means = torch.stack([mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].mean() 
                            for i in range(bsz)])
        vars = torch.stack([mhubert_input_wavs_mono_16k[i, :actual_lengths_16k[i]].var() 
                            for i in range(bsz)])
        mhubert_input_wavs_mono_16k = (mhubert_input_wavs_mono_16k - means.view(-1, 1)) / \
                                    torch.sqrt(vars.view(-1, 1) + 1e-7)
    
        # Step 3: Define chunk size for MHubert (30 seconds at 16kHz)
        chunk_size = 16000 * 30  # 30 seconds = 480,000 samples

        # Step 4: Split audio into chunks
        num_chunks_per_audio = (actual_lengths_16k + chunk_size - 1) // chunk_size  # Ceiling division
        all_chunks = []
        chunk_actual_lengths = []

        for i in range(bsz):
            audio = mhubert_input_wavs_mono_16k[i]
            actual_length = actual_lengths_16k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))  # Pad with zeros
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)
        
        # Step 5: Stack all chunks for batch inference
        all_chunks = torch.stack(all_chunks, dim=0)  # Shape: (total_chunks, chunk_size)

        # Step 6: Batch inference with MHubert model
        with torch.no_grad():
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks).last_hidden_state
            # Shape: (total_chunks, seq_len, hidden_size)

        # Step 7: Compute number of features per chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Step 8: Trim hidden states to remove padding effects
        chunk_hidden_states = [mhubert_ssl_hidden_states[i, :chunk_num_features[i], :] for i in range(len(all_chunks))]

        # Step 9: Reorganize hidden states by original audio
        mhubert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[chunk_idx:chunk_idx + num_chunks_per_audio[i]]
            audio_hidden = torch.cat(audio_chunks, dim=0)  # Concatenate chunks for this audio
            mhubert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states_list

    def preprocess(self, batch, train=True):
        target_wavs = batch["target_wavs"]
        wav_lengths = batch["wav_lengths"]

        dtype = target_wavs.dtype
        bs = target_wavs.shape[0]
        device = target_wavs.device

        # SSL约束
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None
        # is_long = target_wavs.shape[-1] >= 48000 * 150
        if train:
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)
                # mhubert_ssl_hidden_states = self.infer_mhubert_ssl(batch["vocal_wavs"], wav_lengths)
                mhubert_ssl_hidden_states = self.infer_mhubert_ssl(target_wavs, wav_lengths)

        # 1: text embedding
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.lyric_processor.get_text_embeddings(texts, device)
        encoder_text_hidden_states = encoder_text_hidden_states.to(dtype)

        target_latents, _ = self.vae.encode(target_wavs, wav_lengths)
        attention_mask = torch.ones(bs, target_latents.shape[-1], device=device, dtype=dtype)

        speaker_embds = batch["speaker_embs"].to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"]
        
        # pretrain stage 2 需要 cfg
        if train:
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device)
            ).long()
            # N x T x 768
            encoder_text_hidden_states = torch.where(full_cfg_condition_mask.unsqueeze(1).unsqueeze(1).bool(), encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states))
            
        #     full_cfg_condition_mask = torch.where(
        #         (torch.rand(size=(bs,), device=device) < 0.50),
        #         torch.zeros(size=(bs,), device=device),
        #         torch.ones(size=(bs,), device=device)
        #     ).long()
        #     # N x 512
        #     speaker_embds = torch.where(full_cfg_condition_mask.unsqueeze(1).bool(), speaker_embds, torch.zeros_like(speaker_embds))

            # 歌词
            full_cfg_condition_mask = torch.where(
                (torch.rand(size=(bs,), device=device) < 0.15),
                torch.zeros(size=(bs,), device=device),
                torch.ones(size=(bs,), device=device)
            ).long()
            lyric_token_ids = torch.where(full_cfg_condition_mask.unsqueeze(1).bool(), lyric_token_ids, torch.zeros_like(lyric_token_ids))
            lyric_mask = torch.where(full_cfg_condition_mask.unsqueeze(1).bool(), lyric_mask, torch.zeros_like(lyric_mask))

        return (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        )

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        # trainable_parameters = self.transformers.get_trainable_parameters()
        
        # optimizer = get_muon_optimizer(
        #     self.transformers.named_parameters(),
        #     lr=self.hparams.learning_rate,
        #     wd=self.hparams.weight_decay,
        # )
        # optimizer = CAME8BitWrapper(
        #     params=[
        #         {'params': self.transformers.parameters()},
        #     ],
        #     lr=self.hparams.learning_rate,
        #     weight_decay=self.hparams.weight_decay,
        #     betas=(0.8, 0.9),
        # )
        optimizer = torch.optim.AdamW(
            params=[
                {'params': self.transformers.parameters()},
            ],
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        # 训练200k
        decay_interval = int(max_steps * (1 - 0.9) * 0.2)
        lr_scheduler = configure_lr_scheduler(optimizer, total_steps_per_epoch=max_steps, epochs=1, decay_ratio=0.9, decay_interval=decay_interval, warmup_iters=4000)
        return [optimizer], lr_scheduler

    def get_sd3_sigmas(self, timesteps, device, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def get_timestep(self, bsz, device):
        if self.hparams.timestep_densities_type == "logit_normal":
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            # In practice, we sample the random variable u from a normal distribution u ∼ N (u; m, s)
            # and map it through the standard logistic function
            u = torch.normal(mean=self.hparams.logit_mean, std=self.hparams.logit_std, size=(bsz, ), device="cpu")
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.scheduler.config.num_train_timesteps - 1)
            timesteps = self.scheduler.timesteps[indices].to(device)

        if self.hparams.timestep_densities_type == "u_shape":
            # 参数 a 决定 U-shaped 程度，论文中 a=4 效果较好
            a = 4.0
            # 从均匀分布采样 v
            v = torch.rand(bsz)

            # 计算 u：使用上述解析式
            # u = 0.5 + (1/a)*asinh( sinh(a/2)*(2*v -1) )
            s = torch.sinh(torch.tensor(a/2))
            argument = s * (2 * v - 1)
            u = 0.5 + (1.0 / a) * torch.asinh(argument)

            # 数值上可能有极小偏差，保险起见 clamp 一下
            u = torch.clamp(u, 0.0, 1.0)

            # 将连续 [0,1] 的 u 映射到具体的离散 timesteps
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(indices, 0, self.scheduler.config.num_train_timesteps - 1)
            timesteps = self.scheduler.timesteps[indices].to(device)

        return timesteps

    def run_step(self, batch, batch_idx):
        self.plot_step(batch, batch_idx)
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch)

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype
        # check dtype
        # logger.info(f"target_image dtype: {target_image.dtype} model dtype: {self.transformers.dtype}")
        # step 1: 随机生成噪声，初始化设置
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype)
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # clap ssl 约束 和vocal_latent_channel2的约束
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # step 5: predict noise
        transformer_output = self.transformers(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device).to(dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. 只有chunk_mask为1，且无padding的地方才计算loss
        # N x T x 64
        # chunk_masks_to_cat
        # N x T -> N x c x W x T
        mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(-1, target_image.shape[1], target_image.shape[2], -1)

        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(f"{prefix}/denoising_loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True)
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        loss = loss + total_proj_loss * self.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        learning_rate = self.lr_schedulers().get_last_lr()[0]
        self.log(f"{prefix}/learning_rate", learning_rate, on_step=True, on_epoch=False, prog_bar=True)
        # with torch.autograd.detect_anomaly():
        #     self.manual_backward(loss)
        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    @torch.no_grad()
    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generators=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
    ):

        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

        frame_length = int(duration * 44100 / 512 / 8)
        timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None)

        target_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=random_generators, device=device, dtype=dtype)
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            encoder_text_hidden_states = torch.cat([encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)], 0)
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)

            speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)], 0)

            lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0)
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()

        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])
            noise_pred = self.transformers(
                hidden_states=latent_model_input,
                attention_mask=attention_mask,
                encoder_text_hidden_states=encoder_text_hidden_states,
                text_attention_mask=text_attention_mask,
                speaker_embeds=speaker_embds,
                lyric_token_idx=lyric_token_ids,
                lyric_mask=lyric_mask,
                timestep=timestep,
            ).sample

            if do_classifier_free_guidance:
                noise_pred_with_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = apg_forward(
                    pred_cond=noise_pred_with_cond,
                    pred_uncond=noise_pred_uncond,
                    guidance_scale=guidance_scale,
                    momentum_buffer=momentum_buffer,
                )
            
            target_latents = scheduler.step(model_output=noise_pred, timestep=t, sample=target_latents, return_dict=False, omega=omega_scale)[0]
        
        return target_latents
    
    def predict_step(self, batch, batch_idx):
        (
            keys,
            target_latents,
            attention_mask,
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
            mert_ssl_hidden_states,
            mhubert_ssl_hidden_states,
        ) = self.preprocess(batch, train=False)

        infer_steps = 60
        guidance_scale = 15.0
        omega_scale = 10.0
        seed_num = 1234
        random.seed(seed_num)
        bsz = target_latents.shape[0]
        random_generators = [torch.Generator(device=self.device) for _ in range(bsz)]
        seeds = []
        for i in range(bsz):
            seed = random.randint(0, 2**32 - 1)
            random_generators[i].manual_seed(seed)
            seeds.append(seed)
        duration = self.hparams.fix_cut_level * 10
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generators=random_generators,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
        )

        audio_lengths = batch["wav_lengths"]
        sr, pred_wavs = self.vae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)
        return {
            "target_wavs": batch["target_wavs"],
            "pred_wavs": pred_wavs,
            "keys": keys,
            "prompts": batch["prompts"],
            "candidate_lyric_chunks": batch["candidate_lyric_chunks"],
            "sr": sr,
            "seeds": seeds,
        }

    def construct_lyrics(self, candidate_lyric_chunk):
        lyrics = []
        for chunk in candidate_lyric_chunk:
            lyrics.append(chunk["lyric"])

        lyrics = "\n".join(lyrics)
        return lyrics

    def plot_step(self, batch, batch_idx):

        if batch_idx % self.hparams.every_plot_step != 0 or self.local_rank != 0 or torch.distributed.get_rank() != 0 or torch.cuda.current_device() != 0:
            return
        results = self.predict_step(batch, batch_idx)

        target_wavs = results["target_wavs"]
        pred_wavs = results["pred_wavs"]
        keys = results["keys"]
        prompts = results["prompts"]
        candidate_lyric_chunks = results["candidate_lyric_chunks"]
        sr = results["sr"]
        seeds = results["seeds"]
        i = 0
        for key, target_wav, pred_wav, prompt, candidate_lyric_chunk, seed in zip(keys, target_wavs, pred_wavs, prompts, candidate_lyric_chunks, seeds):
            key = key
            prompt = prompt
            lyric = self.construct_lyrics(candidate_lyric_chunk)
            key_prompt_lyric = f"# KEY\n\n{key}\n\n\n# PROMPT\n\n{prompt}\n\n\n# LYRIC\n\n{lyric}\n\n# SEED\n\n{seed}\n\n"
            log_dir = self.logger.log_dir
            save_dir = f"{log_dir}/eval_results/step_{self.global_step}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            torchaudio.save(f"{save_dir}/target_wav_{key}_{i}.flac", target_wav.float().cpu(), sr)
            torchaudio.save(f"{save_dir}/pred_wav_{key}_{i}.flac", pred_wav.float().cpu(), sr)
            with open(f"{save_dir}/key_prompt_lyric_{key}_{i}.txt", "w") as f:
                f.write(key_prompt_lyric)
            i += 1
