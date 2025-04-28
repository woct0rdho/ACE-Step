import random
import time
import os
import re

import torch
import torch.nn as nn
from loguru import logger
from tqdm import tqdm
import json
import math

# from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from transformers import UMT5EncoderModel, AutoTokenizer

from hf_download import download_repo

from language_segmentation import LangSegment
from music_dcae.music_dcae_pipeline import MusicDCAE
from models.fusic_transformer import FusicTransformer2DModel
from models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from apg_guidance import apg_forward, MomentumBuffer, cfg_forward, cfg_zero_star, cfg_double_condition_forward
import torchaudio


SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, 
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, 
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}

structure_pattern = re.compile(r"\[.*?\]")



def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)


REPO_ID = "timedomain/fusic_v1"


# class FusicPipeline(DiffusionPipeline):
class FusicPipeline:

    def __init__(self, checkpoint_dir=None, device_id=0, dtype="bfloat16", text_encoder_checkpoint_path=None, **kwargs):
        # check checkpoint dir exist
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
            if not os.path.exists(checkpoint_dir):
                # huggingface download
                download_repo(
                    repo_id=REPO_ID,
                    save_path=checkpoint_dir
                )

        self.checkpoint_dir = checkpoint_dir
        device = torch.device(f"cuda:{device_id}") if torch.cuda.is_available() else torch.device("cpu")
        self.dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        self.device = device
        self.loaded = False

    def load_checkpoint(self, checkpoint_dir=None):
        device = self.device
        dcae_checkpoint_path = os.path.join(checkpoint_dir, "music_dcae_f8c8")
        vocoder_checkpoint_path = os.path.join(checkpoint_dir, "music_vocoder")
        self.music_dcae = MusicDCAE(dcae_checkpoint_path=dcae_checkpoint_path, vocoder_checkpoint_path=vocoder_checkpoint_path)
        self.music_dcae.to(device).eval().to(self.dtype)

        fusic_checkpoint_path = os.path.join(checkpoint_dir, "fusic_transformer")
        self.fusic_transformer = FusicTransformer2DModel.from_pretrained(fusic_checkpoint_path)
        self.fusic_transformer.to(device).eval().to(self.dtype)

        lang_segment = LangSegment()

        lang_segment.setfilters([
            'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
            'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
            'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
            'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
            'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
            'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
        ])
        self.lang_segment = lang_segment
        self.lyric_tokenizer = VoiceBpeTokenizer()
        text_encoder_checkpoint_path = os.path.join(checkpoint_dir, "umt5-base")
        text_encoder_model = UMT5EncoderModel.from_pretrained(text_encoder_checkpoint_path).eval()
        text_encoder_model = text_encoder_model.to(device).to(self.dtype)
        text_encoder_model.requires_grad_(False)
        self.text_encoder_model = text_encoder_model
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_encoder_checkpoint_path)
        self.loaded = True

        # compile
        self.music_dcae = torch.compile(self.music_dcae)
        self.fusic_transformer = torch.compile(self.fusic_transformer)
        self.text_encoder_model = torch.compile(self.text_encoder_model)

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask
    
    def get_text_embeddings_null(self, texts, device, text_max_length=256, tau=0.01, l_min=8, l_max=10):
        inputs = self.text_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=text_max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        if self.text_encoder_model.device != device:
            self.text_encoder_model.to(device)
        
        def forward_with_temperature(inputs, tau=0.01, l_min=8, l_max=10):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
        
            for i in range(l_min, l_max):
                handler = self.text_encoder_model.encoder.block[i].layer[0].SelfAttention.q.register_forward_hook(hook)
                handlers.append(handler)
        
            with torch.no_grad():
                outputs = self.text_encoder_model(**inputs)
                last_hidden_states = outputs.last_hidden_state
        
            for hook in handlers:
                hook.remove()
        
            return last_hidden_states
    
        last_hidden_states = forward_with_temperature(inputs, tau, l_min, l_max)
        return last_hidden_states

    def set_seeds(self, batch_size, manual_seeds=None):
        seeds = None
        if manual_seeds is not None:
            if isinstance(manual_seeds, str):
                if "," in manual_seeds:
                    seeds = list(map(int, manual_seeds.split(",")))
                elif manual_seeds.isdigit():
                    seeds = int(manual_seeds)

        random_generators = [torch.Generator(device=self.device) for _ in range(batch_size)]
        actual_seeds = []
        for i in range(batch_size):
            seed = None
            if seeds is None:
                seed = torch.randint(0, 2**32, (1,)).item()
            if isinstance(seeds, int):
                seed = seeds
            if isinstance(seeds, list):
                seed = seeds[i]
            random_generators[i].manual_seed(seed)
            actual_seeds.append(seed)
        return random_generators, actual_seeds

    def get_lang(self, text):
        language = "en"
        try:    
            _ = self.lang_segment.getTexts(text)
            langCounts = self.lang_segment.getCounts()
            language = langCounts[0][0]
            if len(langCounts) > 1 and language == "en":
                language = langCounts[1][0]
        except Exception as err:
            language = "en"
        return language

    def tokenize_lyrics(self, lyrics, debug=False):
        lines = lyrics.split("\n")
        lyric_token_idx = [261]
        for line in lines:
            line = line.strip()
            if not line:
                lyric_token_idx += [2]
                continue

            lang = self.get_lang(line)

            if lang not in SUPPORT_LANGUAGES:
                lang = "en"
            if "zh" in lang:
                lang = "zh"
            if "spa" in lang:
                lang = "es"

            try:
                if structure_pattern.match(line):
                    token_idx = self.lyric_tokenizer.encode(line, "en")
                else:
                    token_idx = self.lyric_tokenizer.encode(line, lang)
                if debug:
                    toks = self.lyric_tokenizer.batch_decode([[tok_id] for tok_id in token_idx])
                    logger.info(f"debbug {line} --> {lang} --> {toks}")
                lyric_token_idx = lyric_token_idx + token_idx + [2]
            except Exception as e:
                print("tokenize error", e, "for line", line, "major_language", lang)
        return lyric_token_idx


    @torch.no_grad()
    def text2music_diffusion_process(
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
        scheduler_type="euler",
        cfg_type="apg",
        zero_steps=1,
        use_zero_init=True,
        guidance_interval=0.5,
        guidance_interval_decay=1.0,
        min_guidance_scale=3.0,
        oss_steps=[],
        encoder_text_hidden_states_null=None,
        use_erg_lyric=False,
        use_erg_diffusion=False,
        retake_random_generators=None,
        retake_variance=0.5,
        add_retake_noise=False,
        guidance_scale_text=0.0,
        guidance_scale_lyric=0.0,
    ):

        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(cfg_type, guidance_scale, omega_scale))
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False
        
        do_double_condition_guidance = False
        if guidance_scale_text is not None and guidance_scale_text > 1.0 and guidance_scale_lyric is not None and guidance_scale_lyric > 1.0:
            do_double_condition_guidance = True
            logger.info("do_double_condition_guidance: {}, guidance_scale_text: {}, guidance_scale_lyric: {}".format(do_double_condition_guidance, guidance_scale_text, guidance_scale_lyric))

        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        if scheduler_type == "euler":
            scheduler = FlowMatchEulerDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        elif scheduler_type == "heun":
            scheduler = FlowMatchHeunDiscreteScheduler(
                num_train_timesteps=1000,
                shift=3.0,
            )
        frame_length = int(duration * 44100 / 512 / 8)

        if len(oss_steps) > 0:
            infer_steps = max(oss_steps)
            scheduler.set_timesteps
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None)
            new_timesteps = torch.zeros(len(oss_steps), dtype=dtype, device=device)
            for idx in range(len(oss_steps)):
                new_timesteps[idx] = timesteps[oss_steps[idx]-1]
            num_inference_steps = len(oss_steps)
            sigmas = (new_timesteps / 1000).float().cpu().numpy()
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=num_inference_steps, device=device, sigmas=sigmas)
            logger.info(f"oss_steps: {oss_steps}, num_inference_steps: {num_inference_steps} after remapping to timesteps {timesteps}")
        else:
            timesteps, num_inference_steps = retrieve_timesteps(scheduler, num_inference_steps=infer_steps, device=device, timesteps=None)
        
        target_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=random_generators, device=device, dtype=dtype)
        if add_retake_noise:
            retake_variance = torch.tensor(retake_variance * math.pi/2).to(device).to(dtype)
            retake_latents = randn_tensor(shape=(bsz, 8, 16, frame_length), generator=retake_random_generators, device=device, dtype=dtype)
            # to make sure mean = 0, std = 1
            target_latents = torch.cos(retake_variance) * target_latents + torch.sin(retake_variance) * retake_latents
        
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)
        
        # guidance interval逻辑
        start_idx = int(num_inference_steps * ((1 - guidance_interval) / 2))
        end_idx = int(num_inference_steps * (guidance_interval / 2 + 0.5))
        logger.info(f"start_idx: {start_idx}, end_idx: {end_idx}, num_inference_steps: {num_inference_steps}")

        momentum_buffer = MomentumBuffer()

        def forward_encoder_with_temperature(self, inputs, tau=0.01, l_min=4, l_max=6):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
            
            for i in range(l_min, l_max):
                handler = self.fusic_transformer.lyric_encoder.encoders[i].self_attn.linear_q.register_forward_hook(hook)
                handlers.append(handler)
        
            encoder_hidden_states, encoder_hidden_mask = self.fusic_transformer.encode(**inputs)
            
            for hook in handlers:
                hook.remove()
            
            return encoder_hidden_states

        # P(speaker, text, lyric)
        encoder_hidden_states, encoder_hidden_mask = self.fusic_transformer.encode(
            encoder_text_hidden_states,
            text_attention_mask,
            speaker_embds,
            lyric_token_ids,
            lyric_mask,
        )

        if use_erg_lyric:
            # P(null_speaker, text_weaker, lyric_weaker)
            encoder_hidden_states_null = forward_encoder_with_temperature(
                self,
                inputs={
                    "encoder_text_hidden_states": encoder_text_hidden_states_null if encoder_text_hidden_states_null is not None else torch.zeros_like(encoder_text_hidden_states),
                    "text_attention_mask": text_attention_mask,
                    "speaker_embeds": torch.zeros_like(speaker_embds),
                    "lyric_token_idx": lyric_token_ids,
                    "lyric_mask": lyric_mask,
                }
            )
        else:
            # P(null_speaker, null_text, null_lyric)
            encoder_hidden_states_null, _ = self.fusic_transformer.encode(
                torch.zeros_like(encoder_text_hidden_states),
                text_attention_mask,
                torch.zeros_like(speaker_embds),
                torch.zeros_like(lyric_token_ids),
                lyric_mask,
            )
        
        encoder_hidden_states_no_lyric = None
        if do_double_condition_guidance:
            # P(null_speaker, text, lyric_weaker)
            if use_erg_lyric:
                encoder_hidden_states_no_lyric = forward_encoder_with_temperature(
                    self,
                    inputs={
                        "encoder_text_hidden_states": encoder_text_hidden_states,
                        "text_attention_mask": text_attention_mask,
                        "speaker_embeds": torch.zeros_like(speaker_embds),
                        "lyric_token_idx": lyric_token_ids,
                        "lyric_mask": lyric_mask,
                    }
                )
            # P(null_speaker, text, no_lyric)
            else:
                encoder_hidden_states_no_lyric, _ = self.fusic_transformer.encode(
                    encoder_text_hidden_states,
                    text_attention_mask,
                    torch.zeros_like(speaker_embds),
                    torch.zeros_like(lyric_token_ids),
                    lyric_mask,
                )

        def forward_diffusion_with_temperature(self, hidden_states, timestep, inputs, tau=0.01, l_min=15, l_max=20):
            handlers = []
            
            def hook(module, input, output):
                output[:] *= tau
                return output
            
            for i in range(l_min, l_max):
                handler = self.fusic_transformer.transformer_blocks[i].attn.to_q.register_forward_hook(hook)
                handlers.append(handler)
                handler = self.fusic_transformer.transformer_blocks[i].cross_attn.to_q.register_forward_hook(hook)
                handlers.append(handler)

            sample = self.fusic_transformer.decode(hidden_states=hidden_states, timestep=timestep, **inputs).sample
            
            for hook in handlers:
                hook.remove()
            
            return sample

    
        for i, t in tqdm(enumerate(timesteps), total=num_inference_steps):
            # expand the latents if we are doing classifier free guidance
            latents = target_latents

            is_in_guidance_interval = start_idx <= i < end_idx
            if is_in_guidance_interval and do_classifier_free_guidance:
                # compute current guidance scale
                if guidance_interval_decay > 0:
                    # Linearly interpolate to calculate the current guidance scale
                    progress = (i - start_idx) / (end_idx - start_idx - 1)  # 归一化到[0,1]
                    current_guidance_scale = guidance_scale - (guidance_scale - min_guidance_scale) * progress * guidance_interval_decay
                else:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                output_length = latent_model_input.shape[-1]
                # P(x|speaker, text, lyric)
                noise_pred_with_cond = self.fusic_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=output_length,
                    timestep=timestep,
                ).sample

                noise_pred_with_only_text_cond = None
                if do_double_condition_guidance and encoder_hidden_states_no_lyric is not None:
                    noise_pred_with_only_text_cond = self.fusic_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_no_lyric,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if use_erg_diffusion:
                    noise_pred_uncond = forward_diffusion_with_temperature(
                        self,
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        inputs={
                            "encoder_hidden_states": encoder_hidden_states_null,
                            "encoder_hidden_mask": encoder_hidden_mask,
                            "output_length": output_length,
                            "attention_mask": attention_mask,
                        },
                    )
                else:
                    noise_pred_uncond = self.fusic_transformer.decode(
                        hidden_states=latent_model_input,
                        attention_mask=attention_mask,
                        encoder_hidden_states=encoder_hidden_states_null,
                        encoder_hidden_mask=encoder_hidden_mask,
                        output_length=output_length,
                        timestep=timestep,
                    ).sample

                if do_double_condition_guidance and noise_pred_with_only_text_cond is not None:
                    noise_pred = cfg_double_condition_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        only_text_cond_output=noise_pred_with_only_text_cond,
                        guidance_scale_text=guidance_scale_text,
                        guidance_scale_lyric=guidance_scale_lyric,
                    )

                elif cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                elif cfg_type == "cfg":
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=current_guidance_scale,
                    )
                elif cfg_type == "cfg_star":
                    noise_pred = cfg_zero_star(
                        noise_pred_with_cond=noise_pred_with_cond,
                        noise_pred_uncond=noise_pred_uncond,
                        guidance_scale=current_guidance_scale,
                        i=i,
                        zero_steps=zero_steps,
                        use_zero_init=use_zero_init
                    )
            else:
                latent_model_input = latents
                timestep = t.expand(latent_model_input.shape[0])
                noise_pred = self.fusic_transformer.decode(
                    hidden_states=latent_model_input,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_mask=encoder_hidden_mask,
                    output_length=latent_model_input.shape[-1],
                    timestep=timestep,
                ).sample

            target_latents = scheduler.step(model_output=noise_pred, timestep=t, sample=target_latents, return_dict=False, omega=omega_scale)[0]
        
        return target_latents

    def latents2audio(self, latents, target_wav_duration_second=30, sample_rate=48000, save_path=None, format="flac"):
        output_audio_paths = []
        bs = latents.shape[0]
        audio_lengths = [target_wav_duration_second * sample_rate] * bs
        pred_latents = latents
        with torch.no_grad():
            _, pred_wavs = self.music_dcae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs]
        for i in tqdm(range(bs)):
            output_audio_path = self.save_wav_file(pred_wavs[i], i, sample_rate=sample_rate)
            output_audio_paths.append(output_audio_path)
        return output_audio_paths

    def save_wav_file(self, target_wav, idx, save_path=None, sample_rate=48000, format="flac"):
        if save_path is None:
            logger.warning("save_path is None, using default path ./outputs/")
            base_path = f"./outputs/"
            ensure_directory_exists(base_path)
        else:
            base_path = save_path
            ensure_directory_exists(base_path)

        output_path_flac = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.{format}"
        target_wav = target_wav.float()
        torchaudio.save(output_path_flac, target_wav, sample_rate=sample_rate, format=format, backend="ffmpeg", compression=torchaudio.io.CodecConfig(bit_rate=320000))
        return output_path_flac

    def __call__(
        self,
        audio_duration: float = 60.0,
        prompt: str = None,
        lyrics: str = None,
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler",
        cfg_type: str = "apg",
        omega_scale: int = 10.0,
        manual_seeds: list = None,
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None,
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        retake_seeds: list = None,
        retake_variance: float = 0.5,
        task: str = "text2music",
        save_path: str = None,
        format: str = "flac",
        batch_size: int = 1,
    ):

        start_time = time.time()

        if not self.loaded:
            logger.warning("Checkpoint not loaded, loading checkpoint...")
            self.load_checkpoint(self.checkpoint_dir)
            load_model_cost = time.time() - start_time
            logger.info(f"Model loaded in {load_model_cost:.2f} seconds.")

        start_time = time.time()

        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)
        retake_random_generators, actual_retake_seeds = self.set_seeds(batch_size, retake_seeds)

        if isinstance(oss_steps, str) and len(oss_steps) > 0:
            oss_steps = list(map(int, oss_steps.split(",")))
        else:
            oss_steps = []
        
        texts = [prompt]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(texts, self.device)
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

        encoder_text_hidden_states_null = None
        if use_erg_tag:
            encoder_text_hidden_states_null = self.get_text_embeddings_null(texts, self.device)
            encoder_text_hidden_states_null = encoder_text_hidden_states_null.repeat(batch_size, 1, 1)

        # not support for released checkpoint
        speaker_embeds = torch.zeros(batch_size, 512).to(self.device).to(self.dtype)

        # 6 lyric
        lyric_token_idx = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        lyric_mask = torch.tensor([0]).repeat(batch_size, 1).to(self.device).long()
        if len(lyrics) > 0:
            lyric_token_idx = self.tokenize_lyrics(lyrics, debug=True)
            lyric_mask = [1] * len(lyric_token_idx)
            lyric_token_idx = torch.tensor(lyric_token_idx).unsqueeze(0).to(self.device).repeat(batch_size, 1)
            lyric_mask = torch.tensor(lyric_mask).unsqueeze(0).to(self.device).repeat(batch_size, 1)

        if audio_duration <= 0:
            audio_duration = random.uniform(30.0, 240.0)
            logger.info(f"random audio duration: {audio_duration}")

        end_time = time.time()
        preprocess_time_cost = end_time - start_time
        start_time = end_time

        target_latents = self.text2music_diffusion_process(
            duration=audio_duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embeds,
            lyric_token_ids=lyric_token_idx,
            lyric_mask=lyric_mask,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
            infer_steps=infer_step,
            random_generators=random_generators,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            oss_steps=oss_steps,
            encoder_text_hidden_states_null=encoder_text_hidden_states_null,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            retake_random_generators=retake_random_generators,
            retake_variance=retake_variance,
            add_retake_noise=task == "retake",
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
        )

        end_time = time.time()
        diffusion_time_cost = end_time - start_time
        start_time = end_time

        output_paths = self.latents2audio(
            latents=target_latents,
            target_wav_duration_second=audio_duration,
            save_path=save_path,
            format=format,
        )

        end_time = time.time()
        latent2audio_time_cost = end_time - start_time
        timecosts = {
            "preprocess": preprocess_time_cost,
            "diffusion": diffusion_time_cost,
            "latent2audio": latent2audio_time_cost,
        }

        input_params_json = {
            "task": task,
            "prompt": prompt,
            "lyrics": lyrics,
            "audio_duration": audio_duration,
            "infer_step": infer_step,
            "guidance_scale": guidance_scale,
            "scheduler_type": scheduler_type,
            "cfg_type": cfg_type,
            "omega_scale": omega_scale,
            "guidance_interval": guidance_interval,
            "guidance_interval_decay": guidance_interval_decay,
            "min_guidance_scale": min_guidance_scale,
            "use_erg_tag": use_erg_tag,
            "use_erg_lyric": use_erg_lyric,
            "use_erg_diffusion": use_erg_diffusion,
            "oss_steps": oss_steps,
            "timecosts": timecosts,
            "actual_seeds": actual_seeds,
            "retake_seeds": actual_retake_seeds,
            "retake_variance": retake_variance,
            "guidance_scale_text": guidance_scale_text,
            "guidance_scale_lyric": guidance_scale_lyric,
        }
        # save input_params_json
        for output_audio_path in output_paths:
            input_params_json_save_path = output_audio_path.replace(f".{format}", "_input_params.json")
            input_params_json["audio_path"] = output_audio_path
            with open(input_params_json_save_path, "w") as f:
                json.dump(input_params_json, f, indent=4, ensure_ascii=False)

        return output_paths + [input_params_json]
