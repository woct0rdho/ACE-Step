import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="checkpoints/epoch=22-step=460k_pretrained_ft_80k.ckpt")
parser.add_argument("--port", type=int, default=7862)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--share", action='store_true', default=False)
parser.add_argument("--bf16", action='store_true', default=True)
parser.add_argument("--hide_dataset_sampler", action='store_true', default=False)

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

import torch
import torchaudio
import torch.nn.functional as F
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from diffusers.utils.torch_utils import randn_tensor
from pathlib import Path
import time
from tqdm import tqdm
from loguru import logger
import json
from ui.auth import same_auth
from ui.text2music_large_lyric_components_v3 import create_main_demo_ui

from models.lyrics_utils.lyric_tokenizer import VoiceBpeTokenizer
from schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from schedulers.scheduling_flow_match_heun_discrete import FlowMatchHeunDiscreteScheduler  
from apg_guidance import apg_forward, MomentumBuffer, cfg_forward
from language_segmentation import LangSegment
import random
import re


logger.add("demo_v3.log", level="INFO")


def ensure_directory_exists(directory):
    directory = str(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

VALID_STRUCTURE_PATTERN = ["hook", "break", "pre-chorus", "solo", "inst", "end", "outro", "bridge", "chorus", "verse", "intro", "start"]

def is_structure_tag(lin):
    lin = lin.lower()
    pattern = re.compile(r"\[.*\]")
    for tag in VALID_STRUCTURE_PATTERN:
        if tag in lin and pattern.match(lin):
            return True
    return False


# 重新tokenize的逻辑
SUPPORT_LANGUAGES = {
    "en": 259, "de": 260, "fr": 262, "es": 284, "it": 285, 
    "pt": 286, "pl": 294, "tr": 295, "ru": 267, "cs": 293, 
    "nl": 297, "ar": 5022, "zh": 5023, "ja": 5412, "hu": 5753,
    "ko": 6152, "hi": 6680
}

structure_pattern = re.compile(r"\[.*?\]")


class InferDemo:
    def __init__(self, args):
        logger.info(f"init model with checkpoint: {args.checkpoint_path}")
        model_checkpoint_name = "AceFlow3_250401" + Path(args.checkpoint_path).stem
        if args.bf16:
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        self.device = "cuda:0"

        self.model_checkpoint_name = model_checkpoint_name

        self.checkpoint_path = ""

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

    def reload_model(self, checkpoint_path):
        if checkpoint_path in self.checkpoint_path or self.checkpoint_path == checkpoint_path:
            return

        logger.info(f"re-init model with checkpoint: {checkpoint_path}")
        model_checkpoint_name = "AceFlow3_250401" + Path(checkpoint_path).stem
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        from main_text2music_large_sana_dcae_0331_finetune import Pipeline

        model = Pipeline(infer=True, train=False)
        model.load_state_dict(checkpoint, strict=False)

        self.model = model.eval().to(self.device).to(self.dtype)
        self.model_checkpoint_name = model_checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.tokenizer = VoiceBpeTokenizer()

    def save_wav_file(self, target_wav, idx, sample_rate=48000):
        base_path = f"./test_results/{self.model_checkpoint_name}/demo_outputs"
        ensure_directory_exists(base_path)
        # 压缩成mp3
        output_path_flac = f"{base_path}/output_{time.strftime('%Y%m%d%H%M%S')}_{idx}.flac"
        target_wav = target_wav.float()
        torchaudio.save(output_path_flac, target_wav, sample_rate=sample_rate, format='flac', backend="ffmpeg", compression=torchaudio.io.CodecConfig(bit_rate=320000))
        return output_path_flac

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
            logger.info(f"batch idx: {i}, seed: {seed}")
            random_generators[i].manual_seed(seed)
            actual_seeds.append(seed)
        return random_generators, actual_seeds

    def latents2audio(self, latents, target_wav_duration_second=30, sample_rate=48000):
        output_audio_paths = []
        bs = latents.shape[0]
        audio_lengths = [target_wav_duration_second * sample_rate] * bs
        pred_latents = latents
        with torch.no_grad():
            _, pred_wavs = self.model.vae.decode(pred_latents, sr=sample_rate)
        pred_wavs = [pred_wav.cpu().float() for pred_wav in pred_wavs]
        for i in tqdm(range(bs)):
            output_audio_path = self.save_wav_file(pred_wavs[i], i, sample_rate=sample_rate)
            output_audio_paths.append(output_audio_path)
        return output_audio_paths

    def get_lang(self, text):
        language = "en"
        try:    
            langs = self.lang_segment.getTexts(text)
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
    ):

        logger.info("cfg_type: {}, guidance_scale: {}, omega_scale: {}".format(cfg_type, guidance_scale, omega_scale))
        do_classifier_free_guidance = True
        if guidance_scale == 0.0 or guidance_scale == 1.0:
            do_classifier_free_guidance = False

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
            noise_pred = self.model.transformers(
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
                if cfg_type == "apg":
                    noise_pred = apg_forward(
                        pred_cond=noise_pred_with_cond,
                        pred_uncond=noise_pred_uncond,
                        guidance_scale=guidance_scale,
                        momentum_buffer=momentum_buffer,
                    )
                else:
                    noise_pred = cfg_forward(
                        cond_output=noise_pred_with_cond,
                        uncond_output=noise_pred_uncond,
                        cfg_strength=guidance_scale,
                    )
            
            target_latents = scheduler.step(model_output=noise_pred, timestep=t, sample=target_latents, return_dict=False, omega=omega_scale)[0]
        
        return target_latents
    
    @torch.no_grad()
    def process_text2music(
        self,
        audio_duration,
        prompt,
        lyrics,
        input_params_json,
        selected_checkpoint,
        scheduler_type,
        cfg_type,
        infer_step,
        guidance_scale,
        omega_scale,
        manual_seeds,
    ):
        # 1 check if need to reload model
        if selected_checkpoint is not None and self.checkpoint_path != selected_checkpoint:
            self.reload_model(selected_checkpoint)

        batch_size = 2

        # 2 set seed
        random_generators, actual_seeds = self.set_seeds(batch_size, manual_seeds)

        # 8 x 16 x T//8
        # 4 prompt
        texts = [prompt]
        encoder_text_hidden_states, text_attention_mask = self.model.lyric_processor.get_text_embeddings(texts, self.device)
        encoder_text_hidden_states = encoder_text_hidden_states.repeat(batch_size, 1, 1)
        text_attention_mask = text_attention_mask.repeat(batch_size, 1)

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
            audio_duration = random.uniform(30.0, 300.0)
            logger.info(f"random audio duration: {audio_duration}")

        # 7. encode
        target_latents = self.text2music_diffusion_process(
            audio_duration,
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
        )

        # 8 latents2audio
        output_paths = self.latents2audio(latents=target_latents, target_wav_duration_second=audio_duration)
        if input_params_json is None:
            input_params_json = {}
        input_params_json["prompt"] = prompt
        input_params_json["lyrics"] = lyrics
        input_params_json["infer_steps"] = infer_step
        input_params_json["guidance_scale"] = guidance_scale
        input_params_json["manual_seeds"] = manual_seeds
        input_params_json["actual_seeds"] = actual_seeds
        input_params_json["checkpoint_path"] = self.checkpoint_path
        input_params_json["omega_scale"] = omega_scale
        input_params_json["scheduler_type"] = scheduler_type
        input_params_json["cfg_type"] = cfg_type
        input_params_json["audio_duration"] = audio_duration
        logger.info(json.dumps(input_params_json, indent=4, ensure_ascii=False))

        return output_paths + [input_params_json]


def main(args):
 
    model_demo = InferDemo(args)

    demo = create_main_demo_ui(
        checkpoint_path=args.checkpoint_path,
        text2music_process_func=model_demo.process_text2music,
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        auth=same_auth,
        share=args.share
    )


if __name__ == "__main__":
    main(args)
