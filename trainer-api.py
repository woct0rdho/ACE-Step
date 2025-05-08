import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import os
import random
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from acestep.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.apg_guidance import apg_forward, MomentumBuffer
from transformers import AutoTokenizer
from loguru import logger
import uvicorn
import time
from datetime import datetime

app = FastAPI(title="Text-to-Music API")

class TextToMusicRequest(BaseModel):
    prompt: str
    duration: int = 240  # Duration in seconds (default 240s)
    infer_steps: int = 60
    guidance_scale: float = 15.0
    omega_scale: float = 10.0
    seed: Optional[int] = None

class TextToMusicResponse(BaseModel):
    audio_path: str
    prompt: str
    seed: int
    sample_rate: int

class InferencePipeline:
    def __init__(self, checkpoint_dir: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing model on device: {self.device}")

        # Load the ACEStepPipeline
        self.acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        self.acestep_pipeline.load_checkpoint(checkpoint_dir)

        # Initialize components
        self.transformers = self.acestep_pipeline.ace_step_transformer.float().to(self.device).eval()
        self.dcae = self.acestep_pipeline.music_dcae.float().to(self.device).eval()
        self.text_encoder_model = self.acestep_pipeline.text_encoder_model.float().to(self.device).eval()
        self.text_tokenizer = self.acestep_pipeline.text_tokenizer

        # Ensure no gradients are computed
        self.transformers.requires_grad_(False)
        self.dcae.requires_grad_(False)
        self.text_encoder_model.requires_grad_(False)

        # Initialize scheduler
        self.scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=3.0,
        )

    def get_text_embeddings(self, texts, device, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    def diffusion_process(
        self,
        duration,
        encoder_text_hidden_states,
        text_attention_mask,
        speaker_embds,
        lyric_token_ids,
        lyric_mask,
        random_generator=None,
        infer_steps=60,
        guidance_scale=15.0,
        omega_scale=10.0,
    ):
        do_classifier_free_guidance = guidance_scale > 1.0
        device = encoder_text_hidden_states.device
        dtype = encoder_text_hidden_states.dtype
        bsz = encoder_text_hidden_states.shape[0]

        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps=infer_steps, device=device
        )

        frame_length = int(duration * 44100 / 512 / 8)
        target_latents = randn_tensor(
            shape=(bsz, 8, 16, frame_length),
            generator=random_generator,
            device=device,
            dtype=dtype,
        )
        attention_mask = torch.ones(bsz, frame_length, device=device, dtype=dtype)

        if do_classifier_free_guidance:
            attention_mask = torch.cat([attention_mask] * 2, dim=0)
            encoder_text_hidden_states = torch.cat(
                [encoder_text_hidden_states, torch.zeros_like(encoder_text_hidden_states)],
                0,
            )
            text_attention_mask = torch.cat([text_attention_mask] * 2, dim=0)
            speaker_embds = torch.cat([speaker_embds, torch.zeros_like(speaker_embds)], 0)
            lyric_token_ids = torch.cat([lyric_token_ids, torch.zeros_like(lyric_token_ids)], 0)
            lyric_mask = torch.cat([lyric_mask, torch.zeros_like(lyric_mask)], 0)

        momentum_buffer = MomentumBuffer()

        for t in timesteps:
            latent_model_input = (
                torch.cat([target_latents] * 2) if do_classifier_free_guidance else target_latents
            )
            timestep = t.expand(latent_model_input.shape[0])
            with torch.no_grad():
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

            target_latents = self.scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=target_latents,
                omega=omega_scale,
            )[0]

        return target_latents

    def generate_audio(
        self,
        prompt: str,
        duration: int,
        infer_steps: int,
        guidance_scale: float,
        omega_scale: float,
        seed: Optional[int],
    ):
        # Set random seed
        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)
        else:
            seed = random.randint(0, 2**32 - 1)
            random.seed(seed)
            torch.manual_seed(seed)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        # Get text embeddings
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            [prompt], self.device
        )

        # Dummy speaker embeddings and lyrics (since not provided in API request)
        bsz = 1
        speaker_embds = torch.zeros(bsz, 512, device=self.device, dtype=encoder_text_hidden_states.dtype)
        lyric_token_ids = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)
        lyric_mask = torch.zeros(bsz, 256, device=self.device, dtype=torch.long)

        # Run diffusion process
        pred_latents = self.diffusion_process(
            duration=duration,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embds=speaker_embds,
            lyric_token_ids=lyric_token_ids,
            lyric_mask=lyric_mask,
            random_generator=generator,
            infer_steps=infer_steps,
            guidance_scale=guidance_scale,
            omega_scale=omega_scale,
        )

        # Decode latents to audio
        audio_lengths = torch.tensor([int(duration * 44100)], device=self.device)
        sr, pred_wavs = self.dcae.decode(pred_latents, audio_lengths=audio_lengths, sr=48000)

        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "generated_audio"
        os.makedirs(output_dir, exist_ok=True)
        audio_path = f"{output_dir}/generated_{timestamp}_{seed}.wav"
        torchaudio.save(audio_path, pred_wavs.float().cpu(), sr)

        return audio_path, sr, seed

# Global model instance
model = None

@app.on_event("startup")
async def startup_event():
    global model
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    model = InferencePipeline(checkpoint_dir=checkpoint_dir)
    logger.info("Model loaded successfully")

@app.post("/generate", response_model=TextToMusicResponse)
async def generate_music(request: TextToMusicRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        start_time = time.time()
        audio_path, sr, seed = model.generate_audio(
            prompt=request.prompt,
            duration=request.duration,
            infer_steps=request.infer_steps,
            guidance_scale=request.guidance_scale,
            omega_scale=request.omega_scale,
            seed=request.seed,
        )
        logger.info(f"Generation completed in {time.time() - start_time:.2f} seconds")
        return TextToMusicResponse(
            audio_path=audio_path,
            prompt=request.prompt,
            seed=seed,
            sample_rate=sr,
        )
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
