#!/usr/bin/env python3

import os

# Avoid some common network problems
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import shutil
from glob import glob

import torch
import torch.nn.functional as F
import torch.utils.data
from datasets import Dataset
from natsort import natsorted
from peft import LoraConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.core import LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch._dynamo.config.recompile_limit = 64


def _to_bf16(x):
    if isinstance(x, list):
        return [_to_bf16(y) for y in x]
    elif isinstance(x, torch.Tensor) and x.dtype == torch.float32:
        return x.to(torch.bfloat16)
    else:
        return x


class Pipeline(LightningModule):
    def __init__(
        self,
        # Model
        checkpoint_dir: str = None,
        T: int = 1000,
        shift: float = 3.0,
        timestep_densities_type: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        lora_config_path: str = None,
        # Data
        dataset_path: str = "./data/your_dataset_path",
        batch_size: int = 1,
        num_workers: int = 0,
        # Optimizer
        ssl_coeff: float = 1.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        max_steps: int = 10000,
        warmup_steps: int = 10,
        # Others
        adapter_name: str = "lora_adapter",
        save_last: int = 5,
        every_plot_step: int = 1000,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize scheduler
        self.scheduler = self.get_scheduler()

        # Load model
        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        self.transformer = acestep_pipeline.ace_step_transformer.to(torch.bfloat16)
        self.transformer.train()
        self.transformer.enable_gradient_checkpointing()
        del acestep_pipeline

        # Load LoRA
        assert lora_config_path is not None, "Please provide a LoRA config path"
        with open(lora_config_path, encoding="utf-8") as f:
            lora_config = json.load(f)
        lora_config = LoraConfig(**lora_config)
        self.transformer.add_adapter(
            adapter_config=lora_config,
            adapter_name=adapter_name,
        )
        self.adapter_name = adapter_name

    def get_scheduler(self):
        return FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=self.hparams.T,
            shift=self.hparams.shift,
        )

    def configure_optimizers(self):
        trainable_params = [
            p for name, p in self.transformer.named_parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW(
            params=trainable_params,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            betas=(0.8, 0.9),
        )
        max_steps = self.hparams.max_steps
        warmup_steps = self.hparams.warmup_steps  # New hyperparameter for warmup steps

        # Create a scheduler that first warms up linearly, then decays linearly
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from 0 to learning_rate
                return float(current_step) / float(max(1, warmup_steps))
            else:
                # Linear decay from learning_rate to 0
                progress = float(current_step - warmup_steps) / float(
                    max(1, max_steps - warmup_steps)
                )
                return max(0.0, 1.0 - progress)

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda, last_epoch=-1
        )
        return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]

    def train_dataloader(self):
        ds = Dataset.load_from_disk(self.hparams.dataset_path).with_format(
            "torch", device="cuda:0"
        )
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            # pin_memory=True,
            # persistent_workers=True,
        )

    def get_sd3_sigmas(self, timesteps, device, n_dim, dtype):
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
            u = torch.normal(
                mean=self.hparams.logit_mean,
                std=self.hparams.logit_std,
                size=(bsz,),
                device="cpu",
            )
            u = torch.nn.functional.sigmoid(u)
            indices = (u * self.scheduler.config.num_train_timesteps).long()
            indices = torch.clamp(
                indices, 0, self.scheduler.config.num_train_timesteps - 1
            )
            timesteps = self.scheduler.timesteps[indices].to(device)
        else:
            raise ValueError(
                f"Unknown timestep_densities_type: {self.hparams.timestep_densities_type}"
            )
        return timesteps

    def run_step(self, batch, batch_idx):
        batch = {k: _to_bf16(v) for k, v in batch.items()}

        keys = batch["keys"]
        target_latents = batch["target_latents"]
        attention_mask = batch["attention_mask"]
        encoder_text_hidden_states = batch["encoder_text_hidden_states"]
        text_attention_mask = batch["text_attention_mask"]
        speaker_embds = batch["speaker_embds"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_mask"]
        mert_ssl_hidden_states = batch["mert_ssl_hidden_states"]
        mhubert_ssl_hidden_states = batch["mhubert_ssl_hidden_states"]

        target_image = target_latents
        device = target_image.device
        dtype = target_image.dtype

        # Step 1: Generate random noise, initialize settings
        noise = torch.randn_like(target_image, device=device)
        bsz = target_image.shape[0]
        timesteps = self.get_timestep(bsz, device)

        # Add noise according to flow matching.
        sigmas = self.get_sd3_sigmas(
            timesteps=timesteps, device=device, n_dim=target_image.ndim, dtype=dtype
        )
        noisy_image = sigmas * noise + (1.0 - sigmas) * target_image

        # This is the flow-matching target for vanilla SD3.
        target = target_image

        # SSL constraints for CLAP and vocal_latent_channel2
        all_ssl_hiden_states = []
        if mert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mert_ssl_hidden_states)
        if mhubert_ssl_hidden_states is not None:
            all_ssl_hiden_states.append(mhubert_ssl_hidden_states)

        # N x H -> N x c x W x H
        x = noisy_image
        # Step 5: Predict noise
        transformer_output = self.transformer(
            hidden_states=x,
            attention_mask=attention_mask,
            encoder_text_hidden_states=encoder_text_hidden_states,
            text_attention_mask=text_attention_mask,
            speaker_embeds=speaker_embds,
            lyric_token_idx=lyric_token_ids,
            lyric_mask=lyric_mask,
            timestep=timesteps.to(device, dtype),
            ssl_hidden_states=all_ssl_hiden_states,
        )
        model_pred = transformer_output.sample
        proj_losses = transformer_output.proj_losses

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        model_pred = model_pred * (-sigmas) + noisy_image

        # Compute loss. Only calculate loss where chunk_mask is 1 and there is no padding
        # N x T x 64
        # N x T -> N x c x W x T
        mask = (
            attention_mask.unsqueeze(1)
            .unsqueeze(1)
            .expand(-1, target_image.shape[1], target_image.shape[2], -1)
        )

        # TODO: Check if the masked mean is correct
        selected_model_pred = (model_pred * mask).reshape(bsz, -1).contiguous()
        selected_target = (target * mask).reshape(bsz, -1).contiguous()

        loss = F.mse_loss(selected_model_pred, selected_target, reduction="none")
        loss = loss.mean(1)
        loss = loss * mask.reshape(bsz, -1).mean(1)
        loss = loss.mean()

        prefix = "train"

        self.log(
            f"{prefix}/denoising_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
        )

        total_proj_loss = 0.0
        for k, v in proj_losses:
            self.log(
                f"{prefix}/{k}_loss", v, on_step=True, on_epoch=False, prog_bar=True
            )
            total_proj_loss += v

        if len(proj_losses) > 0:
            total_proj_loss = total_proj_loss / len(proj_losses)

        loss = loss + total_proj_loss * self.hparams.ssl_coeff
        self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=False, prog_bar=True)

        # Log learning rate if scheduler exists
        if self.lr_schedulers() is not None:
            learning_rate = self.lr_schedulers().get_last_lr()[0]
            self.log(
                f"{prefix}/learning_rate",
                learning_rate,
                on_step=True,
                on_epoch=False,
                prog_bar=True,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.run_step(batch, batch_idx)

    def on_save_checkpoint(self, checkpoint):
        checkpoint_dir = "./checkpoints"

        epoch = self.current_epoch
        step = self.global_step
        lora_name = f"epoch={epoch}-step={step}_lora"
        lora_path = os.path.join(checkpoint_dir, lora_name)
        os.makedirs(lora_path, exist_ok=True)
        self.transformer.save_lora_adapter(lora_path, adapter_name=self.adapter_name)

        # Clean up old loras and only save the last few loras
        lora_paths = glob(os.path.join(checkpoint_dir, "*_lora"))
        lora_paths = natsorted(lora_paths)
        if len(lora_paths) > self.hparams.save_last:
            shutil.rmtree(lora_paths[0])

        # Don't save the full model
        checkpoint.clear()
        return checkpoint


def main(args):
    model = Pipeline(
        # Model
        checkpoint_dir=args.checkpoint_dir,
        shift=args.shift,
        lora_config_path=args.lora_config_path,
        # Data
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # Optimizer
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        # Others
        adapter_name=args.exp_name,
        save_last=args.save_last,
        every_plot_step=args.every_plot_step,
    )
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.every_n_train_steps,
    )
    logger_callback = WandbLogger(
        project="ace-step-lora",
        name=args.exp_name,
    )
    trainer = Trainer(
        accelerator="gpu",
        # strategy="ddp_find_unused_parameters_true",
        # devices=args.devices,
        # num_nodes=args.num_nodes,
        precision=args.precision,
        log_every_n_steps=1,
        logger=logger_callback,
        callbacks=[checkpoint_callback],
        max_epochs=args.epochs,
        max_steps=args.max_steps,
        val_check_interval=args.val_check_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        # reload_dataloaders_every_n_epochs=args.reload_dataloaders_every_n_epochs,
    )

    trainer.fit(
        model,
        # ckpt_path=args.ckpt_path,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()

    # Model
    args.add_argument("--checkpoint_dir", type=str, default=None)
    args.add_argument("--shift", type=float, default=3.0)
    args.add_argument("--lora_config_path", type=str, default="./config/lora_config.json")

    # Data
    args.add_argument("--dataset_path", type=str, default=r"C:\data\sawano_prep")
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--num_workers", type=int, default=0)

    # Optimizer
    args.add_argument("--learning_rate", type=float, default=1e-4)
    args.add_argument("--epochs", type=int, default=-1)
    args.add_argument("--max_steps", type=int, default=10000)
    args.add_argument("--warmup_steps", type=int, default=10)
    args.add_argument("--accumulate_grad_batches", type=int, default=1)
    args.add_argument("--gradient_clip_val", type=float, default=1)
    args.add_argument("--gradient_clip_algorithm", type=str, default="norm")

    # Others
    # args.add_argument("--devices", type=int, default=1)
    # args.add_argument("--num_nodes", type=int, default=1)
    args.add_argument("--exp_name", type=str, default="sawano")
    args.add_argument("--precision", type=str, default="bf16-mixed")
    args.add_argument("--every_n_train_steps", type=int, default=100)
    args.add_argument("--save_last", type=int, default=5)
    args.add_argument("--every_plot_step", type=int, default=1000)
    args.add_argument("--val_check_interval", type=int, default=None)
    # args.add_argument("--ckpt_path", type=str, default=None)
    # args.add_argument("--reload_dataloaders_every_n_epochs", type=int, default=0)

    args = args.parse_args()
    main(args)
