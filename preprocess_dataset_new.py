#!/usr/bin/env python3

import argparse

import torch
import torch.nn
import torch.nn.functional as F
import torchaudio
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.text2music_dataset import Text2MusicDataset

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
# torch._dynamo.config.recompile_limit = 64


class Preprocessor(torch.nn.Module):
    def __init__(self, checkpoint_dir=None):
        super().__init__()

        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")

        acestep_pipeline = ACEStepPipeline(checkpoint_dir)
        acestep_pipeline.load_checkpoint(acestep_pipeline.checkpoint_dir)
        self.dcae = acestep_pipeline.music_dcae
        self.dcae.dcae.encoder = torch.compile(self.dcae.dcae.encoder, dynamic=True)
        self.text_encoder_model = acestep_pipeline.text_encoder_model
        self.text_encoder_model = torch.compile(self.text_encoder_model, dynamic=True)
        self.text_tokenizer = acestep_pipeline.text_tokenizer
        del acestep_pipeline

        self.mert_model = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
        )
        self.resampler_mert = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=24000
        )
        self.processor_mert = Wav2Vec2FeatureExtractor.from_pretrained(
            "m-a-p/MERT-v1-330M", trust_remote_code=True, cache_dir=checkpoint_dir
        )

        self.hubert_model = AutoModel.from_pretrained(
            "utter-project/mHuBERT-147", cache_dir=checkpoint_dir
        )
        self.resampler_mhubert = torchaudio.transforms.Resample(
            orig_freq=48000, new_freq=16000
        )
        self.processor_mhubert = Wav2Vec2FeatureExtractor.from_pretrained(
            "utter-project/mHuBERT-147", cache_dir=checkpoint_dir
        )

        self.to(self.device, self.dtype)
        self.eval()

    def infer_mert_ssl(self, target_wavs, wav_lengths):
        # Input is N x 2 x T (48kHz), convert to N x T (24kHz), mono
        mert_input_wavs_mono_24k = self.resampler_mert(target_wavs.mean(dim=1))
        bsz = target_wavs.shape[0]
        actual_lengths_24k = wav_lengths // 2  # 48kHz -> 24kHz

        # Normalize the actual audio part
        means = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].mean()
                for i in range(bsz)
            ]
        )
        _vars = torch.stack(
            [
                mert_input_wavs_mono_24k[i, : actual_lengths_24k[i]].var()
                for i in range(bsz)
            ]
        )
        mert_input_wavs_mono_24k = (
            mert_input_wavs_mono_24k - means.view(-1, 1)
        ) / torch.sqrt(_vars.view(-1, 1) + 1e-7)

        # MERT SSL constraint
        # Define the length of each chunk (5 seconds of samples)
        chunk_size = 24000 * 5  # 5 seconds, 24000 samples per second

        num_chunks_per_audio = (actual_lengths_24k + chunk_size - 1) // chunk_size

        # Process chunks
        all_chunks = []
        chunk_actual_lengths = []
        for i in range(bsz):
            audio = mert_input_wavs_mono_24k[i]
            actual_length = actual_lengths_24k[i]
            for start in range(0, actual_length, chunk_size):
                end = min(start + chunk_size, actual_length)
                chunk = audio[start:end]
                if len(chunk) < chunk_size:
                    # Pad insufficient parts with zeros
                    chunk = F.pad(chunk, (0, chunk_size - len(chunk)))
                all_chunks.append(chunk)
                chunk_actual_lengths.append(end - start)

        # Stack all chunks to (total_chunks, chunk_size)
        all_chunks = torch.stack(all_chunks, dim=0)

        # Batch inference
        with torch.no_grad():
            # Output shape: (total_chunks, seq_len, hidden_size)
            mert_ssl_hidden_states = self.mert_model(all_chunks).last_hidden_state

        # Calculate the number of features for each chunk
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Trim the hidden states of each chunk
        chunk_hidden_states = [
            mert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Organize hidden states by audio
        mert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            audio_hidden = torch.cat(
                audio_chunks, dim=0
            )  # Concatenate chunks of the same audio
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
        means = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].mean()
                for i in range(bsz)
            ]
        )
        _vars = torch.stack(
            [
                mhubert_input_wavs_mono_16k[i, : actual_lengths_16k[i]].var()
                for i in range(bsz)
            ]
        )
        mhubert_input_wavs_mono_16k = (
            mhubert_input_wavs_mono_16k - means.view(-1, 1)
        ) / torch.sqrt(_vars.view(-1, 1) + 1e-7)

        # Step 3: Define chunk size for MHubert (30 seconds at 16kHz)
        chunk_size = 16000 * 30  # 30 seconds = 480,000 samples

        # Step 4: Split audio into chunks
        # Ceiling division
        num_chunks_per_audio = (actual_lengths_16k + chunk_size - 1) // chunk_size
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
            # Shape: (total_chunks, seq_len, hidden_size)
            mhubert_ssl_hidden_states = self.hubert_model(all_chunks).last_hidden_state

        # Step 7: Compute number of features per chunk (assuming model stride of 320)
        chunk_num_features = [(length + 319) // 320 for length in chunk_actual_lengths]

        # Step 8: Trim hidden states to remove padding effects
        chunk_hidden_states = [
            mhubert_ssl_hidden_states[i, : chunk_num_features[i], :]
            for i in range(len(all_chunks))
        ]

        # Step 9: Reorganize hidden states by original audio
        mhubert_ssl_hidden_states_list = []
        chunk_idx = 0
        for i in range(bsz):
            audio_chunks = chunk_hidden_states[
                chunk_idx : chunk_idx + num_chunks_per_audio[i]
            ]
            # Concatenate chunks for this audio
            audio_hidden = torch.cat(audio_chunks, dim=0)
            mhubert_ssl_hidden_states_list.append(audio_hidden)
            chunk_idx += num_chunks_per_audio[i]
        return mhubert_ssl_hidden_states_list

    def get_text_embeddings(self, texts, text_max_length=256):
        inputs = self.text_tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=text_max_length,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.text_encoder_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        return last_hidden_states, attention_mask

    def preprocess(self, batch, train=True):
        dtype = self.dtype
        device = self.device

        target_wavs = batch["target_wavs"].to(device, dtype)
        wav_lengths = batch["wav_lengths"].to(device)

        bs = target_wavs.shape[0]

        # SSL constraints
        mert_ssl_hidden_states = None
        mhubert_ssl_hidden_states = None
        if train:
            mert_ssl_hidden_states = self.infer_mert_ssl(target_wavs, wav_lengths)
            mert_ssl_hidden_states = [x.cpu() for x in mert_ssl_hidden_states]
            mhubert_ssl_hidden_states = self.infer_mhubert_ssl(target_wavs, wav_lengths)
            mhubert_ssl_hidden_states = [x.cpu() for x in mhubert_ssl_hidden_states]

        # text embedding
        texts = batch["prompts"]
        encoder_text_hidden_states, text_attention_mask = self.get_text_embeddings(
            texts
        )
        encoder_text_hidden_states = encoder_text_hidden_states.cpu()
        text_attention_mask = text_attention_mask.cpu().to(dtype)

        target_latents, _ = self.dcae.encode(target_wavs, wav_lengths)
        target_latents = target_latents.cpu()

        # The followings are on CPU

        attention_mask = torch.ones(bs, target_latents.shape[-1], dtype=dtype)

        speaker_embds = batch["speaker_embs"].to(dtype)
        keys = batch["keys"]
        lyric_token_ids = batch["lyric_token_ids"]
        lyric_mask = batch["lyric_masks"].to(dtype)

        # cfg
        if train:
            # TODO: Check the dropout rates
            prompt_dropout = 0
            speaker_dropout = 0
            lyric_dropout = 0

            # N x T x 768
            encoder_text_hidden_states = torch.where(
                (torch.rand(bs) > prompt_dropout).unsqueeze(1).unsqueeze(1),
                encoder_text_hidden_states,
                torch.zeros_like(encoder_text_hidden_states),
            )

            # N x 512
            speaker_embds = torch.where(
                (torch.rand(bs) > speaker_dropout).unsqueeze(1),
                speaker_embds,
                torch.zeros_like(speaker_embds),
            )

            # Lyrics
            full_cfg_condition_mask = (torch.rand(bs) > lyric_dropout).unsqueeze(1)
            lyric_token_ids = torch.where(
                full_cfg_condition_mask,
                lyric_token_ids,
                torch.zeros_like(lyric_token_ids),
            )
            lyric_mask = torch.where(
                full_cfg_condition_mask,
                lyric_mask,
                torch.zeros_like(lyric_mask),
            )

        return {
            "keys": keys,
            "target_latents": target_latents,
            "attention_mask": attention_mask,
            "encoder_text_hidden_states": encoder_text_hidden_states,
            "text_attention_mask": text_attention_mask,
            "speaker_embds": speaker_embds,
            "lyric_token_ids": lyric_token_ids,
            "lyric_mask": lyric_mask,
            "mert_ssl_hidden_states": mert_ssl_hidden_states,
            "mhubert_ssl_hidden_states": mhubert_ssl_hidden_states,
        }


def get_generator(input_name, checkpoint_dir):
    def gen():
        ds = Text2MusicDataset(
            train=True,
            train_dataset_path=input_name,
        )
        dl = DataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=ds.collate_fn,
            # persistent_workers=True,
        )
        prep = Preprocessor(checkpoint_dir)
        for batch in tqdm(dl):
            batch = prep.preprocess(batch)
            batch = {k: v[0] for k, v in batch.items()}
            yield batch

    return gen


@torch.inference_mode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_name",
        type=str,
        default=r"C:\data\sawano_filenames",
        help="The filenames-only dataset.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=r"C:\data\sawano_prep",
        help="The preprocessed dataset.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Directory to cache model checkpoints.",
    )
    args = parser.parse_args()

    ds = Dataset.from_generator(get_generator(args.input_name, args.checkpoint_dir))
    ds.save_to_disk(args.output_name)


if __name__ == "__main__":
    main()
