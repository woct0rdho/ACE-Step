# [ACE-Step](https://github.com/ace-step/ACE-Step) fork

## Progress

* Separate data preprocessing (music and text encoding) and training
* Enable gradient checkpointing
* Cast everything to bf16

Now I can run the training on a single RTX 3080 with < 10 GB VRAM and 0.3 it/s speed, using music duration < 360 seconds and LoRA rank = 64.

## Usage

For example, I want to train a LoRA for Sawano Hiroyuki (澤野 弘之)'s style.

1. Create a dataset that only contains the filenames, not the audio data:
    ```pwsh
    python convert2hf_dataset_new.py --data_dir C:\data\sawano --output_name C:\data\sawano_filenames"
    ```
    where `--data_dir` is a directory containing audio files.

2. Load the audios, do the preprocessing, save to a new dataset:
    ```pwsh
    python preprocess_dataset_new.py --input_name C:\data\sawano_filenames --output_name C:\data\sawano_prep
    ```
    Currently this will take a lot of disk space. 100 songs of < 360 seconds take ~8 GB. This can be optimized.

    If you modify the data files or the code and re-generate the dataset, you may need to clear the cache like `~/.cache/huggingface/datasets/generator`.

3. Do the training:
    ```pwsh
    python trainer_new.py --dataset_path C:\data\sawano_prep --exp_name sawano
    ```
    The LoRA will be saved to the directory `checkpoints`. I recommend to clear this directory before training, otherwise the LoRA may not be correctly saved.

    Note that my script uses Wandb rather than TensorBoard. If you don't need it, you can remove the `WandbLogger`.

4. LoRA strength:

    At this point, when loading the LoRA in ComfyUI, you need to set the lora strength to `alpha / sqrt(rank)` (for rslora) or `alpha / rank` (for non-rslora). For example, if rank = 64, alpha = 1, rslora is enabled, then the lora strength should be `1 / sqrt(64) = 0.125`.

    To avoid manually setting this, you can run:
    ```pwsh
    python bake_alpha_in_lora.py --input_name checkpoints/epoch=0-step=100_lora/pytorch_lora_weights.safetensors --output_name out.safetensors --lora_config_path config/lora_config_transformer_only.json
    ```
    Then load `out.safetensors` in ComfyUI and set the lora strength to 1. But it may not be loaded correctly in diffusers.

## Tips

* If you don't have experience, you can first try to train with a single audio and make sure that it can be overfitted. This is a sanity check of the training pipeline
* You can freeze the lyrics decoder and only train the transformer using `config/lora_config_transformer_only.json`. I think training the lyrics decoder is needed only when adding a new language

## TODO

* How to normalize the audio loudness before preprocessing? It seems the audios generated by ACE-Step usually have loudness in -16 .. -12 LUFS, and they don't follow prompts like 'loud' and 'quiet'
* Before preprocessing, the official ACE-Step uses Qwen2.5-Omni-7B to generate prompts (tags) for audios, but this is too slow on consumer GPUs. Maybe we need to train a specialized tagger like [WD Tagger](https://huggingface.co/SmilingWolf)
    * The statistics of the tags used to train the base model is shared on [Discord](https://discord.com/channels/1369256267645849741/1372633881215500429/1374037211145830442). We may also use the tags from MusicBrainz
* Use regularization audios, like regularization images for image LoRA training
