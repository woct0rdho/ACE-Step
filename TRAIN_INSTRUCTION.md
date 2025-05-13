# Training Instruction

## 1. Data Preparation
1. First, check the format of data preparation in the `data` directory under the root directory of the project page.
Prepare your audio. If you already have well-labeled audio, that's great.
If you don't have labels, you can use the following prompt and utilize Qwen Omini to label your audio. The community welcomes contributions of better prompts, as well as annotation tools and UI.

How to get an audio's reception? 
You can use `Qwen-Omini` https://chat.qwen.ai/ to describe an audio.
Here we share the prompt we used.

```python
sys_prompt_without_tag = """Analyze the input audio and generate 6 description variants. Each variant must be <200 characters. Follow these exact definitions:

1.  `simplified`: Use only one most representative tag from the valid set.
2.  `expanded`: Broaden valid tags to include related sub-genres/techniques.
3.  `descriptive`: Convert tags into a sensory-rich sentence based *only on the sound*. DO NOT transcribe or reference the lyrics.
4.  `synonyms`: Replace tags with equivalent terms (e.g., 'strings' → 'orchestral').
5.  `use_cases`: Suggest practical applications based on audio characteristics.
6.  `analysis`: Analyze the audio's genre, instruments, tempo, and mood **based strictly on the audible musical elements**. Technical breakdown in specified format.
    *   For the `instruments` list: **Only include instruments that are actually heard playing in the audio recording.** **Explicitly ignore any instruments merely mentioned or sung about in the lyrics.** Cover all audibly present instruments.
7. `lyrical_rap_check`: if the audio is lyrical rap
**Strictly ignore any information derived solely from the lyrics when performing the analysis, especially for identifying instruments.**

**Output Format:**
```json
{
  "simplified": <str>,
  "expanded": <str>,
  "descriptive": <str>,
  "synonyms": <str>,
  "use_cases": <str>,
  "analysis": {
    "genre": <str list>,
    "instruments": <str list>,
    "tempo": <str>,
    "mood": <str list>
  },
  "lyrical_rap_check": <bool>
}
"""
```

## 2. Convert to Huggingface Dataset Format
2. Run `python convert2hf_dataset.py --data_dir "./data" --repeat_count 2000 --output_name "zh_lora_dataset"`. (Since there is only one piece of sample data, it is repeated 2000 times. You can adjust it according to the size of your data.)

## 3. Configure Lora Parameters
Refer to `config/zh_rap_lora_config.json` for configuring Lora parameters.

If your VRAM is not sufficient, you can reduce the `r` and `lora_alpha` parameters in the configuration file. Such as:
```json
{
	"r": 16,
	"lora_alpha": 32,
	"target_modules": [
		"linear_q",
		"linear_k",
		"linear_v",
		"to_q",
		"to_k",
		"to_v",
		"to_out.0"
	]
}
```

## 4. Run Training
Run `python trainer.py` with the following parameter introduction:

# Trainer Parameter Interpretation

## 1. General Settings
1. **`--num_nodes`**: This parameter specifies the number of nodes for the training process. It is an integer value, and the default is set to 1. In scenarios where distributed training across multiple nodes is applicable, this parameter determines how many nodes will be utilized. For example, if you have a cluster of machines and want to distribute the training load, you can increase this number. However, for most single-machine or basic training setups, the default value of 1 is sufficient.
2. **`--shift`**: It is a floating-point parameter with a default value of 3.0. Although its specific function depends on the implementation details of the model, it is likely used for some internal calculations related to the model, such as adjusting certain biases or offsets in the neural network architecture during the training process.

## 2. Training Hyperparameters
1. **`--learning_rate`**: This is a crucial hyperparameter for the training process. It is a floating-point value with a default of 1e-4 (0.0001). The learning rate determines the step size at each iteration while updating the model's weights. A smaller learning rate will make the training process more stable but may require more training steps to converge. On the other hand, a larger learning rate can lead to faster convergence but might cause the model to overshoot the optimal solution and result in unstable training or even divergence.
2. **`--num_workers`**: This parameter defines the number of worker processes that will be used for data loading. It is an integer with a default value of 8. Having multiple workers can significantly speed up the data loading process, especially when dealing with large datasets. However, it also consumes additional system resources, so you may need to adjust this value based on the available resources of your machine (e.g., CPU cores and memory).
3. **`--epochs`**: It represents the number of times the entire training dataset will be passed through the model. It is an integer, and the default value is set to -1. When set to -1, the training will continue until another stopping condition (such as reaching the maximum number of steps) is met. If you set a positive integer value, the training will stop after that number of epochs.
4. **`--max_steps`**: This parameter specifies the maximum number of training steps. It is an integer with a default value of 2000000. Once the model has completed this number of training steps, the training process will stop, regardless of whether the model has fully converged or not. This is useful for setting a limit on the training duration in terms of the number of steps.
5. **`--every_n_train_steps`**: It is an integer parameter with a default of 2000. It determines how often certain operations (such as saving checkpoints, logging training progress, etc.) will be performed during the training. For example, with a value of 2000, these operations will occur every 2000 training steps.

## 3. Dataset and Experiment Settings
1. **`--dataset_path`**: This is a string parameter that indicates the path to the dataset in the Huggingface dataset format. The default value is "./zh_lora_dataset". You need to ensure that the dataset at this path is correctly formatted and contains the necessary data for training.
2. **`--exp_name`**: It is a string parameter used to name the experiment. The default value is "chinese_rap_lora". This name can be used to distinguish different training experiments, and it is often used in logging and saving checkpoints to organize and identify the results of different runs.

## 4. Training Precision and Gradient Settings
1. **`--precision`**: This parameter specifies the precision of the training. It is a string with a default value of "32", which usually means 32-bit floating-point precision. Higher precision can lead to more accurate training but may also consume more memory and computational resources. You can adjust this value depending on your hardware capabilities and the requirements of your model.
2. **`--accumulate_grad_batches`**: It is an integer parameter with a default value of 1. It determines how many batches of data will be used to accumulate gradients before performing an optimization step. For example, if you set it to 4, the gradients from 4 consecutive batches will be accumulated, and then the model's weights will be updated. This can be useful in scenarios where you want to simulate larger batch sizes when your available memory does not allow for actual large batch training.
3. **`--gradient_clip_val`**: This is a floating-point parameter with a default value of 0.5. It is used to clip the gradients during the backpropagation process. Clipping the gradients helps prevent the issue of gradient explosion, where the gradients become extremely large and cause the model to become unstable. By setting a clip value, the gradients will be adjusted to be within a certain range.
4. **`--gradient_clip_algorithm`**: It is a string parameter with a default value of "norm". This parameter specifies the algorithm used for gradient clipping. The "norm" algorithm is one common method, but there may be other algorithms available depending on the implementation of the training framework.

## 5. Checkpoint and Logging Settings
1. **`--devices`**: This is an integer parameter with a default value of 1. It specifies the number of devices (such as GPUs) that will be used for training. If you have multiple GPUs available and want to use them for parallel training, you can increase this number accordingly.
2. **`--logger_dir`**: It is a string parameter with a default value of "./exps/logs/". This parameter indicates the directory where the training logs will be saved. The logs can be useful for monitoring the training progress, analyzing the performance of the model during training, and debugging any issues that may arise.
3. **`--ckpt_path`**: It is a string parameter with a default value of None. If you want to resume training from a previously saved checkpoint, you can specify the path to the checkpoint file using this parameter. If set to None, the training will start from scratch.
4. **`--checkpoint_dir`**: This is a string parameter with a default value of None. It specifies the directory where the checkpoints of the model will be saved during the training process. If set to None, checkpoints may not be saved or may be saved in a default location depending on the training framework.

## 6. Validation and Reloading Settings
1. **`--reload_dataloaders_every_n_epochs`**: It is an integer parameter with a default value of 1. It determines how often the data loaders will be reloaded during the training process. Reloading the data loaders can be useful when you want to ensure that the data is shuffled or processed differently for each epoch, especially when dealing with datasets that may change or have some specific requirements.
2. **`--every_plot_step`**: It is an integer parameter with a default value of 2000. It specifies how often some visualizations or plots (such as loss curves, accuracy plots, etc.) will be generated during the training process. For example, with a value of 2000, the plots will be updated every 2000 training steps.
3. **`--val_check_interval`**: This is an integer parameter with a default value of None. It determines how often the validation process will be performed during the training. If set to a positive integer, the model will be evaluated on the validation dataset every specified number of steps. If set to None, no regular validation checks will be performed.
4. **`--lora_config_path`**: It is a string parameter with a default value of "config/zh_rap_lora_config.json". This parameter specifies the path to the configuration file for the Lora (Low-Rank Adaptation) module. The Lora configuration file contains settings related to the Lora module, such as the rank of the low-rank matrices, the learning rate for the Lora parameters, etc. 
