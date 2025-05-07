import os
import shutil
import tempfile
import torchaudio
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.ui.components import TAG_DEFAULT, LYRIC_DEFAULT # For default prompt/lyrics
import torch

class AudioOutput:
    """
    Represents the output of an audio generation process.
    """
    def __init__(self, audio_file_path: str, sample_rate: int, input_params: dict):
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"Generated audio file not found: {audio_file_path}")
        self._audio_file_path = audio_file_path
        self._sample_rate = sample_rate
        self.input_params = input_params # Store for reference

    def save_wav(self, output_path: str):
        """
        Saves the generated audio to the specified path.
        Ensures the output path has a .wav extension.

        Args:
            output_path (str): The desired path to save the audio file (e.g., "my_song.wav").
        """
        if not output_path.lower().endswith(".wav"):
            output_path += ".wav"
        
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        shutil.copyfile(self._audio_file_path, output_path)
        print(f"Audio saved to {output_path}")

    def get_raw_data(self) -> tuple[torch.Tensor, int]:
        """
        Loads and returns the raw audio waveform and sample rate.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the audio waveform (Tensor) and sample rate (int).
        """
        waveform, sr = torchaudio.load(self._audio_file_path)
        return waveform, sr

    def __del__(self):
        # Clean up the temporary file when this object is deleted
        if hasattr(self, '_audio_file_path') and self._audio_file_path and \
           self._audio_file_path.startswith(tempfile.gettempdir()):
            try:
                os.remove(self._audio_file_path)
                # print(f"Cleaned up temporary file: {self._audio_file_path}")
            except OSError:
                # print(f"Error cleaning up temporary file: {self._audio_file_path}")
                pass


class ACEStep:
    """
    A simple API for the ACE-Step music generation model.
    """
    def __init__(
        self,
        checkpoint_path: str = None,
        device_id: int = 0,
        dtype: str = "bfloat16", # "bfloat16" or "float32"
        torch_compile: bool = False,
        persistent_storage_path: str = None,
    ):
        """
        Initializes the ACEStep model.

        Args:
            checkpoint_path (str, optional): Path to the checkpoint directory. 
                                            If None, uses default or persistent_storage_path.
            device_id (int, optional): Device ID to use (e.g., 0 for CUDA). Defaults to 0.
            dtype (str, optional): Data type ("bfloat16" or "float32"). Defaults to "bfloat16".
            torch_compile (bool, optional): Whether to use torch compile. Defaults to False.
            persistent_storage_path (str, optional): Path for persistent storage of models if checkpoint_dir is None.
        """
        self.pipeline = ACEStepPipeline(
            checkpoint_dir=checkpoint_path,
            device_id=device_id,
            dtype=dtype,
            torch_compile=torch_compile,
            persistent_storage_path=persistent_storage_path,
        )
        # Ensure models are loaded if not already (idempotent call within pipeline)
        if not self.pipeline.loaded:
            print("Loading ACE-Step models...")
            self.pipeline.load_checkpoint(self.pipeline.checkpoint_dir)
            print("Models loaded.")
            
        self.default_sample_rate = 48000 # From pipeline_ace_step.py latents2audio

    def infer(
        self,
        prompt: str = TAG_DEFAULT,
        lyrics: str = LYRIC_DEFAULT,
        audio_duration: float = 30.0, # Default to 30s as a common case
        infer_step: int = 60,
        guidance_scale: float = 15.0,
        scheduler_type: str = "euler", # "euler" or "heun"
        cfg_type: str = "apg", # "cfg", "apg", "cfg_star"
        omega_scale: float = 10.0, # Renamed from int to float based on UI
        manual_seeds: str = None, # e.g., "1,2,3,4" or "123"
        guidance_interval: float = 0.5,
        guidance_interval_decay: float = 0.0,
        min_guidance_scale: float = 3.0,
        use_erg_tag: bool = True,
        use_erg_lyric: bool = True,
        use_erg_diffusion: bool = True,
        oss_steps: str = None, # e.g., "16,29,52"
        guidance_scale_text: float = 0.0,
        guidance_scale_lyric: float = 0.0,
        # Parameters for more advanced tasks (retake, repaint, edit, extend)
        task: str = "text2music",
        retake_seeds: str = None,
        retake_variance: float = 0.5,
        repaint_start: float = 0.0, # Renamed from int to float
        repaint_end: float = 0.0, # Renamed from int to float
        src_audio_path: str = None,
        edit_target_prompt: str = None,
        edit_target_lyrics: str = None,
        edit_n_min: float = 0.0,
        edit_n_max: float = 1.0,
        # edit_n_avg: int = 1, # Not exposed in UI components, default in pipeline
        # batch_size: int = 1, # API will simplify to batch_size 1 for now
        # debug: bool = False, # Internal debugging
    ) -> AudioOutput:
        """
        Generates audio based on the provided parameters.

        Args:
            prompt (str): Text prompt (tags, description).
            lyrics (str): Lyrics with structure tags like [verse], [chorus].
            audio_duration (float): Desired audio duration in seconds.
            infer_step (int): Number of inference steps.
            guidance_scale (float): Classifier-free guidance scale.
            scheduler_type (str): Scheduler type ("euler" or "heun").
            cfg_type (str): CFG type ("cfg", "apg", "cfg_star").
            omega_scale (float): Granularity scale.
            manual_seeds (str): Comma-separated seeds or a single seed.
            guidance_interval (float): Guidance interval (0.0 to 1.0).
            guidance_interval_decay (float): Guidance interval decay.
            min_guidance_scale (float): Minimum guidance scale for decay.
            use_erg_tag (bool): Use Entropy Rectifying Guidance for tag.
            use_erg_lyric (bool): Use ERG for lyric.
            use_erg_diffusion (bool): Use ERG for diffusion.
            oss_steps (str): Optimal steps string (e.g., "16,29,52").
            guidance_scale_text (float): Guidance scale for text condition.
            guidance_scale_lyric (float): Guidance scale for lyric condition.
            task (str): Generation task type ("text2music", "retake", "repaint", "edit", "extend").
            retake_seeds (str): Seeds for retake task.
            retake_variance (float): Variance for retake task.
            repaint_start (float): Start time for repaint task.
            repaint_end (float): End time for repaint task.
            src_audio_path (str): Path to source audio for repaint, edit, extend.
            edit_target_prompt (str): Target prompt for edit task.
            edit_target_lyrics (str): Target lyrics for edit task.
            edit_n_min (float): Edit n_min for edit task.
            edit_n_max (float): Edit n_max for edit task.

        Returns:
            AudioOutput: An object containing the generated audio.
        """
        
        # The pipeline saves the file itself. We'll save it to a temp location.
        # The pipeline appends "_0" for batch index 0.
        temp_dir = tempfile.mkdtemp(prefix="acestep_api_")
        temp_file_basename = "generated_audio"
        temp_file_stem = os.path.join(temp_dir, temp_file_basename)

        # For simplicity, this API handles batch_size=1.
        # The pipeline's __call__ expects list for manual_seeds and retake_seeds if they are strings.
        # The pipeline handles string-to-list conversion for manual_seeds internally
        # but it's safer to be explicit or ensure pipeline handles it for all seed types.
        # From pipeline: `if isinstance(manual_seeds, str): ... seeds = list(map(int, manual_seeds.split(",")))`
        # This is handled correctly by the pipeline.
        
        # The pipeline's __call__ expects oss_steps as a string, it converts to list of ints.

        results = self.pipeline(
            audio_duration=audio_duration,
            prompt=prompt,
            lyrics=lyrics,
            infer_step=infer_step,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            omega_scale=omega_scale, # pipeline expects int, but UI has float, let pipeline handle
            manual_seeds=manual_seeds,
            guidance_interval=guidance_interval,
            guidance_interval_decay=guidance_interval_decay,
            min_guidance_scale=min_guidance_scale,
            use_erg_tag=use_erg_tag,
            use_erg_lyric=use_erg_lyric,
            use_erg_diffusion=use_erg_diffusion,
            oss_steps=oss_steps,
            guidance_scale_text=guidance_scale_text,
            guidance_scale_lyric=guidance_scale_lyric,
            
            task=task,
            retake_seeds=retake_seeds,
            retake_variance=retake_variance,
            repaint_start=repaint_start, # pipeline expects int, but UI has float
            repaint_end=repaint_end,   # pipeline expects int, but UI has float
            src_audio_path=src_audio_path,
            edit_target_prompt=edit_target_prompt,
            edit_target_lyrics=edit_target_lyrics,
            edit_n_min=edit_n_min,
            edit_n_max=edit_n_max,
            # edit_n_avg will use pipeline default
            
            save_path=temp_file_stem, # Pipeline will append _0.format
            format="wav",             # Output WAV for AudioOutput consumption
            batch_size=1,             # Simplified for this API
            # debug=False
        )

        if not results["output_paths"]:
            # Cleanup temp dir if something went wrong and no file was created
            shutil.rmtree(temp_dir)
            raise RuntimeError("Audio generation failed, no output paths returned.")

        # Assuming batch_size=1, so only one output path
        generated_audio_path = results["output_paths"][0]
        
        # Move the generated file out of the subdirectory created by save_path stem
        # The pipeline creates something like /tmp/acestep_api_xxxx/generated_audio_0.wav
        # We want AudioOutput to manage this file directly.
        # However, the current temp_file_stem logic for save_path in pipeline might already
        # handle this correctly. If save_path is `/tmp/dir/file_stem`, it saves as `/tmp/dir/file_stem_0.wav`.
        # So generated_audio_path should be correct.

        # The AudioOutput object will be responsible for cleaning up this temp file.
        # To achieve this, we need to make sure the _audio_file_path in AudioOutput
        # is this exact temporary path.
        
        # We need to pass the full temporary path to AudioOutput
        # and AudioOutput's __del__ will remove it.
        # No, AudioOutput's __del__ should remove its own _audio_file_path.
        # The temp_dir itself can be an issue if not cleaned.
        # Let's have AudioOutput take ownership of the file and its original temp_dir.

        final_temp_path = os.path.join(tempfile.gettempdir(), f"acestep_{os.path.basename(generated_audio_path)}")
        shutil.move(generated_audio_path, final_temp_path)
        
        # Clean up the temporary directory that pipeline might have used if save_path was a dir stem
        # If generated_audio_path was inside temp_dir, temp_dir is its dirname.
        original_output_dir = os.path.dirname(generated_audio_path)
        if original_output_dir == temp_dir and os.path.exists(temp_dir):
             # Check if directory is empty before removing, to be safe
            if not os.listdir(temp_dir):
                os.rmdir(temp_dir)
            # else:
                # print(f"Warning: temp dir {temp_dir} not empty after moving file.")

        return AudioOutput(
            audio_file_path=final_temp_path, 
            sample_rate=self.default_sample_rate, # TODO: Get from pipeline if it varies
            input_params=results["input_params_json"]
        )

if __name__ == "__main__":
    print("Starting ACE-Step API Demo...")

    # This demo assumes checkpoints are downloaded or available in the default location.
    # You might need to set checkpoint_path if they are elsewhere.
    # persistent_storage_path can be used if you want models downloaded to a specific shared location.
    # e.g., model = ACEStep(persistent_storage_path="/path/to/my/models/acestep")
    
    try:
        model = ACEStep() # Uses default checkpoint path logic
        print("ACEStep model initialized.")

        print("\nGenerating short audio (text2music)...")
        # Using a very short duration and fewer steps for a quick demo
        audio_out_1 = model.infer(
            prompt="upbeat pop, catchy melody, female singer",
            lyrics="[verse]\nSun is shining bright today\nFeeling happy, come what may",
            audio_duration=5.0, # 5 seconds
            infer_step=20        # Fewer steps for speed
        )
        print("Audio generation complete.")

        output_filename_1 = "demo_song_1.wav"
        audio_out_1.save_wav(output_filename_1)
        
        waveform, sr = audio_out_1.get_raw_data()
        print(f"Raw data: Waveform shape: {waveform.shape}, Sample rate: {sr}")
        print(f"Input params used for generation: {audio_out_1.input_params}")

        # --- Example for 'edit' task (requires a source audio) ---
        # First, generate a base track
        print("\nGenerating a base track for edit demo (10s)...")
        base_audio_for_edit = model.infer(
            prompt="acoustic folk, gentle guitar, male vocal",
            lyrics="[verse]\nOld road calling, wind is low\n[chorus]\nHomeward bound, I softly go",
            audio_duration=10.0,
            infer_step=30
        )
        base_audio_path = "base_for_edit.wav"
        base_audio_for_edit.save_wav(base_audio_path)
        print(f"Base track saved to {base_audio_path}")

        # Now, edit the lyrics/prompt
        print(f"\nEditing the track '{base_audio_path}'...")
        edited_audio_out = model.infer(
            task="edit",
            src_audio_path=base_audio_path, # Path to the audio generated above
            # Original prompt/lyrics are taken from src_audio_path's metadata by pipeline if not re-specified
            # but we provide the *original* ones that correspond to src_audio_path here for clarity
            prompt="acoustic folk, gentle guitar, male vocal", # Original prompt for src_audio_path
            lyrics="[verse]\nOld road calling, wind is low\n[chorus]\nHomeward bound, I softly go", # Original lyrics for src_audio_path
            
            edit_target_prompt="acoustic folk, gentle guitar, male vocal, with violin", # New prompt
            edit_target_lyrics="[verse]\nNew path calling, sun is high\n[chorus]\nJourney onward, to the sky", # New lyrics
            audio_duration=10.0, # Must match src_audio_path duration for edit
            infer_step=30,
            edit_n_min=0.6, # Controls how much of the original is kept vs. re-generated
            edit_n_max=0.8
        )
        edited_audio_filename = "edited_song.wav"
        edited_audio_out.save_wav(edited_audio_filename)
        print(f"Edited audio saved to {edited_audio_filename}")
        
        # Clean up the base audio file used for edit demo
        # if os.path.exists(base_audio_path):
        #     os.remove(base_audio_path)

    except Exception as e:
        print(f"An error occurred during the demo: {e}")
        import traceback
        traceback.print_exc()

    finally:
        print("\nDemo finished.")
        # Note: Temporary files created by AudioOutput are cleaned up by its __del__ method.
        # Any other demo files (demo_song_1.wav, base_for_edit.wav, edited_song.wav) remain.
        # For a real application, you might want to manage these demo outputs more explicitly.
        # Example cleanup:
        # for f in ["demo_song_1.wav", "base_for_edit.wav", "edited_song.wav"]:
        #     if os.path.exists(f):
        #         os.remove(f)

