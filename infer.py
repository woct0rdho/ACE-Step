import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--bf16", type=bool, default=True)
parser.add_argument("--torch_compile", type=bool, default=False)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--output_path", type=str, default=None)
args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

from pipeline_ace_step import ACEStepPipeline
from data_sampler import DataSampler


def sample_data(json_data):
    return (
            json_data["audio_duration"],
            json_data["prompt"],
            json_data["lyrics"],
            json_data["infer_step"],
            json_data["guidance_scale"],
            json_data["scheduler_type"],
            json_data["cfg_type"],
            json_data["omega_scale"],
            ", ".join(map(str, json_data["actual_seeds"])),
            json_data["guidance_interval"],
            json_data["guidance_interval_decay"],
            json_data["min_guidance_scale"],
            json_data["use_erg_tag"],
            json_data["use_erg_lyric"],
            json_data["use_erg_diffusion"],
            ", ".join(map(str, json_data["oss_steps"])),
            json_data["guidance_scale_text"] if "guidance_scale_text" in json_data else 0.0,
            json_data["guidance_scale_lyric"] if "guidance_scale_lyric" in json_data else 0.0,
            )

def main(args):

    model_demo = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32",
        torch_compile=args.torch_compile
    )
    print(model_demo)

    data_sampler = DataSampler()

    json_data = data_sampler.sample()
    json_data = sample_data(json_data)
    print(json_data)


    audio_duration,\
    prompt, \
    lyrics,\
    infer_step, \
    guidance_scale,\
    scheduler_type, \
    cfg_type, \
    omega_scale, \
    manual_seeds, \
    guidance_interval, \
    guidance_interval_decay, \
    min_guidance_scale, \
    use_erg_tag, \
    use_erg_lyric, \
    use_erg_diffusion, \
    oss_steps, \
    guidance_scale_text, \
    guidance_scale_lyric = json_data


    model_demo(audio_duration, \
            prompt, \
            lyrics,\
            infer_step, \
            guidance_scale,\
            scheduler_type, \
            cfg_type, \
            omega_scale, \
            manual_seeds, \
            guidance_interval, \
            guidance_interval_decay, \
            min_guidance_scale, \
            use_erg_tag, \
            use_erg_lyric, \
            use_erg_diffusion, \
            oss_steps, \
            guidance_scale_text, \
            guidance_scale_lyric,
            save_path=args.output_path)

if __name__ == "__main__":
    main(args)
