import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint_path", type=str, default="")
parser.add_argument("--port", type=int, default=7865)
parser.add_argument("--device_id", type=int, default=0)
parser.add_argument("--share", action='store_true', default=False)
parser.add_argument("--bf16", action='store_true', default=False)

args = parser.parse_args()

import os

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)


from ui.components import create_main_demo_ui
from pipeline_ace_step import ACEStepPipeline
from data_sampler import DataSampler


def main(args):

    model_demo = ACEStepPipeline(
        checkpoint_dir=args.checkpoint_path,
        dtype="bfloat16" if args.bf16 else "float32"
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
    )
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share
    )


if __name__ == "__main__":
    main(args)
