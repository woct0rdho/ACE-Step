"""
ACE-Step: A Step Towards Music Generation Foundation Model

https://github.com/ace-step/ACE-Step

Apache 2.0 License
"""

import os
import click

from acestep.ui.components import create_main_demo_ui
from acestep.pipeline_ace_step import ACEStepPipeline
from acestep.data_sampler import DataSampler


@click.command()
@click.option(
    "--checkpoint_path",
    type=str,
    default="",
    help="Path to the checkpoint directory. Downloads automatically if empty.",
)
@click.option(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="The server name to use for the Gradio app.",
)
@click.option(
    "--port", type=int, default=7865, help="The port to use for the Gradio app."
)
@click.option("--device_id", type=int, default=0, help="The CUDA device ID to use.")
@click.option(
    "--share",
    is_flag=True,
    default=False,
    help="Whether to create a public, shareable link for the Gradio app.",
)
@click.option(
    "--bf16",
    is_flag=True,
    default=True,
    help="Whether to use bfloat16 precision. Turn off if using MPS.",
)
@click.option(
    "--torch_compile", is_flag=True, default=False, help="Whether to use torch.compile."
)
def main(checkpoint_path, server_name, port, device_id, share, bf16, torch_compile):
    """
    Main function to launch the ACE Step pipeline demo.
    """

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    model_demo = ACEStepPipeline(
        checkpoint_dir=checkpoint_path,
        dtype="bfloat16" if bf16 else "float32",
        torch_compile=torch_compile,
    )
    data_sampler = DataSampler()

    demo = create_main_demo_ui(
        text2music_process_func=model_demo.__call__,
        sample_data_func=data_sampler.sample,
    )
    demo.launch(server_name=server_name, server_port=port, share=share)


if __name__ == "__main__":
    main()
