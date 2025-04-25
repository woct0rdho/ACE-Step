import gradio as gr
from pathlib import Path


TAG_PLACEHOLDER = "jazz, deep, bass-driven, conscious, acid jazz, electronic, 90 bpm"
LYRIC_PLACEHOLDER = """[verse]
夜风轻轻吹过脸颊，
城市霓虹闪烁如画。
脚步追随节奏的牵挂，
思绪飘荡，梦的火花。

[chorus]
夜色中，心如电光闪耀，
节奏深，带我飞向云霄。
音符跳跃，击退烦忧来到，
在这梦里，我们自由起舞不老。

[verse]
低音如潮冷冷拉扯，
心跳同步，扣人心弦如歌。
键盘奏出泪隐藏的选择，
鼓点敲醒藏于夜里的角色。

[chorus]
夜色中，心如电光闪耀，
节奏深，带我飞向云霄。
音符跳跃，击退烦忧来到，
在这梦里，我们自由起舞不老。

[bridge]
每一滴汗珠画出轨迹，
肢体语言无声诉心意。
触碰世界不可言语的秘语，
在旋律中找回真实的自己。

[chorus]
夜色中，心如电光闪耀，
节奏深，带我飞向云霄。
音符跳跃，击退烦忧来到，
在这梦里，我们自由起舞不老。
"""


def create_output_ui():
    # For many consumer-grade GPU devices, only one batch can be run
    output_audio1 = gr.Audio(type="filepath", label="Generated Audio 1")
    # output_audio2 = gr.Audio(type="filepath", label="Generated Audio 2")
    with gr.Accordion("Input Parameters", open=False):
        input_params_json = gr.JSON(label="Input Parameters")
    # outputs = [output_audio1, output_audio2]
    outputs = [output_audio1]
    return outputs, input_params_json


def dump_func(*args):
    print(args)
    return []


def create_text2music_ui(
    gr,
    text2music_process_func,
    sample_bnt=None,
    sample_data_func=None,
):
    with gr.Row():
        with gr.Column():

            audio_duration = gr.Slider(-1, 300.0, step=0.1, value=60, label="Audio Duration", interactive=True, info="Duration of the audio in seconds. -1 means random duration (30 ~ 300).")

            prompt = gr.Textbox(lines=2, label="Tags", max_lines=4, placeholder=TAG_PLACEHOLDER)
            lyrics = gr.Textbox(lines=9, label="Lyrics", max_lines=13, placeholder=LYRIC_PLACEHOLDER)

            with gr.Accordion("Basic Settings", open=False):
                infer_step = gr.Slider(minimum=1, maximum=1000, step=1, value=60, label="Infer Steps", interactive=True)
                guidance_scale = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=15.0, label="Guidance Scale", interactive=True)
                scheduler_type = gr.Radio(["euler", "heun"], value="euler", label="Scheduler Type", elem_id="scheduler_type", )
                manual_seeds = gr.Textbox(label="manual seeds (default None)", placeholder="1,2,3,4", value=None)
            
            with gr.Accordion("Advanced Settings", open=False):
                cfg_type = gr.Radio(["cfg", "apg", "cfg_star"], value="apg", label="CFG Type", elem_id="cfg_type", )
                use_erg_tag = gr.Checkbox(label="use ERG for tag", value=True, )
                use_erg_lyric = gr.Checkbox(label="use ERG for lyric", value=True, )
                use_erg_diffusion = gr.Checkbox(label="use ERG for diffusion", value=True, )
        
                omega_scale = gr.Slider(minimum=-100.0, maximum=100.0, step=0.1, value=10.0, label="Granularity Scale", interactive=True, )

                guidance_interval = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Guidance Interval", interactive=True, )
                guidance_interval_decay = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Guidance Interval Decay", interactive=True, )
                min_guidance_scale = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=3.0, label="Min Guidance Scale", interactive=True, )
                oss_steps = gr.Textbox(label="OSS Steps", placeholder="16, 29, 52, 96, 129, 158, 172, 183, 189, 200", value=None)

            text2music_bnt = gr.Button(variant="primary")

        with gr.Column():
            outputs, input_params_json = create_output_ui()

        # sample_bnt.click(
        #     sample_data_func,
        #     inputs=[dataset_example_idx, dataset_source, source],
        #     outputs=[dataset_example_idx, prompt, lyrics, target_audio, audio_duration, input_params_json],
        # )

    text2music_bnt.click(
        fn=text2music_process_func,
        inputs=[
            audio_duration,
            prompt,
            lyrics,
            infer_step,
            guidance_scale,
            scheduler_type,
            cfg_type,
            omega_scale,
            manual_seeds,
            guidance_interval,
            guidance_interval_decay,
            min_guidance_scale,
            use_erg_tag,
            use_erg_lyric,
            use_erg_diffusion,
            oss_steps,
        ], outputs=outputs + [input_params_json]
    )


def create_main_demo_ui(
    text2music_process_func=dump_func,
    sample_data_func=dump_func,
):
    with gr.Blocks(
        title="FusicModel 1.0 DEMO",
    ) as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center;">FusicModel 1.0 DEMO</h1>
            """
        )
    
        sample_bnt = None

        with gr.Tab("text2music"):
            create_text2music_ui(
                gr=gr,
                text2music_process_func=text2music_process_func,
                sample_bnt=sample_bnt,
                sample_data_func=sample_data_func,
            )

    return demo


if __name__ == "__main__":
    demo = create_main_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
