import gradio as gr


TAG_PLACEHOLDER = "funk, pop, soul, rock, melodic, guitar, drums, bass, keyboard, percussion, 105 BPM, energetic, upbeat, groovy, vibrant, dynamic"
LYRIC_PLACEHOLDER = """[verse]
Neon lights they flicker bright
City hums in dead of night
Rhythms pulse through concrete veins
Lost in echoes of refrains

[verse]
Bassline groovin' in my chest
Heartbeats match the city's zest
Electric whispers fill the air
Synthesized dreams everywhere

[chorus]
Turn it up and let it flow
Feel the fire let it grow
In this rhythm we belong
Hear the night sing out our song

[verse]
Guitar strings they start to weep
Wake the soul from silent sleep
Every note a story told
In this night we’re bold and gold

[bridge]
Voices blend in harmony
Lost in pure cacophony
Timeless echoes timeless cries
Soulful shouts beneath the skies

[verse]
Keyboard dances on the keys
Melodies on evening breeze
Catch the tune and hold it tight
In this moment we take flight
"""


def create_output_ui(task_name="Text2Music"):
    # For many consumer-grade GPU devices, only one batch can be run
    output_audio1 = gr.Audio(type="filepath", label=f"{task_name} Generated Audio 1")
    # output_audio2 = gr.Audio(type="filepath", label="Generated Audio 2")
    with gr.Accordion(f"{task_name} Parameters", open=False):
        input_params_json = gr.JSON(label=f"{task_name} Parameters")
    # outputs = [output_audio1, output_audio2]
    outputs = [output_audio1]
    return outputs, input_params_json


def dump_func(*args):
    print(args)
    return []


def create_text2music_ui(
    gr,
    text2music_process_func,
    sample_data_func=None,
):
    with gr.Row():
        with gr.Column():
            
            with gr.Row(equal_height=True):
                audio_duration = gr.Slider(-1, 240.0, step=0.00001, value=180, label="Audio Duration", interactive=True, info="-1 means random duration (30 ~ 240).", scale=9)
                sample_bnt = gr.Button("Sample", variant="primary", scale=1)
            
            prompt = gr.Textbox(lines=2, label="Tags", max_lines=4, placeholder=TAG_PLACEHOLDER, info="Support tags, descriptions, and scene. Use commas to separate different tags.")
            lyrics = gr.Textbox(lines=9, label="Lyrics", max_lines=13, placeholder=LYRIC_PLACEHOLDER, info="Support lyric structure tags like [verse], [chorus], and [bridge] to separate different parts of the lyrics.\nUse [instrumental] or [inst] to generate instrumental music. Not support genre structure tag in lyrics")

            with gr.Accordion("Basic Settings", open=True):
                infer_step = gr.Slider(minimum=1, maximum=1000, step=1, value=60, label="Infer Steps", interactive=True)
                guidance_scale = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=15.0, label="Guidance Scale", interactive=True, info="When guidance_scale_lyric > 1 and guidance_scale_text > 1, the guidance scale will not be applied.")
                guidance_scale_text = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=5.0, label="Guidance Scale Text", interactive=True, info="Guidance scale for text condition. It can only apply to cfg. set guidance_scale_text=5.0, guidance_scale_lyric=1.5 for start")
                guidance_scale_lyric = gr.Slider(minimum=0.0, maximum=10.0, step=0.1, value=1.5, label="Guidance Scale Lyric", interactive=True)

                manual_seeds = gr.Textbox(label="manual seeds (default None)", placeholder="1,2,3,4", value=None, info="Seed for the generation")
            
            with gr.Accordion("Advanced Settings", open=False):
                scheduler_type = gr.Radio(["euler", "heun"], value="euler", label="Scheduler Type", elem_id="scheduler_type", info="Scheduler type for the generation. euler is recommended. heun will take more time.")
                cfg_type = gr.Radio(["cfg", "apg", "cfg_star"], value="apg", label="CFG Type", elem_id="cfg_type", info="CFG type for the generation. apg is recommended. cfg and cfg_star are almost the same.")
                use_erg_tag = gr.Checkbox(label="use ERG for tag", value=True, info="Use Entropy Rectifying Guidance for tag. It will multiple a temperature to the attention to make a weaker tag condition and make better diversity.")
                use_erg_lyric = gr.Checkbox(label="use ERG for lyric", value=True, info="The same but apply to lyric encoder's attention.")
                use_erg_diffusion = gr.Checkbox(label="use ERG for diffusion", value=True, info="The same but apply to diffusion model's attention.")
        
                omega_scale = gr.Slider(minimum=-100.0, maximum=100.0, step=0.1, value=10.0, label="Granularity Scale", interactive=True, info="Granularity scale for the generation. Higher values can reduce artifacts")

                guidance_interval = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.5, label="Guidance Interval", interactive=True, info="Guidance interval for the generation. 0.5 means only apply guidance in the middle steps (0.25 * infer_steps to 0.75 * infer_steps)")
                guidance_interval_decay = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.0, label="Guidance Interval Decay", interactive=True, info="Guidance interval decay for the generation. Guidance scale will decay from guidance_scale to min_guidance_scale in the interval. 0.0 means no decay.")
                min_guidance_scale = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=3.0, label="Min Guidance Scale", interactive=True, info="Min guidance scale for guidance interval decay's end scale")
                oss_steps = gr.Textbox(label="OSS Steps", placeholder="16, 29, 52, 96, 129, 158, 172, 183, 189, 200", value=None, info="Optimal Steps for the generation. But not test well")

            text2music_bnt = gr.Button(variant="primary")

        with gr.Column():
            outputs, input_params_json = create_output_ui()
            with gr.Tab("retake"):
                retake_variance = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, value=0.2, label="variance", info="Variance for the retake. 0.0 means no variance. 1.0 means full variance.")
                retake_seeds = gr.Textbox(label="retake seeds (default None)", placeholder="", value=None, info="Seed for the retake.")
                retake_bnt = gr.Button(variant="primary")
                retake_outputs, retake_input_params_json = create_output_ui("Retake")
                
                def retake_process_func(json_data, retake_variance, retake_seeds):
                    return text2music_process_func(
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
                        retake_seeds=retake_seeds,
                        retake_variance=retake_variance,
                        task="retake",
                    )
                
                retake_bnt.click(
                    fn=retake_process_func,
                    inputs=[
                        input_params_json,
                        retake_variance,
                        retake_seeds,
                    ],
                    outputs=retake_outputs + [retake_input_params_json],
                )
            with gr.Tab("repainting"):
                pass
            with gr.Tab("edit"):
                pass

        def sample_data():
            json_data = sample_data_func()
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
    
        sample_bnt.click(
            sample_data,
            outputs=[
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
                guidance_scale_text,
                guidance_scale_lyric,
            ],
        )

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
            guidance_scale_text,
            guidance_scale_lyric,
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

        with gr.Tab("text2music"):
            create_text2music_ui(
                gr=gr,
                text2music_process_func=text2music_process_func,
                sample_data_func=sample_data_func,
            )
    return demo


if __name__ == "__main__":
    demo = create_main_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
