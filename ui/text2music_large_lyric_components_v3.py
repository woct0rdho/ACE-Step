import gradio as gr
from pathlib import Path
import json
from collections import OrderedDict, Counter
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from language_segmentation import LangSegment

MAX_GENERATE_LEN = 60


SUPPORT_LANGUAGES = [
    "af", "sq", "am", "ar", "an", "hy", "az", "ba", "eu", "be", "bn", "bs", "bg", "my", "ca", "zh", "cs", "da", "nl", "en", "eo", "et", "fi", "fr", "gd", "ka", "de", "el", "gn", "gu", "hi", "hu", "io", "id", "ia", "it", "ja", "kk", "km", "ko", "ku", "la", "lt", "lb", "mk", "mt", "nb", "no", "or", "fa", "pl", "pt", "ro", "ru", "sa", "sr", "sd", "sk", "sl", "es", "sw", "sv", "tl", "ta", "tt", "th", "tr", "tk", "uk", "vi", "cy", "is", "ga", "gl", "se", "yue"
]


langseg = LangSegment()

langseg.setfilters([
    'af', 'am', 'an', 'ar', 'as', 'az', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'cs', 'cy', 'da', 'de', 'dz', 'el',
    'en', 'eo', 'es', 'et', 'eu', 'fa', 'fi', 'fo', 'fr', 'ga', 'gl', 'gu', 'he', 'hi', 'hr', 'ht', 'hu', 'hy',
    'id', 'is', 'it', 'ja', 'jv', 'ka', 'kk', 'km', 'kn', 'ko', 'ku', 'ky', 'la', 'lb', 'lo', 'lt', 'lv', 'mg',
    'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'nb', 'ne', 'nl', 'nn', 'no', 'oc', 'or', 'pa', 'pl', 'ps', 'pt', 'qu',
    'ro', 'ru', 'rw', 'se', 'si', 'sk', 'sl', 'sq', 'sr', 'sv', 'sw', 'ta', 'te', 'th', 'tl', 'tr', 'ug', 'uk',
    'ur', 'vi', 'vo', 'wa', 'xh', 'zh', 'zu'
])

keyscale_idx_mapping = OrderedDict({
    "C major": 1,
    "C# major": 2,
    "D major": 3,
    "Eb major": 4,
    "E major": 5,
    "F major": 6,
    "F# major": 7,
    "G major": 8,
    "Ab major": 9,
    "A major": 10,
    "Bb major": 11,
    "B major": 12,
    "A minor": 13,
    "Bb minor": 14,
    "B minor": 15,
    "C minor": 16,
    "C# minor": 17,
    "D minor": 18,
    "Eb minor": 19,
    "E minor": 20,
    "F minor": 21,
    "F# minor": 22,
    "G minor": 23,
    "Ab minor": 24
})


def get_checkpoint_paths(checkpoint_path):
    # 获取指定目录中的所有checkpoint文件路径
    directory = Path(checkpoint_path).parent
    checkpoints = [str(p) for p in directory.glob("*.ckpt")]
    print(checkpoints)
    return checkpoints


def create_list_checkpoint_path_ui(checkpoint_path):
    with gr.Column():
        gr.Markdown("Checkpoint Selection")
        with gr.Group():
            with gr.Row(equal_height=True):
                with gr.Column(scale=9):
                    selected_checkpoint = gr.Dropdown(
                        choices=get_checkpoint_paths(checkpoint_path),
                        label="Select Model",
                        interactive=True,
                        value=checkpoint_path,
                    )
                with gr.Column(scale=1):
                    refresh_button = gr.Button("Refresh Checkpoints", elem_id="refresh_button", variant="primary")
                    refresh_button.click(
                        fn=lambda: gr.update(choices=get_checkpoint_paths(checkpoint_path)),
                        inputs=None,
                        outputs=[selected_checkpoint]
                    )
    return selected_checkpoint


def create_keyscale_bpm_time_signature_input_ui(options=["auto", "manual"]):
    gr.Markdown("### Time and Keyscale Control")
    with gr.Group():
        results = [
            ["keyscale", 0],
            ["bpm", 0],
            ["timesignature", 0],
            ["is_music_start", 0],
            ["is_music_end", 0],
        ]
        keyscale_bpm_time_signature_input = gr.List(visible=False, elem_id="keyscale_bpm_time_signature_input", value=results)
        audio_duration = gr.Slider(10, 600, step=1, value=MAX_GENERATE_LEN, label="Audio Duration", interactive=True)
        with gr.Row():
            is_music_start_input = gr.Radio(["auto", "start", "not_start"], value="auto", label="Is Music Start", elem_id="is_music_start_input")
            is_music_end_input = gr.Radio(["auto", "end", "not_end"], value="auto", label="Is Music End", elem_id="is_music_end_input")

        def when_is_music_start_input_change(
            is_music_start_input,
        ):
            nonlocal results
            if is_music_start_input == "auto":
                is_music_start = 0
            elif is_music_start_input == "start":
                is_music_start = 1
            else:
                is_music_start = 2
            results[3][1] = is_music_start
            return gr.update(elem_id="keyscale_bpm_time_signature_input", value=results)

        is_music_start_input.change(
            when_is_music_start_input_change,
            inputs=[is_music_start_input],
            outputs=[keyscale_bpm_time_signature_input]
        )

        def when_is_music_end_input_change(
            is_music_end_input,
        ):
            nonlocal results
            if is_music_end_input == "auto":
                is_music_end = 0
            elif is_music_end_input == "end":
                is_music_end = 1
            else:
                is_music_end = 2
            results[4][1] = is_music_end
            return gr.update(elem_id="keyscale_bpm_time_signature_input", value=results)

        is_music_end_input.change(
            when_is_music_end_input_change,
            inputs=[is_music_end_input],
            outputs=[keyscale_bpm_time_signature_input]
        )

        with gr.Row():
            keyscale_control = gr.Radio(options, value="auto", label="Keyscale", elem_id="keyscale_control")
            bpm_control = gr.Radio(options, value="auto", label="BPM", elem_id="bpm_control")
            time_signature_control = gr.Radio(options, value="auto", label="Time Signature", elem_id="time_signature_control")

        keyscale_input = gr.Dropdown(list(keyscale_idx_mapping.keys()), label="Keyscale", info="the keyscale of the music", visible=False, elem_id="keyscale_input")

        def when_keyscale_change(
            keyscale_input,
            keyscale_control,
        ):
            nonlocal results
            keyscale = keyscale_input
            if keyscale_control == "auto":
                keyscale = 0
            results[0][1] = keyscale
            return [gr.update(elem_id="keyscale_bpm_time_signature_input", value=results), gr.update(elem_id="keyscale_input", visible=(keyscale_control == "manual"))]

        keyscale_input.change(
            when_keyscale_change,
            inputs=[keyscale_input, keyscale_control],
            outputs=[keyscale_bpm_time_signature_input, keyscale_input]
        )
        keyscale_control.change(
            fn=when_keyscale_change,
            inputs=[keyscale_input, keyscale_control],
            outputs=[keyscale_bpm_time_signature_input, keyscale_input]
        )

        bpm_input = gr.Slider(30, 200, step=1, value=120, label="BPM", info="the beats per minute of the music", visible=False, interactive=True, elem_id="bpm_input")

        def when_bmp_change(
            bpm_input,
            bpm_control,
        ):
            nonlocal results
            bpm = bpm_input
            if bpm_control == "auto":
                bpm = 0
            results[1][1] = bpm
            updates = [gr.update(elem_id="keyscale_bpm_time_signature_input", value=results), gr.update(elem_id="bpm_input", visible=(bpm_control == "manual"))]
            return updates

        bpm_control.change(
            fn=when_bmp_change,
            inputs=[bpm_input, bpm_control],
            outputs=[keyscale_bpm_time_signature_input, bpm_input]
        )

        bpm_input.change(
            when_bmp_change,
            inputs=[bpm_input, bpm_control],
            outputs=[keyscale_bpm_time_signature_input, bpm_input]
        )

        time_signature_input = gr.Slider(1, 12, step=1, value=4, label="Time Signature", info="the time signature of the music", visible=False, interactive=True, elem_id="time_signature_input")

        def when_time_signature_change(
            time_signature_input,
            time_signature_control,
        ):
            nonlocal results
            time_signature = time_signature_input
            if time_signature_control == "auto":
                time_signature = 0
            results[2][1] = time_signature
            return [gr.update(elem_id="keyscale_bpm_time_signature_input", value=results), gr.update(elem_id="time_signature_input", visible=(time_signature_control == "manual"))]

        time_signature_input.change(
            when_time_signature_change,
            inputs=[time_signature_input, time_signature_control],
            outputs=[keyscale_bpm_time_signature_input, time_signature_input]
        )
        time_signature_control.change(
            fn=when_time_signature_change,
            inputs=[time_signature_input, time_signature_control],
            outputs=[keyscale_bpm_time_signature_input, time_signature_input]
        )

    return [audio_duration, keyscale_bpm_time_signature_input]


def detect_language(lyrics: str) -> list:
    lyrics = lyrics.strip()
    if not lyrics:
        return gr.update(value="en")
    langs = langseg.getTexts(lyrics)
    lang_counter = Counter()
    for lang in langs:
        lang_counter[lang["lang"]] += len(lang["text"])
    lang = lang_counter.most_common(1)[0][0]
    return lang


def create_output_ui():
    target_audio = gr.Audio(type="filepath", label="Target Audio")
    output_audio1 = gr.Audio(type="filepath", label="Generated Audio 1")
    output_audio2 = gr.Audio(type="filepath", label="Generated Audio 2")
    input_params_json = gr.JSON(label="Input Parameters")
    outputs = [output_audio1, output_audio2]
    return outputs, target_audio, input_params_json


def dump_func(*args):
    print(args)
    return []


def create_main_demo_ui(
    checkpoint_path="checkpoints/aceflow3_0311/1d_epoch=16-step=140k.ckpt",
    text2music_process_func=dump_func,
    sample_data_func=dump_func,
):
    with gr.Blocks(
        title="AceFlow 3.0 DEMO (3.5B)",
    ) as demo:
        gr.Markdown(
            """
            <h1 style="text-align: center;">AceFlow 3.0 DEMO</h1>
            """
        )
        selected_checkpoint = create_list_checkpoint_path_ui(checkpoint_path)

        gr.Markdown("Dataset Filter")
        with gr.Group():
            with gr.Row(equal_height=True):
                language = gr.Dropdown(["en", "zh"], label="Language", value="en", elem_id="language")
                dataset_example_idx = gr.Number(
                    value=-1,
                    label="Dataset Example Index",
                    interactive=True
                )
                sample_bnt = gr.Button(value="Sample Data", elem_id="sample_bnt", variant="primary")

        with gr.Row():
            with gr.Column():
                audio_duration = gr.Slider(10, 600, step=1, value=MAX_GENERATE_LEN, label="Audio Duration", interactive=True)

                prompt = gr.Textbox(lines=2, label="Tags", max_lines=4)
                lyrics = gr.Textbox(lines=9, label="Lyrics", max_lines=9)

                scheduler_type = gr.Radio(["euler", "heun"], value="euler", label="Scheduler Type", elem_id="scheduler_type")
                cfg_type = gr.Radio(["cfg", "apg"], value="apg", label="CFG Type", elem_id="cfg_type")
                infer_step = gr.Slider(minimum=1, maximum=1000, step=1, value=60, label="Infer Steps", interactive=True)
                guidance_scale = gr.Slider(minimum=0.0, maximum=200.0, step=0.1, value=15.0, label="Guidance Scale", interactive=True)
                omega_scale = gr.Slider(minimum=-100.0, maximum=100.0, step=0.1, value=10.0, label="Granularity Scale", interactive=True)
                manual_seeds = gr.Textbox(label="manual seeds (default None)", placeholder="1,2,3,4", value=None)

                text2music_bnt = gr.Button(variant="primary")
            with gr.Column():
                outputs, target_audio, input_params_json = create_output_ui()

        sample_bnt.click(
            sample_data_func,
            inputs=[dataset_example_idx, audio_duration],
            outputs=[target_audio, prompt, lyrics, input_params_json],
        )
        text2music_bnt.click(
            fn=text2music_process_func,
            inputs=[
                audio_duration,
                prompt,
                lyrics,
                input_params_json,
                selected_checkpoint,
                scheduler_type,
                cfg_type,
                infer_step,
                guidance_scale,
                omega_scale,
                manual_seeds,
            ], outputs=outputs + [input_params_json]
        )

    return demo


if __name__ == "__main__":
    demo = create_main_demo_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
