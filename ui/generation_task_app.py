import os
import gradio as gr
from dnallm import load_config
from dnallm import load_model_and_tokenizer, DNAInference
import argparse

os.environ["GRADIO_TEMP_DIR"] = "tmp/gradio"


GLOBAL_STATE = {"engine": None, "current_model_name": None}

DEFAULT_CONFIG_PATH = "./inference_config.yaml"


def load_user_model(model_name, model_source, config_path):
    if not model_name:
        return (
            "⚠️ Please input model name or path",
            gr.update(interactive=True),
        )

    print(f"Loading model: {model_name} (Source: {model_source})...")

    try:
        # load configuration
        if not os.path.exists(config_path):
            print(
                f"Warning: Config file {config_path} not found, "
                f"use void config"
            )
            configs = {"task": {}}
        else:
            configs = load_config(config_path)

        # load model and tokenizer
        # source: huggingface, modelscope, local
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            task_config=configs.get("task", {}),
            source=model_source,
            use_mirror=True,
        )

        # inference engine
        engine = DNAInference(model=model, tokenizer=tokenizer, config=configs)

        # update state
        GLOBAL_STATE["engine"] = engine
        GLOBAL_STATE["current_model_name"] = model_name

        msg = f"✅ Model '{model_name}' ({model_source}) loaded!"
        print(msg)
        # return msg, gr.update(interactive=True)  # activate button
        return gr.update(interactive=True)

    except Exception as e:
        error_msg = f"❌ Failed to load model: {str(e)}"  # noqa: RUF010
        print(error_msg)
        return error_msg, gr.update(interactive=False)  # disable button


def generate_dna(prompt, n_tokens, n_samples, temperature, top_k, top_p):
    engine = GLOBAL_STATE["engine"]

    if engine is None:
        return "❌ Warning: please set configuration and load model first."

    if not prompt:
        return "⚠️ Warning: please input a prompt DNA sequence."

    try:
        outputs = engine.generate(
            inputs=[prompt],
            n_tokens=int(n_tokens),
            n_samples=int(n_samples),
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            batched=True,
        )

        result_text = ""
        for i, seq_data in enumerate(outputs):
            gen_seq = seq_data.get("Output", "")
            header = f"> Sample_{i + 1} | "
            header += f"Length: {len(gen_seq)}"
            result_text += f"{header}\n{gen_seq}\n\n"

        return result_text

    except Exception as e:
        return f"❌ Generation error: {str(e)}"  # noqa: RUF010


css = """
.info {font-size: 27px; !important;}
.info * {font-size: 27px; !important;}
.dna-output span {font-size: 27px; !important;}
.dna-output textarea {
    font-family: "Consolas", "Courier New", monospace !important;
    font-size: 25px !important;
}
.section-header {
    margin-bottom: 10px;
    padding-bottom: 5px;
    border-bottom: 1px solid #e5e7eb;
}
"""


def main():
    """Main function to launch the app"""
    parser = argparse.ArgumentParser(
        description="Launch DNALLM DNA Generation Task Gradio App"
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1, localhost only)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to bind the server to (default: 7860)",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the app (default: False)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (default: False)",
    )

    args = parser.parse_args()

    with gr.Blocks(
        title="DNA LLM for Generation",
        # theme=gr.themes.Ocean(),
        theme=gr.themes.Default(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.violet,
        ),
        css=css,
    ) as demo:
        # gr.Markdown("# 🧬 DNA LLM App for Generation Task")

        # set config and load model
        with gr.Group(elem_classes="info"):
            gr.Markdown("### Model Setup")
            with gr.Row():
                source_dropdown = gr.Dropdown(
                    choices=["modelscope", "huggingface", "local"],
                    value="modelscope",
                    label="Source",
                    scale=1,
                )
                model_name_input = gr.Textbox(
                    value="lgq12697/megaDNA_updated",
                    label="Model ID or Path",
                    placeholder="eg. lgq12697/megaDNA_updated",
                    scale=3,
                )
                # optional
                config_path_input = gr.Textbox(
                    value=DEFAULT_CONFIG_PATH,
                    label="Config Path",
                    visible=False,  # hide here
                )

            load_btn = gr.Button("🔄 Load Model", variant="secondary")
            # load_status = gr.Textbox(
            #     label="load state", value="wait for loading...",
            #     lines=1, interactive=False
            # )

        # generate config and input
        gr.Markdown("### Generation", elem_classes="info")
        with gr.Row(equal_height=True, elem_classes="info"):
            # prompt
            with gr.Column(scale=0.95):
                input_prompt = gr.Textbox(
                    label="Prompt",
                    lines=5,
                    value="ACGT",
                    placeholder="Input initial DNA sequence...",
                    elem_classes="dna-output",
                )

            # arguments
            with gr.Column(scale=1):
                with gr.Accordion("Arguments", open=True):
                    with gr.Row():
                        n_tokens_slider = gr.Slider(
                            10, 131072, value=1024, step=10, label="Max Tokens"
                        )
                        n_samples_slider = gr.Slider(
                            1, 10, value=1, step=1, label="Num Samples"
                        )
                    with gr.Row():
                        top_k_num = gr.Number(value=4, label="Top-K")
                        top_p_num = gr.Number(value=0.1, label="Top-P")
                        temp_slider = gr.Slider(
                            0.1,
                            2.0,
                            value=0.95,
                            step=0.05,
                            label="Temperature",
                        )

        # execute and output
        submit_btn = gr.Button(
            "🚀 Start generation",
            variant="primary",
            size="lg",
            interactive=False,
            elem_classes="info",
        )

        output_area = gr.Textbox(
            label="Outputs",
            lines=12,
            show_copy_button=True,
            elem_classes="dna-output",
        )

        # Bind function

        load_btn.click(
            fn=load_user_model,
            inputs=[model_name_input, source_dropdown, config_path_input],
            # outputs=[load_status, submit_btn]
            outputs=[submit_btn],
        )

        submit_btn.click(
            fn=generate_dna,
            inputs=[
                input_prompt,
                n_tokens_slider,
                n_samples_slider,
                temp_slider,
                top_k_num,
                top_p_num,
            ],
            outputs=output_area,
        )

    # Launch with custom settings
    print("🚀 Launching DNALLM DNA Generation Task App...")
    launch_kwargs = {
        "server_name": args.host,
        "server_port": args.port,
        "share": args.share,
        "show_error": True,
    }

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
