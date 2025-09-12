#!/usr/bin/env python3
"""
DNALLM Configuration Generator Gradio UI

This Gradio application provides a web-based interface for generating configuration files for:
- Fine-tuning tasks
- Inference tasks
- Benchmarking tasks
"""

import gradio as gr
import yaml
from pathlib import Path
import sys

# Add dnallm package to path to import dnallm modules
current_file = Path(__file__).resolve()
dnallm_path = current_file.parent.parent / "dnallm"
if str(dnallm_path) not in sys.path:
    sys.path.insert(0, str(dnallm_path))

try:
    # Try to import directly from the file to avoid relative import issues
    import importlib.util

    modeling_auto_path = dnallm_path / "models" / "modeling_auto.py"
    spec = importlib.util.spec_from_file_location(
        "modeling_auto", modeling_auto_path
    )
    modeling_auto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modeling_auto)

    MODEL_INFO = modeling_auto.MODEL_INFO
    PRETRAIN_MODEL_MAPS = modeling_auto.PRETRAIN_MODEL_MAPS
    print(
        "✅ Successfully loaded MODEL_INFO"
            "and PRETRAIN_MODEL_MAPS from modeling_auto.py"
    )

except ImportError as e:
    print(f"Warning: Could not import DNALLM models: {e}")
    print("Using mock data for demonstration...")

    # Mock data for demonstration
    MODEL_INFO = {
        "Nucleotide Transformer": {
            "title": "Nucleotide Transformer: building and evaluating"
                "robust foundation models for human genomics",
            "model_architecture": "EsmForMaskedLM",
            "model_tags": [
                "500m-human-ref",
                "500m-1000g",
                "2.5b-1000g",
                "2.5b-multi-species",
            ],
        },
        "DNABERT": {
            "title": "DNABERT: pre-trained Bidirectional Encoder Representations"
                "from Transformers model for DNA-language in genome",
            "model_architecture": "BertForMaskedLM",
            "model_tags": ["3mer", "4mer", "5mer", "6mer"],
        },
        "HyenaDNA": {
            "title": "HyenaDNA: Long-Range Genomic Sequence"
                "Modeling at Single Nucleotide Resolution",
            "model_architecture": "HyenaDNAForCausalLM",
            "model_tags": [
                "tiny-1k-seqlen",
                "small-32k-seqlen",
                "medium-160k-seqlen",
            ],
        },
    }

    PRETRAIN_MODEL_MAPS = {}


# Load model info from YAML file
def load_model_info_from_yaml():
    """Load model information from model_info.yaml file"""
    try:
        yaml_path = dnallm_path / "models" / "model_info.yaml"
        if yaml_path.exists():
            with open(yaml_path, encoding="utf-8") as f:
                model_info = yaml.safe_load(f)
            return model_info
        else:
            print(f"Warning: model_info.yaml not found at {yaml_path}")
            return None
    except Exception as e:
        print(f"Warning: Could not load model_info.yaml: {e}")
        return None


class GradioConfigGenerator:
    """Gradio-based configuration generator for DNALLM"""

    def __init__(self):
        self.config = {}
        self.config_type = None

        # Load model info from YAML
        self.yaml_model_info = load_model_info_from_yaml()

        # Task types mapping
        self.TASK_TYPES = {
            "binary_classification": "binary",
            "multiclass_classification": "multiclass",
            "multilabel_classification": "multilabel",
            "regression": "regression",
            "token_classification": "token",
            "embedding": "embedding",
            "masked_language_modeling": "mask",
            "generation": "generation",
        }

        # Model task types
        self.MODEL_TASK_TYPES = {
            "classification": "classification",
            "generation": "generation",
            "masked_language_modeling": "masked_language_modeling",
            "embedding": "embedding",
            "regression": "regression",
        }

        # Dataset task types
        self.DATASET_TASK_TYPES = {
            "binary_classification": "binary_classification",
            "multiclass_classification": "multiclass_classification",
            "regression": "regression",
            "generation": "generation",
        }

        # Available models from modeling_auto.py
        self.available_models = list(MODEL_INFO.keys())

        # Available model tags for each model
        self.model_tags = {}
        for model_name, model_info in MODEL_INFO.items():
            if "model_tags" in model_info:
                self.model_tags[model_name] = model_info["model_tags"]
            else:
                self.model_tags[model_name] = []

        # Load models from YAML file
        self._load_yaml_models()

    def _load_yaml_models(self):
        """Load models from YAML file and merge with existing models"""
        if not self.yaml_model_info:
            return

        # Add pretrained models
        if "pretrained" in self.yaml_model_info:
            for model in self.yaml_model_info["pretrained"]:
                model_name = model["name"]
                if model_name not in self.available_models:
                    self.available_models.append(model_name)

                # Store model info for later use
                if not hasattr(self, "yaml_model_details"):
                    self.yaml_model_details = {}
                self.yaml_model_details[model["model"]] = model

        # Add finetuned models (these are the most important ones)
        if "finetuned" in self.yaml_model_info:
            for model in self.yaml_model_info["finetuned"]:
                model_name = model["name"]
                if model_name not in self.available_models:
                    self.available_models.append(model_name)

                # Store model info for later use
                if not hasattr(self, "yaml_model_details"):
                    self.yaml_model_details = {}
                self.yaml_model_details[model["model"]] = model

    def get_model_info(self, model_path):
        """Get detailed information about a model from YAML"""
        if (
            hasattr(self, "yaml_model_details")
            and model_path in self.yaml_model_details
        ):
            return self.yaml_model_details[model_path]
        return None

    def auto_fill_task_config(self, model_path):
        """Auto-fill task configuration based on model path"""
        model_info = self.get_model_info(model_path)
        if not model_info or "task" not in model_info:
            return None, None, None, None, None

        task = model_info["task"]

        # Map YAML task types to our internal types
        task_type_mapping = {
            "binary": "binary_classification",
            "multiclass": "multiclass_classification",
            "regression": "regression",
        }

        task_type = task_type_mapping.get(
            task.get("task_type"), "binary_classification"
        )
        num_labels = task.get("num_labels", 2)
        label_names = task.get("label_names", "")
        threshold = task.get("threshold", 0.5)
        describe = task.get("describe", "")

        # Convert label_names list to string if needed
        if isinstance(label_names, list):
            label_names = ", ".join(label_names)

        return task_type, num_labels, label_names, threshold, describe

    def _auto_fill_finetune_config(self, model_path):
        """Auto-fill fine-tuning configuration from model path"""
        if not model_path:
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(visible=False),
            )

        result = self.auto_fill_task_config(model_path)
        if result[0] is None:
            # No model info found
            return (
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(),
                gr.update(
                    value="❌ Model not found or no task information available",
                    visible=True,
                ),
            )

        task_type, num_labels, label_names, threshold, describe = result

        # Create description markdown
        desc_md = f"**Model Description:** {describe}\n\n"
        desc_md += f"**Task Type:** {task_type}\n"
        desc_md += f"**Number of Labels:** {num_labels}\n"
        if label_names:
            desc_md += f"**Label Names:** {label_names}\n"
        desc_md += f"**Threshold:** {threshold}"

        return (
            gr.update(value=task_type),
            gr.update(value=num_labels),
            gr.update(value=threshold),
            gr.update(value=label_names),
            gr.update(value=desc_md, visible=True),
        )

    def _auto_fill_inference_config(self, model_path):
        """Auto-fill inference configuration from model path"""
        if not model_path:
            return gr.update(), gr.update(), gr.update(visible=False)

        result = self.auto_fill_task_config(model_path)
        if result[0] is None:
            # No model info found
            return (
                gr.update(),
                gr.update(),
                gr.update(
                    value="❌ Model not found or no task information available",
                    visible=True,
                ),
            )

        task_type, num_labels, label_names, threshold, describe = result

        # Create description markdown
        desc_md = f"**Model Description:** {describe}\n\n"
        desc_md += f"**Task Type:** {task_type}\n"
        desc_md += f"**Number of Labels:** {num_labels}\n"
        if label_names:
            desc_md += f"**Label Names:** {label_names}\n"
        desc_md += f"**Threshold:** {threshold}"

        return (
            gr.update(value=task_type),
            gr.update(value=num_labels),
            gr.update(value=desc_md, visible=True),
        )

    def _auto_fill_benchmark_model(self, model_path):
        """Auto-fill benchmark model information from model path"""
        if not model_path:
            return gr.update(), gr.update(), gr.update(visible=False)

        model_info = self.get_model_info(model_path)
        if not model_info:
            return (
                gr.update(),
                gr.update(),
                gr.update(value="❌ Model not found", visible=True),
            )

        # Extract model name and task type
        model_name = model_info.get("name", "")
        task = model_info.get("task", {})
        task_type = task.get("task_type", "classification")
        describe = task.get("describe", "")

        # Map task types
        task_type_mapping = {
            "binary": "classification",
            "multiclass": "classification",
            "regression": "regression",
            "mask": "masked_language_modeling",
            "generation": "generation",
        }

        mapped_task_type = task_type_mapping.get(task_type, "classification")

        # Create description markdown
        desc_md = f"**Model:** {model_name}\n\n"
        desc_md += f"**Description:** {describe}\n\n"
        desc_md += f"**Task Type:** {task_type} → {mapped_task_type}"

        return (
            gr.update(value=model_name),
            gr.update(value=mapped_task_type),
            gr.update(value=desc_md, visible=True),
        )

    def create_interface(self):
        """Create the main Gradio interface"""

        with gr.Blocks(
            title="DNALLM Configuration Generator", theme=gr.themes.Soft()
        ) as interface:
            gr.Markdown("# 🚀 DNALLM Configuration Generator")
            gr.Markdown(
                "Generate configuration files for"
                    "fine-tuning, inference, and benchmarking tasks"
            )

            # Output section - define this first so it can be passed to tabs
            with gr.Row():
                with gr.Column(scale=2):
                    output_text = gr.Textbox(
                        label="Generated Configuration",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        show_copy_button=True,
                    )

                with gr.Column(scale=1):
                    # Simplified download functionality - just show filename
                    download_info = gr.Textbox(
                        label="Download Info",
                        value="",
                        interactive=False,
                        visible=False,
                    )

                    save_btn = gr.Button(
                        "Save Configuration", variant="primary"
                    )
                    clear_btn = gr.Button("Clear", variant="secondary")

            with gr.Tabs():
                # Fine-tuning Configuration Tab
                with gr.TabItem("🔧 Fine-tuning Configuration"):
                    self._create_finetune_tab(output_text)

                # Inference Configuration Tab
                with gr.TabItem("🔮 Inference Configuration"):
                    self._create_inference_tab(output_text)

                # Benchmark Configuration Tab
                with gr.TabItem("📊 Benchmark Configuration"):
                    self._create_benchmark_tab(output_text)

            # Event handlers
            save_btn.click(
                fn=self._save_config,
                inputs=[output_text],
                outputs=[download_info],
            )

            clear_btn.click(
                fn=self._clear_output, outputs=[output_text, download_info]
            )

        return interface

    def _create_finetune_tab(self, output_text):
        """Create the fine-tuning configuration tab"""
        with gr.Group():
            gr.Markdown("## Model Selection & Auto-Configuration")

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path/Identifier",
                    placeholder="e.g., zhangtaolab/plant-dnabert-BPE-promoter",
                    interactive=True,
                )

                auto_fill_btn = gr.Button(
                    "🔍 Auto-Fill from Model", variant="secondary"
                )

            # Model description display
            model_description = gr.Markdown(value="", visible=False)

            gr.Markdown("## Task Configuration")

            with gr.Row():
                task_type = gr.Dropdown(
                    choices=list(self.TASK_TYPES.keys()),
                    label="Task Type",
                    value="binary_classification",
                    interactive=True,
                )

                num_labels = gr.Number(
                    label="Number of Labels/Classes",
                    value=2,
                    minimum=1,
                    maximum=100,
                    interactive=True,
                )

            with gr.Row():
                threshold = gr.Number(
                    label="Classification Threshold",
                    value=0.5,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.1,
                    interactive=True,
                )

                label_names = gr.Textbox(
                    label="Label Names (comma-separated)",
                    placeholder="e.g., negative, positive",
                    interactive=True,
                )

            gr.Markdown("## Fine-tuning Configuration")

            with gr.Row():
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./outputs",
                    interactive=True,
                )

                num_epochs = gr.Number(
                    label="Number of Training Epochs",
                    value=3,
                    minimum=1,
                    maximum=1000,
                    interactive=True,
                )

            with gr.Row():
                train_batch_size = gr.Number(
                    label="Training Batch Size per Device",
                    value=8,
                    minimum=1,
                    maximum=128,
                    interactive=True,
                )

                eval_batch_size = gr.Number(
                    label="Evaluation Batch Size per Device",
                    value=16,
                    minimum=1,
                    maximum=128,
                    interactive=True,
                )

            with gr.Row():
                learning_rate = gr.Number(
                    label="Learning Rate",
                    value=2e-5,
                    minimum=1e-7,
                    maximum=1e-1,
                    step=1e-6,
                    interactive=True,
                )

                weight_decay = gr.Number(
                    label="Weight Decay",
                    value=0.01,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.001,
                    interactive=True,
                )

            with gr.Row():
                warmup_ratio = gr.Number(
                    label="Warmup Ratio",
                    value=0.1,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    interactive=True,
                )

                gradient_accumulation_steps = gr.Number(
                    label="Gradient Accumulation Steps",
                    value=1,
                    minimum=1,
                    maximum=100,
                    interactive=True,
                )

            with gr.Row():
                max_grad_norm = gr.Number(
                    label="Max Gradient Norm",
                    value=1.0,
                    minimum=0.1,
                    maximum=10.0,
                    step=0.1,
                    interactive=True,
                )

                seed = gr.Number(
                    label="Random Seed",
                    value=42,
                    minimum=0,
                    maximum=999999,
                    interactive=True,
                )

            with gr.Row():
                use_bf16 = gr.Checkbox(label="Use bfloat16", value=False)
                use_fp16 = gr.Checkbox(label="Use float16", value=False)
                use_epoch_based = gr.Checkbox(
                    label="Use Epoch-based Logging/Saving", value=False
                )

            with gr.Row():
                logging_steps = gr.Number(
                    label="Logging Steps",
                    value=100,
                    minimum=1,
                    maximum=10000,
                    interactive=True,
                )

                eval_steps = gr.Number(
                    label="Evaluation Steps",
                    value=100,
                    minimum=1,
                    maximum=10000,
                    interactive=True,
                )

                save_steps = gr.Number(
                    label="Save Steps",
                    value=500,
                    minimum=1,
                    maximum=10000,
                    interactive=True,
                )

            save_total_limit = gr.Number(
                label="Save Total Limit",
                value=20,
                minimum=1,
                maximum=100,
                interactive=True,
            )

            # Generate button
            generate_finetune_btn = gr.Button(
                "Generate Fine-tuning Configuration", variant="primary"
            )

            # Event handlers
            auto_fill_btn.click(
                fn=self._auto_fill_finetune_config,
                inputs=[model_path],
                outputs=[
                    task_type,
                    num_labels,
                    threshold,
                    label_names,
                    model_description,
                ],
            )

            generate_finetune_btn.click(
                fn=self._generate_finetune_config,
                inputs=[
                    task_type,
                    num_labels,
                    threshold,
                    label_names,
                    output_dir,
                    num_epochs,
                    train_batch_size,
                    eval_batch_size,
                    learning_rate,
                    weight_decay,
                    warmup_ratio,
                    gradient_accumulation_steps,
                    max_grad_norm,
                    seed,
                    use_bf16,
                    use_fp16,
                    use_epoch_based,
                    logging_steps,
                    eval_steps,
                    save_steps,
                    save_total_limit,
                ],
                outputs=[output_text],
            )

    def _create_inference_tab(self, output_text):
        """Create the inference configuration tab"""
        with gr.Group():
            gr.Markdown("## Model Selection & Auto-Configuration")

            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path/Identifier",
                    placeholder="e.g., zhangtaolab/plant-dnabert-BPE-promoter",
                    interactive=True,
                )

                auto_fill_btn = gr.Button(
                    "🔍 Auto-Fill from Model", variant="secondary"
                )

            # Model description display
            model_description = gr.Markdown(value="", visible=False)

            gr.Markdown("## Task Configuration")

            with gr.Row():
                task_type = gr.Dropdown(
                    choices=list(self.TASK_TYPES.keys()),
                    label="Task Type",
                    value="binary_classification",
                    interactive=True,
                )

                num_labels = gr.Number(
                    label="Number of Labels/Classes",
                    value=2,
                    minimum=1,
                    maximum=100,
                    interactive=True,
                )

            gr.Markdown("## Inference Configuration")

            with gr.Row():
                batch_size = gr.Number(
                    label="Batch Size",
                    value=16,
                    minimum=1,
                    maximum=128,
                    interactive=True,
                )

                max_length = gr.Number(
                    label="Maximum Sequence Length",
                    value=512,
                    minimum=64,
                    maximum=8192,
                    interactive=True,
                )

            with gr.Row():
                device = gr.Dropdown(
                    choices=[
                        "auto",
                        "cpu",
                        "cuda",
                        "mps",
                        "rocm",
                        "xpu",
                        "tpu",
                    ],
                    label="Device",
                    value="auto",
                    interactive=True,
                )

                num_workers = gr.Number(
                    label="Number of Workers",
                    value=4,
                    minimum=0,
                    maximum=32,
                    interactive=True,
                )

            with gr.Row():
                use_fp16 = gr.Checkbox(label="Use float16", value=False)
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value="./results",
                    interactive=True,
                )

            # Generate button
            generate_inference_btn = gr.Button(
                "Generate Inference Configuration", variant="primary"
            )

            # Event handlers
            auto_fill_btn.click(
                fn=self._auto_fill_inference_config,
                inputs=[model_path],
                outputs=[task_type, num_labels, model_description],
            )

            generate_inference_btn.click(
                fn=self._generate_inference_config,
                inputs=[
                    task_type,
                    num_labels,
                    batch_size,
                    max_length,
                    device,
                    num_workers,
                    use_fp16,
                    output_dir,
                ],
                outputs=[output_text],
            )

    def _create_benchmark_tab(self, output_text):
        """Create the benchmark configuration tab"""
        with gr.Group():
            gr.Markdown("## Basic Benchmark Configuration")

            with gr.Row():
                benchmark_name = gr.Textbox(
                    label="Benchmark Name",
                    value="DNA Model Benchmark",
                    interactive=True,
                )

                benchmark_description = gr.Textbox(
                    label="Benchmark Description",
                    value="Comparing DNA language models on various tasks",
                    interactive=True,
                )

            gr.Markdown("## Models Configuration")

            with gr.Row():
                model_name = gr.Dropdown(
                    choices=self.available_models,
                    label="Model Name",
                    interactive=True,
                )

                model_tag = gr.Dropdown(
                    choices=[], label="Model Tag", interactive=True
                )

            with gr.Row():
                model_source = gr.Dropdown(
                    choices=["huggingface", "modelscope", "local"],
                    label="Model Source",
                    value="huggingface",
                    interactive=True,
                )

                model_path = gr.Textbox(
                    label="Model Path/Identifier",
                    placeholder="e.g., zhangtaolab/plant-dnabert-BPE-promoter",
                    interactive=True,
                )

                auto_fill_model_btn = gr.Button(
                    "🔍 Auto-Fill Model Info", variant="secondary"
                )

            # Model description display
            model_description = gr.Markdown(value="", visible=False)

            model_task_type = gr.Dropdown(
                choices=list(self.MODEL_TASK_TYPES.keys()),
                label="Model Task Type",
                value="classification",
                interactive=True,
            )

            with gr.Row():
                trust_remote_code = gr.Checkbox(
                    label="Trust Remote Code", value=True
                )
                torch_dtype = gr.Dropdown(
                    choices=["float32", "float16", "bfloat16"],
                    label="Data Type",
                    value="float32",
                    interactive=True,
                )

            gr.Markdown("## Datasets Configuration")

            with gr.Row():
                dataset_name = gr.Textbox(
                    label="Dataset Name", interactive=True
                )

                dataset_path = gr.Textbox(
                    label="Dataset File Path", interactive=True
                )

            with gr.Row():
                dataset_format = gr.Dropdown(
                    choices=[
                        "csv",
                        "tsv",
                        "json",
                        "fasta",
                        "arrow",
                        "parquet",
                    ],
                    label="Dataset Format",
                    value="csv",
                    interactive=True,
                )

                dataset_task = gr.Dropdown(
                    choices=list(self.DATASET_TASK_TYPES.keys()),
                    label="Dataset Task",
                    value="binary_classification",
                    interactive=True,
                )

            with gr.Row():
                text_column = gr.Textbox(
                    label="Text/Sequence Column Name",
                    value="sequence",
                    interactive=True,
                )

                label_column = gr.Textbox(
                    label="Label Column Name", value="label", interactive=True
                )

            gr.Markdown("## Evaluation Configuration")

            with gr.Row():
                eval_batch_size = gr.Number(
                    label="Evaluation Batch Size",
                    value=32,
                    minimum=1,
                    maximum=128,
                    interactive=True,
                )

                eval_max_length = gr.Number(
                    label="Maximum Sequence Length",
                    value=512,
                    minimum=64,
                    maximum=8192,
                    interactive=True,
                )

            with gr.Row():
                eval_device = gr.Dropdown(
                    choices=[
                        "auto",
                        "cpu",
                        "cuda",
                        "mps",
                        "rocm",
                        "xpu",
                        "tpu",
                    ],
                    label="Device",
                    value="auto",
                    interactive=True,
                )

                eval_seed = gr.Number(
                    label="Random Seed",
                    value=42,
                    minimum=0,
                    maximum=999999,
                    interactive=True,
                )

            with gr.Row():
                use_eval_fp16 = gr.Checkbox(label="Use float16", value=True)
                use_eval_bf16 = gr.Checkbox(label="Use bfloat16", value=False)
                deterministic = gr.Checkbox(
                    label="Enable Deterministic Mode", value=True
                )

            gr.Markdown("## Output Configuration")

            with gr.Row():
                output_format = gr.Dropdown(
                    choices=["html", "csv", "json", "pdf"],
                    label="Output Format",
                    value="html",
                    interactive=True,
                )

                output_path = gr.Textbox(
                    label="Output Directory",
                    value="benchmark_results",
                    interactive=True,
                )

            with gr.Row():
                save_predictions = gr.Checkbox(
                    label="Save Predictions", value=True
                )
                save_embeddings = gr.Checkbox(
                    label="Save Embeddings", value=False
                )
                generate_plots = gr.Checkbox(
                    label="Generate Plots", value=True
                )

            # Generate button
            generate_benchmark_btn = gr.Button(
                "Generate Benchmark Configuration", variant="primary"
            )

            # Event handlers
            model_name.change(
                fn=self._update_model_tags,
                inputs=[model_name],
                outputs=[model_tag],
            )

            auto_fill_model_btn.click(
                fn=self._auto_fill_benchmark_model,
                inputs=[model_path],
                outputs=[model_name, model_task_type, model_description],
            )

            generate_benchmark_btn.click(
                fn=self._generate_benchmark_config,
                inputs=[
                    benchmark_name,
                    benchmark_description,
                    model_name,
                    model_tag,
                    model_source,
                    model_path,
                    model_task_type,
                    trust_remote_code,
                    torch_dtype,
                    dataset_name,
                    dataset_path,
                    dataset_format,
                    dataset_task,
                    text_column,
                    label_column,
                    eval_batch_size,
                    eval_max_length,
                    eval_device,
                    eval_seed,
                    use_eval_fp16,
                    use_eval_bf16,
                    deterministic,
                    output_format,
                    output_path,
                    save_predictions,
                    save_embeddings,
                    generate_plots,
                ],
                outputs=[output_text],
            )

    def _update_model_tags(self, model_name):
        """Update model tags based on selected model"""
        if model_name in self.model_tags:
            return gr.Dropdown(choices=self.model_tags[model_name])
        return gr.Dropdown(choices=[])

    def _generate_finetune_config(
        self,
        task_type,
        num_labels,
        threshold,
        label_names,
        output_dir,
        num_epochs,
        train_batch_size,
        eval_batch_size,
        learning_rate,
        weight_decay,
        warmup_ratio,
        gradient_accumulation_steps,
        max_grad_norm,
        seed,
        use_bf16,
        use_fp16,
        use_epoch_based,
        logging_steps,
        eval_steps,
        save_steps,
        save_total_limit,
    ):
        """Generate fine-tuning configuration"""
        try:
            config = {
                "task": {
                    "task_type": self.TASK_TYPES[task_type],
                    "num_labels": num_labels,
                },
                "finetune": {
                    "output_dir": output_dir,
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": train_batch_size,
                    "per_device_eval_batch_size": eval_batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_ratio": warmup_ratio,
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "max_grad_norm": max_grad_norm,
                    "seed": seed,
                    "bf16": use_bf16,
                    "fp16": use_fp16,
                },
                "inference": {
                    "batch_size": 16,
                    "max_length": 512,
                    "num_workers": 4,
                    "device": "auto",
                    "use_fp16": False,
                    "output_dir": "./results",
                },
            }

            # Add task-specific configurations
            if task_type in [
                "binary_classification",
                "multilabel_classification",
            ]:
                config["task"]["threshold"] = threshold

            if label_names:  # label_names
                config["task"]["label_names"] = [
                    name.strip() for name in label_names.split(",")
                ]

            # Add logging and saving strategies
            if use_epoch_based:
                config["finetune"].update({
                    "logging_strategy": "epoch",
                    "eval_strategy": "epoch",
                    "save_strategy": "epoch",
                })
            else:
                config["finetune"].update({
                    "logging_strategy": "steps",
                    "logging_steps": logging_steps,
                    "eval_strategy": "steps",
                    "eval_steps": eval_steps,
                    "save_strategy": "steps",
                    "save_steps": save_steps,
                })

            config["finetune"]["save_total_limit"] = save_total_limit

            # Add advanced settings
            if gradient_accumulation_steps > 1:
                config["finetune"]["gradient_accumulation_steps"] = (
                    gradient_accumulation_steps
                )

            if max_grad_norm != 1.0:
                config["finetune"]["max_grad_norm"] = max_grad_norm

            config["finetune"]["lr_scheduler_type"] = "linear"

            self.config = config
            return yaml.dump(
                config, default_flow_style=False, allow_unicode=True, indent=2
            )

        except Exception as e:
            return f"Error generating configuration: {e!s}"

    def _generate_inference_config(
        self,
        task_type,
        num_labels,
        batch_size,
        max_length,
        device,
        num_workers,
        use_fp16,
        output_dir,
    ):
        """Generate inference configuration"""
        try:
            config = {
                "task": {
                    "task_type": self.TASK_TYPES[task_type],
                    "num_labels": num_labels,
                },
                "inference": {
                    "batch_size": batch_size,
                    "max_length": max_length,
                    "device": device,
                    "num_workers": num_workers,
                    "use_fp16": use_fp16,
                    "output_dir": output_dir,
                },
            }

            self.config = config
            return yaml.dump(
                config, default_flow_style=False, allow_unicode=True, indent=2
            )

        except Exception as e:
            return f"Error generating configuration: {e!s}"

    def _generate_benchmark_config(
        self,
        benchmark_name,
        benchmark_description,
        model_name,
        model_tag,
        model_source,
        model_path,
        model_task_type,
        trust_remote_code,
        torch_dtype,
        dataset_name,
        dataset_path,
        dataset_format,
        dataset_task,
        text_column,
        label_column,
        eval_batch_size,
        eval_max_length,
        eval_device,
        eval_seed,
        use_eval_fp16,
        use_eval_bf16,
        deterministic,
        output_format,
        output_path,
        save_predictions,
        save_embeddings,
        generate_plots,
    ):
        """Generate benchmark configuration"""
        try:
            config = {
                "benchmark": {
                    "name": benchmark_name,
                    "description": benchmark_description,
                },
                "models": [
                    {
                        "name": model_name,
                        "tag": model_tag,
                        "source": model_source,
                        "path": model_path,
                        "task_type": model_task_type,
                        "trust_remote_code": trust_remote_code,
                        "torch_dtype": torch_dtype,
                    }
                ],
                "datasets": [
                    {
                        "name": dataset_name,
                        "path": dataset_path,
                        "format": dataset_format,
                        "task": dataset_task,
                        "text_column": text_column,
                        "label_column": label_column,
                    }
                ],
                "evaluation": {
                    "batch_size": eval_batch_size,
                    "max_length": eval_max_length,
                    "device": eval_device,
                    "seed": eval_seed,
                    "use_fp16": use_eval_fp16,
                    "use_bf16": use_eval_bf16,
                    "deterministic": deterministic,
                },
                "output": {
                    "format": output_format,
                    "path": output_path,
                    "save_predictions": save_predictions,
                    "save_embeddings": save_embeddings,
                    "generate_plots": generate_plots,
                },
            }

            # Add default metrics
            config["metrics"] = ["accuracy", "f1_score", "precision", "recall"]

            self.config = config
            return yaml.dump(
                config, default_flow_style=False, allow_unicode=True, indent=2
            )

        except Exception as e:
            return f"Error generating configuration: {e!s}"

    def _save_config(self, config_text):
        """Save configuration to file and return file info"""
        if not config_text or config_text.startswith("Error"):
            return gr.update(value="", visible=False)

        try:
            # Parse the configuration
            config = yaml.safe_load(config_text)

            # Determine file name based on configuration type
            if "finetune" in config:
                filename = "finetune_config.yaml"
            elif "inference" in config and "inference" not in config:
                filename = "inference_config.yaml"
            else:
                filename = "benchmark_config.yaml"

            # Save to file
            with open(filename, "w", encoding="utf-8") as f:
                yaml.dump(
                    config,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    indent=2,
                )

            success_msg = f"✅ Configuration saved to: {filename}"
            return gr.update(value=success_msg, visible=True)

        except Exception as e:
            error_msg = f"❌ Error saving configuration: {e!s}"
            return gr.update(value=error_msg, visible=True)

    def _clear_output(self):
        """Clear the output and hide download button"""
        return gr.update(value=""), gr.update(visible=False)


def main():
    """Main function to launch the Gradio app"""
    generator = GradioConfigGenerator()
    interface = generator.create_interface()

    # Launch the app
    interface.launch(
        server_name="127.0.0.1", server_port=7860, share=False, show_error=True
    )


if __name__ == "__main__":
    main()
