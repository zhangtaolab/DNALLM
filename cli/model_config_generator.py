#!/usr/bin/env python3
"""
DNALLM Configuration Generator CLI Tool

This tool provides an interactive way to generate configuration files for:
- Fine-tuning tasks
- Inference tasks
- Benchmarking tasks

For detailed TrainingArguments documentation, please refer to:
https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
"""

import click
import yaml
import os
from pathlib import Path
from typing import Any, ClassVar


class ConfigGenerator:
    """Interactive configuration generator for DNALLM with model template support

    For detailed TrainingArguments documentation, please refer to:
    https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
    """

    # Define consistent task types across all methods
    TASK_TYPES: ClassVar[dict[str, str]] = {
        "1": "binary_classification",
        "2": "multiclass_classification",
        "3": "multilabel_classification",
        "4": "regression",
        "5": "token_classification",
        "6": "embedding",
        "7": "masked_language_modeling",
        "8": "generation",
    }
    TASK_ALIAS: ClassVar[dict[str, str]] = {
        "binary_classification": "binary",
        "multiclass_classification": "multiclass",
        "multilabel_classification": "multilabel",
        "regression": "regression",
        "token_classification": "token",
        "embedding": "embedding",
        "masked_language_modeling": "mask",
        "generation": "generation",
    }

    MODEL_TASK_TYPES: ClassVar[dict[str, str]] = {
        "1": "classification",
        "2": "generation",
        "3": "masked_language_modeling",
        "4": "embedding",
        "5": "regression",
    }

    DATASET_TASK_TYPES: ClassVar[dict[str, str]] = {
        "1": "binary_classification",
        "2": "multiclass_classification",
        "3": "regression",
        "4": "generation",
    }

    def __init__(self):
        self.config = {}
        self.config_type = None
        self.model_info = {}
        self.available_models = []
        self.model_templates = {}

        # Load model information from YAML file
        self._load_model_info()

    def _load_model_info(self):
        """Load model information from model_info.yaml file"""
        try:
            # Try to find model_info.yaml in the models directory
            current_file = Path(__file__).resolve()
            yaml_path = (
                current_file.parent.parent
                / "dnallm"
                / "models"
                / "model_info.yaml"
            )

            if yaml_path.exists():
                with open(yaml_path, encoding="utf-8") as f:
                    self.model_info = yaml.safe_load(f)

                # Extract available models and templates
                self._extract_model_templates()
                click.echo(
                    f"✅ Loaded {len(self.available_models)}"
                        "model templates from {yaml_path}"
                )
            else:
                click.echo(f"⚠️  Model info file not found at {yaml_path}")
                click.echo("   Continuing with manual configuration mode...")

        except Exception as e:
            click.echo(f"⚠️  Warning: Could not load model info: {e}")
            click.echo("   Continuing with manual configuration mode...")

    def _extract_model_templates(self):
        """Extract model templates from loaded YAML data"""
        self.available_models = []
        self.model_templates = {}

        # Process pretrained models
        if "pretrained" in self.model_info:
            for model in self.model_info["pretrained"]:
                model_name = model["name"]
                model_path = model["model"]
                self.available_models.append(model_name)
                self.model_templates[model_name] = {
                    "path": model_path,
                    "type": "pretrained",
                    "task": model.get("task", {}),
                }

        # Process finetuned models (these are more useful for configuration)
        if "finetuned" in self.model_info:
            for model in self.model_info["finetuned"]:
                model_name = model["name"]
                model_path = model["model"]
                self.available_models.append(model_name)
                self.model_templates[model_name] = {
                    "path": model_path,
                    "type": "finetuned",
                    "task": model.get("task", {}),
                }

        # Sort models: zhangtaolab models first, then others
        self._sort_models_by_priority()

    def _sort_models_by_priority(self):
        """Sort models with zhangtaolab models first"""

        def get_model_priority(model_name):
            model_info = self.model_templates.get(model_name, {})
            model_path = model_info.get("path", "")

            # zhangtaolab models get highest priority
            if "zhangtaolab" in model_path:
                return 0
            # finetuned models get medium priority
            elif model_info.get("type") == "finetuned":
                return 1
            # pretrained models get lowest priority
            else:
                return 2

        # Sort by priority
        self.available_models.sort(key=get_model_priority)

    def _search_models(self, query: str) -> list[str]:
        """Search models by name or description"""
        if not query:
            return self.available_models[:10]  # Return first 10 if no query

        query = query.lower()
        results = []

        for model_name in self.available_models:
            if query in model_name.lower():
                results.append(model_name)
                continue

            # Search in model path
            model_info = self.model_templates.get(model_name, {})
            if "path" in model_info and query in model_info["path"].lower():
                results.append(model_name)
                continue

            # Search in task description
            task = model_info.get("task", {})
            if "describe" in task and query in task["describe"].lower():
                results.append(model_name)

        return results[:20]  # Limit results

    def _select_model_from_templates(
        self, config_type: str | None = None
    ) -> dict[str, Any] | None:
        """Let user select a model from"
            "available templates based on configuration type"""
        if not self.available_models:
            return None

        # Filter models based on configuration type
        models_to_show = self._filter_models_by_config_type(config_type)
        if models_to_show is None:
            return None

        # Display models and get user choice
        self._display_model_overview(models_to_show)

        while True:
            choice = click.prompt("Enter your choice (1-4)", type=int)

            result = self._handle_model_choice(choice, models_to_show)
            if result == "continue":
                continue  # Try again
            elif isinstance(result, str):
                # This should not happen based on our logic, but handle it for safety
                continue
            else:
                # result is either dict[str, Any] or None (custom model path)
                return result

    def _filter_models_by_config_type(
        self, config_type: str | None
    ) -> list[str] | None:
        """Filter available models based on configuration type"""
        if config_type == "finetune":
            return self._filter_pretrained_models()
        elif config_type == "inference":
            return self._filter_finetuned_models()
        elif config_type == "benchmark":
            click.echo(
                "   Showing all model types (suitable for benchmarking)"
            )
            return self.available_models
        else:
            click.echo("   Showing all available models")
            return self.available_models

    def _filter_pretrained_models(self) -> list[str] | None:
        """Filter and return pretrained models for fine-tuning"""
        filtered_models = [
            model_name
            for model_name in self.available_models
            if self.model_templates.get(model_name, {}).get("type")
            == "pretrained"
        ]

        if not filtered_models:
            click.echo("❌ No pretrained models found for fine-tuning")
            return None

        click.echo(
            "   Showing pretrained models only (base models for fine-tuning)"
        )
        return filtered_models

    def _filter_finetuned_models(self) -> list[str] | None:
        """Filter and return finetuned models for inference"""
        filtered_models = [
            model_name
            for model_name in self.available_models
            if self.model_templates.get(model_name, {}).get("type")
            == "finetuned"
        ]

        if not filtered_models:
            click.echo("❌ No finetuned models found for inference")
            return None

        click.echo("   Showing finetuned models only (ready for prediction)")
        return filtered_models

    def _display_model_overview(self, models_to_show: list[str]) -> None:
        """Display model overview and options"""
        click.echo(
            f"\n🔍 Available model templates ({len(models_to_show)} models):"
        )

        # Show first 10 models
        click.echo("\n📋 First 10 models (zhangtaolab models prioritized):")
        self._display_model_list(models_to_show[:10])

        if len(models_to_show) > 10:
            click.echo(f"  ... and {len(models_to_show) - 10} more")

        self._display_choice_options()

    def _display_model_list(self, models: list[str]) -> None:
        """Display a list of models with their information"""
        for i, model_name in enumerate(models, 1):
            model_info = self.model_templates.get(model_name, {})
            task_type = model_info.get("task", {}).get("task_type", "unknown")
            model_path = model_info.get("path", "N/A")
            click.echo(f"  {i}. {model_name}")
            click.echo(f"     Path: {model_path}")
            click.echo(f"     Task: {task_type}")
            click.echo()

    def _display_choice_options(self) -> None:
        """Display available choice options"""
        click.echo("\nOptions:")
        click.echo("1. Select from list above (1-10)")
        click.echo("2. Search models")
        click.echo("3. Enter custom model path")
        click.echo("4. Show all models")

    def _handle_model_choice(
        self, choice: int, models_to_show: list[str]
    ) -> dict[str, Any] | str | None:
        """Handle user's model selection choice

        Returns:
            dict: Selected model template
            None: Custom model path selected
            str: 'continue' to indicate invalid choice and retry
        """
        if choice == 1:
            return self._handle_list_selection(models_to_show)
        elif choice == 2:
            return self._handle_search_selection()
        elif choice == 3:
            return None  # Custom model path
        elif choice == 4:
            return self._handle_show_all_selection(models_to_show)
        else:
            click.echo("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
            return "continue"  # Special value to continue loop

    def _handle_list_selection(
        self, models_to_show: list[str]
    ) -> dict[str, Any] | str:
        """Handle selection from the displayed list"""
        max_choice = min(10, len(models_to_show))
        model_choice = click.prompt(
            f"Enter model number (1-{max_choice})", type=int
        )

        if 1 <= model_choice <= max_choice:
            selected_model = models_to_show[model_choice - 1]
            return self.model_templates.get(selected_model, {})
        else:
            click.echo("❌ Invalid choice. Please try again.")
            return "continue"

    def _handle_search_selection(self) -> dict[str, Any] | str:
        """Handle model search and selection"""
        search_query = click.prompt("Enter search term")
        search_results = self._search_models(search_query)

        if not search_results:
            click.echo("❌ No models found matching your search")
            return "continue"

        click.echo(f"\n🔍 Search results ({len(search_results)} models):")
        self._display_model_list(search_results)

        if len(search_results) == 1:
            selected_model = search_results[0]
            click.echo(f"✅ Auto-selected: {selected_model}")
            return self.model_templates.get(selected_model, {})

        model_choice = click.prompt(
            f"Enter model number (1-{len(search_results)})", type=int
        )
        if 1 <= model_choice <= len(search_results):
            selected_model = search_results[model_choice - 1]
            return self.model_templates.get(selected_model, {})
        else:
            click.echo("❌ Invalid choice. Please try again.")
            return "continue"

    def _handle_show_all_selection(
        self, models_to_show: list[str]
    ) -> dict[str, Any] | str:
        """Handle showing all models and selection"""
        click.echo(
            f"\n📋 All {len(models_to_show)}"
                "models (zhangtaolab models prioritized):"
        )
        self._display_model_list(models_to_show)

        model_choice = click.prompt(
            f"Enter model number (1-{len(models_to_show)})", type=int
        )
        if 1 <= model_choice <= len(models_to_show):
            selected_model = models_to_show[model_choice - 1]
            return self.model_templates.get(selected_model, {})
        else:
            click.echo("❌ Invalid choice. Please try again.")
            return "continue"

    def _auto_fill_from_template(
        self, model_template: dict[str, Any]
    ) -> dict[str, Any]:
        """Auto-fill configuration from model"
            "template with comprehensive defaults"""
        auto_config = {}
        task = model_template.get("task", {})

        # Auto-fill task configuration
        if task:
            task_type = task.get("task_type", "binary")
            auto_config["task_type"] = task_type

            # Map YAML task types to our internal types
            task_type_mapping = {
                "binary": "binary_classification",
                "multiclass": "multiclass_classification",
                "regression": "regression",
                "mask": "masked_language_modeling",
                "generation": "generation",
            }

            auto_config["mapped_task_type"] = task_type_mapping.get(
                task_type, "binary_classification"
            )
            auto_config["num_labels"] = task.get("num_labels", 2)
            auto_config["label_names"] = task.get("label_names", [])
            auto_config["threshold"] = task.get("threshold", 0.5)
            auto_config["description"] = task.get("describe", "")

        # Auto-fill model path
        auto_config["model_path"] = model_template.get("path", "")
        auto_config["model_type"] = model_template.get("type", "unknown")

        # Add comprehensive default parameters based on configuration type
        auto_config["defaults"] = self._get_default_parameters()

        return auto_config

    def _get_default_parameters(self) -> dict[str, Any]:
        """Get comprehensive default parameters"
            "for different configuration types"""
        defaults = {}

        # Common task defaults
        defaults["task"] = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["negative", "positive"],
            "threshold": 0.5,
        }

        # Fine-tuning defaults (based on finetune_config.yaml)
        # For detailed arguments, please refer to https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
        defaults["finetune"] = {
            "output_dir": "./outputs",
            "num_train_epochs": 3,
            "per_device_train_batch_size": 16,  # Updated to match example
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "max_steps": -1,
            "logging_strategy": "steps",
            "logging_steps": 1000,  # Updated to match example
            "eval_strategy": "steps",
            "eval_steps": 1000,  # Updated to match example
            "save_strategy": "steps",
            "save_steps": 1000,  # Updated to match example
            "save_total_limit": 20,
            "save_safetensors": True,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "max_grad_norm": 1.0,
            "warmup_ratio": 0.1,
            "lr_scheduler_type": "linear",
            "lr_scheduler_kwargs": {},  # Always include empty dict for lr_scheduler_kwargs
            "seed": 42,
            "bf16": False,
            "fp16": False,
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "report_to": "tensorboard",
        }

        # Inference defaults (based on inference_config.yaml)
        defaults["inference"] = {
            "batch_size": 16,
            "max_length": 512,
            "device": "auto",
            "num_workers": 4,
            "use_fp16": False,
            "output_dir": "./results",
        }

        # Benchmark defaults (based on benchmark_config.yaml)
        defaults["benchmark"] = {
            "name": "Model Benchmark",
            "description": "Comparing DNA models on various tasks",
            "metrics": ["accuracy", "f1_score", "precision", "recall"],
            "plot": {"format": "pdf"},
            "inference": {
                "batch_size": 16,
                "max_length": 512,
                "device": "auto",
                "num_workers": 4,
                "use_fp16": False,
                "output_dir": "benchmark_results",
            },
        }

        return defaults

    def _confirm_auto_fill(self, auto_config: dict[str, Any]) -> bool:
        """Show auto-filled configuration and ask for confirmation"""
        click.echo("\n🎯 Auto-filled configuration from template:")
        click.echo("=" * 40)

        if "description" in auto_config:
            click.echo(f"📝 Description: {auto_config['description']}")

        click.echo(
            f"🔧 Task Type: {auto_config.get('mapped_task_type', 'Unknown')}"
        )
        click.echo(
            f"🏷️  Number of Labels: {auto_config.get('num_labels', 'Unknown')}"
        )

        if auto_config.get("label_names"):
            if isinstance(auto_config["label_names"], list):
                labels_str = ", ".join(auto_config["label_names"])
            else:
                labels_str = str(auto_config["label_names"])
            click.echo(f"📋 Label Names: {labels_str}")

        if "threshold" in auto_config:
            click.echo(f"⚖️  Threshold: {auto_config['threshold']}")

        click.echo(
            f"🤖 Model Path: {auto_config.get('model_path', 'Unknown')}"
        )
        click.echo(
            f"📚 Model Type: {auto_config.get('model_type', 'Unknown')}"
        )

        # Show available default parameters
        if "defaults" in auto_config:
            click.echo("\n📋 Available default parameters:")
            click.echo("-" * 30)
            click.echo(
                "🔧 Task defaults:"
                    "task_type, num_labels, label_names, threshold"
            )
            click.echo(
                "🎯 Fine-tuning defaults:"
                    "learning_rate, batch_size, epochs, etc."
            )
            click.echo(
                "🔮 Inference defaults: batch_size, max_length, device, etc."
            )
            click.echo("📊 Benchmark defaults: metrics, plot format, etc.")

        return click.confirm("\n✅ Use this auto-filled configuration?")

    def generate_config(self) -> dict[str, Any]:
        """Main method to generate configuration interactively"""
        click.echo("🚀 Welcome to DNALLM Configuration Generator!")
        click.echo("=" * 50)

        # Choose configuration type
        self._choose_config_type()

        # Generate configuration based on type
        if self.config_type == "finetune":
            self._generate_finetune_config()
        elif self.config_type == "inference":
            self._generate_inference_config()
        elif self.config_type == "benchmark":
            self._generate_benchmark_config()

        return self.config

    def _choose_config_type(self):
        """Let user choose the type of configuration to generate"""
        click.echo("\n📋 Choose configuration type:")
        click.echo("1. Fine-tuning configuration (for training models)")
        click.echo("2. Inference configuration (for model prediction)")
        click.echo("3. Benchmark configuration (for model comparison)")

        while True:
            choice = click.prompt("Enter your choice (1-3)", type=int)
            if choice == 1:
                self.config_type = "finetune"
                break
            elif choice == 2:
                self.config_type = "inference"
                break
            elif choice == 3:
                self.config_type = "benchmark"
                break
            else:
                click.echo("❌ Invalid choice. Please enter 1, 2, or 3.")

    def _generate_finetune_config(self):
        """Generate fine-tuning configuration"""
        click.echo("\n🔧 Generating Fine-tuning Configuration...")

        # Model selection with template support
        click.echo("\n🤖 Model Selection:")
        click.echo("1. Use model template (recommended)")
        click.echo("2. Enter custom model path")

        model_choice = click.prompt("Choose option (1-2)", type=int)

        auto_config = None
        if model_choice == 1:
            # Use model template - for finetune, show pretrained models (base for fine-tuning)
            model_template = self._select_model_from_templates("finetune")
            if model_template:
                auto_config = self._auto_fill_from_template(model_template)
                if auto_config and self._confirm_auto_fill(auto_config):
                    click.echo(
                        "✅ Using auto-filled configuration from template"
                    )
                else:
                    click.echo(
                        "❌ Auto-fill cancelled, proceeding with manual input"
                    )
                    auto_config = None

        # Task configuration (use auto-filled if available)
        if auto_config:
            task_config = self._configure_task_with_template(auto_config)
        else:
            task_config = self._configure_task()

        # Fine-tuning configuration
        finetune_config = self._configure_finetune(auto_config)

        # Store the configuration
        self.config["task"] = task_config
        self.config["finetune"] = finetune_config

        click.echo("✅ Fine-tuning configuration generated successfully!")

    def _generate_inference_config(self):
        """Generate inference configuration"""
        click.echo("\n🔮 Generating Inference Configuration...")

        # Model selection with template support
        click.echo("\n🤖 Model Selection:")
        click.echo("1. Use model template (recommended)")
        click.echo("2. Enter custom model path")

        model_choice = click.prompt("Choose option (1-2)", type=int)

        auto_config = None
        if model_choice == 1:
            # Use model template - for inference, show finetuned models (ready for prediction)
            model_template = self._select_model_from_templates("inference")
            if model_template:
                auto_config = self._auto_fill_from_template(model_template)
                if auto_config and self._confirm_auto_fill(auto_config):
                    click.echo(
                        "✅ Using auto-filled configuration from template"
                    )
                else:
                    click.echo(
                        "❌ Auto-fill cancelled, proceeding with manual input"
                    )
                    auto_config = None

        # Task configuration (use auto-filled if available)
        if auto_config:
            task_config = self._configure_task_with_template(auto_config)
        else:
            task_config = self._configure_task()

        # Inference configuration
        inference_config = self._configure_inference(auto_config)

        # Store the configuration
        self.config["task"] = task_config
        self.config["inference"] = inference_config

        click.echo("✅ Inference configuration generated successfully!")

    def _generate_benchmark_config(self):
        """Generate benchmark configuration"""
        click.echo("\n📊 Generating Benchmark Configuration...")

        # Basic benchmark info
        self.config["benchmark"] = self._configure_benchmark_basic()

        # Model selection with template support for benchmarking
        click.echo("\n🤖 Model Selection for Benchmarking:")
        click.echo("1. Use model template (recommended)")
        click.echo("2. Enter custom model path")

        model_choice = click.prompt("Choose option (1-2)", type=int)

        auto_config = None
        if model_choice == 1:
            # Use model template - for benchmark, show all model types
            model_template = self._select_model_from_templates("benchmark")
            if model_template:
                auto_config = self._auto_fill_from_template(model_template)
                if auto_config and self._confirm_auto_fill(auto_config):
                    click.echo(
                        "✅ Using auto-filled configuration from template"
                    )
                else:
                    click.echo(
                        "❌ Auto-fill cancelled, proceeding with manual input"
                    )
                    auto_config = None

        # Models configuration (use auto-filled if available)
        if auto_config:
            models_config = self._configure_models_with_template(auto_config)
        else:
            models_config = self._configure_models()

        # Datasets configuration
        datasets_config = self._configure_datasets()

        # Metrics configuration
        self.config["metrics"] = self._configure_metrics()

        # Evaluation configuration
        evaluation_config = self._configure_evaluation()

        # Output configuration
        output_config = self._configure_output()

        # Store the configuration
        self.config["models"] = models_config
        self.config["datasets"] = datasets_config
        self.config["evaluation"] = evaluation_config
        self.config["output"] = output_config

        click.echo("✅ Benchmark configuration generated successfully!")

    def _configure_task(self) -> dict[str, Any]:
        """Configure task settings"""
        click.echo("\n📝 Task Configuration:")

        # Task type
        click.echo("Available task types:")
        for key, task_type in self.TASK_TYPES.items():
            click.echo(f"  {key}. {task_type}")

        while True:
            choice = click.prompt("Choose task type", type=str)
            if choice in self.TASK_TYPES:
                task_type = self.TASK_TYPES[choice]
                task_alias = self.TASK_ALIAS[task_type]
                break
            click.echo("❌ Invalid choice. Please try again.")

        task_config: dict[str, Any] = {"task_type": task_alias}

        # Number of labels (for classification tasks)
        if "classification" in task_type:
            num_labels = click.prompt(
                "Number of labels/classes",
                type=int,
                default=2 if "binary" in task_type else 3,
            )
            task_config["num_labels"] = num_labels

            # Label names
            if click.confirm("Do you want to specify label names?"):
                label_names = []
                for i in range(num_labels):
                    label_name = click.prompt(f"Label {i} name")
                    label_names.append(label_name)
                task_config["label_names"] = label_names

        # Threshold for binary/multilabel
        if "binary" in task_type or "multilabel" in task_type:
            threshold = click.prompt(
                "Classification threshold", type=float, default=0.5
            )
            task_config["threshold"] = threshold

        return task_config

    def _configure_task_with_template(
        self, auto_config: dict[str, Any]
    ) -> dict[str, Any]:
        """Configure task settings using"
            "template information with smart defaults"""
        click.echo("\n📝 Task Configuration (from template):")

        # Get defaults
        defaults = auto_config.get("defaults", {}).get("task", {})

        # Use template values with fallback to defaults
        task_type = auto_config.get(
            "mapped_task_type",
            defaults.get("task_type", "binary_classification"),
        )
        num_labels = auto_config.get(
            "num_labels", defaults.get("num_labels", 2)
        )
        label_names = auto_config.get(
            "label_names",
            defaults.get("label_names", ["negative", "positive"]),
        )
        threshold = auto_config.get(
            "threshold", defaults.get("threshold", 0.5)
        )

        click.echo(f"✅ Task Type: {task_type}")
        click.echo(f"✅ Number of Labels: {num_labels}")

        if label_names:
            if isinstance(label_names, list):
                labels_str = ", ".join(label_names)
            else:
                labels_str = str(label_names)
            click.echo(f"✅ Label Names: {labels_str}")

        if "classification" in task_type:
            click.echo(f"✅ Threshold: {threshold}")

        # Show available defaults
        click.echo("\n📋 Smart defaults available:")
        click.echo(
            f"• Task: {defaults.get('task_type', 'binary')}"
                "with {defaults.get('num_labels', 2)} labels"
        )
        click.echo(f"   • Threshold: {defaults.get('threshold', 0.5)}")
        click.echo(
            f"• Labels: {',"
                "'.join(defaults.get('label_names', ['negative', 'positive']))}"
        )

        # Ask if user wants to modify any values
        if click.confirm("\n🔧 Modify any of these values?"):
            return self._configure_task()  # Fall back to manual configuration
        else:
            # Create task config from template
            task_config: dict[str, Any] = {
                "task_type": self.TASK_ALIAS.get(task_type, task_type)
            }

            if "classification" in task_type:
                task_config["num_labels"] = num_labels
                if label_names:
                    if isinstance(label_names, list):
                        task_config["label_names"] = label_names
                    else:
                        task_config["label_names"] = [
                            name.strip()
                            for name in str(label_names).split(",")
                        ]

                if "binary" in task_type or "multilabel" in task_type:
                    task_config["threshold"] = threshold

            return task_config

    def _configure_finetune(
        self, auto_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Configure fine-tuning settings with smart defaults

        For detailed arguments, please refer to https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
        """
        click.echo("\n🎯 Fine-tuning Configuration:")

        # Get defaults if available
        defaults = {}
        if auto_config and "defaults" in auto_config:
            defaults = auto_config["defaults"].get("finetune", {})

        finetune_config = {}

        # Basic settings with smart defaults
        finetune_config["output_dir"] = click.prompt(
            "Output directory",
            type=str,
            default=defaults.get("output_dir", "./outputs"),
        )

        finetune_config["num_train_epochs"] = click.prompt(
            "Number of training epochs",
            type=int,
            default=defaults.get("num_train_epochs", 3),
        )

        finetune_config["per_device_train_batch_size"] = click.prompt(
            "Training batch size per device",
            type=int,
            default=defaults.get("per_device_train_batch_size", 16),
        )

        finetune_config["per_device_eval_batch_size"] = click.prompt(
            "Evaluation batch size per device",
            type=int,
            default=defaults.get("per_device_eval_batch_size", 16),
        )

        # Learning rate and optimization with smart defaults
        finetune_config["learning_rate"] = click.prompt(
            "Learning rate",
            type=float,
            default=defaults.get("learning_rate", 2e-5),
        )

        finetune_config["weight_decay"] = click.prompt(
            "Weight decay",
            type=float,
            default=defaults.get("weight_decay", 0.01),
        )

        # Adam optimizer parameters
        finetune_config["adam_beta1"] = click.prompt(
            "Adam beta1", type=float, default=defaults.get("adam_beta1", 0.9)
        )

        finetune_config["adam_beta2"] = click.prompt(
            "Adam beta2", type=float, default=defaults.get("adam_beta2", 0.999)
        )

        finetune_config["adam_epsilon"] = click.prompt(
            "Adam epsilon",
            type=float,
            default=defaults.get("adam_epsilon", 1e-8),
        )

        finetune_config["warmup_ratio"] = click.prompt(
            "Warmup ratio",
            type=float,
            default=defaults.get("warmup_ratio", 0.1),
        )

        # Max steps configuration
        finetune_config["max_steps"] = click.prompt(
            "Maximum training steps (-1 for epoch-based)",
            type=int,
            default=defaults.get("max_steps", -1),
        )

        # Learning rate scheduler configuration (always included)
        # For detailed arguments, please refer to https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments
        click.echo("\n📚 Learning Rate Scheduler Configuration:")
        click.echo("Available scheduler types:")
        scheduler_types = [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "inverse_sqrt",
            "reduce_lr_on_plateau",
        ]
        for i, scheduler in enumerate(scheduler_types, 1):
            click.echo(f"  {i}. {scheduler}")

        while True:
            choice = click.prompt("Choose scheduler type", type=int)
            if 1 <= choice <= len(scheduler_types):
                finetune_config["lr_scheduler_type"] = scheduler_types[
                    choice - 1
                ]
                break
            click.echo("❌ Invalid choice. Please try again.")

        # Always include lr_scheduler_kwargs (required by TrainingArguments)
        finetune_config["lr_scheduler_kwargs"] = {}

        # Configure scheduler-specific parameters
        if click.confirm("Configure scheduler-specific parameters?"):
            scheduler = finetune_config["lr_scheduler_type"]
            click.echo(
                f"\n🔧 Configuring parameters for {scheduler} scheduler:"
            )

            if scheduler in ["cosine", "cosine_with_restarts"]:
                if click.confirm(
                    "Set number of restarts for cosine_with_restarts?"
                ):
                    restarts = click.prompt(
                        "Number of restarts", type=int, default=1
                    )
                    finetune_config["lr_scheduler_kwargs"]["num_restarts"] = (
                        restarts
                    )

            if scheduler == "polynomial":
                if click.confirm("Set power for polynomial decay?"):
                    power = click.prompt(
                        "Power value", type=float, default=1.0
                    )
                    finetune_config["lr_scheduler_kwargs"]["power"] = power

            if scheduler == "reduce_lr_on_plateau":
                if click.confirm("Set patience for reduce_lr_on_plateau?"):
                    patience = click.prompt(
                        "Patience value", type=int, default=10
                    )
                    finetune_config["lr_scheduler_kwargs"]["patience"] = (
                        patience
                    )

                if click.confirm("Set factor for reduce_lr_on_plateau?"):
                    factor = click.prompt(
                        "Factor value", type=float, default=0.1
                    )
                    finetune_config["lr_scheduler_kwargs"]["factor"] = factor

            # Generic scheduler parameters
            if click.confirm("Add custom scheduler parameters?"):
                while True:
                    key = click.prompt(
                        "Enter parameter name (or 'done' to finish)", type=str
                    )
                    if key.lower() == "done":
                        break
                    value = click.prompt(f"Enter value for {key}", type=str)
                    finetune_config["lr_scheduler_kwargs"][key] = value

        # Show available defaults
        if defaults:
            click.echo("\n📋 Smart defaults available for:")
            click.echo(
                f"   • Logging: {defaults.get('logging_steps', 2000)} steps"
            )
            click.echo(
                f"   • Evaluation: {defaults.get('eval_steps', 2000)} steps"
            )
            click.echo(
                f"   • Saving: {defaults.get('save_steps', 2000)} steps"
            )
            click.echo(f"   • Seed: {defaults.get('seed', 42)}")

        # Advanced settings
        if click.confirm("Configure advanced settings?"):
            finetune_config["gradient_accumulation_steps"] = click.prompt(
                "Gradient accumulation steps",
                type=int,
                default=defaults.get("gradient_accumulation_steps", 1),
            )

            finetune_config["max_grad_norm"] = click.prompt(
                "Max gradient norm", type=float, default=1.0
            )

            finetune_config["seed"] = click.prompt(
                "Random seed", type=int, default=42
            )

            finetune_config["bf16"] = click.confirm(
                "Use bfloat16?", default=False
            )
            finetune_config["fp16"] = click.confirm(
                "Use float16?", default=False
            )

        # Logging and saving (Let user judge whether to use epoch or steps)
        if click.confirm("Use epoch-based logging and saving?", default=False):
            finetune_config["logging_strategy"] = "epoch"
            finetune_config["eval_strategy"] = "epoch"
            finetune_config["save_strategy"] = "epoch"
        else:
            finetune_config["logging_strategy"] = "steps"
            finetune_config["logging_steps"] = click.prompt(
                "Logging steps", type=int, default=1000
            )
            finetune_config["eval_strategy"] = "steps"
            finetune_config["eval_steps"] = click.prompt(
                "Evaluation steps", type=int, default=1000
            )
            finetune_config["save_strategy"] = "steps"
            finetune_config["save_steps"] = click.prompt(
                "Save steps", type=int, default=1000
            )

        finetune_config["save_total_limit"] = click.prompt(
            "Save total limit", type=int, default=20
        )

        # Model saving and evaluation options
        finetune_config["save_safetensors"] = click.confirm(
            "Save model in safetensors format?",
            default=defaults.get("save_safetensors", True),
        )

        finetune_config["load_best_model_at_end"] = click.confirm(
            "Load best model at end of training?",
            default=defaults.get("load_best_model_at_end", True),
        )

        if finetune_config["load_best_model_at_end"]:
            finetune_config["metric_for_best_model"] = click.prompt(
                "Metric for best model selection",
                type=str,
                default=defaults.get("metric_for_best_model", "eval_loss"),
            )

        # Reporting options
        report_options = ["tensorboard", "wandb", "none"]
        click.echo("Available reporting options:")
        for i, option in enumerate(report_options, 1):
            click.echo(f"  {i}. {option}")

        while True:
            choice = click.prompt("Choose reporting option", type=int)
            if 1 <= choice <= len(report_options):
                finetune_config["report_to"] = report_options[choice - 1]
                break
            click.echo("❌ Invalid choice. Please try again.")

        # Optional resume from checkpoint
        if click.confirm("Configure resume from checkpoint?"):
            finetune_config["resume_from_checkpoint"] = click.prompt(
                "Checkpoint path to resume from (leave empty for none)",
                type=str,
                default="",
            )
            if not finetune_config["resume_from_checkpoint"]:
                finetune_config["resume_from_checkpoint"] = None

        return finetune_config

    def _configure_device_selection(self) -> str:
        """Common method for device selection"""
        devices = ["auto", "cpu", "cuda", "mps", "rocm", "xpu", "tpu"]
        click.echo("Available devices:")
        for i, device in enumerate(devices, 1):
            click.echo(f"  {i}. {device}")

        while True:
            choice = click.prompt("Choose device", type=int)
            if 1 <= choice <= len(devices):
                return devices[choice - 1]
            click.echo("❌ Invalid choice. Please try again.")

    def _configure_inference(
        self, auto_config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Configure inference settings with smart defaults"""
        click.echo("\n🔮 Inference Configuration:")

        # Get defaults if available
        defaults = {}
        if auto_config and "defaults" in auto_config:
            defaults = auto_config["defaults"].get("inference", {})

        inference_config = {}

        inference_config["batch_size"] = click.prompt(
            "Batch size", type=int, default=defaults.get("batch_size", 16)
        )

        inference_config["max_length"] = click.prompt(
            "Maximum sequence length",
            type=int,
            default=defaults.get("max_length", 512),
        )

        # Device selection using common method
        inference_config["device"] = self._configure_device_selection()

        inference_config["num_workers"] = click.prompt(
            "Number of workers",
            type=int,
            default=defaults.get("num_workers", 4),
        )

        inference_config["use_fp16"] = click.confirm(
            "Use float16?", default=defaults.get("use_fp16", False)
        )

        inference_config["output_dir"] = click.prompt(
            "Output directory",
            type=str,
            default=defaults.get("output_dir", "./results"),
        )

        # Show available defaults
        if defaults:
            click.echo("\n📋 Smart defaults available for:")
            click.echo(f"   • Batch size: {defaults.get('batch_size', 16)}")
            click.echo(f"   • Max length: {defaults.get('max_length', 512)}")
            click.echo(f"   • Workers: {defaults.get('num_workers', 4)}")

        return inference_config

    def _configure_benchmark_basic(self) -> dict[str, Any]:
        """Configure basic benchmark settings"""
        click.echo("\n📊 Basic Benchmark Configuration:")

        benchmark_config = {}

        benchmark_config["name"] = click.prompt(
            "Benchmark name", type=str, default="DNA Model Benchmark"
        )

        benchmark_config["description"] = click.prompt(
            "Benchmark description",
            type=str,
            default="Comparing DNA language models on various tasks",
        )

        return benchmark_config

    def _configure_models(self) -> list[dict[str, Any]]:
        """Configure models for benchmarking"""
        click.echo("\n🤖 Model Configuration:")

        models = []

        while True:
            click.echo(f"\nConfiguring model {len(models) + 1}:")

            model = {}
            model["name"] = click.prompt("Model name", type=str)

            # Model source
            sources = ["huggingface", "modelscope", "local"]
            click.echo("Available sources:")
            for i, source in enumerate(sources, 1):
                click.echo(f"  {i}. {source}")

            while True:
                choice = click.prompt("Choose source", type=int)
                if 1 <= choice <= len(sources):
                    model["source"] = sources[choice - 1]
                    break
                click.echo("❌ Invalid choice. Please try again.")

            if model["source"] == "local":
                model["path"] = click.prompt("Local model path", type=str)
            else:
                model["path"] = click.prompt("Model identifier/path", type=str)

            # Task type using consistent mapping
            click.echo("Available task types:")
            for key, task_type in self.MODEL_TASK_TYPES.items():
                click.echo(f"  {key}. {task_type}")

            while True:
                choice = click.prompt("Choose task type", type=str)
                if choice in self.MODEL_TASK_TYPES:
                    model["task_type"] = self.MODEL_TASK_TYPES[choice]
                    break
                click.echo("❌ Invalid choice. Please try again.")

            # Advanced model settings
            if click.confirm("Configure advanced model settings?"):
                model["revision"] = click.prompt(
                    "Git revision", type=str, default="main"
                )
                model["trust_remote_code"] = click.confirm(
                    "Trust remote code?", default=True
                )

                dtypes = ["float32", "float16", "bfloat16"]
                click.echo("Available data types:")
                for i, dtype in enumerate(dtypes, 1):
                    click.echo(f"  {i}. {dtype}")

                while True:
                    choice = click.prompt("Choose data type", type=int)
                    if 1 <= choice <= len(dtypes):
                        model["torch_dtype"] = dtypes[choice - 1]
                        break
                    click.echo("❌ Invalid choice. Please try again.")

            models.append(model)

            if not click.confirm("Add another model?"):
                break

        return models

    def _configure_models_with_template(
        self, auto_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Configure models for benchmarking using template information"""
        click.echo("\n🤖 Model Configuration (from template):")

        # Use template values
        model_name = (
            auto_config.get("model_path", "").split("/")[-1]
            if "/" in auto_config.get("model_path", "")
            else "Model"
        )
        model_path = auto_config.get("model_path", "")
        task_type = auto_config.get("mapped_task_type", "classification")

        click.echo(f"✅ Model Name: {model_name}")
        click.echo(f"✅ Model Path: {model_path}")
        click.echo(f"✅ Task Type: {task_type}")

        # Ask if user wants to modify any values
        if click.confirm("\n🔧 Modify any of these values?"):
            return (
                self._configure_models()
            )  # Fall back to manual configuration
        else:
            # Create model config from template
            model = {
                "name": model_name,
                "source": "huggingface",
                "path": model_path,
                "task_type": task_type,
                "trust_remote_code": True,
                "torch_dtype": "float32",
            }

            # Ask if user wants to add more models
            models = [model]
            if click.confirm("Add another model?"):
                additional_models = self._configure_models()
                models.extend(additional_models)

            return models

    def _configure_datasets(self) -> list[dict[str, Any]]:
        """Configure datasets for benchmarking"""
        click.echo("\n📁 Dataset Configuration:")

        datasets = []

        while True:
            click.echo(f"\nConfiguring dataset {len(datasets) + 1}:")

            dataset = {}
            dataset["name"] = click.prompt("Dataset name", type=str)
            dataset["path"] = click.prompt("Dataset file path", type=str)

            # Dataset format
            formats = ["csv", "tsv", "json", "fasta", "arrow", "parquet"]
            click.echo("Available formats:")
            for i, fmt in enumerate(formats, 1):
                click.echo(f"  {i}. {fmt}")

            while True:
                choice = click.prompt("Choose format", type=int)
                if 1 <= choice <= len(formats):
                    dataset["format"] = formats[choice - 1]
                    break
                click.echo("❌ Invalid choice. Please try again.")

            # Task type using consistent mapping
            click.echo("Available task types:")
            for key, task_type in self.DATASET_TASK_TYPES.items():
                click.echo(f"  {key}. {task_type}")

            while True:
                choice = click.prompt("Choose task type", type=str)
                if choice in self.DATASET_TASK_TYPES:
                    dataset["task"] = self.DATASET_TASK_TYPES[choice]
                    break
                click.echo("❌ Invalid choice. Please try again.")

            dataset["text_column"] = click.prompt(
                "Text/sequence column name", type=str, default="sequence"
            )
            dataset["label_column"] = click.prompt(
                "Label column name", type=str, default="label"
            )

            # Preprocessing options
            if click.confirm("Configure preprocessing options?"):
                dataset["max_length"] = click.prompt(
                    "Maximum sequence length", type=int, default=512
                )
                dataset["truncation"] = click.confirm(
                    "Enable truncation?", default=True
                )
                dataset["padding"] = click.prompt(
                    "Padding strategy", type=str, default="max_length"
                )

                dataset["test_size"] = click.prompt(
                    "Test set size ratio", type=float, default=0.2
                )
                dataset["val_size"] = click.prompt(
                    "Validation set size ratio", type=float, default=0.1
                )
                dataset["random_state"] = click.prompt(
                    "Random seed", type=int, default=42
                )

            datasets.append(dataset)

            if not click.confirm("Add another dataset?"):
                break

        return datasets

    def _configure_metrics(self) -> list[str]:
        """Configure evaluation metrics"""
        click.echo("\n📏 Metrics Configuration:")

        metric_categories = self._get_metric_categories()
        self._display_metric_categories(metric_categories)

        selected_metrics = []

        while True:
            if self._should_finish_metric_selection(selected_metrics):
                break

        return selected_metrics

    def _get_metric_categories(self) -> dict[str, list[str]]:
        """Get available metrics organized by category"""
        return {
            "Classification": [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "roc_auc",
                "pr_auc",
                "matthews_correlation",
            ],
            "Regression": [
                "mse",
                "mae",
                "rmse",
                "r2_score",
                "pearson_correlation",
                "spearman_correlation",
            ],
            "General": ["perplexity"],
        }

    def _display_metric_categories(
        self, metric_categories: dict[str, list[str]]
    ) -> None:
        """Display available metrics organized by category"""
        click.echo("Available metrics by category:")
        for category, metrics in metric_categories.items():
            click.echo(f"\n{category}:")
            for i, metric in enumerate(metrics, 1):
                click.echo(f"  {i}. {metric}")

    def _should_finish_metric_selection(
        self, selected_metrics: list[str]
    ) -> bool:
        """Handle metric selection menu and return True if finished"""
        click.echo(f"\nSelected metrics: {selected_metrics}")
        click.echo("Options:")
        click.echo("1. Add metric from list")
        click.echo("2. Add custom metric")
        click.echo("3. Finish selection")

        choice = click.prompt("Choose option", type=int)

        if choice == 1:
            self._add_metric_from_list(selected_metrics)
            return False
        elif choice == 2:
            self._add_custom_metric(selected_metrics)
            return False
        elif choice == 3:
            return True
        else:
            click.echo("❌ Invalid choice. Please try again.")
            return False

    def _add_metric_from_list(self, selected_metrics: list[str]) -> None:
        """Add a metric from the predefined list"""
        metric_categories = self._get_metric_categories()
        all_metrics = self._get_all_metrics(metric_categories)

        self._display_all_metrics(all_metrics)
        selected_metric = self._get_metric_choice(all_metrics)

        if selected_metric:
            self._add_metric_if_not_exists(selected_metrics, selected_metric)

    def _get_all_metrics(
        self, metric_categories: dict[str, list[str]]
    ) -> list[str]:
        """Get flattened list of all available metrics"""
        all_metrics = []
        for metrics in metric_categories.values():
            all_metrics.extend(metrics)
        return all_metrics

    def _display_all_metrics(self, all_metrics: list[str]) -> None:
        """Display all available metrics with numbers"""
        click.echo("Available metrics:")
        for i, metric in enumerate(all_metrics, 1):
            click.echo(f"  {i}. {metric}")

    def _get_metric_choice(self, all_metrics: list[str]) -> str | None:
        """Get user's metric choice and validate it"""
        while True:
            metric_choice = click.prompt("Choose metric number", type=int)
            if 1 <= metric_choice <= len(all_metrics):
                return all_metrics[metric_choice - 1]
            click.echo("❌ Invalid choice. Please try again.")

    def _add_custom_metric(self, selected_metrics: list[str]) -> None:
        """Add a custom metric name"""
        custom_metric = click.prompt("Enter custom metric name", type=str)
        self._add_metric_if_not_exists(
            selected_metrics, custom_metric, is_custom=True
        )

    def _add_metric_if_not_exists(
        self, selected_metrics: list[str], metric: str, is_custom: bool = False
    ) -> None:
        """Add metric to list if it doesn't already exist"""
        if metric not in selected_metrics:
            selected_metrics.append(metric)
            prefix = "custom metric: " if is_custom else ""
            click.echo(f"✅ Added {prefix}{metric}")
        else:
            click.echo("⚠️  Metric already selected")

    def _configure_evaluation(self) -> dict[str, Any]:
        """Configure evaluation settings"""
        click.echo("\n⚡ Evaluation Configuration:")

        evaluation_config = {}

        evaluation_config["batch_size"] = click.prompt(
            "Batch size", type=int, default=32
        )

        evaluation_config["max_length"] = click.prompt(
            "Maximum sequence length", type=int, default=512
        )

        # Device selection using common method
        evaluation_config["device"] = self._configure_device_selection()

        evaluation_config["num_workers"] = click.prompt(
            "Number of workers", type=int, default=4
        )

        # Performance optimization
        if click.confirm("Configure performance optimization?"):
            evaluation_config["use_fp16"] = click.confirm(
                "Use float16?", default=True
            )
            evaluation_config["use_bf16"] = click.confirm(
                "Use bfloat16?", default=False
            )
            evaluation_config["mixed_precision"] = click.confirm(
                "Enable mixed precision?", default=True
            )

            evaluation_config["pin_memory"] = click.confirm(
                "Pin memory?", default=True
            )
            evaluation_config["memory_efficient_attention"] = click.confirm(
                "Use memory efficient attention?", default=False
            )

        # Reproducibility
        evaluation_config["seed"] = click.prompt(
            "Random seed", type=int, default=42
        )

        evaluation_config["deterministic"] = click.confirm(
            "Enable deterministic mode?", default=True
        )

        return evaluation_config

    def _configure_output(self) -> dict[str, Any]:
        """Configure output settings"""
        click.echo("\n📤 Output Configuration:")

        output_config = {}

        # Output formats
        formats = ["html", "csv", "json", "pdf"]
        click.echo("Available output formats:")
        for i, fmt in enumerate(formats, 1):
            click.echo(f"  {i}. {fmt}")

        while True:
            choice = click.prompt("Choose output format", type=int)
            if 1 <= choice <= len(formats):
                output_config["format"] = formats[choice - 1]
                break
            click.echo("❌ Invalid choice. Please try again.")

        output_config["path"] = click.prompt(
            "Output directory", type=str, default="benchmark_results"
        )

        # Content options
        output_config["save_predictions"] = click.confirm(
            "Save predictions?", default=True
        )
        output_config["save_embeddings"] = click.confirm(
            "Save embeddings?", default=False
        )
        output_config["save_attention_maps"] = click.confirm(
            "Save attention maps?", default=False
        )
        output_config["generate_plots"] = click.confirm(
            "Generate plots?", default=True
        )

        # Report customization
        if click.confirm("Customize report?"):
            output_config["report_title"] = click.prompt(
                "Report title", type=str, default="DNA Model Benchmark Report"
            )

            output_config["include_summary"] = click.confirm(
                "Include summary?", default=True
            )
            output_config["include_details"] = click.confirm(
                "Include detailed results?", default=True
            )
            output_config["include_recommendations"] = click.confirm(
                "Include recommendations?", default=True
            )

        return output_config

    def save_config(self, filepath: str | None = None) -> str:
        """Save configuration to YAML file"""
        if filepath is None:
            if self.config_type == "finetune":
                filepath = "finetune_model_config.yaml"
            elif self.config_type == "inference":
                filepath = "inference_model_config.yaml"
            elif self.config_type == "benchmark":
                filepath = "benchmark_model_config.yaml"
            else:
                filepath = "model_config.yaml"

        # Ensure filepath is not None at this point
        if filepath is None:
            raise ValueError(
                "filepath should not be None after initialization"
            )

        # Ensure .yaml extension
        if not filepath.endswith((".yaml", ".yml")):
            filepath += ".yaml"

        # Create directory if it doesn't exist
        dir_path = os.path.dirname(filepath)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        # Save configuration
        with open(filepath, "w", encoding="utf-8") as f:
            yaml.dump(
                self.config,
                f,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )

        return filepath

    def preview_config(self):
        """Preview the generated configuration"""
        click.echo("\n📋 Generated Configuration Preview:")
        click.echo("=" * 50)
        click.echo(
            yaml.dump(
                self.config,
                default_flow_style=False,
                allow_unicode=True,
                indent=2,
            )
        )


@click.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for the configuration",
)
@click.option(
    "--preview", "-p", is_flag=True, help="Preview configuration before saving"
)
@click.option(
    "--non-interactive",
    "-n",
    is_flag=True,
    help="Use non-interactive mode with defaults",
)
def main(output, preview, non_interactive):
    """Generate DNALLM configuration files interactively"""

    if non_interactive:
        click.echo(
            "Non-interactive mode not yet implemented. Using interactive mode."
        )

    # Initialize generator
    generator = ConfigGenerator()

    try:
        # Generate configuration
        generator.generate_config()

        # Preview if requested
        if preview:
            generator.preview_config()

            if not click.confirm("Save this configuration?"):
                click.echo("Configuration generation cancelled.")
                return

        # Save configuration
        filepath = generator.save_config(output)
        click.echo(f"\n✅ Configuration saved to: {filepath}")

        # Show next steps
        click.echo("\n🎉 Configuration generated successfully!")
        click.echo("\nNext steps:")
        if generator.config_type == "finetune":
            click.echo("1. Review the generated configuration file")
            click.echo("2. Adjust parameters if needed")
            click.echo(
                "3. Run training with: dnallm-train --config " + filepath
            )
        elif generator.config_type == "inference":
            click.echo("1. Review the generated configuration file")
            click.echo("2. Adjust parameters if needed")
            click.echo(
                "3. Run inference with: dnallm-inference --config " + filepath
            )
        elif generator.config_type == "benchmark":
            click.echo("1. Review the generated configuration file")
            click.echo("2. Adjust parameters if needed")
            click.echo(
                "3. Run benchmark with: dnallm-benchmark --config " + filepath
            )

    except KeyboardInterrupt:
        click.echo("\n\n❌ Configuration generation cancelled by user.")
    except Exception as e:
        click.echo(f"\n❌ Error generating configuration: {e}")
        raise


if __name__ == "__main__":
    main()
