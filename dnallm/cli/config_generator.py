#!/usr/bin/env python3
"""
DNALLM Configuration Generator CLI Tool

This tool provides an interactive way to generate configuration files for:
- Fine-tuning tasks
- Inference tasks  
- Benchmarking tasks
"""

import click
import yaml
import os
from pathlib import Path
from typing import Dict, Any, List, Optional


class ConfigGenerator:
    """Interactive configuration generator for DNALLM"""
    
    # Define consistent task types across all methods
    TASK_TYPES = {
        '1': 'binary_classification',
        '2': 'multiclass_classification', 
        '3': 'multilabel_classification',
        '4': 'regression',
        '5': 'token_classification',
        '6': 'embedding',
        '7': 'masked_language_modeling',
        '8': 'generation'
    }
    TASK_ALIAS = {
        'binary_classification': 'binary',
        'multiclass_classification': 'multiclass',
        'multilabel_classification': 'multilabel',
        'regression': 'regression',
        'token_classification': 'token',
        'embedding': 'embedding',
        'masked_language_modeling': 'mask',
        'generation': 'generation'
    }
    
    MODEL_TASK_TYPES = {
        '1': 'classification',
        '2': 'generation', 
        '3': 'masked_language_modeling',
        '4': 'embedding',
        '5': 'regression'
    }
    
    DATASET_TASK_TYPES = {
        '1': 'binary_classification',
        '2': 'multiclass_classification',
        '3': 'regression',
        '4': 'generation'
    }
    
    def __init__(self):
        self.config = {}
        self.config_type = None
        
    def generate_config(self) -> Dict[str, Any]:
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
        
        # Task configuration
        self.config['task'] = self._configure_task()
        
        # Fine-tuning configuration
        self.config['finetune'] = self._configure_finetune()
        
        # Inference configuration (for evaluation; when do fine-tuning, use default inference config)
        self.config['inference'] = {
            'batch_size': 16,
            'max_length': 512,
            'num_workers': 4,
            'device': 'auto',
            'use_fp16': False,
            'output_dir': './results'
        }
        
        click.echo("✅ Fine-tuning configuration generated successfully!")
    
    def _generate_inference_config(self):
        """Generate inference configuration"""
        click.echo("\n🔮 Generating Inference Configuration...")
        
        # Task configuration
        self.config['task'] = self._configure_task()
        
        # Inference configuration
        self.config['inference'] = self._configure_inference()
        
        click.echo("✅ Inference configuration generated successfully!")
    
    def _generate_benchmark_config(self):
        """Generate benchmark configuration"""
        click.echo("\n📊 Generating Benchmark Configuration...")
        
        # Basic benchmark info
        self.config['benchmark'] = self._configure_benchmark_basic()
        
        # Models configuration
        self.config['models'] = self._configure_models()
        
        # Datasets configuration
        self.config['datasets'] = self._configure_datasets()
        
        # Metrics configuration
        self.config['metrics'] = self._configure_metrics()
        
        # Evaluation configuration
        self.config['evaluation'] = self._configure_evaluation()
        
        # Output configuration
        self.config['output'] = self._configure_output()
        
        click.echo("✅ Benchmark configuration generated successfully!")
    
    def _configure_task(self) -> Dict[str, Any]:
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
        
        task_config = {'task_type': task_alias}
        
        # Number of labels (for classification tasks)
        if 'classification' in task_type:
            num_labels = click.prompt(
                "Number of labels/classes", 
                type=int, 
                default=2 if 'binary' in task_type else 3
            )
            task_config['num_labels'] = num_labels
            
            # Label names
            if click.confirm("Do you want to specify label names?"):
                label_names = []
                for i in range(num_labels):
                    label_name = click.prompt(f"Label {i} name")
                    label_names.append(label_name)
                task_config['label_names'] = label_names
        
        # Threshold for binary/multilabel
        if 'binary' in task_type or 'multilabel' in task_type:
            threshold = click.prompt(
                "Classification threshold", 
                type=float, 
                default=0.5
            )
            task_config['threshold'] = threshold
        
        return task_config
    
    def _configure_finetune(self) -> Dict[str, Any]:
        """Configure fine-tuning settings"""
        click.echo("\n🎯 Fine-tuning Configuration:")
        
        finetune_config = {}
        
        # Basic settings
        finetune_config['output_dir'] = click.prompt(
            "Output directory", 
            type=str, 
            default="./outputs"
        )
        
        finetune_config['num_train_epochs'] = click.prompt(
            "Number of training epochs", 
            type=int, 
            default=3
        )
        
        finetune_config['per_device_train_batch_size'] = click.prompt(
            "Training batch size per device", 
            type=int, 
            default=8
        )
        
        finetune_config['per_device_eval_batch_size'] = click.prompt(
            "Evaluation batch size per device", 
            type=int, 
            default=16
        )
        
        # Learning rate and optimization
        finetune_config['learning_rate'] = click.prompt(
            "Learning rate", 
            type=float, 
            default=2e-5
        )
        
        finetune_config['weight_decay'] = click.prompt(
            "Weight decay", 
            type=float, 
            default=0.01
        )
        
        finetune_config['warmup_ratio'] = click.prompt(
            "Warmup ratio", 
            type=float, 
            default=0.1
        )
        
        # Advanced settings
        if click.confirm("Configure advanced settings?"):
            finetune_config['gradient_accumulation_steps'] = click.prompt(
                "Gradient accumulation steps", 
                type=int, 
                default=1
            )
            
            finetune_config['max_grad_norm'] = click.prompt(
                "Max gradient norm", 
                type=float, 
                default=1.0
            )
            
            finetune_config['lr_scheduler_type'] = click.prompt(
                "Learning rate scheduler type", 
                type=str, 
                default="linear"
            )
            
            finetune_config['seed'] = click.prompt(
                "Random seed", 
                type=int, 
                default=42
            )
            
            finetune_config['bf16'] = click.confirm("Use bfloat16?", default=False)
            finetune_config['fp16'] = click.confirm("Use float16?", default=False)
        
        # Logging and saving (Let user judge whether to use epoch or steps)
        if click.confirm("Use epoch-based logging and saving?", default=False):
            finetune_config['logging_strategy'] = "epoch"
            finetune_config['eval_strategy'] = "epoch"
            finetune_config['save_strategy'] = "epoch"
        else:
            finetune_config['logging_strategy'] = "steps"
            finetune_config['logging_steps'] = click.prompt(
                "Logging steps", 
                type=int, 
                default=100
            )
            finetune_config['eval_strategy'] = "steps"
            finetune_config['eval_steps'] = click.prompt(
                "Evaluation steps", 
                type=int, 
                default=100
            )
            finetune_config['save_strategy'] = "steps"
            finetune_config['save_steps'] = click.prompt(
                "Save steps", 
                type=int, 
                default=500
            )
        
        finetune_config['save_total_limit'] = click.prompt(
            "Save total limit", 
            type=int, 
            default=20
        )
        
        return finetune_config
    
    def _configure_device_selection(self) -> str:
        """Common method for device selection"""
        devices = ['auto', 'cpu', 'cuda', 'mps', 'rocm', 'xpu', 'tpu']
        click.echo("Available devices:")
        for i, device in enumerate(devices, 1):
            click.echo(f"  {i}. {device}")
        
        while True:
            choice = click.prompt("Choose device", type=int)
            if 1 <= choice <= len(devices):
                return devices[choice - 1]
            click.echo("❌ Invalid choice. Please try again.")
    
    def _configure_inference(self) -> Dict[str, Any]:
        """Configure inference settings"""
        click.echo("\n🔮 Inference Configuration:")
        
        inference_config = {}
        
        inference_config['batch_size'] = click.prompt(
            "Batch size", 
            type=int, 
            default=16
        )
        
        inference_config['max_length'] = click.prompt(
            "Maximum sequence length", 
            type=int, 
            default=512
        )
        
        # Device selection using common method
        inference_config['device'] = self._configure_device_selection()
        
        inference_config['num_workers'] = click.prompt(
            "Number of workers", 
            type=int, 
            default=4
        )
        
        inference_config['use_fp16'] = click.confirm("Use float16?", default=False)
        
        inference_config['output_dir'] = click.prompt(
            "Output directory", 
            type=str, 
            default="./results"
        )
        
        return inference_config
    
    def _configure_benchmark_basic(self) -> Dict[str, Any]:
        """Configure basic benchmark settings"""
        click.echo("\n📊 Basic Benchmark Configuration:")
        
        benchmark_config = {}
        
        benchmark_config['name'] = click.prompt(
            "Benchmark name", 
            type=str, 
            default="DNA Model Benchmark"
        )
        
        benchmark_config['description'] = click.prompt(
            "Benchmark description", 
            type=str, 
            default="Comparing DNA language models on various tasks"
        )
        
        return benchmark_config
    
    def _configure_models(self) -> List[Dict[str, Any]]:
        """Configure models for benchmarking"""
        click.echo("\n🤖 Model Configuration:")
        
        models = []
        
        while True:
            click.echo(f"\nConfiguring model {len(models) + 1}:")
            
            model = {}
            model['name'] = click.prompt("Model name", type=str)
            
            # Model source
            sources = ['huggingface', 'modelscope', 'local']
            click.echo("Available sources:")
            for i, source in enumerate(sources, 1):
                click.echo(f"  {i}. {source}")
            
            while True:
                choice = click.prompt("Choose source", type=int)
                if 1 <= choice <= len(sources):
                    model['source'] = sources[choice - 1]
                    break
                click.echo("❌ Invalid choice. Please try again.")
            
            if model['source'] == 'local':
                model['path'] = click.prompt("Local model path", type=str)
            else:
                model['path'] = click.prompt("Model identifier/path", type=str)
            
            # Task type using consistent mapping
            click.echo("Available task types:")
            for key, task_type in self.MODEL_TASK_TYPES.items():
                click.echo(f"  {key}. {task_type}")
            
            while True:
                choice = click.prompt("Choose task type", type=str)
                if choice in self.MODEL_TASK_TYPES:
                    model['task_type'] = self.MODEL_TASK_TYPES[choice]
                    break
                click.echo("❌ Invalid choice. Please try again.")
            
            # Advanced model settings
            if click.confirm("Configure advanced model settings?"):
                model['revision'] = click.prompt("Git revision", type=str, default="main")
                model['trust_remote_code'] = click.confirm("Trust remote code?", default=True)
                
                dtypes = ['float32', 'float16', 'bfloat16']
                click.echo("Available data types:")
                for i, dtype in enumerate(dtypes, 1):
                    click.echo(f"  {i}. {dtype}")
                
                while True:
                    choice = click.prompt("Choose data type", type=int)
                    if 1 <= choice <= len(dtypes):
                        model['torch_dtype'] = dtypes[choice - 1]
                        break
                    click.echo("❌ Invalid choice. Please try again.")
            
            models.append(model)
            
            if not click.confirm("Add another model?"):
                break
        
        return models
    
    def _configure_datasets(self) -> List[Dict[str, Any]]:
        """Configure datasets for benchmarking"""
        click.echo("\n📁 Dataset Configuration:")
        
        datasets = []
        
        while True:
            click.echo(f"\nConfiguring dataset {len(datasets) + 1}:")
            
            dataset = {}
            dataset['name'] = click.prompt("Dataset name", type=str)
            dataset['path'] = click.prompt("Dataset file path", type=str)
            
            # Dataset format
            formats = ['csv', 'tsv', 'json', 'fasta', 'arrow', 'parquet']
            click.echo("Available formats:")
            for i, fmt in enumerate(formats, 1):
                click.echo(f"  {i}. {fmt}")
            
            while True:
                choice = click.prompt("Choose format", type=int)
                if 1 <= choice <= len(formats):
                    dataset['format'] = formats[choice - 1]
                    break
                click.echo("❌ Invalid choice. Please try again.")
            
            # Task type using consistent mapping
            click.echo("Available task types:")
            for key, task_type in self.DATASET_TASK_TYPES.items():
                click.echo(f"  {key}. {task_type}")
            
            while True:
                choice = click.prompt("Choose task type", type=str)
                if choice in self.DATASET_TASK_TYPES:
                    dataset['task'] = self.DATASET_TASK_TYPES[choice]
                    break
                click.echo("❌ Invalid choice. Please try again.")
            
            dataset['text_column'] = click.prompt("Text/sequence column name", type=str, default="sequence")
            dataset['label_column'] = click.prompt("Label column name", type=str, default="label")
            
            # Preprocessing options
            if click.confirm("Configure preprocessing options?"):
                dataset['max_length'] = click.prompt("Maximum sequence length", type=int, default=512)
                dataset['truncation'] = click.confirm("Enable truncation?", default=True)
                dataset['padding'] = click.prompt("Padding strategy", type=str, default="max_length")
                
                dataset['test_size'] = click.prompt("Test set size ratio", type=float, default=0.2)
                dataset['val_size'] = click.prompt("Validation set size ratio", type=float, default=0.1)
                dataset['random_state'] = click.prompt("Random seed", type=int, default=42)
            
            datasets.append(dataset)
            
            if not click.confirm("Add another dataset?"):
                break
        
        return datasets
    
    def _configure_metrics(self) -> List[str]:
        """Configure evaluation metrics"""
        click.echo("\n📏 Metrics Configuration:")
        
        # Available metrics by category
        metric_categories = {
            'Classification': ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc', 'pr_auc', 'matthews_correlation'],
            'Regression': ['mse', 'mae', 'rmse', 'r2_score', 'pearson_correlation', 'spearman_correlation'],
            'General': ['perplexity']
        }
        
        click.echo("Available metrics by category:")
        for category, metrics in metric_categories.items():
            click.echo(f"\n{category}:")
            for i, metric in enumerate(metrics, 1):
                click.echo(f"  {i}. {metric}")
        
        selected_metrics = []
        
        while True:
            click.echo(f"\nSelected metrics: {selected_metrics}")
            click.echo("Options:")
            click.echo("1. Add metric from list")
            click.echo("2. Add custom metric")
            click.echo("3. Finish selection")
            
            choice = click.prompt("Choose option", type=int)
            
            if choice == 1:
                # Add from predefined list
                all_metrics = []
                for metrics in metric_categories.values():
                    all_metrics.extend(metrics)
                
                click.echo("Available metrics:")
                for i, metric in enumerate(all_metrics, 1):
                    click.echo(f"  {i}. {metric}")
                
                while True:
                    metric_choice = click.prompt("Choose metric number", type=int)
                    if 1 <= metric_choice <= len(all_metrics):
                        metric = all_metrics[metric_choice - 1]
                        if metric not in selected_metrics:
                            selected_metrics.append(metric)
                            click.echo(f"✅ Added {metric}")
                        else:
                            click.echo("⚠️  Metric already selected")
                        break
                    click.echo("❌ Invalid choice. Please try again.")
            
            elif choice == 2:
                # Add custom metric
                custom_metric = click.prompt("Enter custom metric name", type=str)
                if custom_metric not in selected_metrics:
                    selected_metrics.append(custom_metric)
                    click.echo(f"✅ Added custom metric: {custom_metric}")
                else:
                    click.echo("⚠️  Metric already selected")
            
            elif choice == 3:
                break
            
            else:
                click.echo("❌ Invalid choice. Please try again.")
        
        return selected_metrics
    
    def _configure_evaluation(self) -> Dict[str, Any]:
        """Configure evaluation settings"""
        click.echo("\n⚡ Evaluation Configuration:")
        
        evaluation_config = {}
        
        evaluation_config['batch_size'] = click.prompt(
            "Batch size", 
            type=int, 
            default=32
        )
        
        evaluation_config['max_length'] = click.prompt(
            "Maximum sequence length", 
            type=int, 
            default=512
        )
        
        # Device selection using common method
        evaluation_config['device'] = self._configure_device_selection()
        
        evaluation_config['num_workers'] = click.prompt(
            "Number of workers", 
            type=int, 
            default=4
        )
        
        # Performance optimization
        if click.confirm("Configure performance optimization?"):
            evaluation_config['use_fp16'] = click.confirm("Use float16?", default=True)
            evaluation_config['use_bf16'] = click.confirm("Use bfloat16?", default=False)
            evaluation_config['mixed_precision'] = click.confirm("Enable mixed precision?", default=True)
            
            evaluation_config['pin_memory'] = click.confirm("Pin memory?", default=True)
            evaluation_config['memory_efficient_attention'] = click.confirm("Use memory efficient attention?", default=False)
        
        # Reproducibility
        evaluation_config['seed'] = click.prompt(
            "Random seed", 
            type=int, 
            default=42
        )
        
        evaluation_config['deterministic'] = click.confirm("Enable deterministic mode?", default=True)
        
        return evaluation_config
    
    def _configure_output(self) -> Dict[str, Any]:
        """Configure output settings"""
        click.echo("\n📤 Output Configuration:")
        
        output_config = {}
        
        # Output formats
        formats = ['html', 'csv', 'json', 'pdf']
        click.echo("Available output formats:")
        for i, fmt in enumerate(formats, 1):
            click.echo(f"  {i}. {fmt}")
        
        while True:
            choice = click.prompt("Choose output format", type=int)
            if 1 <= choice <= len(formats):
                output_config['format'] = formats[choice - 1]
                break
            click.echo("❌ Invalid choice. Please try again.")
        
        output_config['path'] = click.prompt(
            "Output directory", 
            type=str, 
            default="benchmark_results"
        )
        
        # Content options
        output_config['save_predictions'] = click.confirm("Save predictions?", default=True)
        output_config['save_embeddings'] = click.confirm("Save embeddings?", default=False)
        output_config['save_attention_maps'] = click.confirm("Save attention maps?", default=False)
        output_config['generate_plots'] = click.confirm("Generate plots?", default=True)
        
        # Report customization
        if click.confirm("Customize report?"):
            output_config['report_title'] = click.prompt(
                "Report title", 
                type=str, 
                default="DNA Model Benchmark Report"
            )
            
            output_config['include_summary'] = click.confirm("Include summary?", default=True)
            output_config['include_details'] = click.confirm("Include detailed results?", default=True)
            output_config['include_recommendations'] = click.confirm("Include recommendations?", default=True)
        
        return output_config
    
    def save_config(self, filepath: str = None) -> str:
        """Save configuration to YAML file"""
        if filepath is None:
            if self.config_type == "finetune":
                filepath = "finetune_config.yaml"
            elif self.config_type == "inference":
                filepath = "inference_config.yaml"
            elif self.config_type == "benchmark":
                filepath = "benchmark_config.yaml"
        
        # Ensure .yaml extension
        if not filepath.endswith(('.yaml', '.yml')):
            filepath += '.yaml'
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save configuration
        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        return filepath
    
    def preview_config(self):
        """Preview the generated configuration"""
        click.echo("\n📋 Generated Configuration Preview:")
        click.echo("=" * 50)
        click.echo(yaml.dump(self.config, default_flow_style=False, allow_unicode=True, indent=2))


@click.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path for the configuration')
@click.option('--preview', '-p', is_flag=True, help='Preview configuration before saving')
@click.option('--non-interactive', '-n', is_flag=True, help='Use non-interactive mode with defaults')
def main(output, preview, non_interactive):
    """Generate DNALLM configuration files interactively"""
    
    if non_interactive:
        click.echo("Non-interactive mode not yet implemented. Using interactive mode.")
    
    # Initialize generator
    generator = ConfigGenerator()
    
    try:
        # Generate configuration
        config = generator.generate_config()
        
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
            click.echo("3. Run training with: dnallm-train --config " + filepath)
        elif generator.config_type == "inference":
            click.echo("1. Review the generated configuration file")
            click.echo("2. Adjust parameters if needed")
            click.echo("3. Run inference with: dnallm-predict --config " + filepath)
        elif generator.config_type == "benchmark":
            click.echo("1. Review the generated configuration file")
            click.echo("2. Adjust parameters if needed")
            click.echo("3. Run benchmark with: dnallm-benchmark --config " + filepath)
        
    except KeyboardInterrupt:
        click.echo("\n\n❌ Configuration generation cancelled by user.")
    except Exception as e:
        click.echo(f"\n❌ Error generating configuration: {e}")
        raise


if __name__ == '__main__':
    main()
