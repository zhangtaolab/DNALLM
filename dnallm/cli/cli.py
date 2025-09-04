#!/usr/bin/env python3
"""
Main CLI entry point for DNALLM package.
"""

import click
from pathlib import Path
import sys
from ..utils import get_logger

logger = get_logger("dnallm.cli")

@click.group()
@click.version_option()
def cli():
    """
    DNALLM - A toolkit for fine-tuning and inference with DNA Language Models
    
    This toolkit provides tools for:
    - Training DNA language models
    - Running inference and predictions
    - Benchmarking model performance
    - In-silico mutagenesis analysis
    - Generating configuration files
    """
    pass

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to training configuration file')
@click.option('--model', '-m', type=str, 
              help='Model name or path')
@click.option('--data', '-d', type=click.Path(exists=True), 
              help='Path to training data')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for training results')
def train(config, model, data, output):
    """Train a DNA language model"""
    from ..finetune import DNATrainer
    from ..configuration import load_config
    
    if config:
        # Load configuration from file
        config_dict = load_config(config)
        trainer = DNATrainer(config_dict)
        trainer.train()
    else:
        # Use command line arguments
        if not all([model, data, output]):
            click.echo("Error: --model, --data, and --output are required when not using --config")
            sys.exit(1)
        
        # Create minimal config
        config_dict = {
            'model_name_or_path': model,
            'data_path': data,
            'output_dir': output,
            'training_args': {
                'num_train_epochs': 3,
                'per_device_train_batch_size': 4,
                'learning_rate': 5e-5,
                'save_steps': 1000,
                'eval_steps': 1000,
            }
        }
        
        trainer = DNATrainer(config_dict)
        trainer.train()

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to prediction configuration file')
@click.option('--model', '-m', type=str, 
              help='Model name or path')
@click.option('--input', '-i', type=click.Path(exists=True), 
              help='Path to input data file')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path')
def predict(config, model, input, output):
    """Run inference with a trained DNA language model"""
    from ..inference import DNAPredictor
    from ..configuration import load_config
    
    if config:
        # Load configuration from file
        config_dict = load_config(config)
        predictor = DNAPredictor(config_dict)
        results = predictor.predict()
        
        if output:
            predictor.save_results(results, output)
        else:
            logger.info(f"Prediction results: {results}")
    else:
        # Use command line arguments
        if not all([model, input]):
            click.echo("Error: --model and --input are required when not using --config")
            sys.exit(1)
        
        # Create minimal config
        config_dict = {
            'model_name_or_path': model,
            'data_path': input,
        }
        
        predictor = DNAPredictor(config_dict)
        results = predictor.predict()
        
        if output:
            predictor.save_results(results, output)
        else:
            logger.info(f"Prediction results: {results}")

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to benchmark configuration file')
@click.option('--model', '-m', type=str, 
              help='Model name or path')
@click.option('--data', '-d', type=click.Path(exists=True), 
              help='Path to benchmark data')
@click.option('--output', '-o', type=click.Path(), 
              help='Output directory for benchmark results')
def benchmark(config, model, data, output):
    """Run benchmark evaluation on a DNA language model"""
    from ..inference import Benchmark
    from ..configuration import load_config
    
    if config:
        config_dict = load_config(config)
        benchmark = Benchmark(config_dict)
        results = benchmark.run()
        
        if output:
            benchmark.save_results(results, output)
        else:
            print(results)
    else:
        if not all([model, data]):
            click.echo("Error: --model and --data are required when not using --config")
            sys.exit(1)
        
        config_dict = {
            'model_name_or_path': model,
            'data_path': data,
        }
        
        benchmark = Benchmark(config_dict)
        results = benchmark.run()
        
        if output:
            benchmark.save_results(results, output)
        else:
            print(results)

@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), 
              help='Path to mutagenesis configuration file')
@click.option('--model', '-m', type=str, 
              help='Model name or path')
@click.option('--sequence', '-s', type=str, 
              help='DNA sequence for analysis')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path')
def mutagenesis(config, model, sequence, output):
    """Run in-silico mutagenesis analysis"""
    from ..inference import Mutagenesis
    from ..configuration import load_config
    
    if config:
        config_dict = load_config(config)
        mutagenesis = Mutagenesis(config_dict)
        results = mutagenesis.run()
        
        if output:
            mutagenesis.save_results(results, output)
        else:
            print(results)
    else:
        if not all([model, sequence]):
            click.echo("Error: --model and --sequence are required when not using --config")
            sys.exit(1)
        
        config_dict = {
            'model_name_or_path': model,
            'sequence': sequence,
        }
        
        mutagenesis = Mutagenesis(config_dict)
        results = mutagenesis.run()
        
        if output:
            mutagenesis.save_results(results, output)
        else:
            print(results)

@cli.command()
@click.option('--output', '-o', type=click.Path(), help='Output file path for the configuration')
@click.option('--preview', '-p', is_flag=True, help='Preview configuration before saving')
@click.option('--non-interactive', '-n', is_flag=True, help='Use non-interactive mode with defaults')
def model_config_generator(output, preview, non_interactive):
    """Generate DNALLM configuration files interactively"""
    try:
        # Import and run the configuration generator
        from .model_config_generator import main as config_generator_main
        
        # Create sys.argv-like arguments for the config generator
        import sys
        original_argv = sys.argv.copy()
        
        # Build arguments for the config generator
        args = ['model_config_generator']
        if output:
            args.extend(['--output', output])
        if preview:
            args.append('--preview')
        if non_interactive:
            args.append('--non-interactive')
        
        # Temporarily replace sys.argv
        sys.argv = args
        
        try:
            config_generator_main()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv
            
    except ImportError as e:
        click.echo(f"Error importing configuration generator: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Configuration generation failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    cli()
