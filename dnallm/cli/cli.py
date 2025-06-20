import click
import yaml
import os
from ..finetune.trainer import DNATrainer
from ..inference.predictor import DNAPredictor, save_predictions, save_metrics
from ..inference.benchmark import Benchmark
from ..inference.mutagenesis import Mutagenesis
from ..configuration.configs import TaskConfig, TrainingConfig, InferenceConfig

@click.group()
def cli():
    """
    DNALLM CLI: train, predict, benchmark, mutagenesis.
    """
    pass

# ------------------------
# Train subcommand
# ------------------------
@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='YAML config file with sections: task and finetune.')
@click.option('--model-name', type=str, help='Pretrained model name or path.')
@click.option('--train-file', type=click.Path(exists=True), help='Path to training data.')
@click.option('--eval-file', type=click.Path(exists=True), help='Path to evaluation data.')
@click.option('--output-dir', type=click.Path(), default=None, help='Directory for outputs.')
@click.option('--batch-size', type=int, help='Training batch size.')
@click.option('--learning-rate', type=float, help='Learning rate.')
@click.option('--epochs', type=int, help='Number of epochs.')
@click.option('--seed', type=int, help='Random seed.')
def train(config, model_name, train_file, eval_file, output_dir,
          batch_size, learning_rate, epochs, seed):
    """
    Fine-tune a DNA LLM model.
    """
    # Load default configs
    if config:
        data = yaml.safe_load(open(config))
        task_cfg = TaskConfig(**data.get('task', {}))
        train_cfg = TrainingConfig(**data.get('finetune', {}))
    else:
        task_cfg = TaskConfig(task_type='mask')
        train_cfg = TrainingConfig()

    # Override CLI args
    overrides = {
        'model_name': model_name,
        'output_dir': output_dir,
        'seed': seed,
    }
    for field, val in overrides.items():
        if val is not None:
            setattr(train_cfg, field, val)
    extra = {}
    if batch_size: extra['per_device_train_batch_size'] = batch_size
    if learning_rate: extra['learning_rate'] = learning_rate
    if epochs: extra['num_train_epochs'] = epochs
    # Instantiate and run trainer
    trainer = DNATrainer(task_cfg, train_cfg, train_file, eval_file, model_name, extra_args=extra)
    metrics = trainer.train()
    click.echo(f"Training completed. Metrics: {metrics}")

# ------------------------
# Predict subcommand
# ------------------------
@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='YAML config file with inference section.')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to fine-tuned model.')
@click.option('--input-file', type=click.Path(exists=True), required=True,
              help='Sequence file (fasta or txt).')
@click.option('--output-dir', type=click.Path(), default=None,
              help='Directory for predictions.')
@click.option('--batch-size', type=int, help='Inference batch size.')
def predict(config, model_path, input_file, output_dir, batch_size):
    """
    Run inference using a fine-tuned DNA LLM.
    """
    # Load inference config
    if config:
        data = yaml.safe_load(open(config))
        infer_cfg = InferenceConfig(**data.get('inference', {}))
    else:
        infer_cfg = InferenceConfig()
    # Override
    if output_dir: infer_cfg.output_dir = output_dir
    if batch_size: infer_cfg.batch_size = batch_size
    # Instantiate predictor
    predictor = DNAPredictor(model_path, infer_cfg)
    pred = predictor.predict_file(input_file)
    # Save outputs
    outdir = infer_cfg.output_dir or 'predictions'
    os.makedirs(outdir, exist_ok=True)
    save_predictions(pred, outdir)
    click.echo(f"Predictions saved to {outdir}")

# ------------------------
# Benchmark subcommand
# ------------------------
@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='YAML config with task, finetune, inference sections.')
@click.option('--models', '-m', type=str,
              help='Comma-separated list of model identifiers to benchmark.')
@click.option('--output-dir', '-o', type=click.Path(), default='bench_results',
              help='Directory to save metrics and plots.')
def benchmark(config, models, output_dir):
    """
    Benchmark multiple fine-tuned models.
    """
    data = yaml.safe_load(open(config))
    bench = Benchmark(data)
    model_list = models.split(',') if models else None
    metrics = bench.run(model_names=model_list)
    os.makedirs(output_dir, exist_ok=True)
    bench.plot(metrics, output_dir)
    click.echo(f"Benchmark complete. Results in {output_dir}")

# ------------------------
# Mutagenesis subcommand
# ------------------------
@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True),
              help='YAML config with inference parameters.')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to fine-tuned model.')
@click.option('--sequence', type=str, required=True,
              help='Input DNA sequence or path to sequence file.')
@click.option('--output-dir', type=click.Path(), default='mutagenesis',
              help='Directory for mutagenesis results.')
@click.option('--batch-size', type=int, default=None,
              help='Batch size for mutation predictions.')
@click.option('--strategy', type=str, default='last',
              help='Aggregation strategy for evaluation (e.g., last, max).')
def mutagenesis(config, model_path, sequence, output_dir, batch_size, strategy):
    """
    Perform in silico mutagenesis on a DNA sequence.
    """
    # Load inference settings
    cfg = yaml.safe_load(open(config)) if config else {}
    # Instantiate mutagenesis engine
    engine = Mutagenesis(model_path, None, cfg)
    # Run evaluation
    results = engine.evaluate(strategy=strategy)
    # Save and plot
    os.makedirs(output_dir, exist_ok=True)
    engine.plot(results, output_dir)
    click.echo(f"Mutagenesis complete. Outputs in {output_dir}")

if __name__ == '__main__':
    cli()
