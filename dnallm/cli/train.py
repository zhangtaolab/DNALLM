#!/usr/bin/env python3
"""Training CLI for DNALLM."""

import click
import sys
from ..utils import get_logger

logger = get_logger("dnallm.cli.train")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to training configuration YAML file",
)
@click.option(
    "--model",
    "-m",
    type=str,
    help="Model name or path",
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True),
    help="Path to training data",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for training results",
)
def main(config, model, data, output):
    """Train a DNA language model."""
    from ..finetune import DNATrainer
    from ..configuration import load_config

    if config:
        config_dict = load_config(config)
        trainer = DNATrainer(model=None, config=config_dict)
        trainer.train()
    else:
        if not all([model, data, output]):
            click.echo(
                "Error: --model, --data, and --output are required when not using --config",
                err=True,
            )
            sys.exit(1)

        minimal_config = {
            "model_name_or_path": model,
            "data_path": data,
            "output_dir": output,
            "finetune": {
                "num_train_epochs": 3,
                "per_device_train_batch_size": 4,
                "learning_rate": 5e-5,
                "save_steps": 1000,
                "eval_steps": 1000,
            },
        }
        trainer = DNATrainer(model=None, config=minimal_config)  # type: ignore[arg-type]
        trainer.train()


if __name__ == "__main__":
    main()
