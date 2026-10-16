#!/usr/bin/env python3
"""Training CLI for DNALLM."""

import click
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
    """Train a DNA large language model."""
    if not config:
        raise click.UsageError(
            "--config is required: the task section and other core settings "
            "come only from a configuration file, so a flag-only invocation "
            "is not supported. Generate a config with "
            "`dnallm-model-config-generator --output config.yaml` and rerun "
            "with --config. See the CLI usage docs "
            "(docs/user_guide/cli/usage.md)."
        )

    from ..finetune import DNATrainer
    from ..configuration import load_config

    config_dict = load_config(config)
    trainer = DNATrainer(model=None, config=config_dict)
    trainer.train()


if __name__ == "__main__":
    main()
