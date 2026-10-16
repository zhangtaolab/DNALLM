#!/usr/bin/env python3
"""Inference CLI for DNALLM."""

import click
from ..utils import get_logger

logger = get_logger("dnallm.cli.inference")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to inference configuration YAML file",
)
@click.option(
    "--model",
    "-m",
    type=str,
    help="Model name or path",
)
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Path to input data file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path",
)
def main(config, model, input, output):
    """Run inference with a trained DNA large language model."""
    if not config:
        raise click.UsageError(
            "--config is required: the task section and other core settings "
            "come only from a configuration file, so a flag-only invocation "
            "is not supported. Generate a config with "
            "`dnallm-model-config-generator --output config.yaml` and rerun "
            "with --config. See the CLI usage docs "
            "(docs/user_guide/cli/usage.md)."
        )

    from ..inference import DNAInference
    from ..configuration import load_config

    config_dict = load_config(config)
    inference_engine = DNAInference(model=None, tokenizer=None, config=config_dict)
    results = inference_engine.infer()
    if output:
        click.echo(f"Results saved to: {output}")
    else:
        logger.info(f"Inference results: {results}")


if __name__ == "__main__":
    main()
