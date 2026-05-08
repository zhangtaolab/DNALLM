#!/usr/bin/env python3
"""Inference CLI for DNALLM."""

import click
import sys
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
    """Run inference with a trained DNA language model."""
    from ..inference import DNAInference
    from ..configuration import load_config

    if config:
        config_dict = load_config(config)
        inference_engine = DNAInference(
            model=None, tokenizer=None, config=config_dict
        )
        results = inference_engine.infer()
        if output:
            click.echo(f"Results saved to: {output}")
        else:
            logger.info(f"Inference results: {results}")
    else:
        if not all([model, input]):
            click.echo(
                "Error: --model and --input are required when "
                "not using --config",
                err=True,
            )
            sys.exit(1)

        config_dict = {
            "model_name_or_path": model,
            "data_path": input,
        }
        inference_engine = DNAInference(
            model=None, tokenizer=None, config=config_dict
        )
        results = inference_engine.infer()
        if output:
            click.echo(f"Results saved to: {output}")
        else:
            logger.info(f"Inference results: {results}")


if __name__ == "__main__":
    main()
