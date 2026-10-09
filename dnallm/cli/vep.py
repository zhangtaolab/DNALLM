#!/usr/bin/env python3
"""Standalone zero-shot variant effect prediction CLI for DNALLM."""

import json
import sys
from pathlib import Path

import click

from ..utils import get_logger

logger = get_logger("dnallm.cli.vep")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to a YAML config file with a 'vep' section (VepConfig: "
    "paradigm, context_window, output_dir)",
)
@click.option(
    "--vcf",
    type=click.Path(exists=True),
    required=True,
    help="Path to the (optionally gzipped) VCF to score",
)
@click.option(
    "--reference",
    type=click.Path(exists=True),
    required=True,
    help="Path to the reference genome FASTA (plain or .gz)",
)
@click.option(
    "--model-name",
    "-m",
    type=str,
    required=True,
    help="Model name or path for variant scoring",
)
@click.option(
    "--source",
    type=click.Choice(["huggingface", "modelscope"]),
    default="huggingface",
    help="Model registry to load from",
)
@click.option(
    "--paradigm",
    type=click.Choice(["clm", "mlm"]),
    default=None,
    help="Scoring paradigm (default: VepConfig value, else 'mlm')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file path (defaults to stdout)",
)
def main(config, vcf, reference, model_name, source, paradigm, output):
    """Score zero-shot variant effects from a VCF (dnallm-vep)."""
    from ..inference.vep import evaluate_vcf
    from ..models import load_model_and_tokenizer

    # Config: the 'vep' section supplies defaults the CLI flags override.
    context_window = 200  # mirrors VepConfig.context_window
    output_dir = None
    if config:
        try:
            import yaml

            from ..configuration.configs import VepConfig

            with open(config, encoding="utf-8") as handle:
                raw = yaml.safe_load(handle) or {}
            vep_cfg = VepConfig(**(raw.get("vep", raw) if isinstance(raw, dict) else {}))
            context_window = vep_cfg.context_window
            output_dir = vep_cfg.output_dir
            if paradigm is None:
                paradigm = vep_cfg.paradigm
        except Exception as e:
            click.echo(f"Error loading config '{config}': {e}", err=True)
            sys.exit(1)
    if paradigm is None:
        paradigm = "mlm"

    # The paradigm decides which Auto* head loads the model through:
    # mlm -> AutoModelForMaskedLM (task 'mask'), clm -> AutoModelForCausalLM
    # (task 'generation') — the pairing the D-11 guard then verifies.
    task_type = "generation" if paradigm == "clm" else "mask"

    try:
        from ..configuration.configs import TaskConfig

        task_config = TaskConfig(task_type=task_type)
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, task_config=task_config, source=source
        )
    except Exception as e:
        click.echo(f"Error loading model '{model_name}': {e}", err=True)
        sys.exit(1)

    try:
        result = evaluate_vcf(
            model,
            tokenizer,
            vcf,
            reference,
            paradigm=paradigm,
            context_window=context_window,
            output_dir=output_dir,
        )
        payload = result.to_dict()
        payload["model_name"] = model_name
        payload["paradigm"] = paradigm
    except Exception as e:
        click.echo(f"VEP evaluation failed: {e}", err=True)
        sys.exit(1)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        click.echo(f"Results saved to: {output}")
    else:
        click.echo(json.dumps(payload, indent=2))

    logger.info(
        f"dnallm-vep finished: model={model_name} paradigm={paradigm} "
        f"evaluated={result.evaluated} skipped={result.skipped}"
    )


if __name__ == "__main__":  # pragma: no cover - console-script entry
    main()
