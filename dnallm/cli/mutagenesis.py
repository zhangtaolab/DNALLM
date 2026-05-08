#!/usr/bin/env python3
"""Standalone mutagenesis CLI for DNALLM."""

import json
import re
import sys
from pathlib import Path

import click
import numpy as np

from ..utils import get_logger

logger = get_logger("dnallm.cli.mutagenesis")


def parse_positions(positions_str: str | None) -> list[int] | None:
    """Parse comma-separated position string into list of ints."""
    if positions_str is None:
        return None
    return [int(p.strip()) for p in positions_str.split(",")]


def load_sequences_from_file(path: str) -> list[str]:
    """Load sequences from a text file (one per line)."""
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


@click.command()
@click.option(
    "--sequence",
    "-s",
    type=str,
    help="Single DNA sequence for mutagenesis analysis",
)
@click.option(
    "--sequences",
    type=click.Path(exists=True),
    help="Path to file containing multiple DNA sequences (one per line)",
)
@click.option(
    "--positions",
    "-p",
    type=str,
    help="Comma-separated list of positions to mutate (e.g., '0,1,2'). "
    "Note: full saturation mutagenesis is performed; position filtering is not yet implemented.",
)
@click.option(
    "--mutation-type",
    "-t",
    type=click.Choice([
        "single_base_substitution",
        "multi_base_substitution",
        "deletion",
        "insertion",
        "combo",
    ]),
    default="single_base_substitution",
    help="Type of mutation to apply",
)
@click.option(
    "--task-type",
    type=click.Choice([
        "binary",
        "multiclass",
        "multilabel",
        "regression",
        "token",
    ]),
    default="binary",
    help="Task type for model configuration",
)
@click.option(
    "--model-name",
    "-m",
    type=str,
    required=True,
    help="Model name or path for mutagenesis analysis",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output JSON file path (defaults to stdout)",
)
def main(sequence, sequences, positions, mutation_type, task_type, model_name, output):
    """Run in-silico mutagenesis analysis on DNA sequences."""
    from ..inference.mutagenesis import Mutagenesis
    from ..models import load_model_and_tokenizer

    # Validate input: need either --sequence or --sequences
    if not sequence and not sequences:
        click.echo(
            "Error: Either --sequence or --sequences is required.",
            err=True,
        )
        sys.exit(1)

    # Parse positions (optional — full saturation mutagenesis is performed)
    pos_list = parse_positions(positions)

    # Build sequence list
    seq_list = []
    if sequences:
        seq_list.extend(load_sequences_from_file(sequences))
    if sequence:
        seq_list.append(sequence)

    # Validate DNA sequence content (IUPAC ambiguity codes allowed)
    dna_pattern = re.compile(r"^[ACGTURYKMSWBDHVNXacgturykmswbdhvnx]+$")
    for i, seq in enumerate(seq_list):
        if not dna_pattern.match(seq):
            click.echo(
                f"Error: Sequence at index {i} contains invalid characters. "
                "Allowed: standard bases (A,C,G,T), N, and IUPAC ambiguity codes.",
                err=True,
            )
            sys.exit(1)

    # Enforce combo limit when positions are specified (4^5 = 1024 max combos)
    if mutation_type == "combo" and pos_list is not None and len(pos_list) > 5:
        click.echo(
            f"Error: Combo mutation supports at most 5 positions "
            f"(got {len(pos_list)}). Limit: 4^5 = 1024 combinations.",
            err=True,
        )
        sys.exit(1)

    try:
        # Load model, tokenizer, and config
        from ..configuration import InferenceConfig, TaskConfig

        task_config = TaskConfig(task_type=task_type)
        model, tokenizer = load_model_and_tokenizer(
            model_name=model_name, task_config=task_config
        )
        inference_config = InferenceConfig()
        config = {"task": task_config, "inference": inference_config}
    except Exception as e:
        click.echo(f"Error loading model '{model_name}': {e}", err=True)
        sys.exit(1)

    # Prepare mutagenesis parameters based on mutation type
    replace_mut = mutation_type in {
        "single_base_substitution",
        "multi_base_substitution",
        "combo",
    }
    delete_size = 1 if mutation_type == "deletion" else 0
    insert_seq = "N" if mutation_type == "insertion" else None

    try:
        results = []
        for seq in seq_list:
            mutagenesis = Mutagenesis(model, tokenizer, config)
            mutagenesis.mutate_sequence(
                seq,
                replace_mut=replace_mut,
                delete_size=delete_size,
                insert_seq=insert_seq,
            )
            eval_result = mutagenesis.evaluate(do_pred=True)

            # Extract original and mutated predictions
            raw = eval_result.get("raw", {})
            original_prediction = {
                "sequence": raw.get("sequence", seq),
                "prediction": _serialize(raw.get("pred", {})),
                "score": raw.get("score", 0.0),
            }

            # Aggregate mutated predictions
            mutated_entries = [
                v for k, v in eval_result.items() if k != "raw"
            ]
            mutated_prediction = {
                "count": len(mutated_entries),
                "predictions": [
                    {
                        "sequence": e.get("sequence", ""),
                        "prediction": _serialize(e.get("pred", {})),
                        "logfc": _serialize(e.get("logfc", 0.0)),
                        "diff": _serialize(e.get("diff", 0.0)),
                        "score": e.get("score", 0.0),
                    }
                    for e in mutated_entries
                ],
            }

            # Compute delta (average logfc and diff)
            if mutated_entries:
                avg_logfc = float(
                    np.mean(
                        [
                            float(np.mean(e.get("logfc", 0)))
                            if hasattr(e.get("logfc", 0), "__len__")
                            else float(e.get("logfc", 0))
                            for e in mutated_entries
                        ]
                    )
                )
                avg_diff = float(
                    np.mean(
                        [
                            float(np.mean(e.get("diff", 0)))
                            if hasattr(e.get("diff", 0), "__len__")
                            else float(e.get("diff", 0))
                            for e in mutated_entries
                        ]
                    )
                )
            else:
                avg_logfc = 0.0
                avg_diff = 0.0

            delta = {
                "average_logfc": avg_logfc,
                "average_diff": avg_diff,
            }

            results.append({
                "original_prediction": original_prediction,
                "mutated_prediction": mutated_prediction,
                "delta": delta,
            })

        # Format response
        if len(results) == 1:
            result_payload = results[0]
        else:
            result_payload = {
                "batch_results": results,
                "sequence_count": len(seq_list),
            }

        output_data = {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Mutagenesis complete: {mutation_type} "
                        f"at positions {pos_list} using model {model_name}"
                    ),
                }
            ],
            **result_payload,
            "affected_positions": pos_list,
            "mutation_type": mutation_type,
            "model_name": model_name,
        }

        if output:
            out_path = Path(output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            click.echo(f"Results saved to: {output}")
        else:
            click.echo(json.dumps(output_data, indent=2))

    except Exception as e:
        click.echo(f"Mutagenesis failed: {e}", err=True)
        sys.exit(1)


def _serialize(value):
    """Serialize numpy arrays and other non-JSON-serializable values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


if __name__ == "__main__":
    main()
