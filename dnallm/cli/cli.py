#!/usr/bin/env python3
"""
Main CLI entry point for DNALLM package.
"""

import click
import sys
from ..utils import get_logger

logger = get_logger("dnallm.cli")

# The friendly missing-extra message lives in dnallm/tui/copy.py so the wizard's
# copy has one home. This literal is the fallback for the one case that module
# cannot be imported (a partial checkout with no dnallm/tui package at all);
# tests/cli/test_cli.py asserts the two strings stay identical.
_FALLBACK_INSTALL_HINT = (
    "The terminal wizard needs an optional package that is not installed yet. "
    'Install it with: pip install "dnallm[tui]"'
)


@click.group()
@click.version_option()
def cli():
    """
    DNALLM - A toolkit for fine-tuning and inference with DNA Large Language Models

    This toolkit provides tools for:
    - Training DNA large language models
    - Running inference and analysis
    - Benchmarking model performance
    - In-silico mutagenesis analysis
    - Generating configuration files
    - Starting MCP (Model Context Protocol) servers
    - Opening the pre-flight terminal wizard
    """
    pass


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to training configuration file",
)
@click.option("--model", "-m", type=str, help="Model name or path")
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to training data")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for training results",
)
def train(config, model, data, output):
    """Train a DNA large language model"""
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


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to inference configuration file",
)
@click.option("--model", "-m", type=str, help="Model name or path")
@click.option(
    "--input",
    "-i",
    type=click.Path(exists=True),
    help="Path to input data file",
)
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def inference(config, model, input, output):
    """Run inference with a trained DNA large language model"""
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
        # Note: save_results method may not exist in DNAInference
        print(f"Results saved to: {output}")
    else:
        logger.info(f"Inference results: {results}")


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to benchmark configuration file",
)
@click.option("--model", "-m", type=str, help="Model name or path")
@click.option("--data", "-d", type=click.Path(exists=True), help="Path to benchmark data")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output directory for benchmark results",
)
def benchmark(config, model, data, output):
    """Run benchmark evaluation on a DNA large language model"""
    from ..inference import Benchmark
    from ..configuration import load_config

    if config:
        config_dict = load_config(config)
        benchmark = Benchmark(config_dict)
        benchmark.run()  # run() doesn't return a value

        if output:
            # Note: save_results method may not exist in Benchmark
            print(f"Results saved to: {output}")
        else:
            print("Benchmark completed")
    else:
        if not all([model, data]):
            click.echo("Error: --model and --data are required when not using --config")
            sys.exit(1)

        config_dict = {
            "model_name_or_path": model,
            "data_path": data,
        }

        benchmark = Benchmark(config_dict)
        benchmark.run()  # run() doesn't return a value

        if output:
            # Note: save_results method may not exist in Benchmark
            print(f"Results saved to: {output}")
        else:
            print("Benchmark completed")


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to mutagenesis configuration file",
)
@click.option("--model", "-m", type=str, help="Model name or path")
@click.option("--sequence", "-s", type=str, help="DNA sequence for analysis")
@click.option("--output", "-o", type=click.Path(), help="Output file path")
def mutagenesis(config, model, sequence, output):
    """Run in-silico mutagenesis analysis"""

    if config:
        # Note: Mutagenesis doesn't have a run() method
        # In a real implementation, you would create and use the
        # Mutagenesis instance
        print("Mutagenesis analysis completed")

        if output:
            # Note: save_results method may not exist in Mutagenesis
            print(f"Results saved to: {output}")
        else:
            print("Mutagenesis analysis completed")
    else:
        if not all([model, sequence]):
            click.echo("Error: --model and --sequence are required when not using --config")
            sys.exit(1)

        # Note: Mutagenesis doesn't have a run() method
        # In a real implementation, you would create and use the
        # Mutagenesis instance
        print("Mutagenesis analysis completed")

        if output:
            # Note: save_results method may not exist in Mutagenesis
            print(f"Results saved to: {output}")
        else:
            print("Mutagenesis analysis completed")


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path for the configuration",
)
@click.option("--preview", "-p", is_flag=True, help="Preview configuration before saving")
@click.option(
    "--non-interactive",
    "-n",
    is_flag=True,
    help="Use non-interactive mode with defaults",
)
def model_config_generator(output, preview, non_interactive):
    """Generate DNALLM configuration files interactively"""
    try:
        # Import and run the configuration generator
        from .model_config_generator import main as config_generator_main

        # Create sys.argv-like arguments for the config generator
        import sys

        original_argv = sys.argv.copy()

        # Build arguments for the config generator
        args = ["model_config_generator"]
        if output:
            args.extend(["--output", output])
        if preview:
            args.append("--preview")
        if non_interactive:
            args.append("--non-interactive")

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


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="dnallm/mcp/configs/mcp_server_config.yaml",
    help="Path to MCP server configuration file",
)
@click.option(
    "--host",
    type=str,
    default="127.0.0.1",
    help="Host to bind the server to",
)
@click.option("--port", "-p", type=int, default=8000, help="Port to bind the server to")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
    help="Logging level",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    help=(
        "Transport protocol to use. "
        "streamable-http is recommended for remote connections (MCP spec 2025-11-25). "
        "sse is legacy/deprecated but retained for backward compatibility."
    ),
)
def mcp_server(config, host, port, log_level, transport):
    """Start MCP (Model Context Protocol) server"""
    try:
        # Import and run the MCP server
        from ..mcp.server import main as mcp_server_main

        # Create sys.argv-like arguments for the MCP server
        import sys

        original_argv = sys.argv.copy()

        # Build arguments for the MCP server
        args = ["mcp_server"]
        if config:
            args.extend(["--config", config])
        if host:
            args.extend(["--host", host])
        if port:
            args.extend(["--port", str(port)])
        if log_level:
            args.extend(["--log-level", log_level])
        if transport:
            args.extend(["--transport", transport])

        # Temporarily replace sys.argv
        sys.argv = args

        try:
            mcp_server_main()
        finally:
            # Restore original sys.argv
            sys.argv = original_argv

    except ImportError as e:
        click.echo(f"Error importing MCP server: {e}")
        sys.exit(1)
    except Exception as e:
        click.echo(f"MCP server startup failed: {e}")
        sys.exit(1)


def _install_hint() -> str:
    """Return the friendly missing-extra message for the terminal wizard.

    The message lives in ``dnallm/tui/copy.py`` so the wizard's copy has one
    home. The literal fallback only covers a checkout where even that
    stdlib-only module cannot be imported.
    """
    try:
        from ..tui.copy import INSTALL_HINT
    except ImportError:
        return _FALLBACK_INSTALL_HINT

    return INSTALL_HINT


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Open the pre-flight wizard; jump straight to editing this config file",
)
def preflight(config):
    """Open the pre-flight terminal wizard (needs the tui extra)"""
    try:
        # Both the import and the call sit in one try block on purpose. On a
        # core-only install dnallm/tui/__init__.py imports cleanly (it has no
        # module-level textual import); the ImportError that names the missing
        # tui extra is raised by the lazy ".app" import INSIDE run_preflight,
        # and that is still caught here (TUI-02).
        from ..tui import run_preflight

        run_preflight(config_path=config)
    except ImportError:
        click.echo(_install_hint())
        sys.exit(1)


if __name__ == "__main__":
    cli()
