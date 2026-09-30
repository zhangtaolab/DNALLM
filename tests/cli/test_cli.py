"""Tests for every DNALLM CLI entry point through click.testing.CliRunner.

All commands lazy-import their heavy cores inside the command body, so each
test patches the core at the site the lazy from-import resolves (the origin
package attribute, e.g. ``dnallm.finetune.DNATrainer``) and asserts the core
was invoked with the parsed arguments. CliRunner executes everything
in-process — nothing in this file spawns or shells out (AUDIT-04 intact).
"""

from __future__ import annotations

import json
import sys
import types
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from dnallm.cli import inference as inference_module
from dnallm.cli import model_config_generator as generator_module
from dnallm.cli import mutagenesis as mutagenesis_module
from dnallm.cli import train as train_module
from dnallm.cli.cli import cli
from dnallm.cli.mutagenesis import load_sequences_from_file, parse_positions

if TYPE_CHECKING:
    from pathlib import Path

SENTINEL_CONFIG = {"sentinel": "config"}


@pytest.fixture
def runner():
    """Return a CliRunner with exceptions caught (SystemExit -> exit_code)."""
    return CliRunner()


@pytest.fixture
def config_file(tmp_path):
    """Return an existing (content-irrelevant) config path for -c options."""
    path = tmp_path / "config.yaml"
    path.write_text("# stub config; load_config is patched in the tests that use it\n")
    return str(path)


@pytest.fixture
def data_file(tmp_path):
    """Return an existing data file for --data/--input options."""
    path = tmp_path / "data.csv"
    path.write_text("sequence,label\nATCG,0\n")
    return str(path)


class TestCliGroup:
    """The dnallm click group surface."""

    def test_no_args_prints_help_with_all_subcommands(self, runner):
        """Invoking with no args shows help listing every subcommand.

        click emits the help text either way; the exit code for a bare group
        invocation differs across click versions (0 vs 2), so only the help
        content is pinned.
        """
        result = runner.invoke(cli, [])
        assert result.exit_code in (0, 2)
        assert "Usage:" in result.output
        for command in (
            "train",
            "inference",
            "benchmark",
            "mutagenesis",
            "model-config-generator",
            "mcp-server",
        ):
            assert command in result.output

    def test_help_flag_lists_subcommands(self, runner):
        """--help exits 0 with usage text and the subcommand list."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage:" in result.output
        assert "mcp-server" in result.output

    def test_version_flag_reports_package_version(self, runner):
        """--version exits 0 and prints a version line."""
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "version" in result.output


class TestTrainCommand:
    """The `train` subcommand of the dnallm group."""

    def test_config_path_loads_config_and_trains(self, runner, config_file):
        """-c loads the YAML via load_config and drives DNATrainer.train()."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.finetune.DNATrainer") as trainer_cls,
        ):
            result = runner.invoke(cli, ["train", "--config", config_file])

        assert result.exit_code == 0
        trainer_cls.assert_called_once_with(model=None, config=SENTINEL_CONFIG)
        trainer_cls.return_value.train.assert_called_once_with()

    def test_missing_required_options_without_config(self, runner):
        """No --config and no --model/--data/--output exits 1 with usage help."""
        result = runner.invoke(cli, ["train"])
        assert result.exit_code == 1
        assert "--model, --data, and --output are required" in result.output

    def test_minimal_config_from_cli_arguments(self, runner, data_file, tmp_path):
        """--model/--data/--output builds the minimal config dict for the trainer."""
        output_dir = str(tmp_path / "run")
        with patch("dnallm.finetune.DNATrainer") as trainer_cls:
            result = runner.invoke(
                cli,
                ["train", "-m", "tiny-dna", "-d", data_file, "-o", output_dir],
            )

        assert result.exit_code == 0
        config = trainer_cls.call_args.kwargs["config"]
        assert trainer_cls.call_args.kwargs["model"] is None
        assert config["model_name_or_path"] == "tiny-dna"
        assert config["data_path"] == data_file
        assert config["output_dir"] == output_dir
        assert config["finetune"]["num_train_epochs"] == 3
        trainer_cls.return_value.train.assert_called_once_with()

    def test_nonexistent_config_path_is_a_usage_error(self, runner):
        """click.Path(exists=True) rejects a missing config with exit code 2."""
        result = runner.invoke(cli, ["train", "-c", "/no/such/config.yaml"])
        assert result.exit_code == 2
        assert "Usage:" in result.output


class TestInferenceCommand:
    """The `inference` subcommand of the dnallm group."""

    def test_config_path_with_output_prints_save_location(self, runner, config_file):
        """-c wires the loaded config into DNAInference; -o echoes the save path."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.DNAInference") as engine_cls,
        ):
            engine_cls.return_value.infer.return_value = {"done": True}
            result = runner.invoke(cli, ["inference", "-c", config_file, "-o", "out.txt"])

        assert result.exit_code == 0
        engine_cls.assert_called_once_with(model=None, tokenizer=None, config=SENTINEL_CONFIG)
        engine_cls.return_value.infer.assert_called_once_with()
        assert "Results saved to: out.txt" in result.output

    def test_missing_required_options_without_config(self, runner):
        """No --config and no --model/--input exits 1 with the error text."""
        result = runner.invoke(cli, ["inference"])
        assert result.exit_code == 1
        assert "--model and --input are required" in result.output

    def test_minimal_config_builds_two_key_dict(self, runner, data_file, caplog):
        """--model/--input builds the minimal config and logs the results."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            engine_cls.return_value.infer.return_value = {"sentinel": 1}
            result = runner.invoke(cli, ["inference", "-m", "m1", "-i", data_file])

        assert result.exit_code == 0
        config = engine_cls.call_args.kwargs["config"]
        assert config == {"model_name_or_path": "m1", "data_path": data_file}
        assert "Inference results" in caplog.text

    def test_config_path_without_output_logs_results(self, runner, config_file, caplog):
        """-c without -o logs the results instead of printing a save path."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.DNAInference") as engine_cls,
        ):
            engine_cls.return_value.infer.return_value = {"v": 42}
            result = runner.invoke(cli, ["inference", "-c", config_file])

        assert result.exit_code == 0
        assert "Inference results" in caplog.text
        assert "Results saved to" not in result.output

    def test_minimal_config_with_output_prints_save_location(self, runner, data_file):
        """--model/--input with -o echoes the save path on the minimal path."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            engine_cls.return_value.infer.return_value = {"v": 1}
            result = runner.invoke(cli, ["inference", "-m", "m1", "-i", data_file, "-o", "res.txt"])

        assert result.exit_code == 0
        assert "Results saved to: res.txt" in result.output


class TestBenchmarkCommand:
    """The `benchmark` subcommand of the dnallm group."""

    def test_config_path_runs_benchmark(self, runner, config_file):
        """-c loads the config, constructs Benchmark and runs it."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.Benchmark") as benchmark_cls,
        ):
            result = runner.invoke(cli, ["benchmark", "-c", config_file])

        assert result.exit_code == 0
        benchmark_cls.assert_called_once_with(SENTINEL_CONFIG)
        benchmark_cls.return_value.run.assert_called_once_with()
        assert "Benchmark completed" in result.output

    def test_missing_required_options_without_config(self, runner):
        """No --config and no --model/--data exits 1 with the error text."""
        result = runner.invoke(cli, ["benchmark"])
        assert result.exit_code == 1
        assert "--model and --data are required" in result.output

    def test_config_path_with_output_prints_save_location(self, runner, config_file):
        """-c with -o echoes the save path after the run."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.Benchmark"),
        ):
            result = runner.invoke(cli, ["benchmark", "-c", config_file, "-o", "bench/"])

        assert result.exit_code == 0
        assert "Results saved to: bench/" in result.output

    def test_minimal_config_from_cli_arguments(self, runner, data_file, tmp_path):
        """--model/--data builds the minimal config dict and runs the benchmark."""
        with patch("dnallm.inference.Benchmark") as benchmark_cls:
            result = runner.invoke(
                cli,
                ["benchmark", "-m", "tiny-dna", "-d", data_file],
            )

        assert result.exit_code == 0
        config = benchmark_cls.call_args.args[0]
        assert config == {"model_name_or_path": "tiny-dna", "data_path": data_file}
        benchmark_cls.return_value.run.assert_called_once_with()
        assert "Benchmark completed" in result.output

    def test_minimal_config_with_output_prints_save_location(self, runner, data_file):
        """--model/--data with -o echoes the save path on the minimal path."""
        with patch("dnallm.inference.Benchmark"):
            result = runner.invoke(
                cli, ["benchmark", "-m", "tiny-dna", "-d", data_file, "-o", "out/"]
            )

        assert result.exit_code == 0
        assert "Results saved to: out/" in result.output


class TestMutagenesisSubcommand:
    """The `mutagenesis` subcommand of the dnallm group (stub implementation)."""

    def test_config_path_prints_completion(self, runner, config_file):
        """-c prints the completion notice."""
        result = runner.invoke(cli, ["mutagenesis", "-c", config_file])
        assert result.exit_code == 0
        assert "Mutagenesis analysis completed" in result.output

    def test_output_option_prints_save_location(self, runner, config_file):
        """-o additionally prints where results were saved."""
        result = runner.invoke(cli, ["mutagenesis", "-c", config_file, "-o", "muts.json"])
        assert result.exit_code == 0
        assert "Results saved to: muts.json" in result.output

    def test_missing_required_options_without_config(self, runner):
        """No --config and no --model/--sequence exits 1 with the error text."""
        result = runner.invoke(cli, ["mutagenesis"])
        assert result.exit_code == 1
        assert "--model and --sequence are required" in result.output

    def test_model_and_sequence_without_config_prints_completion(self, runner):
        """--model/--sequence without -c prints the completion notice."""
        result = runner.invoke(cli, ["mutagenesis", "-m", "model", "-s", "ATCG"])
        assert result.exit_code == 0
        assert "Mutagenesis analysis completed" in result.output

    def test_model_and_sequence_with_output_prints_save_location(self, runner):
        """--model/--sequence plus -o prints the save path on the minimal path."""
        result = runner.invoke(cli, ["mutagenesis", "-m", "model", "-s", "ATCG", "-o", "out.json"])
        assert result.exit_code == 0
        assert "Results saved to: out.json" in result.output


class TestModelConfigGeneratorCommand:
    """The `model-config-generator` subcommand of the dnallm group."""

    def test_invokes_generator_with_assembled_argv(self, runner, config_file):
        """Options are forwarded as a sys.argv array to the generator main."""
        recorded = {}

        def recording_main():
            recorded["argv"] = list(sys.argv)

        argv_before = list(sys.argv)
        with patch(
            "dnallm.cli.model_config_generator.main", side_effect=recording_main
        ) as generator_main:
            result = runner.invoke(
                cli,
                [
                    "model-config-generator",
                    "-o",
                    "/tmp/gen.yaml",
                    "--preview",
                    "--non-interactive",
                ],
            )

        assert result.exit_code == 0
        generator_main.assert_called_once_with()
        assert recorded["argv"] == [
            "model_config_generator",
            "--output",
            "/tmp/gen.yaml",
            "--preview",
            "--non-interactive",
        ]
        assert sys.argv == argv_before  # restored after the invocation

    def test_generator_failure_exits_one(self, runner):
        """An exception inside the generator exits 1 with the failure text."""
        with patch(
            "dnallm.cli.model_config_generator.main",
            side_effect=ValueError("bad generator state"),
        ):
            result = runner.invoke(cli, ["model-config-generator"])

        assert result.exit_code == 1
        assert "Configuration generation failed: bad generator state" in result.output

    def test_generator_import_error_exits_one(self, runner, monkeypatch):
        """An unimportable generator module exits 1 with the import error text."""
        monkeypatch.setitem(sys.modules, "dnallm.cli.model_config_generator", None)
        result = runner.invoke(cli, ["model-config-generator"])
        assert result.exit_code == 1
        assert "Error importing configuration generator" in result.output


class TestMcpServerCommand:
    """The `mcp-server` subcommand of the dnallm group."""

    def test_invokes_mcp_main_with_assembled_argv(self, runner, config_file):
        """All options are forwarded as a sys.argv array to the server main."""
        recorded = {}

        def recording_main():
            recorded["argv"] = list(sys.argv)

        argv_before = list(sys.argv)
        with patch("dnallm.mcp.server.main", side_effect=recording_main) as mcp_main:
            result = runner.invoke(
                cli,
                [
                    "mcp-server",
                    "--config",
                    config_file,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "9001",
                    "--transport",
                    "sse",
                ],
            )

        assert result.exit_code == 0
        mcp_main.assert_called_once_with()
        assert recorded["argv"] == [
            "mcp_server",
            "--config",
            config_file,
            "--host",
            "127.0.0.1",
            "--port",
            "9001",
            "--log-level",
            "INFO",
            "--transport",
            "sse",
        ]
        assert sys.argv == argv_before

    def test_mcp_server_failure_exits_one(self, runner, config_file):
        """An exception inside the server main exits 1 with the failure text."""
        with (
            patch("dnallm.mcp.server.main", side_effect=RuntimeError("port taken")),
        ):
            result = runner.invoke(cli, ["mcp-server", "--config", config_file])

        assert result.exit_code == 1
        assert "MCP server startup failed: port taken" in result.output

    def test_mcp_server_import_error_exits_one(self, runner, monkeypatch):
        """An unimportable server module exits 1 with the import error text."""
        monkeypatch.setitem(sys.modules, "dnallm.mcp.server", None)
        result = runner.invoke(cli, ["mcp-server"])
        assert result.exit_code == 1
        assert "Error importing MCP server" in result.output


class TestStandaloneTrainModule:
    """The dnallm-train console script module (dnallm.cli.train)."""

    def test_main_with_config_loads_and_trains(self, runner, config_file):
        """The standalone main mirrors the group train wiring."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.finetune.DNATrainer") as trainer_cls,
        ):
            result = runner.invoke(train_module.main, ["-c", config_file])

        assert result.exit_code == 0
        trainer_cls.assert_called_once_with(model=None, config=SENTINEL_CONFIG)
        trainer_cls.return_value.train.assert_called_once_with()

    def test_main_without_config_exits_one(self, runner):
        """Missing --model/--data/--output exits 1 with the error on stderr."""
        result = runner.invoke(train_module.main, [])
        assert result.exit_code == 1
        assert "--model, --data, and --output are required" in result.output

    def test_main_minimal_config_from_cli_arguments(self, runner, data_file, tmp_path):
        """--model/--data/--output builds the minimal config for the trainer."""
        output_dir = str(tmp_path / "standalone-run")
        with patch("dnallm.finetune.DNATrainer") as trainer_cls:
            result = runner.invoke(
                train_module.main,
                ["-m", "tiny-dna", "-d", data_file, "-o", output_dir],
            )

        assert result.exit_code == 0
        config = trainer_cls.call_args.kwargs["config"]
        assert config["model_name_or_path"] == "tiny-dna"
        assert config["data_path"] == data_file
        assert config["output_dir"] == output_dir
        assert config["finetune"]["learning_rate"] == 5e-5
        trainer_cls.return_value.train.assert_called_once_with()


class TestStandaloneInferenceModule:
    """The dnallm-inference console script module (dnallm.cli.inference)."""

    def test_main_minimal_path_builds_config_dict(self, runner, data_file):
        """--model/--input builds the minimal two-key config for DNAInference."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            engine_cls.return_value.infer.return_value = []
            result = runner.invoke(inference_module.main, ["-m", "m2", "-i", data_file])

        assert result.exit_code == 0
        config = engine_cls.call_args.kwargs["config"]
        assert config == {"model_name_or_path": "m2", "data_path": data_file}
        engine_cls.return_value.infer.assert_called_once_with()

    def test_main_with_config_loads_config(self, runner, config_file):
        """-c loads the YAML via load_config and drives DNAInference.infer()."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.DNAInference") as engine_cls,
        ):
            engine_cls.return_value.infer.return_value = []
            result = runner.invoke(inference_module.main, ["-c", config_file])

        assert result.exit_code == 0
        engine_cls.assert_called_once_with(model=None, tokenizer=None, config=SENTINEL_CONFIG)
        engine_cls.return_value.infer.assert_called_once_with()

    def test_main_without_config_exits_one(self, runner):
        """Missing --model/--input exits 1 with the error on stderr."""
        result = runner.invoke(inference_module.main, [])
        assert result.exit_code == 1
        assert "--model and --input are required" in result.output

    def test_main_with_config_and_output_prints_save_location(self, runner, config_file):
        """-c with -o echoes the save path after the config-driven run."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.inference.DNAInference") as engine_cls,
        ):
            engine_cls.return_value.infer.return_value = []
            result = runner.invoke(inference_module.main, ["-c", config_file, "-o", "cfg-out.txt"])

        assert result.exit_code == 0
        assert "Results saved to: cfg-out.txt" in result.output

    def test_main_minimal_path_with_output_prints_save_location(self, runner, data_file):
        """--model/--input with -o echoes the save path."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            engine_cls.return_value.infer.return_value = []
            result = runner.invoke(
                inference_module.main, ["-m", "m2", "-i", data_file, "-o", "o.txt"]
            )

        assert result.exit_code == 0
        assert "Results saved to: o.txt" in result.output


class TestStandaloneModelConfigGeneratorModule:
    """The dnallm-model-config-generator module (dnallm.cli.model_config_generator)."""

    @staticmethod
    def _root_cli_fake(main_mock):
        """Build a fake root cli.model_config_generator module."""
        fake_module = types.ModuleType("cli.model_config_generator")
        fake_module.main = main_mock
        return fake_module

    def test_main_delegates_to_root_cli_generator(self, monkeypatch):
        """main() imports the root cli generator and calls its main()."""
        generate_config = Mock()
        monkeypatch.setitem(
            sys.modules,
            "cli.model_config_generator",
            self._root_cli_fake(generate_config),
        )
        saved_path = list(sys.path)
        try:
            generator_module.main()
        finally:
            sys.path[:] = saved_path  # undo the sys.path.insert done by main()

        generate_config.assert_called_once_with()

    def test_main_import_error_prints_fallback_and_exits(self, monkeypatch, capsys):
        """A missing root cli module prints guidance and exits 1."""
        monkeypatch.setitem(sys.modules, "cli.model_config_generator", None)
        saved_path = list(sys.path)
        try:
            with pytest.raises(SystemExit) as excinfo:
                generator_module.main()
        finally:
            sys.path[:] = saved_path

        assert excinfo.value.code == 1
        out = capsys.readouterr().out
        assert "DNALLM Model Configuration Generator" in out
        assert "python cli/model_config_generator.py" in out

    def test_main_generic_failure_prints_and_exits(self, monkeypatch, capsys):
        """An exception inside the generator prints the failure and exits 1."""
        generate_config = Mock(side_effect=ValueError("boom"))
        monkeypatch.setitem(
            sys.modules,
            "cli.model_config_generator",
            self._root_cli_fake(generate_config),
        )
        saved_path = list(sys.path)
        try:
            with pytest.raises(SystemExit) as excinfo:
                generator_module.main()
        finally:
            sys.path[:] = saved_path

        assert excinfo.value.code == 1
        assert "Configuration generation failed: boom" in capsys.readouterr().out


def _mutagenesis_eval_result(sequence):
    """Build a Mutagenesis.evaluate() result with one array and one scalar mutant."""
    return {
        "raw": {"sequence": sequence, "pred": {"binary": 0.87}, "score": 0.87},
        "mut_0": {
            "sequence": "T" + sequence[1:],
            "pred": np.array([0.42, 0.58]),
            "logfc": np.array([0.11, -0.07]),
            "diff": np.array([0.2, 0.1]),
            "score": 0.58,
        },
        "mut_1": {
            "sequence": "A" + sequence[1:],
            "pred": np.array([0.91, 0.09]),
            "logfc": 0.3,
            "diff": -0.1,
            "score": 0.91,
        },
    }


class TestStandaloneMutagenesModule:
    """The dnallm-mutagenesis console script module (dnallm.cli.mutagenesis)."""

    def test_parse_positions_none_returns_none(self):
        """parse_positions maps None to None (full saturation)."""
        assert parse_positions(None) is None

    def test_parse_positions_splits_and_strips(self):
        """parse_positions parses comma-separated ints with whitespace."""
        assert parse_positions("0, 1,2") == [0, 1, 2]

    def test_load_sequences_from_file_skips_blank_lines(self, tmp_path):
        """Only non-blank stripped lines are loaded."""
        path = tmp_path / "seqs.txt"
        path.write_text("ATCG\n\n   \nGCTA\n")
        assert load_sequences_from_file(str(path)) == ["ATCG", "GCTA"]

    def test_requires_sequence_or_sequences(self, runner):
        """Neither --sequence nor --sequences exits 1."""
        result = runner.invoke(mutagenesis_module.main, ["-m", "model"])
        assert result.exit_code == 1
        assert "Either --sequence or --sequences is required" in result.output

    def test_rejects_invalid_sequence_characters(self, runner):
        """Non-IUPAC characters exit 1 with the index in the message."""
        result = runner.invoke(mutagenesis_module.main, ["-m", "model", "-s", "ATCGZZ"])
        assert result.exit_code == 1
        assert "Sequence at index 0 contains invalid characters" in result.output

    def test_rejects_combo_over_five_positions(self, runner):
        """--mutation-type combo with more than five positions exits 1."""
        result = runner.invoke(
            mutagenesis_module.main,
            ["-m", "model", "-s", "ATCGAA", "-t", "combo", "-p", "0,1,2,3,4,5"],
        )
        assert result.exit_code == 1
        assert "Combo mutation supports at most 5 positions" in result.output

    def test_model_load_failure_exits_one(self, runner):
        """A failing model load exits 1 naming the model."""
        with patch(
            "dnallm.models.load_model_and_tokenizer",
            side_effect=ValueError("Model ghost-model download failed."),
        ):
            result = runner.invoke(mutagenesis_module.main, ["-m", "ghost-model", "-s", "ATCG"])
        assert result.exit_code == 1
        assert "Error loading model 'ghost-model'" in result.output

    def test_mutagenesis_failure_exits_one(self, runner):
        """A failing evaluation exits 1 with the failure text."""
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(Mock(), Mock())),
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            mut_cls.return_value.evaluate.side_effect = RuntimeError("eval crashed")
            result = runner.invoke(mutagenesis_module.main, ["-m", "model", "-s", "ATCG"])
        assert result.exit_code == 1
        assert "Mutagenesis failed: eval crashed" in result.output

    def test_single_sequence_stdout_payload(self, runner):
        """The full happy path emits a structured JSON payload on stdout."""
        with (
            patch("dnallm.models.load_model_and_tokenizer") as load_mock,
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            load_mock.return_value = (Mock(), Mock())
            instance = mut_cls.return_value
            instance.evaluate.return_value = _mutagenesis_eval_result("ATCGG")
            result = runner.invoke(
                mutagenesis_module.main,
                ["-s", "ATCGG", "-m", "test-model", "-p", "0,1"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["model_name"] == "test-model"
        assert payload["mutation_type"] == "single_base_substitution"
        assert payload["affected_positions"] == [0, 1]
        # Single sequence -> a flat result, not batch_results.
        assert "batch_results" not in payload
        assert payload["original_prediction"]["sequence"] == "ATCGG"
        assert payload["original_prediction"]["prediction"] == {"binary": 0.87}
        assert payload["mutated_prediction"]["count"] == 2
        mutant0 = payload["mutated_prediction"]["predictions"][0]
        assert mutant0["logfc"] == pytest.approx([0.11, -0.07])  # ndarray serialized
        assert payload["mutated_prediction"]["predictions"][1]["diff"] == -0.1
        assert payload["delta"]["average_logfc"] == pytest.approx(
            np.mean([np.mean([0.11, -0.07]), 0.3])
        )
        assert payload["delta"]["average_diff"] == pytest.approx(
            np.mean([np.mean([0.2, 0.1]), -0.1])
        )
        instance.mutate_sequence.assert_called_once_with(
            "ATCGG", replace_mut=True, delete_size=0, insert_seq=None
        )
        load_mock.assert_called_once()
        assert load_mock.call_args.kwargs["model_name"] == "test-model"

    @pytest.mark.parametrize(
        ("mutation_type", "expected_kwargs"),
        [
            ("deletion", {"replace_mut": False, "delete_size": 1, "insert_seq": None}),
            ("insertion", {"replace_mut": False, "delete_size": 0, "insert_seq": "N"}),
            (
                "multi_base_substitution",
                {"replace_mut": True, "delete_size": 0, "insert_seq": None},
            ),
        ],
    )
    def test_mutation_type_kwargs(self, runner, mutation_type, expected_kwargs):
        """Each --mutation-type maps to the documented mutate_sequence kwargs."""
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(Mock(), Mock())),
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            instance = mut_cls.return_value
            instance.evaluate.return_value = _mutagenesis_eval_result("ATCG")
            result = runner.invoke(
                mutagenesis_module.main,
                ["-s", "ATCG", "-m", "model", "-t", mutation_type],
            )

        assert result.exit_code == 0
        instance.mutate_sequence.assert_called_once_with("ATCG", **expected_kwargs)

    def test_output_file_written(self, runner, tmp_path):
        """--output writes the JSON payload to the file and echoes the path."""
        out_path = tmp_path / "results" / "mut.json"
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(Mock(), Mock())),
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            mut_cls.return_value.evaluate.return_value = _mutagenesis_eval_result("ATCG")
            result = runner.invoke(
                mutagenesis_module.main,
                ["-s", "ATCG", "-m", "model", "-o", str(out_path)],
            )

        assert result.exit_code == 0
        assert f"Results saved to: {out_path}" in result.output
        payload = json.loads(out_path.read_text())
        assert payload["original_prediction"]["sequence"] == "ATCG"

    def test_empty_mutant_set_reports_zero_delta(self, runner):
        """An eval result with only the raw entry yields zeroed delta averages."""
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(Mock(), Mock())),
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            mut_cls.return_value.evaluate.return_value = {
                "raw": {"sequence": "ATCG", "pred": {"binary": 0.5}, "score": 0.5}
            }
            result = runner.invoke(mutagenesis_module.main, ["-s", "ATCG", "-m", "model"])

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["mutated_prediction"]["count"] == 0
        assert payload["mutated_prediction"]["predictions"] == []
        assert payload["delta"] == {"average_logfc": 0.0, "average_diff": 0.0}

    def test_sequences_file_builds_batch_results(self, runner, tmp_path):
        """--sequences files drive one Mutagenesis run per sequence."""
        seq_file = tmp_path / "seqs.txt"
        seq_file.write_text("ATCG\nGCTA\n")
        eval_results = [
            _mutagenesis_eval_result("ATCG"),
            _mutagenesis_eval_result("GCTA"),
        ]
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(Mock(), Mock())),
            patch("dnallm.inference.mutagenesis.Mutagenesis") as mut_cls,
        ):
            instance = mut_cls.return_value
            instance.evaluate.side_effect = eval_results
            result = runner.invoke(
                mutagenesis_module.main,
                ["--sequences", str(seq_file), "-m", "model"],
            )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["sequence_count"] == 2
        assert len(payload["batch_results"]) == 2
        assert payload["batch_results"][0]["original_prediction"]["sequence"] == "ATCG"
        assert payload["batch_results"][1]["original_prediction"]["sequence"] == "GCTA"
        assert mut_cls.call_count == 2
        assert instance.evaluate.call_count == 2
