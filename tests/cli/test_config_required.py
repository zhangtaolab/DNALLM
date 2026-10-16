"""Regression tests for the --config-required usage-error contract (D-01/D-02).

Every train/inference CLI entry point (the ``dnallm-train`` /
``dnallm-inference`` console scripts and the ``dnallm train`` /
``dnallm inference`` group subcommands) exits with click's usage-error code 2
and a static, actionable message when invoked without ``--config`` — bare or
with any combination of the other flags. The pre-fix behavior constructed a
config dict lacking the task section and crashed with a ``KeyError`` inside
the core class constructor (phase-19 deferred-items #1); these tests pin
that the core class is never constructed on the flag-only path and that no
traceback escapes the invocation.

Message-identity contract: all entry points emit one byte-identical static
message containing ``--config is required``; any wording change must keep
that substring and update these tests in the same change.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from dnallm.cli import inference as inference_module
from dnallm.cli import train as train_module
from dnallm.cli.cli import cli

SENTINEL_CONFIG = {"sentinel": "config"}

CONFIG_REQUIRED_SUBSTRING = "--config is required"


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


def _assert_no_traceback(result):
    """Pin that no exception escaped the invocation.

    click >= 8.2 records the standalone-mode ``SystemExit`` in
    ``Result.exception`` for any nonzero exit code, so a handled usage error
    surfaces as ``None`` (older click) or a ``SystemExit`` (current click).
    Anything else — e.g. the pre-fix ``KeyError('task')`` — is an escaped
    traceback and fails this pin.
    """
    assert result.exception is None or isinstance(result.exception, SystemExit), (
        f"unexpected escaped exception: {result.exception!r}"
    )


def _extract_error_message(result):
    """Return the message text after click's ``Error: `` prefix."""
    for line in result.output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Error:"):
            return stripped[len("Error:"):].strip()
    raise AssertionError(f"no 'Error:' line in output: {result.output!r}")


class TestStandaloneTrainConfigRequired:
    """The dnallm-train console script module (dnallm.cli.train.main)."""

    def test_bare_invocation_is_a_usage_error(self, runner):
        """No options exits 2 naming --config, the generator, and the docs."""
        result = runner.invoke(train_module.main, [])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        assert "dnallm-model-config-generator" in result.output
        assert "docs/user_guide/cli/usage.md" in result.output
        _assert_no_traceback(result)

    def test_full_flags_still_requires_config(self, runner, data_file, tmp_path):
        """-m/-d/-o (the old KeyError repro) exits 2; the core is never built."""
        output_dir = str(tmp_path / "run")
        with patch("dnallm.finetune.DNATrainer") as trainer_cls:
            result = runner.invoke(
                train_module.main,
                ["-m", "tiny-dna", "-d", data_file, "-o", output_dir],
            )

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        trainer_cls.assert_not_called()
        _assert_no_traceback(result)

    def test_config_path_still_loads_and_trains(self, runner, config_file):
        """-c keeps the working path: load_config -> DNATrainer -> train()."""
        with (
            patch("dnallm.configuration.load_config", return_value=SENTINEL_CONFIG),
            patch("dnallm.finetune.DNATrainer") as trainer_cls,
        ):
            result = runner.invoke(train_module.main, ["-c", config_file])

        assert result.exit_code == 0
        trainer_cls.assert_called_once_with(model=None, config=SENTINEL_CONFIG)
        trainer_cls.return_value.train.assert_called_once_with()


class TestGroupTrainConfigRequired:
    """The train subcommand of the dnallm group (dnallm.cli.cli)."""

    def test_bare_invocation_is_a_usage_error(self, runner):
        """`dnallm train` with no options exits 2 naming --config."""
        result = runner.invoke(cli, ["train"])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        _assert_no_traceback(result)

    def test_full_flags_still_requires_config(self, runner, data_file, tmp_path):
        """-m/-d/-o without -c exits 2; the core is never built."""
        output_dir = str(tmp_path / "run")
        with patch("dnallm.finetune.DNATrainer") as trainer_cls:
            result = runner.invoke(
                cli,
                ["train", "-m", "tiny-dna", "-d", data_file, "-o", output_dir],
            )

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        trainer_cls.assert_not_called()
        _assert_no_traceback(result)


class TestGroupInferenceConfigRequired:
    """The inference subcommand of the dnallm group (dnallm.cli.cli)."""

    def test_bare_invocation_is_a_usage_error(self, runner):
        """`dnallm inference` with no options exits 2 naming --config."""
        result = runner.invoke(cli, ["inference"])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        _assert_no_traceback(result)

    def test_full_flags_still_requires_config(self, runner, data_file):
        """-m/-i without -c exits 2; the core is never built."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            result = runner.invoke(cli, ["inference", "-m", "m1", "-i", data_file])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        engine_cls.assert_not_called()
        _assert_no_traceback(result)


class TestStandaloneInferenceConfigRequired:
    """The dnallm-inference console script module (dnallm.cli.inference)."""

    def test_bare_invocation_is_a_usage_error(self, runner):
        """Standalone main with no options exits 2 naming --config."""
        result = runner.invoke(inference_module.main, [])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        _assert_no_traceback(result)

    def test_full_flags_still_requires_config(self, runner, data_file):
        """-m/-i without -c exits 2; the core is never built."""
        with patch("dnallm.inference.DNAInference") as engine_cls:
            result = runner.invoke(inference_module.main, ["-m", "m2", "-i", data_file])

        assert result.exit_code == 2
        assert CONFIG_REQUIRED_SUBSTRING in result.output
        engine_cls.assert_not_called()
        _assert_no_traceback(result)


class TestConfigRequiredMessageConsistency:
    """One static message across all four train/inference entry points."""

    def test_all_four_entry_points_emit_identical_message(self, runner):
        """Bare invocation of each site emits byte-identical error text."""
        invocations = (
            (train_module.main, []),
            (inference_module.main, []),
            (cli, ["train"]),
            (cli, ["inference"]),
        )
        messages = []
        for fn, args in invocations:
            result = runner.invoke(fn, args)
            assert result.exit_code == 2, (fn, args, result.exit_code)
            messages.append(_extract_error_message(result))

        assert CONFIG_REQUIRED_SUBSTRING in messages[0]
        assert "dnallm-model-config-generator" in messages[0]
        assert "docs/user_guide/cli/usage.md" in messages[0]
        assert len(set(messages)) == 1, f"messages diverged: {messages!r}"
