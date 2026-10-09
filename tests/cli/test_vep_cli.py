"""CliRunner smoke tests for the dnallm-vep entry point.

All heavy cores are lazy-imported inside the command body, so each test
patches the core at the site the lazy from-import resolves (the origin
package attribute) and asserts the parsed plumbing. CliRunner executes
everything in-process — nothing here spawns or shells out.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from dnallm.cli.vep import main
from dnallm.inference.vep import VepResult

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def runner():
    """Return a CliRunner with exceptions caught (SystemExit -> exit_code)."""
    return CliRunner()


@pytest.fixture
def vcf_file(tmp_path):
    """Return an existing (content-irrelevant) VCF path; parsing is patched."""
    path = tmp_path / "variants.vcf"
    path.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
    return str(path)


@pytest.fixture
def reference_file(tmp_path):
    """Return an existing (content-irrelevant) reference path."""
    path = tmp_path / "reference.fa"
    path.write_text(">chrT\nACGT\n")
    return str(path)


def _fake_result() -> VepResult:
    """A realistic VepResult: scores + skip accounting + metrics + convention."""
    from dnallm.inference.vep import VepVariantRecord

    return VepResult(
        records=[
            VepVariantRecord("chrT", 6, "G", "C", 1, 1.5, None),
            VepVariantRecord("chrT", 10, "A", "AT", 0, None, "length-changing allele"),
        ],
        skip_counts={"length-changing allele": 1, "multi-slot token difference": 0, "no change": 0},
        evaluated=1,
        skipped=1,
        skip_fraction=0.5,
        metrics={"AUROC": 0.9, "AUPRC": 0.9},
        convention={"star_floor": 1, "variant_type": "single_nucleotide_variant"},
    )


def _mock_model_pair():
    model, tokenizer = Mock(), Mock()
    return model, tokenizer


class TestVepCli:
    """The dnallm-vep click command surface."""

    def test_successful_run_echoes_full_json_shape(
        self, runner, vcf_file, reference_file, tmp_path
    ):
        """A successful run echoes per-variant scores + skip accounting +
        metrics + the convention block as JSON."""
        model, tokenizer = _mock_model_pair()
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(model, tokenizer)),
            patch("dnallm.inference.vep.evaluate_vcf", return_value=_fake_result()) as eval_mock,
        ):
            outcome = runner.invoke(
                main,
                ["--vcf", vcf_file, "--reference", reference_file, "--model-name", "some-model"],
            )

        assert outcome.exit_code == 0, outcome.output
        payload = json.loads(outcome.output)
        assert payload["records"][0]["delta"] == 1.5
        assert payload["skip_counts"]["length-changing allele"] == 1
        assert payload["skip_fraction"] == 0.5
        assert payload["metrics"] == {"AUROC": 0.9, "AUPRC": 0.9}
        assert payload["convention"]["star_floor"] == 1
        assert payload["model_name"] == "some-model"
        assert eval_mock.call_args.kwargs["paradigm"] == "mlm"  # default
        assert eval_mock.call_args.args[:3] == (model, tokenizer, vcf_file)

    def test_model_load_failure_exits_one_with_stderr(self, runner, vcf_file, reference_file):
        """A load error exits 1 with the model name in the stderr message."""
        with patch(
            "dnallm.models.load_model_and_tokenizer",
            side_effect=ValueError("boom: no such model"),
        ):
            outcome = runner.invoke(
                main,
                ["--vcf", vcf_file, "--reference", reference_file, "--model-name", "ghost"],
            )

        assert outcome.exit_code == 1
        assert "Error loading model 'ghost'" in outcome.output

    def test_evaluation_failure_exits_one_with_stderr(self, runner, vcf_file, reference_file):
        """An evaluate_vcf error exits 1 through the 'VEP evaluation failed'
        channel, not a traceback."""
        model, tokenizer = _mock_model_pair()
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(model, tokenizer)),
            patch("dnallm.inference.vep.evaluate_vcf", side_effect=ValueError("bad reference")),
        ):
            outcome = runner.invoke(
                main,
                ["--vcf", vcf_file, "--reference", reference_file, "--model-name", "m"],
            )

        assert outcome.exit_code == 1
        assert "VEP evaluation failed" in outcome.output

    def test_output_option_writes_the_json_file(self, runner, vcf_file, reference_file, tmp_path):
        """--output writes the same payload to a (nested) path instead of
        stdout."""
        model, tokenizer = _mock_model_pair()
        out = tmp_path / "nested" / "result.json"
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(model, tokenizer)),
            patch("dnallm.inference.vep.evaluate_vcf", return_value=_fake_result()),
        ):
            outcome = runner.invoke(
                main,
                [
                    "--vcf",
                    vcf_file,
                    "--reference",
                    reference_file,
                    "--model-name",
                    "m",
                    "--output",
                    str(out),
                ],
            )

        assert outcome.exit_code == 0, outcome.output
        assert out.is_file()
        payload = json.loads(out.read_text(encoding="utf-8"))
        assert payload["metrics"]["AUROC"] == 0.9
        assert "Results saved to" in outcome.output

    def test_paradigm_flag_reaches_evaluate_vcf(self, runner, vcf_file, reference_file):
        """--paradigm clm is threaded through to evaluate_vcf."""
        model, tokenizer = _mock_model_pair()
        with (
            patch(
                "dnallm.models.load_model_and_tokenizer", return_value=(model, tokenizer)
            ) as load_mock,
            patch("dnallm.inference.vep.evaluate_vcf", return_value=_fake_result()) as eval_mock,
        ):
            outcome = runner.invoke(
                main,
                [
                    "--vcf",
                    vcf_file,
                    "--reference",
                    reference_file,
                    "--model-name",
                    "m",
                    "--paradigm",
                    "clm",
                ],
            )

        assert outcome.exit_code == 0, outcome.output
        assert eval_mock.call_args.kwargs["paradigm"] == "clm"
        # clm pairs with the causal head: TaskConfig(task_type="generation").
        task_config = load_mock.call_args.kwargs["task_config"]
        assert task_config.task_type == "generation"

    def test_config_file_supplies_vep_defaults(self, runner, vcf_file, reference_file, tmp_path):
        """A --config YAML's vep section supplies context_window (and the
        paradigm when the flag is absent)."""
        model, tokenizer = _mock_model_pair()
        config = tmp_path / "vep.yaml"
        config.write_text("vep:\n  paradigm: clm\n  context_window: 64\n")
        with (
            patch("dnallm.models.load_model_and_tokenizer", return_value=(model, tokenizer)),
            patch("dnallm.inference.vep.evaluate_vcf", return_value=_fake_result()) as eval_mock,
        ):
            outcome = runner.invoke(
                main,
                [
                    "--config",
                    str(config),
                    "--vcf",
                    vcf_file,
                    "--reference",
                    reference_file,
                    "--model-name",
                    "m",
                ],
            )

        assert outcome.exit_code == 0, outcome.output
        assert eval_mock.call_args.kwargs["paradigm"] == "clm"
        assert eval_mock.call_args.kwargs["context_window"] == 64

    def test_config_load_failure_exits_one_with_stderr(
        self, runner, vcf_file, reference_file, tmp_path
    ):
        """A malformed --config YAML exits 1 through the config error
        channel."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("vep: [not, a, mapping]\n")
        with patch(
            "dnallm.models.load_model_and_tokenizer",
            side_effect=AssertionError("must not be reached"),
        ):
            outcome = runner.invoke(
                main,
                [
                    "--config",
                    str(bad),
                    "--vcf",
                    vcf_file,
                    "--reference",
                    reference_file,
                    "--model-name",
                    "m",
                ],
            )

        assert outcome.exit_code == 1
        assert f"Error loading config '{bad}'" in outcome.output
