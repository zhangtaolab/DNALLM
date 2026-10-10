"""JSON contract tests for the Phase 12 MCP tools (ism_scan, hotspots).

Per-tool contracts with mocked inference engines (MCPE-01 fast lane): the
validation-first bodies, the tool-boundary input caps, the error-dict
boundary (nothing ever raises across the protocol), and the event-loop
liveness guarantee (a concurrent health_check completes while a long
mocked engine call is in flight because the torch work runs in the
executor, never on the loop thread).

The in-memory ASGI round trips for the same tools live in
tests/mcp/test_server_transports.py; this file drives the tool methods
directly against mock managers.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from dnallm.inference.vep import ClinVarFilter, VepResult, VepVariantRecord
from dnallm.mcp.server import (
    DNALLMMCPServer,
    HOTSPOT_MAX_REGION_LENGTH,
    ISM_MAX_POSITIONS,
    ISM_MAX_SEQUENCE_LENGTH,
    ZERO_SHOT_MAX_VCF_BYTES,
    ZERO_SHOT_MAX_VARIANTS,
)


@pytest.fixture
def mock_engine():
    """A mock DNAInference exposing model/tokenizer/config."""
    engine = Mock()
    engine.model = Mock()
    engine.tokenizer = Mock()
    engine.config = {
        "task": Mock(task_type="binary"),
        "inference": Mock(max_length=512, batch_size=8, num_workers=0),
    }
    return engine


@pytest.fixture
def v12_server(mock_engine):
    """A DNALLMMCPServer whose managers are mocks wired for the tool bodies.

    ``get_model_config`` returns a truthy Mock (registry-known model);
    ``get_inference_engine`` returns the mock engine; the fork-safety flight
    lock is a real ``threading.Lock`` so executor closures can acquire it.
    """
    with (
        patch("dnallm.mcp.server.MCPConfigManager") as mock_cfg_mgr,
        patch("dnallm.mcp.server.ModelManager") as mock_model_mgr,
    ):
        mock_cfg = Mock()
        mock_cfg.get_server_config.return_value = None
        mock_cfg.get_enabled_models.return_value = []
        mock_cfg_mgr.return_value = mock_cfg

        mock_mgr = Mock()
        mock_mgr.config_manager = mock_cfg
        mock_mgr.get_model_config.return_value = Mock()
        mock_mgr.get_inference_engine.return_value = mock_engine
        mock_mgr.get_loaded_models.return_value = []
        mock_mgr._infer_thread_lock = threading.Lock()
        mock_model_mgr.return_value = mock_mgr

        server = DNALLMMCPServer("config.yaml")
        server.app = Mock()
        server.model_manager = mock_mgr
        yield server


def _ism_eval_result() -> dict:
    """A minimal Mutagenesis.evaluate-style result for one sequence."""
    return {
        "raw": {"sequence": "ATGC", "pred": np.array([0.1]), "score": 0.0},
        "mut_0_A_T": {
            "sequence": "TTGC",
            "pred": np.array([0.2]),
            "logfc": np.array([0.5]),
            "diff": np.array([0.1]),
            "score": 0.5,
        },
    }


class TestIsmScanContracts:
    """Per-tool JSON contracts for _ism_scan."""

    async def test_happy_path_single_sequence(self, v12_server):
        """A valid scan returns the mutagenesis-shaped payload."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = _ism_eval_result()

            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence="ATGC",
                mutation_type="single_base_substitution",
                positions=[0],
            )

        assert not result.get("isError")
        assert result["model_name"] == "test-model"
        assert result["affected_positions"] == [0]
        assert result["mutation_type"] == "single_base_substitution"
        assert result["original_prediction"]["sequence"] == "ATGC"
        assert result["mutated_prediction"]["count"] == 1
        assert result["delta"] == {"average_logfc": 0.5, "average_diff": 0.1}
        mock_mut.mutate_sequence.assert_called_once_with(
            "ATGC", replace_mut=True, delete_size=0, insert_seq=None
        )

    async def test_engine_call_is_off_the_event_loop(self, v12_server):
        """The Mutagenesis work is submitted to an executor, not the loop."""
        seen_threads: list[str] = []

        def _evaluate(**kwargs):
            seen_threads.append(threading.current_thread().name)
            return _ism_eval_result()

        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.side_effect = _evaluate
            await v12_server._ism_scan(
                model_name="test-model",
                sequence="ATGC",
                positions=[0],
            )

        assert seen_threads, "the engine must actually run"
        assert all(t != threading.main_thread().name for t in seen_threads)

    async def test_model_not_loaded_returns_not_loaded_dict(self, mock_engine):
        """A registry-known model without a loaded engine: matchable dict."""
        with (
            patch("dnallm.mcp.server.MCPConfigManager") as mock_cfg_mgr,
            patch("dnallm.mcp.server.ModelManager") as mock_model_mgr,
        ):
            mock_cfg = Mock()
            mock_cfg.get_server_config.return_value = None
            mock_cfg_mgr.return_value = mock_cfg
            mock_mgr = Mock()
            mock_mgr.config_manager = mock_cfg
            mock_mgr.get_model_config.return_value = Mock()  # registered
            mock_mgr.get_inference_engine.return_value = None  # not loaded
            mock_mgr._infer_thread_lock = threading.Lock()
            mock_model_mgr.return_value = mock_mgr
            server = DNALLMMCPServer("config.yaml")
            server.app = Mock()
            server.model_manager = mock_mgr

            result = await server._ism_scan(
                model_name="registered-model",
                sequence="ATGC",
                positions=[0],
            )

        assert result == {
            "error": "Model registered-model not loaded",
            "isError": True,
        }

    async def test_model_not_configured_returns_registry_dict(self, v12_server):
        """An unregistered model never reaches the engine (T-12-07)."""
        v12_server.model_manager.config_manager.get_model_config.return_value = None

        result = await v12_server._ism_scan(
            model_name="ghost",
            sequence="ATGC",
            positions=[0],
        )

        assert result["isError"] is True
        assert "not configured" in result["error"]
        v12_server.model_manager.get_inference_engine.assert_not_called()

    async def test_invalid_sequence_returns_error_dict(self, v12_server):
        """Non-ACGTN sequence content fails with the index named."""
        result = await v12_server._ism_scan(
            model_name="test-model",
            sequence="ATGZ",
            positions=[0],
        )

        assert result["isError"] is True
        assert "index 0" in result["error"]
        assert "invalid characters" in result["error"]

    async def test_sequence_length_cap_stated_in_error(self, v12_server):
        """The sequence cap is enforced before any engine access."""
        long_seq = "A" * (ISM_MAX_SEQUENCE_LENGTH + 1)
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence=long_seq,
                positions=[0],
            )

        assert result["isError"] is True
        assert str(ISM_MAX_SEQUENCE_LENGTH) in result["error"]
        assert "cap" in result["error"]
        mock_mut_cls.assert_not_called()

    async def test_positions_cap_stated_in_error(self, v12_server):
        """The positions cap is enforced before any engine access."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence="ACGTACGTACGT",
                positions=list(range(ISM_MAX_POSITIONS + 1)),
            )

        assert result["isError"] is True
        assert str(ISM_MAX_POSITIONS) in result["error"]
        mock_mut_cls.assert_not_called()

    async def test_position_out_of_range_named(self, v12_server):
        """A position past the sequence end is rejected with both numbers."""
        result = await v12_server._ism_scan(
            model_name="test-model",
            sequence="ATGC",
            positions=[4],
        )

        assert result["isError"] is True
        assert "out of range" in result["error"]
        assert "4" in result["error"]
        assert "length 4" in result["error"]

    async def test_empty_model_name(self, v12_server):
        """An empty model_name fails before everything else."""
        result = await v12_server._ism_scan(model_name="", sequence="ATGC", positions=[0])
        assert result == {"error": "model_name is required", "isError": True}

    async def test_missing_positions(self, v12_server):
        """positions=None is a matchable validation error."""
        result = await v12_server._ism_scan(
            model_name="test-model", sequence="ATGC", positions=None
        )
        assert result["isError"] is True
        assert "positions" in result["error"]

    async def test_invalid_mutation_type(self, v12_server):
        """An unknown mutation_type names the allowed set."""
        result = await v12_server._ism_scan(
            model_name="test-model", sequence="ATGC", positions=[0], mutation_type="inversion"
        )
        assert result["isError"] is True
        assert "mutation_type" in result["error"]

    async def test_sequence_and_sequences_both_provided_rejected(self, v12_server):
        """WR-04: passing both inputs is a matchable error — `sequence`
        must not be silently dropped in favor of `sequences`."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence="AAAA",
                sequences=["ATGC"],
                positions=[0],
            )

        assert result["isError"] is True
        assert "not both" in result["error"]
        mock_mut_cls.assert_not_called()

    async def test_empty_sequences_list_rejected(self, v12_server):
        """WR-04: an empty sequences list is rejected like missing input —
        consistent with the empty-positions rejection — instead of
        succeeding with an empty batch; empty member strings too."""
        for sequences in ([], [""]):
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequences=sequences,
                positions=[0],
            )
            assert result["isError"] is True, sequences
            assert "non-empty list of non-empty strings" in result["error"]

    async def test_non_string_sequence_rejected_matchably(self, v12_server):
        """WR-04: a non-string sequence is a matchable validation error,
        not a TypeError into the generic failure dict."""
        result = await v12_server._ism_scan(
            model_name="test-model",
            sequence=12345,
            positions=[0],
        )

        assert result["isError"] is True
        assert "error" in result  # matchable dict, not content-only generic
        assert "non-empty strings" in result["error"]

    async def test_engine_exception_returns_generic_error_dict(self, v12_server):
        """A raising engine is caught; nothing crosses the protocol boundary."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut_cls.side_effect = RuntimeError("engine fault")
            result = await v12_server._ism_scan(
                model_name="test-model", sequence="ATGC", positions=[0]
            )

        assert result["isError"] is True
        assert result["content"][0]["text"] == "ISM scan failed. See server logs for details."

    async def test_batch_sequences_payload(self, v12_server):
        """Multiple sequences produce the batch_results shape."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = _ism_eval_result()
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequences=["ATGC", "ATGC"],
                positions=[0],
            )

        assert not result.get("isError")
        assert result["sequence_count"] == 2
        assert len(result["batch_results"]) == 2

    async def test_positions_restrict_reported_entries(self, v12_server):
        """WR-03: only entries at the requested positions are reported —
        count and delta averages come from the filtered set, not the full
        scan the engine performed."""
        eval_result = {
            "raw": {"sequence": "ATGC", "pred": np.array([0.1]), "score": 0.0},
            "mut_0_A_T": {
                "sequence": "TTGC",
                "pred": np.array([0.2]),
                "logfc": np.array([0.5]),
                "diff": np.array([0.1]),
                "score": 0.5,
            },
            "mut_1_T_A": {
                "sequence": "AAGC",
                "pred": np.array([0.3]),
                "logfc": np.array([2.0]),
                "diff": np.array([0.4]),
                "score": 2.0,
            },
        }
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = eval_result
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence="ATGC",
                positions=[1],
            )

        assert not result.get("isError")
        assert result["mutated_prediction"]["count"] == 1
        assert result["mutated_prediction"]["predictions"][0]["sequence"] == "AAGC"
        assert result["delta"] == {"average_logfc": 2.0, "average_diff": 0.4}
        assert result["affected_positions"] == [1]

    async def test_positions_filter_handles_deletion_entry_names(self, v12_server):
        """WR-03: `del_{i}_{n}` entries filter by their position too, and
        entries whose name carries no resolvable position are dropped."""
        eval_result = {
            "raw": {"sequence": "ATGC", "pred": np.array([0.1]), "score": 0.0},
            "del_0_1": {
                "sequence": "TGC",
                "pred": np.array([0.2]),
                "logfc": np.array([0.5]),
                "diff": np.array([0.1]),
                "score": 0.5,
            },
            "del_2_1": {
                "sequence": "ATC",
                "pred": np.array([0.3]),
                "logfc": np.array([1.5]),
                "diff": np.array([0.3]),
                "score": 1.5,
            },
            "weird_entry": {
                "sequence": "NNNN",
                "pred": np.array([0.9]),
                "logfc": np.array([9.0]),
                "diff": np.array([9.0]),
                "score": 9.0,
            },
        }
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = eval_result
            result = await v12_server._ism_scan(
                model_name="test-model",
                sequence="ATGC",
                mutation_type="deletion",
                positions=[2],
            )

        assert not result.get("isError")
        assert result["mutated_prediction"]["count"] == 1
        assert result["mutated_prediction"]["predictions"][0]["sequence"] == "ATC"


class TestHotspotsContracts:
    """Per-tool JSON contracts for _hotspots."""

    @pytest.fixture
    def reference_fasta(self, tmp_path) -> Path:
        """A real 64-base single-chromosome FASTA (vep._load_reference reads it)."""
        fasta = tmp_path / "ref.fasta"
        fasta.write_text(">chr1 test reference\n" + "ACGT" * 16 + "\n")
        return fasta

    async def test_happy_path_derives_windows_from_model_and_coordinates(
        self, v12_server, reference_fasta
    ):
        """Windows come from ISM over the coordinates slice (D-05)."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {"raw": {"sequence": "ACGT", "score": 0.0}}
            mock_mut.find_hotspots.return_value = [(2, 12), (30, 40)]

            result = await v12_server._hotspots(
                model_name="test-model",
                coordinates={"chrom": "chr1", "start": 4, "end": 40},
                fasta_path=str(reference_fasta),
            )

        assert not result.get("isError")
        assert result["hotspots"] == [[2, 12], [30, 40]]
        assert result["hotspots_genomic"] == [
            {"chrom": "chr1", "start": 6, "end": 16},
            {"chrom": "chr1", "start": 34, "end": 44},
        ]
        assert result["window_count"] == 2
        assert result["sequence_length"] == 36
        assert result["coordinates"] == {"chrom": "chr1", "start": 4, "end": 40}
        # The ISM'd slice is the requested region (0-based half-open),
        # uppercased by the vep window convention.
        call = mock_mut.mutate_sequence.call_args
        assert call.args[0] == "ACGT" * 9  # 36 bases from offset 4
        assert call.kwargs["replace_mut"] is True
        hs_kwargs = mock_mut.find_hotspots.call_args.kwargs
        assert hs_kwargs == {
            "strategy": "maxabs",
            "window_size": 10,
            "percentile_threshold": 90.0,
        }

    async def test_missing_fasta_names_the_tool_and_path(self, v12_server, tmp_path):
        """A missing FASTA is a matchable error naming hotspots + the path."""
        missing = tmp_path / "nope.fasta"
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chr1", "start": 0, "end": 50},
            fasta_path=str(missing),
        )

        assert result["isError"] is True
        assert "hotspots" in result["error"]
        assert str(missing) in result["error"]
        assert "not found" in result["error"]

    async def test_fasta_suffix_allowlist(self, v12_server, tmp_path):
        """A non-FASTA suffix is rejected before any file access."""
        bogus = tmp_path / "genome.txt"
        bogus.write_text("not a fasta")
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chr1", "start": 0, "end": 50},
            fasta_path=str(bogus),
        )

        assert result["isError"] is True
        assert ".fasta" in result["error"]
        assert "genome.txt" in result["error"]

    async def test_coordinates_field_validation(self, v12_server, reference_fasta):
        """Each malformed coordinates shape names the failing field."""
        base = {"fasta_path": str(reference_fasta), "model_name": "test-model"}
        cases = [
            ({"chrom": "chr1"}, "start"),
            ({"chrom": "chr1", "start": 0}, "end"),
            ({"chrom": "", "start": 0, "end": 10}, "chrom"),
            ({"chrom": "chr1", "start": -1, "end": 10}, "start"),
            ({"chrom": "chr1", "start": 10, "end": 10}, "end"),
        ]
        for coordinates, field in cases:
            result = await v12_server._hotspots(coordinates=coordinates, **base)
            assert result["isError"] is True, coordinates
            assert field in result["error"], (coordinates, result["error"])

    async def test_region_length_cap(self, v12_server, reference_fasta):
        """The region cap is enforced before the engine (T-12-06)."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            result = await v12_server._hotspots(
                model_name="test-model",
                coordinates={
                    "chrom": "chr1",
                    "start": 0,
                    "end": HOTSPOT_MAX_REGION_LENGTH + 1,
                },
                fasta_path=str(reference_fasta),
            )

        assert result["isError"] is True
        assert str(HOTSPOT_MAX_REGION_LENGTH) in result["error"]
        mock_mut_cls.assert_not_called()

    async def test_unknown_chromosome_is_a_matchable_dict(self, v12_server, reference_fasta):
        """A chromosome absent from the reference surfaces the vep error."""
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chrZ", "start": 0, "end": 10},
            fasta_path=str(reference_fasta),
        )

        assert result["isError"] is True
        assert "chrZ" in result["error"]
        assert "not found in the reference" in result["error"]

    async def test_region_beyond_reference_length(self, v12_server, reference_fasta):
        """end past the chromosome length reports the assembly mismatch."""
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chr1", "start": 0, "end": 100},
            fasta_path=str(reference_fasta),
        )

        assert result["isError"] is True
        assert "exceeds" in result["error"]
        assert "64" in result["error"]

    async def test_invalid_strategy(self, v12_server, reference_fasta):
        """An unknown aggregation strategy names the allowed set."""
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chr1", "start": 0, "end": 10},
            fasta_path=str(reference_fasta),
            strategy="median",
        )

        assert result["isError"] is True
        assert "strategy" in result["error"]

    async def test_model_not_loaded_returns_not_loaded_dict(self, v12_server, reference_fasta):
        """Engine-missing maps to the matchable not-loaded dict."""
        v12_server.model_manager.get_inference_engine.return_value = None
        result = await v12_server._hotspots(
            model_name="test-model",
            coordinates={"chrom": "chr1", "start": 0, "end": 10},
            fasta_path=str(reference_fasta),
        )

        assert result == {"error": "Model test-model not loaded", "isError": True}

    async def test_engine_exception_returns_generic_error_dict(self, v12_server, reference_fasta):
        """A raising engine never crosses the protocol boundary."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut_cls.side_effect = RuntimeError("boom")
            result = await v12_server._hotspots(
                model_name="test-model",
                coordinates={"chrom": "chr1", "start": 0, "end": 10},
                fasta_path=str(reference_fasta),
            )

        assert result["isError"] is True
        assert result["content"][0]["text"] == ("Hotspot scan failed. See server logs for details.")


class TestEventLoopLiveness:
    """The event loop stays responsive during a long engine call."""

    async def test_health_check_completes_during_long_ism_call(self, v12_server):
        """A blocked engine flight cannot stall a concurrent health_check.

        The mocked Mutagenesis.evaluate parks in the executor until released;
        if the tool ran the torch work on the loop thread, the 0.5s
        wait_for on health_check would time out.
        """
        release = threading.Event()

        def _parked_evaluate(**kwargs):
            release.wait(timeout=10.0)
            return _ism_eval_result()

        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.side_effect = _parked_evaluate

            task = asyncio.create_task(
                v12_server._ism_scan(
                    model_name="test-model",
                    sequence="ATGC",
                    positions=[0],
                )
            )
            try:
                await asyncio.sleep(0.1)  # the tool reaches its executor flight
                health = await asyncio.wait_for(v12_server._health_check(), timeout=0.5)
                assert health["health"]["status"] == "healthy"
            finally:
                release.set()

            result = await asyncio.wait_for(task, timeout=5.0)

        assert not result.get("isError")
        assert result["model_name"] == "test-model"


class TestZeroShotScoreContracts:
    """Per-tool JSON contracts for _zero_shot_score (dual-mode, D-04)."""

    @pytest.fixture
    def reference_fasta(self, tmp_path) -> Path:
        """A real single-chromosome FASTA for the per-call fasta_path."""
        fasta = tmp_path / "ref.fasta"
        fasta.write_text(">chr1 test reference\n" + "ACGT" * 16 + "\n")
        return fasta

    @staticmethod
    def _vep_result(**overrides) -> VepResult:
        """A realistic VepResult whose skip/convention blocks are distinctive."""
        defaults: dict = {
            "records": [
                VepVariantRecord(
                    chrom="chr1",
                    pos=10,
                    ref="A",
                    alt="G",
                    label=1,
                    delta=-0.25,
                    skip_reason=None,
                ),
                VepVariantRecord(
                    chrom="chr1",
                    pos=62,
                    ref="T",
                    alt="A",
                    label=1,
                    delta=None,
                    skip_reason="no change",  # window-edge skip-as-data
                ),
            ],
            "skip_counts": {
                "length-changing allele": 0,
                "multi-slot token difference": 0,
                "no change": 1,
            },
            "evaluated": 1,
            "skipped": 1,
            "skip_fraction": 0.5,
            "metrics": None,
            "convention": {
                "cohort": "pass-through",
                "exclusion_counts": {
                    "non_snv_clnvc": 0,
                    "unlabeled_clnsig": 0,
                    "below_star_floor": 0,
                },
                "clnrevstat_counts": {"criteria_provided": 2},
            },
        }
        defaults.update(overrides)
        return VepResult(**defaults)

    async def test_zero_shot_inline_happy_path_routes_through_kernel(
        self, v12_server, reference_fasta
    ):
        """Inline variants materialize to a fixed-name temp VCF and the
        VepResult blocks surface verbatim (D-04 locked specificity)."""
        result_obj = self._vep_result()
        with patch("dnallm.mcp.server.evaluate_vcf", return_value=result_obj) as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[
                    {"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"},
                    {"chrom": "chr1", "pos": 62, "ref": "T", "alt": "A"},
                ],
                paradigm="clm",
                context_window=50,
            )

        assert not result.get("isError")
        assert result["input_mode"] == "inline_variants"
        assert result["paradigm"] == "clm"
        assert result["model_name"] == "test-model"
        # skip accounting + convention VERBATIM
        assert result["skip_counts"] == result_obj.skip_counts
        assert result["skipped"] == 1
        assert result["skip_fraction"] == 0.5
        assert result["convention"] == result_obj.convention
        # per-record scores incl. the window-edge skip-as-data record
        assert result["records"][0]["delta"] == -0.25
        assert result["records"][1]["skip_reason"] == "no change"

        kernel_vcf = mock_kernel.call_args.args[2]
        assert Path(kernel_vcf).name == "inline_variants.vcf"
        kernel_kwargs = mock_kernel.call_args.kwargs
        assert kernel_kwargs["paradigm"] == "clm"
        assert kernel_kwargs["context_window"] == 50
        assert kernel_kwargs["clnsig_filter"].variant_type == "inline_variant"
        assert kernel_kwargs["clnsig_filter"].positive_labels == frozenset({"not_analyzed"})

    async def test_zero_shot_inline_temp_vcf_content_contract(self, v12_server, reference_fasta):
        """The temp VCF rows carry the 1-based pos, uppercase alleles, and
        the pass-through sentinel INFO (read from inside the kernel call,
        before the tool's cleanup deletes the file)."""
        captured: dict = {}

        def _fake_evaluate(model, tokenizer, vcf_path, reference, **kwargs):
            captured["text"] = Path(vcf_path).read_text(encoding="utf-8")
            captured["path"] = str(vcf_path)
            return self._vep_result()

        with patch("dnallm.mcp.server.evaluate_vcf", side_effect=_fake_evaluate):
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[
                    {"chrom": "chr1", "pos": 10, "ref": "ac", "alt": "GT"},
                    {"chrom": "chr1", "pos": 62, "ref": "T", "alt": "A"},
                ],
            )

        assert not result.get("isError")
        text = captured["text"]
        assert "##fileformat=VCFv4.2" in text
        assert "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO" in text
        data_line = next(line for line in text.splitlines() if line.startswith("chr1\t10\t"))
        fields = data_line.split("\t")
        assert fields[0] == "chr1"
        assert fields[1] == "10"  # 1-based VCF coordinate preserved
        assert fields[3] == "AC"  # alleles uppercased
        assert fields[4] == "GT"
        assert "CLNSIG=not_analyzed" in fields[7]
        assert "CLNVC=inline_variant" in fields[7]
        # Belt-and-braces (CR-01): exactly one data line per accepted
        # variant — no field can smuggle extra rows past the variant cap.
        data_lines = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
        assert len(data_lines) == 2
        # temp dir cleaned up after the call
        assert not Path(captured["path"]).exists()

    async def test_zero_shot_security_no_record_derived_string_in_any_path(
        self, v12_server, reference_fasta
    ):
        """T-12-05 security invariant: no record-derived string in any path.

        Distinctive chrom/ref/alt values; the kernel capture asserts the
        fixed sanitized basename, the tool-owned temp dir prefix, and that
        none of the record values appear anywhere in the path.
        """
        captured: dict = {}

        def _fake_evaluate(model, tokenizer, vcf_path, reference, **kwargs):
            captured["path"] = str(vcf_path)
            return self._vep_result()

        with patch("dnallm.mcp.server.evaluate_vcf", side_effect=_fake_evaluate):
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[
                    {
                        "chrom": "chrZZSECURITYPROBE",
                        "pos": 10,
                        "ref": "ACGTTACG",
                        "alt": "TTTTGGGG",
                    }
                ],
            )

        assert not result.get("isError")
        path = captured["path"]
        assert Path(path).name == "inline_variants.vcf"  # fixed sanitized name
        assert Path(path).parent.name.startswith("dnallm_zero_shot_")
        for record_string in ("chrZZSECURITYPROBE", "ACGTTACG", "TTTTGGGG"):
            assert record_string not in path

    async def test_zero_shot_security_tab_injected_chrom_rejected(
        self, v12_server, reference_fasta
    ):
        """CR-01 injection regression: a tab-bearing chrom must be rejected
        before the temp VCF is written — materialized, the tab row would
        override the validated POS/REF/ALT with client-chosen fields and
        attach a client-chosen ClinVar label."""
        injected_chrom = (
            "chr1\t20\t.\tA\tT\t.\t.\t"
            "CLNSIG=Pathogenic;CLNREVSTAT=criteria_provided;"
            "CLNVC=single_nucleotide_variant"
        )
        with patch("dnallm.mcp.server.evaluate_vcf") as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": injected_chrom, "pos": 10, "ref": "A", "alt": "G"}],
            )

        assert result["isError"] is True
        assert "whitespace-free" in result["error"]
        assert "variants[0].chrom" in result["error"]
        mock_kernel.assert_not_called()

    async def test_zero_shot_security_newline_injected_chrom_rejected(
        self, v12_server, reference_fasta
    ):
        """CR-01 injection regression: one accepted variant whose chrom
        embeds newline-separated fully-formatted rows must be rejected —
        otherwise the variant cap and the inline no-ClinVar guarantee are
        bypassable by a single record."""
        smuggled_chrom = "\n".join([
            "chr1",
            *[
                "chr1\t20\t.\tA\tT\t.\t.\tCLNSIG=Benign;"
                "CLNREVSTAT=criteria_provided;CLNVC=single_nucleotide_variant"
                for _ in range(30)
            ],
        ])
        with patch("dnallm.mcp.server.evaluate_vcf") as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": smuggled_chrom, "pos": 10, "ref": "A", "alt": "G"}],
            )

        assert result["isError"] is True
        assert "whitespace-free" in result["error"]
        mock_kernel.assert_not_called()

    async def test_zero_shot_vcf_path_happy_path_uses_d17_default_filter(
        self, v12_server, reference_fasta, tmp_path
    ):
        """vcf_path mode passes clnsig_filter=None (kernel D-17 defaults)."""
        vcf = tmp_path / "clinvar.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\n")
        with patch(
            "dnallm.mcp.server.evaluate_vcf", return_value=self._vep_result()
        ) as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                vcf_path=str(vcf),
            )

        assert not result.get("isError")
        assert result["input_mode"] == "vcf_path"
        assert mock_kernel.call_args.args[2] == str(vcf)
        assert mock_kernel.call_args.kwargs["clnsig_filter"] is None

    async def test_zero_shot_explicit_clnsig_filter_override(
        self, v12_server, reference_fasta, tmp_path
    ):
        """A caller filter maps to ClinVarFilter fields (non-ClinVar
        opt-out). Full override is the documented vcf_path-mode use."""
        vcf = tmp_path / "clinvar.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\n")
        with patch(
            "dnallm.mcp.server.evaluate_vcf", return_value=self._vep_result()
        ) as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                vcf_path=str(vcf),
                clnsig_filter={
                    "variant_type": "my_type",
                    "positive_labels": ["a", "b"],
                    "negative_labels": ["c"],
                    "star_floor": 2,
                },
            )

        assert not result.get("isError")
        kernel_filter = mock_kernel.call_args.kwargs["clnsig_filter"]
        assert kernel_filter.variant_type == "my_type"
        assert kernel_filter.positive_labels == frozenset({"a", "b"})
        assert kernel_filter.negative_labels == frozenset({"c"})
        assert kernel_filter.star_floor == 2

    async def test_zero_shot_inline_conflicting_variant_type_rejected(
        self, v12_server, reference_fasta
    ):
        """WR-02: inline mode keeps the inline_variant sentinel. A caller
        filter replacing variant_type (e.g. a plausible "turn off
        filtering" attempt) would revert to the D-17 default and exclude
        every inline row as non_snv_clnvc — rejected with a matchable
        error naming the required value."""
        with patch("dnallm.mcp.server.evaluate_vcf") as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"}],
                clnsig_filter={"variant_type": "single_nucleotide_variant"},
            )

        assert result["isError"] is True
        assert "clnsig_filter.variant_type" in result["error"]
        assert "inline_variant" in result["error"]
        mock_kernel.assert_not_called()

    async def test_zero_shot_inline_partial_filter_keeps_sentinel_base(
        self, v12_server, reference_fasta
    ):
        """WR-02: a partial inline-mode override merges with the sentinel,
        not the D-17 defaults — variant_type stays inline_variant (rows stay
        admitted) and unset fields fall back to the sentinel values."""
        with patch(
            "dnallm.mcp.server.evaluate_vcf", return_value=self._vep_result()
        ) as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"}],
                clnsig_filter={"star_floor": 0},
            )

        assert not result.get("isError")
        kernel_filter = mock_kernel.call_args.kwargs["clnsig_filter"]
        assert kernel_filter.variant_type == "inline_variant"
        assert kernel_filter.positive_labels == frozenset({"not_analyzed"})
        assert kernel_filter.negative_labels == frozenset()
        assert kernel_filter.star_floor == 0

    async def test_zero_shot_both_modes_rejected(self, v12_server, reference_fasta, tmp_path):
        """Passing both variants and vcf_path is a matchable error."""
        vcf = tmp_path / "clin.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n")
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"}],
            vcf_path=str(vcf),
        )

        assert result["isError"] is True
        assert "not both" in result["error"]

    async def test_zero_shot_neither_mode_rejected(self, v12_server, reference_fasta):
        """Passing neither input mode is a matchable error."""
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
        )

        assert result["isError"] is True
        assert "either variants" in result["error"]

    @pytest.mark.parametrize(
        ("variant", "field"),
        [
            ({"pos": 10, "ref": "A", "alt": "G"}, "chrom"),
            ({"chrom": "chr1", "pos": 0, "ref": "A", "alt": "G"}, "pos"),
            ({"chrom": "chr1", "pos": "ten", "ref": "A", "alt": "G"}, "pos"),
            ({"chrom": "chr1", "pos": 10, "ref": "XYZ", "alt": "G"}, "ref"),
            ({"chrom": "chr1", "pos": 10, "ref": "A"}, "alt"),
            ({"chrom": "chr1", "pos": 10, "ref": "A", "alt": ""}, "alt"),
        ],
    )
    async def test_zero_shot_per_field_validation_errors(
        self, v12_server, reference_fasta, variant, field
    ):
        """Each malformed field is named in the error dict."""
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[variant],
        )

        assert result["isError"] is True
        assert field in result["error"]
        assert "variants[0]" in result["error"]

    async def test_zero_shot_variant_cap_stated_in_error(self, v12_server, reference_fasta):
        """The variant-count cap is enforced before the kernel."""
        too_many = [
            {"chrom": "chr1", "pos": i + 1, "ref": "A", "alt": "G"}
            for i in range(ZERO_SHOT_MAX_VARIANTS + 1)
        ]
        with patch("dnallm.mcp.server.evaluate_vcf") as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=too_many,
            )

        assert result["isError"] is True
        assert str(ZERO_SHOT_MAX_VARIANTS) in result["error"]
        mock_kernel.assert_not_called()

    async def test_zero_shot_vcf_suffix_allowlist(self, v12_server, reference_fasta, tmp_path):
        """A non-VCF suffix is rejected before any file read."""
        bogus = tmp_path / "variants.txt"
        bogus.write_text("not a vcf")
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            vcf_path=str(bogus),
        )

        assert result["isError"] is True
        assert ".vcf" in result["error"]

    async def test_zero_shot_missing_vcf_named(self, v12_server, reference_fasta, tmp_path):
        """A missing VCF is a matchable error naming the tool + path."""
        missing = tmp_path / "ghost.vcf"
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            vcf_path=str(missing),
        )

        assert result["isError"] is True
        assert "zero_shot_score" in result["error"]
        assert str(missing) in result["error"]

    async def test_zero_shot_vcf_size_cap(self, v12_server, reference_fasta, tmp_path):
        """An over-size VCF is rejected by the byte cap (T-12-04)."""
        big = tmp_path / "big.vcf"
        with open(big, "wb") as handle:
            handle.truncate(ZERO_SHOT_MAX_VCF_BYTES + 1)  # sparse file
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            vcf_path=str(big),
        )

        assert result["isError"] is True
        assert "byte cap" in result["error"]

    async def test_zero_shot_fasta_validated(self, v12_server, tmp_path):
        """fasta_path suffix + existence are enforced, naming the tool."""
        wrong_suffix = tmp_path / "ref.txt"
        wrong_suffix.write_text(">chr1\nACGT")
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(wrong_suffix),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
        )
        assert result["isError"] is True
        assert ".fasta" in result["error"]

        missing = tmp_path / "nope.fasta"
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(missing),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
        )
        assert result["isError"] is True
        assert "not found" in result["error"]
        assert "zero_shot_score" in result["error"]

    async def test_zero_shot_invalid_paradigm_and_context_window(self, v12_server, reference_fasta):
        """paradigm and context_window are validated up front."""
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
            paradigm="random",
        )
        assert result["isError"] is True
        assert "paradigm" in result["error"]

        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
            context_window=0,
        )
        assert result["isError"] is True
        assert "context_window" in result["error"]

    async def test_zero_shot_clnsig_filter_validation(self, v12_server, reference_fasta):
        """Unknown keys and wrong types are matchable errors."""
        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
            clnsig_filter={"unknown_key": 1},
        )
        assert result["isError"] is True
        assert "clnsig_filter" in result["error"]

        result = await v12_server._zero_shot_score(
            model_name="test-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
            clnsig_filter={"positive_labels": [1, 2]},
        )
        assert result["isError"] is True
        assert "positive_labels" in result["error"]

    async def test_zero_shot_clnsig_filter_null_members_treated_as_absent(
        self, v12_server, reference_fasta, tmp_path
    ):
        """WR-01: keys present with a JSON null fall back to the defaults
        instead of crashing into the generic error (frozenset(None)) or
        silently passing None into the kernel."""
        vcf = tmp_path / "clinvar.vcf"
        vcf.write_text("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\n")
        with patch(
            "dnallm.mcp.server.evaluate_vcf", return_value=self._vep_result()
        ) as mock_kernel:
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                vcf_path=str(vcf),
                clnsig_filter={
                    "variant_type": None,
                    "positive_labels": None,
                    "negative_labels": None,
                    "star_floor": None,
                },
            )

        assert not result.get("isError")
        kernel_filter = mock_kernel.call_args.kwargs["clnsig_filter"]
        assert kernel_filter == ClinVarFilter()  # D-17 defaults, no crash

    async def test_zero_shot_model_errors(self, v12_server, reference_fasta):
        """Registry and engine gates produce their matchable dicts."""
        v12_server.model_manager.config_manager.get_model_config.return_value = None
        result = await v12_server._zero_shot_score(
            model_name="ghost",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
        )
        assert "not configured" in result["error"]

        v12_server.model_manager.config_manager.get_model_config.return_value = Mock()
        v12_server.model_manager.get_inference_engine.return_value = None
        result = await v12_server._zero_shot_score(
            model_name="registered-model",
            fasta_path=str(reference_fasta),
            variants=[{"chrom": "chr1", "pos": 2, "ref": "C", "alt": "G"}],
        )
        assert result == {"error": "Model registered-model not loaded", "isError": True}

    async def test_zero_shot_kernel_value_error_is_matchable(self, v12_server, reference_fasta):
        """Kernel ValueErrors (assembly mismatch, unreadable VCF) surface
        their text; nothing raises across the boundary."""
        with patch(
            "dnallm.mcp.server.evaluate_vcf",
            side_effect=ValueError("VCF position 99 on chr1 exceeds the reference length 64"),
        ):
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"}],
            )

        assert result["isError"] is True
        assert "zero_shot_score" in result["error"]
        assert "exceeds the reference length" in result["error"]

    async def test_zero_shot_kernel_other_exception_is_generic_dict(
        self, v12_server, reference_fasta
    ):
        """Non-ValueError kernel faults return the generic isError dict."""
        with patch("dnallm.mcp.server.evaluate_vcf", side_effect=RuntimeError("kernel fault")):
            result = await v12_server._zero_shot_score(
                model_name="test-model",
                fasta_path=str(reference_fasta),
                variants=[{"chrom": "chr1", "pos": 10, "ref": "A", "alt": "G"}],
            )

        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "Zero-shot scoring failed. See server logs for details."
        )
