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

from dnallm.mcp.server import (
    DNALLMMCPServer,
    HOTSPOT_MAX_REGION_LENGTH,
    ISM_MAX_POSITIONS,
    ISM_MAX_SEQUENCE_LENGTH,
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
