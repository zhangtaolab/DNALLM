"""Unit tests for the dna_mutagenesis MCP tool."""

import pytest
from unittest.mock import Mock, patch
import numpy as np


@pytest.fixture
def mock_inference_engine_for_mutagenesis():
    """Return a mock inference engine with model, tokenizer, and config."""
    mock_model = Mock()
    mock_model.device = "cpu"
    mock_model.parameters.return_value = iter([Mock()])

    mock_tokenizer = Mock()
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.all_special_ids = [0, 1, 2]
    mock_tokenizer.mask_token_id = 3
    mock_tokenizer.decode = Mock(return_value="A")
    mock_tokenizer.convert_tokens_to_ids = Mock(return_value=5)
    mock_tokenizer.__call__ = Mock(
        return_value={
            "input_ids": Mock(),
            "attention_mask": Mock(),
        }
    )

    mock_task_config = Mock()
    mock_task_config.task_type = "binary"

    mock_pred_config = Mock()
    mock_pred_config.max_length = 512
    mock_pred_config.batch_size = 8
    mock_pred_config.num_workers = 0
    mock_pred_config.device = "cpu"

    mock_config = {
        "task": mock_task_config,
        "inference": mock_pred_config,
    }

    mock_engine = Mock()
    mock_engine.model = mock_model
    mock_engine.tokenizer = mock_tokenizer
    mock_engine.config = mock_config

    return mock_engine


@pytest.fixture
def mock_server(mock_inference_engine_for_mutagenesis):
    """Return a mock-configured DNALLMMCPServer."""
    from dnallm.mcp.server import DNALLMMCPServer

    with (
        patch("dnallm.mcp.server.MCPConfigManager") as mock_cfg_mgr,
        patch("dnallm.mcp.server.ModelManager") as mock_model_mgr,
    ):
        mock_cfg = Mock()
        mock_cfg.get_server_config.return_value = Mock(
            mcp=Mock(name="test", description="test", version="1.0"),
            server=Mock(host="127.0.0.1", port=8000),
        )
        mock_cfg.get_enabled_models.return_value = []
        mock_cfg_mgr.return_value = mock_cfg

        mock_mgr = Mock()
        mock_mgr.get_inference_engine.return_value = mock_inference_engine_for_mutagenesis
        mock_model_mgr.return_value = mock_mgr

        server = DNALLMMCPServer("config.yaml")
        server.app = Mock()
        server.model_manager = mock_mgr
        return server


@pytest.mark.asyncio
class TestDNAMutagenesisTool:
    """Tests for the _dna_mutagenesis MCP tool."""

    async def test_single_base_substitution(self, mock_server):
        """Test single base substitution mutation type."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "mut_0_A_T": {
                    "sequence": "TTGC",
                    "pred": np.array([0.2]),
                    "logfc": np.array([0.5]),
                    "diff": np.array([0.1]),
                    "score": 0.5,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequence="ATGC",
                mutation_type="single_base_substitution",
                positions=[0],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "original_prediction" in result
            assert "mutated_prediction" in result
            assert "delta" in result
            assert result["affected_positions"] == [0]
            assert result["mutation_type"] == "single_base_substitution"
            assert result["model_name"] == "test-model"

    async def test_multi_base_substitution(self, mock_server):
        """Test multi base substitution mutation type."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "mut_0_A_T": {
                    "sequence": "TTGC",
                    "pred": np.array([0.2]),
                    "logfc": np.array([0.5]),
                    "diff": np.array([0.1]),
                    "score": 0.5,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequence="ATGC",
                mutation_type="multi_base_substitution",
                positions=[0, 1],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "original_prediction" in result
            assert "mutated_prediction" in result
            assert result["affected_positions"] == [0, 1]
            assert result["mutation_type"] == "multi_base_substitution"

    async def test_deletion_mutation(self, mock_server):
        """Test deletion mutation type."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "del_0_1": {
                    "sequence": "TGC",
                    "pred": np.array([0.15]),
                    "logfc": np.array([0.3]),
                    "diff": np.array([0.05]),
                    "score": 0.3,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequence="ATGC",
                mutation_type="deletion",
                positions=[0],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "original_prediction" in result
            assert "mutated_prediction" in result
            assert result["mutation_type"] == "deletion"

    async def test_insertion_mutation(self, mock_server):
        """Test insertion mutation type."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "ins_0_N": {
                    "sequence": "NATGC",
                    "pred": np.array([0.12]),
                    "logfc": np.array([0.2]),
                    "diff": np.array([0.02]),
                    "score": 0.2,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequence="ATGC",
                mutation_type="insertion",
                positions=[0],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "original_prediction" in result
            assert "mutated_prediction" in result
            assert result["mutation_type"] == "insertion"

    async def test_combo_mutation(self, mock_server):
        """Test combo mutation type within limit."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "mut_0_A_T": {
                    "sequence": "TTGC",
                    "pred": np.array([0.2]),
                    "logfc": np.array([0.5]),
                    "diff": np.array([0.1]),
                    "score": 0.5,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequence="ATGC",
                mutation_type="combo",
                positions=[0, 1],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "original_prediction" in result
            assert "mutated_prediction" in result
            assert result["mutation_type"] == "combo"

    async def test_combo_too_many_positions(self, mock_server):
        """Test combo mutation with >5 positions returns error."""
        result = await mock_server._dna_mutagenesis(
            sequence="ATGCATGCATGCATGCATGC",
            mutation_type="combo",
            positions=[0, 1, 2, 3, 4, 5],
            model_name="test-model",
        )

        assert result.get("isError") is True
        assert "combo" in result["error"].lower()
        assert "5" in result["error"]

    async def test_missing_sequence_input(self, mock_server):
        """Test error when both sequence and sequences are None."""
        result = await mock_server._dna_mutagenesis(
            sequence=None,
            sequences=None,
            mutation_type="single_base_substitution",
            positions=[0],
            model_name="test-model",
        )

        assert result.get("isError") is True
        assert "sequence" in result["error"].lower()

    async def test_invalid_mutation_type(self, mock_server):
        """Test error for invalid mutation_type."""
        result = await mock_server._dna_mutagenesis(
            sequence="ATGC",
            mutation_type="invalid_type",
            positions=[0],
            model_name="test-model",
        )

        assert result.get("isError") is True
        assert "mutation_type" in result["error"].lower()

    async def test_missing_positions(self, mock_server):
        """Test error when positions is None."""
        result = await mock_server._dna_mutagenesis(
            sequence="ATGC",
            mutation_type="single_base_substitution",
            positions=None,
            model_name="test-model",
        )

        assert result.get("isError") is True
        assert "positions" in result["error"].lower()

    async def test_empty_positions(self, mock_server):
        """Test error when positions is empty."""
        result = await mock_server._dna_mutagenesis(
            sequence="ATGC",
            mutation_type="single_base_substitution",
            positions=[],
            model_name="test-model",
        )

        assert result.get("isError") is True
        assert "positions" in result["error"].lower()

    async def test_model_not_loaded(self, mock_server):
        """Test error when model is not loaded."""
        mock_server.model_manager.get_inference_engine.return_value = None

        result = await mock_server._dna_mutagenesis(
            sequence="ATGC",
            mutation_type="single_base_substitution",
            positions=[0],
            model_name="nonexistent-model",
        )

        assert result.get("isError") is True
        assert "not loaded" in result["error"].lower()

    async def test_batch_sequences(self, mock_server):
        """Test processing multiple sequences."""
        with patch("dnallm.mcp.server.Mutagenesis") as mock_mut_cls:
            mock_mut = Mock()
            mock_mut_cls.return_value = mock_mut
            mock_mut.evaluate.return_value = {
                "raw": {
                    "sequence": "ATGC",
                    "pred": np.array([0.1]),
                    "logfc": np.zeros(1),
                    "diff": np.zeros(1),
                    "score": 0.0,
                },
                "mut_0_A_T": {
                    "sequence": "TTGC",
                    "pred": np.array([0.2]),
                    "logfc": np.array([0.5]),
                    "diff": np.array([0.1]),
                    "score": 0.5,
                },
            }

            result = await mock_server._dna_mutagenesis(
                sequences=["ATGC", "CGTA"],
                mutation_type="single_base_substitution",
                positions=[0],
                model_name="test-model",
            )

            assert "isError" not in result or result.get("isError") is False
            assert "batch_results" in result
            assert result["sequence_count"] == 2
