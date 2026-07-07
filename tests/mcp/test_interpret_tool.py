"""Unit tests for the dna_interpret MCP tool."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
import torch


@pytest.fixture
def mock_inference_engine_for_interpret():
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
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        }
    )
    mock_tokenizer.convert_ids_to_tokens = Mock(return_value=["A", "T", "G", "C"])

    mock_task_config = Mock()
    mock_task_config.task_type = "binary"
    mock_task_config.label_names = ["negative", "positive"]

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
def mock_server(mock_inference_engine_for_interpret):
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
        mock_mgr.get_inference_engine.return_value = mock_inference_engine_for_interpret
        mock_model_mgr.return_value = mock_mgr

        server = DNALLMMCPServer("config.yaml")
        server.app = Mock()
        server.model_manager = mock_mgr
        return server


@pytest.mark.asyncio
class TestDNAInterpretTool:
    """Tests for the _dna_interpret MCP tool."""

    @pytest.fixture(autouse=True)
    def setup_mock_predict(self, mock_server):
        """Setup mock predict_sequence for auto target_class selection."""
        import asyncio

        async def mock_predict(*args, **kwargs):  # noqa: RUF029
            return {"probabilities": [0.3, 0.7]}

        mock_server.model_manager.predict_sequence = mock_predict

    async def test_lig_method(self, mock_server):
        """Test LIG (Layer Integrated Gradients) method."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=1,
            )

            assert "isError" not in result or result.get("isError") is False
            assert "attributions" in result
            assert "raw" in result["attributions"]
            assert "normalized" in result["attributions"]
            assert result["tokens"] == ["A", "T", "G", "C"]
            assert result["method"] == "lig"
            assert result["target_class"] == 1
            assert result["model_name"] == "test-model"
            assert result["sequence"] == "ATGC"

    async def test_deeplift_method(self, mock_server):
        """Test DeepLIFT method."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="deeplift",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert "attributions" in result
            assert result["method"] == "deeplift"

    async def test_occlusion_method(self, mock_server):
        """Test Occlusion method."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="occlusion",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "occlusion"

    async def test_feature_ablation_method(self, mock_server):
        """Test Feature Ablation method."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="feature_ablation",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "feature_ablation"

    async def test_layer_conductance_method(self, mock_server):
        """Test Layer Conductance method with auto-detected embedding layer."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_embedding_layer = Mock()
            mock_interp._find_embedding_layer.return_value = mock_embedding_layer
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="layer_conductance",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "layer_conductance"
            # Verify that interpret was called with target_layer
            call_kwargs = mock_interp.interpret.call_args[1]
            assert "target_layer" in call_kwargs

    async def test_gradient_shap_method(self, mock_server):
        """Test Gradient SHAP method (mapped from gradient_shap to gradshap)."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="gradient_shap",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "gradient_shap"
            # Verify internal method mapping
            call_kwargs = mock_interp.interpret.call_args[1]
            assert call_kwargs["method"] == "gradshap"

    async def test_noise_tunnel_method(self, mock_server):
        """Test Noise Tunnel method."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="noise_tunnel",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "noise_tunnel"

    async def test_integrated_gradients_method(self, mock_server):
        """Test Integrated Gradients method (mapped to lig)."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="integrated_gradients",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["method"] == "integrated_gradients"
            # Verify internal method mapping
            call_kwargs = mock_interp.interpret.call_args[1]
            assert call_kwargs["method"] == "lig"

    async def test_auto_target_class(self, mock_server):
        """Test auto-selection of target_class when None."""

        async def mock_predict(*args, **kwargs):  # noqa: RUF029
            return {"probabilities": [0.2, 0.8]}

        mock_server.model_manager.predict_sequence = mock_predict

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=None,
            )

            assert "isError" not in result or result.get("isError") is False
            # Should auto-select class with max probability (class 1)
            assert result["target_class"] == 1

    async def test_invalid_method(self, mock_server):
        """Test error for invalid method."""
        result = await mock_server._dna_interpret(
            sequence="ATGC",
            model_name="test-model",
            method="invalid_method",
            target_class=0,
        )

        assert result.get("isError") is True
        assert "method" in result["error"].lower()

    async def test_model_not_loaded(self, mock_server):
        """Test error when model is not loaded."""
        mock_server.model_manager.get_inference_engine.return_value = None

        result = await mock_server._dna_interpret(
            sequence="ATGC",
            model_name="nonexistent-model",
            method="lig",
            target_class=0,
        )

        assert result.get("isError") is True
        assert "not loaded" in result["error"].lower()

    async def test_normalization_with_zero_range(self, mock_server):
        """Test normalization when all attribution scores are equal."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            # All zeros - should produce normalized zeros
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.0, 0.0, 0.0, 0.0]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            assert result["attributions"]["normalized"] == [0.0, 0.0, 0.0, 0.0]

    async def test_normalization_with_nonzero_range(self, mock_server):
        """Test normalization when attribution scores have range."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.0, 0.5, 1.0, 0.25]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
            )

            assert "isError" not in result or result.get("isError") is False
            normalized = result["attributions"]["normalized"]
            # min=0, max=1, so normalized should be [0, 0.5, 1, 0.25]
            assert normalized[0] == pytest.approx(0.0, abs=1e-6)
            assert normalized[1] == pytest.approx(0.5, abs=1e-6)
            assert normalized[2] == pytest.approx(1.0, abs=1e-6)
            assert normalized[3] == pytest.approx(0.25, abs=1e-6)

    async def test_max_length_parameter(self, mock_server):
        """Test that max_length is passed through to interpret."""
        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
                max_length=256,
            )

            assert "isError" not in result or result.get("isError") is False
            call_kwargs = mock_interp.interpret.call_args[1]
            assert call_kwargs["max_length"] == 256
