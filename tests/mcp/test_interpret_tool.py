"""Unit tests for the dna_interpret MCP tool."""

import asyncio
import threading
import time

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

        async def mock_predict(*args, **kwargs):  # ruff: ignore[unused-async]
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

    async def test_mamba_architecture_guard_returns_clean_error(self, mock_server):
        """Mamba-family models refuse interpretation instead of dying (261003-csd).

        Captum gradient backward on Mamba-family models (pure-PyTorch
        fallback backend) exhausts memory and the whole serving process is
        SIGKILLed at driver level (repro: lig and layer_conductance on the
        open_chromatin DNAMamba model exit 137 in-process), so the tool must
        return a typed error and never instantiate the interpreter.
        """
        cfg = Mock()
        cfg.model.task_info.architecture = "DNAMamba"
        mock_server.model_manager.config_manager.get_model_config.return_value = cfg

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="mamba-model",
                method="lig",
                target_class=0,
            )

        assert result.get("isError") is True
        assert "not supported" in result["error"]
        assert "mamba-model" in result["error"]
        mock_interp_cls.assert_not_called()

    async def test_non_mamba_architecture_interprets_normally(self, mock_server):
        """Non-mamba architectures are unaffected by the guard."""
        cfg = Mock()
        cfg.model.task_info.architecture = "DNABERT"
        mock_server.model_manager.config_manager.get_model_config.return_value = cfg

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="bert-model",
                method="lig",
                target_class=1,
            )

        assert "isError" not in result or result.get("isError") is False
        mock_interp_cls.assert_called_once()

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

        async def mock_predict(*args, **kwargs):  # ruff: ignore[unused-async]
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

    async def test_invalid_sequence_characters_return_error(self, mock_server):
        """Non-ACGTN characters fail validation before engine access."""
        result = await mock_server._dna_interpret(
            sequence="ATGZ",
            model_name="test-model",
            method="lig",
            target_class=0,
        )

        assert result["isError"] is True
        assert "invalid characters" in result["error"]
        assert "A, C, G, T, N" in result["error"]

    async def test_target_class_fallback_on_empty_probabilities(self, mock_server):
        """Auto-selection falls back to class 0 when probabilities are empty."""

        async def mock_predict(*args, **kwargs):  # ruff: ignore[unused-async]
            return {"probabilities": []}

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

        assert result["target_class"] == 0

    async def test_target_class_fallback_on_none_prediction(self, mock_server):
        """Auto-selection falls back to class 0 when prediction fails."""

        async def mock_predict(*args, **kwargs):  # ruff: ignore[unused-async]
            return None

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

        assert result["target_class"] == 0

    async def test_interpret_exception_returns_error_dict(self, mock_server):
        """A raising interpretation is caught with the verbatim text."""

        async def mock_predict(*args, **kwargs):  # ruff: ignore[unused-async]
            return {"probabilities": [0.2, 0.8]}

        mock_server.model_manager.predict_sequence = mock_predict

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp_cls.side_effect = RuntimeError("captum fault")

            result = await mock_server._dna_interpret(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=1,
            )

        assert result["isError"] is True
        assert result["content"][0]["text"] == (
            "Interpretation failed. See server logs for details."
        )


@pytest.mark.asyncio
class TestDNAInterpretLoopOffloading:
    """WR-01 (05-REVIEW.md:198): dna_interpret must never block the event loop.

    Before the fix, ``_dna_interpret`` called the synchronous captum
    attribution ``interpreter.interpret(...)`` directly on the event-loop
    thread (observed 172s on a non-mamba model against a 30s tool cap).
    ``_with_timeout_wrapper``'s ``asyncio.wait_for`` only fires at await
    points, so while the single loop thread ran a long interpretation the
    dna_interpret timeout was dead code and every client on every transport
    was frozen.

    The fix submits the whole blocking body to the default executor behind
    a dedicated ``_interpret_thread_lock`` acquired INSIDE the
    executor-submitted closure (the CR-01 / 261003-hhj pattern).  Accepted
    property (same as CR-01): on timeout only the await is cancelled — the
    orphaned executor thread runs to completion and keeps the flight.  The
    assertions below cover loop responsiveness and flight retention, never
    thread cancellation.
    """

    @staticmethod
    def _blocked_interpret(state: dict, started: threading.Event, release: threading.Event):
        """Return an interpret() fake that blocks until released.

        The bounded ``release.wait(timeout=5)`` stands in for a long real
        attribution (172s class) and keeps a failing run from wedging
        pytest-timeout: on the pre-fix code it is the only thing that
        unblocks the frozen loop thread.
        """
        tokens = ["A", "T", "G", "C"]
        scores = np.array([0.1, -0.2, 0.3, -0.1])

        def fake_interpret(*args, **kwargs):
            state["in_flight"] = True
            started.set()
            release.wait(timeout=5)
            state["in_flight"] = False
            return (tokens, scores)

        return fake_interpret

    async def test_concurrent_tool_completes_while_interpretation_blocked(self, mock_server):
        """A trivial tool call completes WHILE an attribution is blocked.

        RED on the pre-fix code: ``interpret`` runs inline on the loop
        thread, so the health-check heartbeat can only run after the 5s
        block releases — by then the fake has already flipped
        ``state["in_flight"]`` back to False and the assertion below
        fails.  That is the proof the loop was frozen (WR-01).
        """
        state = {"in_flight": False}
        started = threading.Event()
        release = threading.Event()

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret = Mock(
                side_effect=self._blocked_interpret(state, started, release)
            )

            task = asyncio.create_task(
                mock_server._dna_interpret(
                    sequence="ATGC",
                    model_name="test-model",
                    method="lig",
                    target_class=0,
                )
            )

            # Poll for the fake to start without ever blocking the loop
            # thread under test on a threading wait.  The 7s window must
            # exceed the fake's 5s bounded block: since Python 3.12
            # asyncio.wait_for drives a raw coroutine inline in the
            # caller's task, a shorter window would fail at this poll with
            # TimeoutError on the pre-fix code instead of proving the
            # heartbeat property at the assertion below.
            async def _poll_started():
                while not started.is_set():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(_poll_started(), timeout=7)

            # A cheap pure-async tool (two config lookups) must complete
            # while the attribution is still blocked — the serving-liveness
            # property WR-01 restores.  A bare Mock has no len(), so give
            # get_loaded_models a real empty list first.
            mock_server.model_manager.get_loaded_models.return_value = []
            heartbeat = await asyncio.wait_for(mock_server._health_check(), timeout=2)
            assert heartbeat.get("health", {}).get("status") == "healthy"
            assert state["in_flight"] is True, (
                "health_check only ran after the interpretation unblocked — "
                "captum work occupied the event-loop thread (WR-01)"
            )

            release.set()
            result = await asyncio.wait_for(task, timeout=5)
            assert "isError" not in result or result.get("isError") is False

    async def test_timeout_wrapper_fires_while_interpretation_blocked(self, mock_server):
        """The tool timeout returns its timeout dict while interpret still runs.

        RED on the pre-fix code: ``asyncio.wait_for`` cannot fire while the
        loop thread is blocked inside the sync attribution, so the wrapped
        call returns a SUCCESS dict only after the 5s block releases and
        the ``error_type == "timeout"`` assertion fails — the timeout
        wrapper was dead code for dna_interpret (WR-01).
        """
        state = {"in_flight": False}
        started = threading.Event()
        release = threading.Event()

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret = Mock(
                side_effect=self._blocked_interpret(state, started, release)
            )

            mock_server._tool_timeout_seconds = 0.5
            mock_server._log_format = "text"
            wrapped = mock_server._with_timeout_wrapper(mock_server._dna_interpret, "dna_interpret")

            wall_start = time.monotonic()
            result = await wrapped(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
            )
            elapsed = time.monotonic() - wall_start

            assert result.get("error_type") == "timeout", (
                "the dna_interpret timeout wrapper cannot fire while the "
                "attribution occupies the loop thread (WR-01)"
            )
            assert result.get("isError") is True
            assert elapsed < 2, f"wrapped call took {elapsed:.1f}s, not the 0.5s cap"
            assert state["in_flight"] is True, (
                "the wrapper must return while the attribution is still running"
            )

            # The server stays usable after a timeout fired: release the
            # orphan, restore a working interpret, call the tool again.
            release.set()
            mock_server._tool_timeout_seconds = 10
            mock_interp.interpret.side_effect = None
            mock_interp.interpret.return_value = (
                ["A", "T", "G", "C"],
                np.array([0.1, -0.2, 0.3, -0.1]),
            )

            retry = await wrapped(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
            )
            assert not retry.get("isError")
            assert "attributions" in retry

    async def test_timeout_cancelled_interpret_holds_single_flight_for_retry(self, mock_server):
        """A timeout→retry loop cannot stack concurrent attributions (CR-01 mirror).

        RED on the pre-fix code right at the first assertion: the wrapped
        call cannot time out while the loop thread is blocked, so call 1
        returns a success dict instead of the timeout dict.  After the fix
        this test also pins the dedicated interpret flight lock: without
        it, a bare executor move would let the immediate retry enter
        ``interpret`` concurrently with the orphaned, uncancellable first
        attribution (max_active 2 on one shared torch model).
        """
        counts = {"active": 0, "max_active": 0}
        guard = threading.Lock()
        started = threading.Event()
        release = threading.Event()
        tokens = ["A", "T", "G", "C"]
        scores = np.array([0.1, -0.2, 0.3, -0.1])

        def counting_interpret(*args, **kwargs):
            with guard:
                counts["active"] += 1
                counts["max_active"] = max(counts["max_active"], counts["active"])
            started.set()
            release.wait(timeout=5)
            with guard:
                counts["active"] -= 1
            return (tokens, scores)

        with patch("dnallm.mcp.server.DNAInterpret") as mock_interp_cls:
            mock_interp = Mock()
            mock_interp_cls.return_value = mock_interp
            mock_interp.interpret = Mock(side_effect=counting_interpret)

            mock_server._tool_timeout_seconds = 0.3
            mock_server._log_format = "text"
            wrapped = mock_server._with_timeout_wrapper(mock_server._dna_interpret, "dna_interpret")

            # Call 1 hits the tool timeout and abandons its attribution to
            # the executor thread — the uncancellable orphan holds the flight.
            first = await wrapped(
                sequence="ATGC",
                model_name="test-model",
                method="lig",
                target_class=0,
            )
            assert first.get("error_type") == "timeout", (
                "call 1 could not be timed out — the wrapper is dead code "
                "while the attribution occupies the loop thread (WR-01)"
            )

            # The client's immediate retry must stay OUT of interpret while
            # the orphan is still inside it.
            mock_server._tool_timeout_seconds = 10
            retry_task = asyncio.create_task(
                wrapped(
                    sequence="ATGC",
                    model_name="test-model",
                    method="lig",
                    target_class=0,
                )
            )
            await asyncio.sleep(0.3)
            assert counts["active"] == 1, (
                "retry entered interpret while the orphaned attribution was "
                "still running — single-flight was lost across the timeout "
                "cancellation boundary"
            )

            release.set()
            retry = await asyncio.wait_for(retry_task, timeout=5)
            assert not retry.get("isError")
            assert "attributions" in retry
            assert counts["max_active"] == 1, (
                f"concurrent interpretations observed (max {counts['max_active']})"
            )
