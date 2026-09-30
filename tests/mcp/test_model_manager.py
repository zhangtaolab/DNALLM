"""Lifecycle and routing tests for the MCP ModelManager.

Model loading runs against a real YAML configuration directory (the
established dnallm/mcp/tests/test_server_integration.py seam) with the
``load_model_and_tokenizer`` boundary patched at its import site — the
dispatch, executor bridge, and status routing under test are all real.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
import yaml

from dnallm.mcp.config_manager import MCPConfigManager
from dnallm.mcp.model_manager import ModelManager


def _write_configs(
    temp_dir: Path,
    *,
    broken_model: bool = False,
) -> Path:
    """Write a server config plus one model config into temp_dir.

    Args:
        temp_dir: Directory the YAML files are written into.
        broken_model: When True, add a second enabled model entry whose
            config file does not exist (loading it must fail gracefully).

    Returns:
        Path to temp_dir (the MCPConfigManager config directory).
    """
    models: dict[str, dict[str, Any]] = {
        "test_model": {
            "name": "test_model",
            "model_name": "Test Model",
            "config_path": "./test_model_config.yaml",
            "enabled": True,
            "priority": 1,
        }
    }
    if broken_model:
        models["broken_model"] = {
            "name": "broken_model",
            "model_name": "Broken Model",
            "config_path": "./does_not_exist.yaml",
            "enabled": True,
            "priority": 2,
        }

    server_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 8123,
            "workers": 1,
            "log_level": "INFO",
            "debug": False,
        },
        "mcp": {
            "name": "Model Manager Test",
            "version": "0.1.0",
            "description": "test",
        },
        "models": models,
        "multi_model": {},
        "sse": {
            "heartbeat_interval": 30,
            "max_connections": 100,
            "connection_timeout": 300,
            "enable_compression": True,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file": "./logs/test.log",
            "max_size": "10MB",
            "backup_count": 5,
        },
    }
    model_config = {
        "task": {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["Not promoter", "Core promoter"],
            "threshold": 0.5,
            "description": "Test promoter prediction",
        },
        "inference": {
            "batch_size": 16,
            "max_length": 512,
            "device": "auto",
            "num_workers": 4,
            "precision": "float16",
            "output_dir": str(temp_dir / "out"),
            "save_predictions": True,
        },
        "model": {
            "name": "test_model",
            "path": "test/path",
            "source": "huggingface",
            "task_info": {
                "architecture": "DNABERT",
                "tokenizer": "BPE",
                "species": "plant",
                "task_category": "promoter_prediction",
            },
        },
    }

    with open(temp_dir / "mcp_server_config.yaml", "w") as f:
        yaml.dump(server_config, f)
    with open(temp_dir / "test_model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    return temp_dir


@pytest.fixture
def manager(tmp_path: Path) -> ModelManager:
    """ModelManager over a real config directory."""
    config_dir = _write_configs(tmp_path)
    return ModelManager(MCPConfigManager(str(config_dir)))


_DEFAULT_INFER_RESULT = {"probabilities": [0.5, 0.5]}


def _fake_engine(infer_result: Any = None) -> Mock:
    """Build a mock DNAInference engine with the predict surface."""
    engine = Mock()
    engine.infer_seqs = Mock(
        return_value=_DEFAULT_INFER_RESULT if infer_result is None else infer_result
    )
    engine.estimate_memory_usage = Mock(return_value=128)
    return engine


class TestLoadModelLifecycle:
    """Test async load/unload routing and status transitions."""

    @pytest.mark.asyncio
    async def test_load_model_success(self, manager):
        """A successful load registers a loaded engine with status loaded."""
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            ok = await manager.load_model("test_model")

        assert ok is True
        assert manager.get_loaded_models() == ["test_model"]
        assert manager.get_model_status("test_model") == "loaded"
        assert manager.get_inference_engine("test_model") is not None

    @pytest.mark.asyncio
    async def test_load_model_is_lazy_and_idempotent(self, manager):
        """A second load returns True without touching the load boundary."""
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            assert await manager.load_model("test_model") is True
            assert await manager.load_model("test_model") is True

        assert mock_load.call_count == 1

    @pytest.mark.asyncio
    async def test_load_model_unknown_config_fails(self, manager):
        """An unconfigured model name fails and is marked error."""
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            ok = await manager.load_model("ghost_model")

        assert ok is False
        assert manager.get_model_status("ghost_model") == "error"
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_previously_failed_short_circuits(self, manager):
        """A model already marked error is not retried."""
        manager.model_loading_status["test_model"] = "error"
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            ok = await manager.load_model("test_model")

        assert ok is False
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_in_flight_short_circuits(self, manager):
        """A model already being loaded is not double-loaded."""
        manager.model_loading_status["test_model"] = "loading"
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            ok = await manager.load_model("test_model")

        assert ok is False
        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_model_loader_failure_marks_error(self, manager):
        """A raising loader is recorded as an error status, not raised."""
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.side_effect = RuntimeError("download exploded")
            ok = await manager.load_model("test_model")

        assert ok is False
        assert manager.get_model_status("test_model") == "error"
        assert manager.get_loaded_models() == []

    @pytest.mark.asyncio
    async def test_load_model_sync_passes_config_through(self, manager):
        """The executor bridge forwards path, task config, and source."""
        from dnallm.configuration.configs import TaskConfig

        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            task_config = TaskConfig(task_type="binary", num_labels=2, label_names=["a", "b"])
            result = manager._load_model_sync("some/path", task_config, "modelscope")

        assert isinstance(result, tuple)
        mock_load.assert_called_once_with(
            model_name="some/path", task_config=task_config, source="modelscope"
        )


class TestLoadAllEnabledModels:
    """Test aggregate loading of every enabled model."""

    @pytest.mark.asyncio
    async def test_aggregates_success_and_failure(self, tmp_path):
        """A broken model entry fails while the good one loads."""
        config_dir = _write_configs(tmp_path, broken_model=True)
        manager = ModelManager(MCPConfigManager(str(config_dir)))

        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            results = await manager.load_all_enabled_models()

        assert results == {"test_model": True, "broken_model": False}

    @pytest.mark.asyncio
    async def test_exception_from_load_recorded_as_failure(self, manager):
        """An exception escaping load_model is captured as a False result."""

        async def exploding_load(model_name: str) -> bool:
            await asyncio.sleep(0)
            raise RuntimeError("aggregate fault")

        manager.config_manager.get_enabled_models = Mock(return_value=["test_model"])
        with patch.object(manager, "load_model", exploding_load):
            results = await manager.load_all_enabled_models()

        assert results == {"test_model": False}


class TestStatusRouting:
    """Test status lookup fallbacks."""

    def test_unknown_model_status_is_not_found(self, manager):
        """An unknown model reports not_found."""
        assert manager.get_model_status("ghost") == "not_found"


class TestPredictionRouting:
    """Test predict_sequence / predict_batch / predict_multi_model."""

    @pytest.mark.asyncio
    async def test_predict_sequence_not_loaded_returns_none(self, manager):
        """Predicting with an unloaded model returns None."""
        assert await manager.predict_sequence("ghost", "ATCG") is None

    @pytest.mark.asyncio
    async def test_predict_sequence_runs_inference(self, manager):
        """A loaded engine's infer_seqs result is returned verbatim."""
        engine = _fake_engine({"probabilities": [0.9, 0.1]})
        manager.loaded_models["test_model"] = engine

        result = await manager.predict_sequence("test_model", "ATCG")

        assert result == {"probabilities": [0.9, 0.1]}
        engine.infer_seqs.assert_called_once_with("ATCG")

    @pytest.mark.asyncio
    async def test_predict_sequence_infer_failure_returns_none(self, manager):
        """A raising inference call is swallowed into None."""
        engine = _fake_engine()
        engine.infer_seqs = Mock(side_effect=RuntimeError("bad tensors"))
        manager.loaded_models["test_model"] = engine

        assert await manager.predict_sequence("test_model", "ATCG") is None

    @pytest.mark.asyncio
    async def test_predict_batch_not_loaded_returns_none(self, manager):
        """Batch prediction with an unloaded model returns None."""
        assert await manager.predict_batch("ghost", ["ATCG"]) is None

    @pytest.mark.asyncio
    async def test_predict_batch_runs_inference(self, manager):
        """A loaded engine receives the full sequence list."""
        engine = _fake_engine({"batch": True})
        manager.loaded_models["test_model"] = engine

        result = await manager.predict_batch("test_model", ["ATCG", "GGCC"])

        assert result == {"batch": True}
        engine.infer_seqs.assert_called_once_with(["ATCG", "GGCC"])

    @pytest.mark.asyncio
    async def test_predict_batch_infer_failure_returns_none(self, manager):
        """A raising batch inference is swallowed into None."""
        engine = _fake_engine()
        engine.infer_seqs = Mock(side_effect=RuntimeError("bad batch"))
        manager.loaded_models["test_model"] = engine

        assert await manager.predict_batch("test_model", ["ATCG"]) is None

    @pytest.mark.asyncio
    async def test_predict_multi_model_merges_results(self, manager):
        """Each model's result lands under its own key."""
        good = _fake_engine({"probabilities": [0.9, 0.1]})
        bad = _fake_engine()
        bad.infer_seqs = Mock(side_effect=RuntimeError("engine fault"))
        manager.loaded_models["model-a"] = good
        manager.loaded_models["model-b"] = bad

        result = await manager.predict_multi_model(["model-a", "model-b"], "ATCG")

        assert result["model-a"] == {"probabilities": [0.9, 0.1]}
        # predict_sequence swallows the engine fault into None
        assert result["model-b"] is None

    @pytest.mark.asyncio
    async def test_predict_multi_model_exception_recorded(self, manager):
        """An exception escaping predict_sequence becomes an error entry."""

        async def exploding_predict(model_name: str, sequence: str, **kwargs):
            await asyncio.sleep(0)
            raise RuntimeError("transport gone")

        with patch.object(manager, "predict_sequence", exploding_predict):
            result = await manager.predict_multi_model(["model-a"], "ATCG")

        assert result == {"model-a": {"error": "transport gone"}}


class TestModelInfo:
    """Test get_model_info and get_all_models_info."""

    def test_get_model_info_unknown_returns_none(self, manager):
        """An unknown model has no info."""
        assert manager.get_model_info("ghost") is None

    def test_get_model_info_with_engine_includes_memory(self, manager):
        """A loaded engine contributes its memory estimate."""
        engine = _fake_engine()
        manager.loaded_models["test_model"] = engine
        manager.model_loading_status["test_model"] = "loaded"

        info = manager.get_model_info("test_model")

        assert info["name"] == "test_model"
        assert info["status"] == "loaded"
        assert info["loaded"] is True
        assert info["memory_usage"] == 128
        assert info["task_type"] == "binary"
        assert info["architecture"] == "DNABERT"

    def test_get_model_info_estimate_failure_omits_memory(self, manager):
        """A failing memory estimate degrades gracefully (no key)."""
        engine = _fake_engine()
        engine.estimate_memory_usage = Mock(side_effect=RuntimeError("no cuda"))
        manager.loaded_models["test_model"] = engine

        info = manager.get_model_info("test_model")

        assert "memory_usage" not in info
        assert info["name"] == "test_model"

    def test_get_all_models_info_lists_enabled(self, manager):
        """Every enabled model with a loaded config appears in the info map."""
        all_info = manager.get_all_models_info()
        assert set(all_info) == {"test_model"}


class TestUnload:
    """Test unload_model and unload_all_models."""

    @pytest.mark.asyncio
    async def test_unload_model_removes_engine_and_status(self, manager):
        """Unloading drops the engine and its status entry."""
        with patch("dnallm.mcp.model_manager.load_model_and_tokenizer") as mock_load:
            mock_load.return_value = (Mock(), Mock())
            await manager.load_model("test_model")

        assert manager.unload_model("test_model") is True
        assert manager.get_loaded_models() == []
        assert manager.get_model_status("test_model") == "not_found"

    def test_unload_unknown_model_returns_false(self, manager):
        """Unloading a model that is not loaded returns False."""
        assert manager.unload_model("ghost") is False

    @pytest.mark.asyncio
    async def test_unload_all_models_counts(self, manager):
        """unload_all_models returns the number of models removed."""
        manager.loaded_models["m1"] = _fake_engine()
        manager.loaded_models["m2"] = _fake_engine()

        assert manager.unload_all_models() == 2
        assert manager.get_loaded_models() == []
