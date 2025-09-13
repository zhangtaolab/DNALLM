"""Tests for DNA model loading and management utilities.

This module tests the functions for downloading, loading, and managing
DNA language models from various sources.

"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Any

from dnallm.models.model import (
    download_model,
    is_fp8_capable,
    load_model_and_tokenizer,
    load_preset_model,
    _handle_evo2_models,
    _setup_huggingface_mirror,
    _get_model_path_and_imports,
    _create_label_mappings,
    _load_model_by_task_type,
    _configure_model_padding,
)
from dnallm.configuration.configs import TaskConfig


class TestDownloadModel:
    """Test download_model function with retry mechanism."""

    def test_download_success_first_attempt(self):
        """Test successful download on first attempt."""
        mock_downloader = Mock(return_value="/path/to/model")

        result = download_model("test-model", mock_downloader, max_try=3)

        assert result == "/path/to/model"
        assert mock_downloader.call_count == 1

    def test_download_success_after_retries(self):
        """Test successful download after multiple retries."""
        mock_downloader = Mock(
            side_effect=[
                Exception("connection error"),
                Exception("connection error"),
                "/path/to/model",
            ]
        )

        with patch("time.sleep"):  # Mock sleep to speed up test
            result = download_model("test-model", mock_downloader, max_try=3)

        assert result == "/path/to/model"
        assert mock_downloader.call_count == 3

    def test_download_failure_after_max_retries(self):
        """Test download failure after maximum retries."""
        mock_downloader = Mock(side_effect=Exception("connection error"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(
                ValueError,
                match=r"Model test-model download failed.",
            ):
                download_model("test-model", mock_downloader, max_try=2)

    def test_download_model_not_found_huggingface(self):
        """Test download failure for model not found in HuggingFace."""
        mock_downloader = Mock(side_effect=Exception("not found"))

        with pytest.raises(
            ValueError,
            match=r"Model test-model download failed.",
        ):
            download_model("test-model", mock_downloader, max_try=1)

    def test_download_model_not_found_modelscope(self):
        """Test download failure for model not found in ModelScope."""
        mock_downloader = Mock(side_effect=Exception("response [404]"))

        with pytest.raises(
            ValueError, match="Model test-model download failed"
        ):
            download_model("test-model", mock_downloader, max_try=1)

    def test_download_other_error(self):
        """Test download with other types of errors."""
        mock_downloader = Mock(side_effect=Exception("unknown error"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(
                ValueError, match="Model test-model download failed"
            ):
                download_model("test-model", mock_downloader, max_try=2)

    @pytest.mark.slow
    def test_download_real_huggingface_connection(self):
        """Test real HuggingFace connection (may be skipped
        if network unavailable).
        """
        try:
            from huggingface_hub import snapshot_download

            # Try to download a small test model
            result = download_model(
                "microsoft/DialoGPT-small", snapshot_download, max_try=1
            )
            assert result is not None
            assert os.path.exists(result)
        except Exception as e:
            if "connection" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Skipping due to network connection issue: {e}")
            else:
                raise

    @pytest.mark.slow
    def test_download_real_modelscope_connection(self):
        """Test real ModelScope connection (may be skipped if
        network unavailable).
        """
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            # Try to download a small test model
            result = download_model(
                "ZhejiangLab-LifeScience/DNA_bert_4",
                snapshot_download,
                max_try=1,
            )
            assert result is not None
            assert os.path.exists(result)
        except Exception as e:
            if "connection" in str(e).lower() or "network" in str(e).lower():
                pytest.skip(f"Skipping due to network connection issue: {e}")
            else:
                raise


class TestIsFp8Capable:
    """Test is_fp8_capable function for hardware detection."""

    @patch("torch.cuda.get_device_capability")
    def test_fp8_capable_hopper(self, mock_capability):
        """Test FP8 capability detection for Hopper (H100) architecture."""
        mock_capability.return_value = (9, 0)

        result = is_fp8_capable()

        assert result is True

    @patch("torch.cuda.get_device_capability")
    def test_fp8_capable_newer_architecture(self, mock_capability):
        """Test FP8 capability detection for newer architecture."""
        mock_capability.return_value = (9, 1)

        result = is_fp8_capable()

        assert result is True

    @patch("torch.cuda.get_device_capability")
    def test_fp8_not_capable_older_architecture(self, mock_capability):
        """Test FP8 capability detection for older architecture."""
        mock_capability.return_value = (8, 0)

        result = is_fp8_capable()

        assert result is False

    @patch("torch.cuda.get_device_capability")
    def test_fp8_not_capable_much_older_architecture(self, mock_capability):
        """Test FP8 capability detection for much older architecture."""
        mock_capability.return_value = (7, 5)

        result = is_fp8_capable()

        assert result is False


class TestHandleEvo2Models:
    """Test _handle_evo2_models function for EVO2 model handling."""

    def test_handle_evo2_models_not_evo2(self):
        """Test handling of non-EVO2 models."""
        result = _handle_evo2_models("regular-model", "huggingface")

        assert result is None

    def test_handle_evo2_models_evo2_local(self):
        """Test handling of EVO2 models with local source."""
        with patch("glob.glob", return_value=["/path/to/model.pt"]):
            with patch("os.path.isdir", return_value=True):
                with patch("evo2.Evo2") as mock_evo2:
                    mock_model = Mock()
                    mock_tokenizer = Mock()
                    mock_evo2.return_value = mock_model
                    mock_model.tokenizer = mock_tokenizer

                    with patch(
                        "dnallm.models.model.is_fp8_capable", return_value=True
                    ):
                        result = _handle_evo2_models("evo2_1b_base", "local")

                    assert result == (mock_model, mock_tokenizer)
                    mock_evo2.assert_called_once_with(
                        "evo2_1b_base",
                        local_path="/path/to/model.pt",
                        use_fp8=True,
                    )

    def test_handle_evo2_models_evo2_remote(self):
        """Test handling of EVO2 models with remote source."""
        with patch("evo2.Evo2") as mock_evo2:
            mock_model = Mock()
            mock_tokenizer = Mock()
            mock_evo2.return_value = mock_model
            mock_model.tokenizer = mock_tokenizer

            with patch(
                "dnallm.models.model.is_fp8_capable", return_value=False
            ):
                result = _handle_evo2_models("evo2_7b_base", "huggingface")

            assert result == (mock_model, mock_tokenizer)
            mock_evo2.assert_called_once_with("evo2_7b_base", use_fp8=False)

    def test_handle_evo2_models_import_error(self):
        """Test handling of EVO2 models when EVO2 package is not available."""
        with patch("evo2.Evo2", side_effect=ImportError("EVO2 not installed")):
            with pytest.raises(ImportError, match="EVO2 package is required"):
                _handle_evo2_models("evo2_1b_base", "huggingface")


class TestSetupHuggingfaceMirror:
    """Test _setup_huggingface_mirror function."""

    def test_setup_mirror_enabled(self):
        """Test setting up HuggingFace mirror when enabled."""
        with patch.dict(os.environ, {}, clear=True):
            _setup_huggingface_mirror(True)
            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    def test_setup_mirror_disabled(self):
        """Test setting up HuggingFace mirror when disabled."""
        with patch.dict(
            os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}, clear=True
        ):
            _setup_huggingface_mirror(False)
            assert "HF_ENDPOINT" not in os.environ

    def test_setup_mirror_disabled_no_existing_env(self):
        """Test handling evo2 models with local path and fp8 support."""
        with patch.dict(os.environ, {}, clear=True):
            _setup_huggingface_mirror(False)
            assert "HF_ENDPOINT" not in os.environ


class TestGetModelPathAndImports:
    """Test _get_model_path_and_imports function."""

    def test_get_model_path_local_exists(self):
        """Test getting model path for existing local model."""
        with patch("os.path.exists", return_value=True):
            model_path, modules = _get_model_path_and_imports(
                "/path/to/model", "local"
            )

            assert model_path == "/path/to/model"
            assert "AutoModel" in modules
            assert "AutoTokenizer" in modules

    def test_get_model_path_local_not_exists(self):
        """Test getting model path for non-existing local model."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(
                ValueError, match="Model /path/to/model not found locally"
            ):
                _get_model_path_and_imports("/path/to/model", "local")

    def test_get_model_path_huggingface(self):
        """Test getting model path for HuggingFace model."""
        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ) as mock_download:
            with patch(
                "huggingface_hub.snapshot_download"
            ) as mock_hf_download:
                model_path, modules = _get_model_path_and_imports(
                    "test-model", "huggingface"
                )

                assert model_path == "/downloaded/model"
                mock_download.assert_called_once_with(
                    "test-model", downloader=mock_hf_download
                )
                assert "AutoModel" in modules

    def test_get_model_path_modelscope(self):
        """Test getting model path for ModelScope model."""
        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ) as mock_download:
            with patch(
                "modelscope.hub.snapshot_download.snapshot_download"
            ) as mock_ms_download:
                with patch("modelscope.AutoModel") as mock_auto_model:
                    model_path, modules = _get_model_path_and_imports(
                        "test-model", "modelscope"
                    )

                    assert model_path == "/downloaded/model"
                    mock_download.assert_called_once_with(
                        "test-model", downloader=mock_ms_download
                    )
                    assert "AutoModel" in modules

    def test_get_model_path_unsupported_source(self):
        """Test getting model path for unsupported source."""
        with pytest.raises(ValueError, match="Unsupported source: unknown"):
            _get_model_path_and_imports("test-model", "unknown")

    def test_get_model_path_import_error_transformers(self):
        """Test import error for transformers library."""
        with patch(
            "transformers.AutoModel",
            side_effect=ImportError("transformers not installed"),
        ):
            with patch("os.path.exists", return_value=True):
                with pytest.raises(
                    ImportError,
                    match="Transformers is required but not available",
                ):
                    _get_model_path_and_imports("/path/to/model", "local")

    def test_get_model_path_import_error_modelscope(self):
        """Test import error for modelscope library."""
        with patch(
            "modelscope.AutoModel",
            side_effect=ImportError("modelscope not installed"),
        ):
            with patch(
                "dnallm.models.model.download_model",
                return_value="/downloaded/model",
            ):
                with patch(
                    "modelscope.hub.snapshot_download.snapshot_download"
                ):
                    with pytest.raises(
                        ImportError,
                        match="ModelScope is required but not available",
                    ):
                        _get_model_path_and_imports("test-model", "modelscope")


class TestCreateLabelMappings:
    """Test _create_label_mappings function."""

    def test_create_label_mappings_with_labels(self):
        """Test creating label mappings with provided labels."""
        task_config = TaskConfig(
            task_type="binary",
            num_labels=2,
            label_names=["negative", "positive"],
        )

        id2label, label2id = _create_label_mappings(task_config)

        expected_id2label = {0: "negative", 1: "positive"}
        expected_label2id = {"negative": 0, "positive": 1}

        assert id2label == expected_id2label
        assert label2id == expected_label2id

    def test_create_label_mappings_no_labels(self):
        """Test creating label mappings without labels."""
        task_config = TaskConfig(
            task_type="mask", num_labels=None, label_names=None
        )

        id2label, label2id = _create_label_mappings(task_config)

        assert id2label == {}
        assert label2id == {}


class TestLoadModelByTaskType:
    """Test _load_model_by_task_type function."""

    def test_load_model_mask_task(self):
        """Test loading model for mask task."""
        modules = {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()}
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForMaskedLM"
        ].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type(
            "mask", "test-model", 1, {}, {}, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoTokenizer"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True
        )
        modules[
            "AutoModelForMaskedLM"
        ].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation="eager"
        )

    def test_load_model_generation_task(self):
        """Test loading model for generation task."""
        modules = {"AutoTokenizer": Mock(), "AutoModelForCausalLM": Mock()}
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForCausalLM"
        ].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type(
            "generation", "test-model", 1, {}, {}, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer

    def test_load_model_binary_classification_task(self):
        """Test loading model for binary classification task."""
        modules = {
            "AutoTokenizer": Mock(),
            "AutoModelForSequenceClassification": Mock(),
        }
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.return_value = mock_model

        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}

        model, tokenizer = _load_model_by_task_type(
            "binary", "test-model", 2, id2label, label2id, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.assert_called_once_with(
            "test-model",
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
            problem_type="single_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )

    def test_load_model_multilabel_task(self):
        """Test loading model for multilabel classification task."""
        modules = {
            "AutoTokenizer": Mock(),
            "AutoModelForSequenceClassification": Mock(),
        }
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type(
            "multilabel", "test-model", 3, {}, {}, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.assert_called_once_with(
            "test-model",
            num_labels=3,
            problem_type="multi_label_classification",
            trust_remote_code=True,
            attn_implementation="eager",
        )

    def test_load_model_regression_task(self):
        """Test loading model for regression task."""
        modules = {
            "AutoTokenizer": Mock(),
            "AutoModelForSequenceClassification": Mock(),
        }
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type(
            "regression", "test-model", 1, {}, {}, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.assert_called_once_with(
            "test-model",
            num_labels=1,
            problem_type="regression",
            trust_remote_code=True,
            attn_implementation="eager",
        )

    def test_load_model_token_task(self):
        """Test loading model for token classification task."""
        modules = {
            "AutoTokenizer": Mock(),
            "AutoModelForTokenClassification": Mock(),
        }
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules[
            "AutoModelForTokenClassification"
        ].from_pretrained.return_value = mock_model

        id2label = {0: "O", 1: "B-GENE", 2: "I-GENE"}
        label2id = {"O": 0, "B-GENE": 1, "I-GENE": 2}

        model, tokenizer = _load_model_by_task_type(
            "token", "test-model", 3, id2label, label2id, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoTokenizer"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, add_prefix_space=True
        )
        modules[
            "AutoModelForTokenClassification"
        ].from_pretrained.assert_called_once_with(
            "test-model",
            num_labels=3,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
            attn_implementation="eager",
        )

    def test_load_model_default_task(self):
        """Test loading model for default task type."""
        modules = {"AutoTokenizer": Mock(), "AutoModel": Mock()}
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules["AutoModel"].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type(
            "embedding", "test-model", 1, {}, {}, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoModel"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation="eager"
        )


class TestConfigureModelPadding:
    """Test _configure_model_padding function."""

    def test_configure_padding_token_not_set(self):
        """Test configuring padding token when not set."""
        mock_model = Mock()
        mock_model.config.pad_token_id = None
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        _configure_model_padding(mock_model, mock_tokenizer)

        assert mock_model.config.pad_token_id == 0

    def test_configure_padding_token_already_set(self):
        """Test configuring padding token when already set."""
        mock_model = Mock()
        mock_model.config.pad_token_id = 1
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token_id = 0

        _configure_model_padding(mock_model, mock_tokenizer)

        assert mock_model.config.pad_token_id == 1  # Should remain unchanged


class TestLoadModelAndTokenizer:
    """Test load_model_and_tokenizer function."""

    def test_load_model_evo2(self):
        """Test loading EVO2 model."""
        task_config = TaskConfig(task_type="generation", num_labels=None)

        with patch(
            "dnallm.models.model._handle_evo2_models",
            return_value=("model", "tokenizer"),
        ):
            model, tokenizer = load_model_and_tokenizer(
                "evo2_1b_base", task_config
            )

            assert model == "model"
            assert tokenizer == "tokenizer"

    def test_load_model_regular_huggingface(self):
        """Test loading regular HuggingFace model."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.model._handle_evo2_models", return_value=None
        ):
            with patch("dnallm.models.model._setup_huggingface_mirror"):
                with patch(
                    "dnallm.models.model._get_model_path_and_imports",
                    return_value=("/path", {}),
                ):
                    with patch(
                        "dnallm.models.model._create_label_mappings",
                        return_value=({}, {}),
                    ):
                        with patch(
                            "dnallm.models.model._load_model_by_task_type",
                            return_value=("model", "tokenizer"),
                        ):
                            with patch(
                                "dnallm.models.model._configure_model_padding"
                            ):
                                model, tokenizer = load_model_and_tokenizer(
                                    "test-model",
                                    task_config,
                                    source="huggingface",
                                )

                                assert model == "model"
                                assert tokenizer == "tokenizer"

    def test_load_model_missing_num_labels_classification(self):
        """Test that correct problem types are set for different task types."""
        task_config = TaskConfig(task_type="binary", num_labels=None)

        with patch(
            "dnallm.models.model._handle_evo2_models", return_value=None
        ):
            with patch("dnallm.models.model._setup_huggingface_mirror"):
                with patch(
                    "dnallm.models.model._get_model_path_and_imports",
                    return_value=("/path", {}),
                ):
                    with pytest.raises(
                        ValueError,
                        match="num_labels is required for task type\
                            'binary' but is None",
                    ):
                        load_model_and_tokenizer("test-model", task_config)

    def test_load_model_loading_error(self):
        """Test loading model with loading error."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.model._handle_evo2_models", return_value=None
        ):
            with patch("dnallm.models.model._setup_huggingface_mirror"):
                with patch(
                    "dnallm.models.model._get_model_path_and_imports",
                    return_value=("/path", {}),
                ):
                    with patch(
                        "dnallm.models.model._create_label_mappings",
                        return_value=({}, {}),
                    ):
                        with patch(
                            "dnallm.models.model._load_model_by_task_type",
                            side_effect=Exception("Loading failed"),
                        ):
                            with pytest.raises(
                                ValueError,
                                match="Failed to load model: Loading failed",
                            ):
                                load_model_and_tokenizer(
                                    "test-model", task_config
                                )


class TestLoadPresetModel:
    """Test load_preset_model function."""

    def test_load_preset_model_success(self):
        """Test successful loading of preset model."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.model.MODEL_INFO",
            {"test-model": {"default": "actual-model"}},
        ):
            with patch(
                "dnallm.models.model.load_model_and_tokenizer",
                return_value=("model", "tokenizer"),
            ):
                result = load_preset_model("test-model", task_config)

                assert result == ("model", "tokenizer")

    def test_load_preset_model_not_found(self):
        """Test loading preset model that is not found."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model.MODEL_INFO", {}):
            with patch("builtins.print") as mock_print:
                result = load_preset_model("unknown-model", task_config)

                assert result == 0
                mock_print.assert_called_once()

    def test_load_preset_model_preset_name(self):
        """Test loading preset model by preset name."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.model.MODEL_INFO",
            {"test-model": {"preset": ["preset1", "preset2"]}},
        ):
            with patch(
                "dnallm.models.model.load_model_and_tokenizer",
                return_value=("model", "tokenizer"),
            ):
                result = load_preset_model("preset1", task_config)

                assert result == ("model", "tokenizer")

    def test_load_preset_model_key_error(self):
        """Test loading preset model with KeyError in MODEL_INFO."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model.MODEL_INFO", {}):
            with patch("builtins.print") as mock_print:
                result = load_preset_model("test-model", task_config)

                assert result == 0
                mock_print.assert_called_once()


@pytest.mark.parametrize(
    ("task_type", "expected_problem_type"),
    [
        ("binary", "single_label_classification"),
        ("multiclass", "single_label_classification"),
        ("multilabel", "multi_label_classification"),
        ("regression", "regression"),
    ],
)
def test_load_model_by_task_type_problem_types(
    task_type, expected_problem_type
):
    """Test that correct problem types are set for different task types."""
    modules = {
        "AutoTokenizer": Mock(),
        "AutoModelForSequenceClassification": Mock(),
    }
    mock_tokenizer = Mock()
    mock_model = Mock()
    modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
    modules[
        "AutoModelForSequenceClassification"
    ].from_pretrained.return_value = mock_model

    _load_model_by_task_type(task_type, "test-model", 2, {}, {}, modules)

    call_args = modules[
        "AutoModelForSequenceClassification"
    ].from_pretrained.call_args
    assert call_args[1]["problem_type"] == expected_problem_type


@pytest.mark.parametrize(
    ("task_type", "expected_attn_implementation"),
    [
        ("mask", "eager"),
        ("generation", "eager"),
        ("binary", "eager"),
        ("multiclass", "eager"),
        ("multilabel", "eager"),
        ("regression", "eager"),
        ("token", "eager"),
        ("embedding", "eager"),
    ],
)
def test_load_model_by_task_type_attention_implementation(
    task_type, expected_attn_implementation
):
    """Test that eager attention implementation is used for all task types."""
    modules = {
        "AutoTokenizer": Mock(),
        "AutoModelForMaskedLM": Mock(),
        "AutoModelForCausalLM": Mock(),
        "AutoModelForSequenceClassification": Mock(),
        "AutoModelForTokenClassification": Mock(),
        "AutoModel": Mock(),
    }

    for module in modules.values():
        module.from_pretrained.return_value = Mock()

    modules["AutoTokenizer"].from_pretrained.return_value = Mock()

    _load_model_by_task_type(task_type, "test-model", 2, {}, {}, modules)

    # Check that the appropriate model class was called with eager attention
    if task_type == "mask":
        call_args = modules["AutoModelForMaskedLM"].from_pretrained.call_args
    elif task_type == "generation":
        call_args = modules["AutoModelForCausalLM"].from_pretrained.call_args
    elif task_type in ["binary", "multiclass", "multilabel", "regression"]:
        call_args = modules[
            "AutoModelForSequenceClassification"
        ].from_pretrained.call_args
    elif task_type == "token":
        call_args = modules[
            "AutoModelForTokenClassification"
        ].from_pretrained.call_args
    else:
        call_args = modules["AutoModel"].from_pretrained.call_args

    assert call_args[1]["attn_implementation"] == expected_attn_implementation
