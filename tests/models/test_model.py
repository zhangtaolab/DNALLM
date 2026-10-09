"""Tests for DNA model loading and management utilities.

This module tests the functions for downloading, loading, and managing
DNA language models from various sources.

"""

import hashlib
import importlib
import json
import logging
import os
import sys
import types
from pathlib import Path
import pytest
import torch
import torch.nn as nn
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import ANY, Mock, patch, MagicMock
from typing import Any

from transformers import PretrainedConfig

from dnallm.models.model import (
    download_model,
    load_model_and_tokenizer,
    load_preset_model,
    clear_model_cache,
    peft_forward_compatiable,
    _setup_huggingface_mirror,
    _get_model_path_and_imports,
    _create_label_mappings,
    _load_model_by_task_type,
    _configure_model_padding,
    _fix_bnb_quantized_layers,
    _get_device,
    _safe_num_labels,
    _detect_special_family,
    _gate_random_init,
    _get_auto_modules_for_source,
    _tensor_digest,
    _log_random_init_fingerprint,
    _load_random_init_model,
    RANDOM_INIT_SUPPORTED_FAMILIES,
    DNALLMforSequenceClassification,
)
from dnallm.models.losses import FocalLoss
from dnallm.utils.support import is_fp8_capable
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

        with pytest.raises(ValueError, match="Model test-model download failed"):
            download_model("test-model", mock_downloader, max_try=1)

    def test_download_other_error(self):
        """Test download with other types of errors."""
        mock_downloader = Mock(side_effect=Exception("unknown error"))

        with patch("time.sleep"):  # Mock sleep to speed up test
            with pytest.raises(ValueError, match="Model test-model download failed"):
                download_model("test-model", mock_downloader, max_try=2)

    def test_download_404_classified_single_call_no_retry(self):
        """The ModelScope 404 branch must break without retrying or sleeping."""
        # NOTE: the message must contain "response [404]" but NOT "not found",
        # otherwise the earlier not-found branch wins the classification.
        mock_downloader = Mock(side_effect=Exception("HTTP Response [404] model absent"))

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, max_try=3)

        assert mock_downloader.call_count == 1
        assert mock_sleep.call_count == 0

    def test_download_not_found_breaks_after_single_call(self):
        """The HuggingFace not-found branch must stop after exactly one attempt."""
        mock_downloader = Mock(side_effect=Exception("Repository not found"))

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, max_try=5)

        assert mock_downloader.call_count == 1
        assert mock_sleep.call_count == 0

    def test_download_no_revision_error_resets_revision(self):
        """A no-revision error must retry with revision=None on the next call."""
        mock_downloader = Mock(side_effect=Exception("no revision found: deadbeef"))

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, revision="deadbeef", max_try=2)

        assert mock_downloader.call_count == 2
        assert mock_downloader.call_args_list[0].kwargs["revision"] == "deadbeef"
        assert mock_downloader.call_args_list[1].kwargs["revision"] is None
        assert mock_sleep.call_count == 2

    def test_download_incomplete_status_exhausts_without_sleep(self):
        """Perpetual 'incomplete' status exhausts max_try without sleeping."""
        mock_downloader = Mock(return_value="incomplete")

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, max_try=3)

        assert mock_downloader.call_count == 3
        # sleep() lives inside the except arm only; no exception -> no sleep
        assert mock_sleep.call_count == 0

    def test_download_connection_exhaustion_call_and_sleep_counts(self):
        """Connection-classified failures retry once per attempt and sleep per failure."""
        mock_downloader = Mock(side_effect=Exception("connection reset by peer"))

        with patch("time.sleep") as mock_sleep:
            with pytest.raises(ValueError, match=r"Model test-model download failed."):
                download_model("test-model", mock_downloader, max_try=3)

        assert mock_downloader.call_count == 3
        assert mock_sleep.call_count == 3

    def test_download_success_never_sleeps(self):
        """A first-attempt success must not sleep at all."""
        mock_downloader = Mock(return_value="/cache/model")

        with patch("time.sleep") as mock_sleep:
            result = download_model("test-model", mock_downloader, max_try=3)

        assert result == "/cache/model"
        assert mock_downloader.call_count == 1
        assert mock_sleep.call_count == 0

    def test_download_forwards_allow_patterns_kwarg(self):
        """allow_patterns set -> forwarded as a downloader kwarg on the success path."""
        mock_downloader = Mock(return_value="/path/to/model")

        result = download_model(
            "test-model", mock_downloader, max_try=3, allow_patterns=["*.safetensors"]
        )

        assert result == "/path/to/model"
        assert mock_downloader.call_args.kwargs["allow_patterns"] == ["*.safetensors"]

    def test_download_omits_allow_patterns_kwarg_when_none(self):
        """allow_patterns omitted -> the downloader kwargs carry no allow_patterns key.

        This is the byte-identical-when-omitted half of the CI-05 contract:
        no existing caller (and no family) may silently inherit a pattern set.
        """
        mock_downloader = Mock(return_value="/path/to/model")

        result = download_model("test-model", mock_downloader, max_try=3)

        assert result == "/path/to/model"
        assert "allow_patterns" not in mock_downloader.call_args.kwargs

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_download_real_huggingface_connection(self):
        """Test real HuggingFace connection (requires network)."""
        from huggingface_hub import snapshot_download

        # Try to download a small test model
        result = download_model("microsoft/DialoGPT-small", snapshot_download, max_try=1)
        assert result is not None
        assert os.path.exists(result)

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_download_real_modelscope_connection(self):
        """Test real ModelScope connection (requires network)."""
        from modelscope.hub.snapshot_download import snapshot_download

        # Try to download a small test model
        result = download_model(
            "ZhejiangLab-LifeScience/DNA_bert_4",
            snapshot_download,
            max_try=1,
        )
        assert result is not None
        assert os.path.exists(result)


class TestGetModelPathAndImportsAllowPatterns:
    """allow_patterns passthrough on the hub branch (CI-05 load-time half)."""

    def test_hub_branch_forwards_allow_patterns_to_download_model(self):
        """Hub branch threads allow_patterns into download_model when given."""
        with (
            patch("huggingface_hub.snapshot_download"),
            patch("dnallm.models.model.download_model", return_value="/hf/model") as mock_download,
        ):
            _get_model_path_and_imports(
                "test-model", "huggingface", allow_patterns=["*.safetensors"]
            )

        assert mock_download.call_args.kwargs["allow_patterns"] == ["*.safetensors"]

    def test_hub_branch_omits_allow_patterns_when_none(self):
        """Hub branch carries no allow_patterns key when the parameter is None."""
        with (
            patch("huggingface_hub.snapshot_download"),
            patch("dnallm.models.model.download_model", return_value="/hf/model") as mock_download,
        ):
            _get_model_path_and_imports("test-model", "huggingface")

        assert "allow_patterns" not in mock_download.call_args.kwargs


class TestIsFp8Capable:
    """Test is_fp8_capable function for hardware detection."""

    @patch("dnallm.utils.support.get_device_capability")
    def test_fp8_capable_hopper(self, mock_capability):
        """Test FP8 capability detection for Hopper (H100) architecture."""
        mock_capability.return_value = (9, 0)

        result = is_fp8_capable()

        assert result is True

    @patch("dnallm.utils.support.get_device_capability")
    def test_fp8_capable_newer_architecture(self, mock_capability):
        """Test FP8 capability detection for newer architecture."""
        mock_capability.return_value = (9, 1)

        result = is_fp8_capable()

        assert result is True

    @patch("dnallm.utils.support.get_device_capability")
    def test_fp8_not_capable_older_architecture(self, mock_capability):
        """Test FP8 capability detection for older architecture."""
        mock_capability.return_value = (8, 0)

        result = is_fp8_capable()

        assert result is False

    @patch("dnallm.utils.support.get_device_capability")
    def test_fp8_not_capable_much_older_architecture(self, mock_capability):
        """Test FP8 capability detection for much older architecture."""
        mock_capability.return_value = (7, 5)

        result = is_fp8_capable()

        assert result is False


class TestSetupHuggingfaceMirror:
    """Test _setup_huggingface_mirror function."""

    def test_setup_mirror_enabled(self):
        """Test setting up HuggingFace mirror when enabled."""
        with patch.dict(os.environ, {}, clear=True):
            _setup_huggingface_mirror(True)
            assert os.environ["HF_ENDPOINT"] == "https://hf-mirror.com"

    def test_setup_mirror_disabled(self):
        """Test setting up HuggingFace mirror when disabled."""
        with patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.com"}, clear=True):
            _setup_huggingface_mirror(False)
            assert "HF_ENDPOINT" not in os.environ

    def test_setup_mirror_disabled_no_existing_env(self):
        """Test setting up HuggingFace mirror when
        disabled and no existing env.
        """
        with patch.dict(os.environ, {}, clear=True):
            _setup_huggingface_mirror(False)
            assert "HF_ENDPOINT" not in os.environ


class TestGetModelPathAndImports:
    """Test _get_model_path_and_imports function."""

    def test_get_model_path_local_exists(self):
        """Test getting model path for existing local model."""
        with patch("os.path.exists", return_value=True):
            model_path, modules = _get_model_path_and_imports("/path/to/model", "local")

            assert model_path == "/path/to/model"
            assert "AutoModel" in modules
            assert "AutoTokenizer" in modules

    def test_get_model_path_local_not_exists(self):
        """Test getting model path for non-existing local model."""
        with patch("os.path.exists", return_value=False):
            with pytest.raises(ValueError, match="Model /path/to/model not found locally"):
                _get_model_path_and_imports("/path/to/model", "local")

    def test_get_model_path_huggingface(self):
        """Test getting model path for HuggingFace model."""
        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ) as mock_download:
            with patch("huggingface_hub.snapshot_download") as mock_hf_download:
                model_path, modules = _get_model_path_and_imports("test-model", "huggingface")

                assert model_path == "/downloaded/model"
                mock_download.assert_called_once_with(
                    "test-model", downloader=mock_hf_download, revision=None
                )
                assert "AutoModel" in modules

    def test_get_model_path_modelscope(self):
        """Test getting model path for ModelScope model."""
        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ) as mock_download:
            snapshot_module = importlib.import_module("modelscope.hub.snapshot_download")
            with patch.object(snapshot_module, "snapshot_download") as mock_ms_download:
                with patch("modelscope.AutoModel") as mock_auto_model:
                    model_path, modules = _get_model_path_and_imports("test-model", "modelscope")

                    assert model_path == "/downloaded/model"
                    mock_download.assert_called_once_with(
                        "test-model",
                        downloader=mock_ms_download,
                        revision=None,
                    )
                    assert "AutoModel" in modules

    def test_get_model_path_unsupported_source(self):
        """Test getting model path for unsupported source."""
        with pytest.raises(ValueError, match="Unsupported source: unknown"):
            _get_model_path_and_imports("test-model", "unknown")

    def test_get_model_path_local_real_directory(self, tmp_path):
        """Test local source resolution against a real directory."""
        model_path, modules = _get_model_path_and_imports(str(tmp_path), "local")

        assert model_path == str(tmp_path)
        assert "AutoConfig" in modules
        assert "AutoModelForSequenceClassification" in modules

    def test_get_model_path_huggingface_forwards_revision(self):
        """Test that the revision is forwarded to the HuggingFace downloader."""
        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ) as mock_download:
            with patch("huggingface_hub.snapshot_download"):
                _get_model_path_and_imports("test-model", "huggingface", revision="main")

                mock_download.assert_called_once_with(
                    "test-model",
                    downloader=ANY,
                    revision="main",
                )

    def test_get_model_path_transformers_missing_raises_importerror(self, monkeypatch):
        """A missing transformers install must surface the wrapped ImportError."""
        monkeypatch.setitem(sys.modules, "transformers", None)

        with patch("os.path.exists", return_value=True):
            with pytest.raises(ImportError, match="Transformers is required"):
                _get_model_path_and_imports("/path/to/model", "local")

    def test_get_model_path_modelscope_extra_imports_missing(self, monkeypatch):
        """ModelScope without the Auto* classes must raise the wrapped ImportError."""
        fake_modelscope = types.ModuleType("modelscope")
        fake_hub = types.ModuleType("modelscope.hub")
        fake_snapshot = types.ModuleType("modelscope.hub.snapshot_download")
        fake_snapshot.snapshot_download = Mock()
        monkeypatch.setitem(sys.modules, "modelscope", fake_modelscope)
        monkeypatch.setitem(sys.modules, "modelscope.hub", fake_hub)
        monkeypatch.setitem(sys.modules, "modelscope.hub.snapshot_download", fake_snapshot)

        with patch(
            "dnallm.models.model.download_model",
            return_value="/downloaded/model",
        ):
            with pytest.raises(ImportError, match="ModelScope is required"):
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
        task_config = TaskConfig(task_type="mask", num_labels=None, label_names=None)

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
        modules["AutoModelForMaskedLM"].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type("mask", "test-model", 1, {}, {}, modules)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoTokenizer"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True
        )
        modules["AutoModelForMaskedLM"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation="eager"
        )

    def test_load_model_generation_task(self):
        """Test loading model for generation task."""
        modules = {"AutoTokenizer": Mock(), "AutoModelForCausalLM": Mock()}
        mock_tokenizer = Mock()
        mock_model = Mock()
        modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
        modules["AutoModelForCausalLM"].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type("generation", "test-model", 1, {}, {}, modules)

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
        modules["AutoModelForSequenceClassification"].from_pretrained.return_value = mock_model

        id2label = {0: "negative", 1: "positive"}
        label2id = {"negative": 0, "positive": 1}

        model, tokenizer = _load_model_by_task_type(
            "binary", "test-model", 2, id2label, label2id, modules
        )

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoModelForSequenceClassification"].from_pretrained.assert_called_once_with(
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
        modules["AutoModelForSequenceClassification"].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type("multilabel", "test-model", 3, {}, {}, modules)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoModelForSequenceClassification"].from_pretrained.assert_called_once_with(
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
        modules["AutoModelForSequenceClassification"].from_pretrained.return_value = mock_model

        model, tokenizer = _load_model_by_task_type("regression", "test-model", 1, {}, {}, modules)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoModelForSequenceClassification"].from_pretrained.assert_called_once_with(
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
        modules["AutoModelForTokenClassification"].from_pretrained.return_value = mock_model

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
        modules["AutoModelForTokenClassification"].from_pretrained.assert_called_once_with(
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

        model, tokenizer = _load_model_by_task_type("embedding", "test-model", 1, {}, {}, modules)

        assert model == mock_model
        assert tokenizer == mock_tokenizer
        modules["AutoModel"].from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation="eager"
        )

    def test_load_model_custom_tokenizer_factory(self):
        """Test that a custom tokenizer callable replaces the fallback chain."""
        modules = {
            "AutoTokenizer": Mock(),
            "AutoModel": Mock(**{"from_pretrained.return_value": Mock()}),
        }
        custom_tokenizer = Mock(return_value="custom-tokenizer")

        _model, tokenizer = _load_model_by_task_type(
            "embedding",
            "test-model",
            1,
            {},
            {},
            modules,
            custom_tokenizer=custom_tokenizer,
        )

        assert tokenizer == "custom-tokenizer"
        custom_tokenizer.assert_called_once_with()

    def test_load_model_head_config_builds_wrapper(self):
        """A head_config routes through DNALLMforSequenceClassification.from_base_model."""
        mock_tokenizer = Mock()
        modules = {
            "AutoTokenizer": Mock(**{"from_pretrained.return_value": mock_tokenizer}),
            "AutoConfig": Mock(),
            "AutoModel": Mock(),
        }
        head_config = SimpleNamespace(head="basic-mlp", num_classes=2)

        with patch("dnallm.models.model.DNALLMforSequenceClassification") as mock_wrapper:
            mock_wrapper.from_base_model.return_value = "wrapped-model"
            model, tokenizer = _load_model_by_task_type(
                "binary",
                "test-model",
                2,
                {},
                {},
                modules,
                head_config=head_config,
            )

        assert model == "wrapped-model"
        assert tokenizer is mock_tokenizer
        base_config = modules["AutoConfig"].from_pretrained.return_value
        assert base_config.head_config == head_config.__dict__
        mock_wrapper.from_base_model.assert_called_once_with(
            "test-model",
            config=base_config,
            module=modules["AutoModel"],
            quantization_config=None,
        )

    def test_load_model_default_task_retries_with_ignore_mismatched_sizes(self):
        """Unknown task types retry AutoModel loading with ignore_mismatched_sizes."""
        modules = {"AutoTokenizer": Mock(**{"from_pretrained.return_value": Mock()})}
        mock_model = Mock()
        auto_model = Mock(**{"from_pretrained.side_effect": [Exception("boom"), mock_model]})
        modules["AutoModel"] = auto_model

        model, _tokenizer = _load_model_by_task_type("embedding", "test-model", 1, {}, {}, modules)

        assert model == mock_model
        assert auto_model.from_pretrained.call_count == 2
        retry_kwargs = auto_model.from_pretrained.call_args_list[1].kwargs
        assert retry_kwargs["ignore_mismatched_sizes"] is True

    def test_load_model_bnb_config_adds_quantization_kwargs(self):
        """A bitsandbytes config is forwarded to the from_pretrained call."""
        modules = {
            "AutoTokenizer": Mock(**{"from_pretrained.return_value": Mock()}),
            "AutoModelForMaskedLM": Mock(),
        }
        bnb_config = Mock()

        _load_model_by_task_type("mask", "test-model", 1, {}, {}, modules, bnb_config=bnb_config)

        call_kwargs = modules["AutoModelForMaskedLM"].from_pretrained.call_args.kwargs
        assert call_kwargs["quantization_config"] is bnb_config
        assert call_kwargs["device_map"] == "auto"


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

    def test_configure_padding_falls_back_to_pad_token_type_id(self):
        """Without pad_token_id the pad_token_type_id is used."""
        mock_model = Mock()
        mock_model.config.pad_token_id = None
        mock_tokenizer = Mock(spec=["pad_token_id", "pad_token_type_id"])
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.pad_token_type_id = 5

        _configure_model_padding(mock_model, mock_tokenizer)

        assert mock_model.config.pad_token_id == 5

    def test_configure_padding_falls_back_to_pad_token_type_id_only(self):
        """A tokenizer exposing only pad_token_type_id still configures padding."""
        mock_model = Mock()
        mock_model.config.pad_token_id = None
        mock_tokenizer = Mock(spec=["pad_token_type_id"])
        mock_tokenizer.pad_token_type_id = 3

        _configure_model_padding(mock_model, mock_tokenizer)

        assert mock_model.config.pad_token_id == 3

    def test_configure_padding_falls_back_to_eos_token_id(self):
        """With no padding attributes at all the EOS token id is used."""
        mock_model = Mock()
        mock_model.config.pad_token_id = None
        mock_tokenizer = Mock(spec=["eos_token_id"])
        mock_tokenizer.eos_token_id = 7

        _configure_model_padding(mock_model, mock_tokenizer)

        assert mock_model.config.pad_token_id == 7


class TestLoadModelAndTokenizer:
    """Test load_model_and_tokenizer function."""

    def test_load_model_regular_huggingface(self):
        """Test loading regular HuggingFace model."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=(
                    "/path",
                    {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()},
                ),
            ),
            patch(
                "dnallm.models.model._create_label_mappings",
                return_value=({}, {}),
            ),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(Mock(), "tokenizer"),
            ),
            patch("dnallm.models.model._configure_model_padding"),
        ):
            model, tokenizer = load_model_and_tokenizer(
                "test-model", task_config, source="huggingface"
            )

            assert model is not None
            assert tokenizer == "tokenizer"

    def test_load_model_crossdna_result_not_overwritten(self):
        """CrossDNA handler result must survive the dispatch chain verbatim."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        sentinel_model = Mock()
        # load_model_and_tokenizer rebinds the model via .to(device); a plain
        # Mock would return a fresh child mock and break the identity check.
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        other_model, other_tokenizer = Mock(), Mock()

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch("dnallm.models.model._handle_evo2_models", return_value=None),
            patch("dnallm.models.model._handle_evo1_models", return_value=None),
            patch("dnallm.models.model._handle_gpn_models", return_value=None),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=(
                    "/models/CrossDNA-8.1M",
                    {"AutoTokenizer": Mock(), "AutoModelForMaskedLM": Mock()},
                ),
            ),
            patch(
                "dnallm.models.model._create_label_mappings",
                return_value=({}, {}),
            ),
            patch(
                "dnallm.models.model._handle_crossdna_models",
                return_value=(sentinel_model, sentinel_tokenizer),
            ),
            patch(
                "dnallm.models.model._handle_dnabert2_models",
                return_value=(other_model, other_tokenizer),
            ),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                side_effect=AssertionError("generic loader must not run"),
            ),
            patch("dnallm.models.model._configure_model_padding"),
        ):
            model, tokenizer = load_model_and_tokenizer(
                "CrossDNA-8.1M", task_config, source="local"
            )

            assert model is sentinel_model
            assert tokenizer is sentinel_tokenizer

    def test_load_model_missing_num_labels_classification(self):
        """Test that correct problem types are set for different task types."""
        # Create a task config that bypasses Pydantic validation
        task_config = TaskConfig(task_type="regression", num_labels=1)
        # Manually set num_labels to None to test the validation
        task_config.num_labels = None

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch("dnallm.models.model._handle_evo2_models", return_value=None),
            patch("dnallm.models.model._handle_evo1_models", return_value=None),
            patch("dnallm.models.model._handle_gpn_models", return_value=None),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=(
                    "/path",
                    {
                        "AutoTokenizer": Mock(),
                        "AutoModelForSequenceClassification": Mock(),
                    },
                ),
            ),
            patch(
                "dnallm.models.model._create_label_mappings",
                return_value=({}, {}),
            ),
        ):
            with pytest.raises(
                ValueError,
                match=("num_labels is required for task type 'regression' but is None"),
            ):
                load_model_and_tokenizer("test-model", task_config)

    def test_load_model_loading_error(self):
        """Test loading model with loading error."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/path", {}),
            ),
            patch(
                "dnallm.models.model._create_label_mappings",
                return_value=({}, {}),
            ),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                side_effect=Exception("Loading failed"),
            ),
        ):
            with pytest.raises(
                ValueError,
                match="Failed to load model: Loading failed",
            ):
                load_model_and_tokenizer("test-model", task_config)


# ───────────────────────── random_init (BASE-01) ─────────────────────────


class _TinyScratchNN(nn.Module):
    """Real tiny torch module that consumes the CPU RNG during construction.

    Carries an int buffer and a bool buffer so the per-tensor hash table's
    buffer rows and the non-float exception rules are exercisable; ``tied``
    adds an ``lm_head`` weight aliased onto the embedding (shared storage).
    """

    def __init__(self, tied: bool = False):
        super().__init__()
        self.embedding = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 4)
        self.register_buffer("position_ids", torch.arange(4, dtype=torch.long))
        self.register_buffer("is_causal", torch.tensor(True))
        if tied:
            self.lm_head = nn.Linear(8, 16, bias=False)
            self.lm_head.weight = self.embedding.weight
        self.config = SimpleNamespace(pad_token_id=None)

    def forward(self, input_ids=None, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids).mean(dim=1)))


class _FakeAutoConfig:
    """Fake AutoConfig capturing from_pretrained calls."""

    def __init__(self, config=None):
        self.config = config if config is not None else PretrainedConfig(
            model_type="bert",
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=16,
        )
        self.calls = []

    def from_pretrained(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return self.config


class _FakeAutoClass:
    """Fake Auto* class whose from_config builds a fresh tiny module."""

    def __init__(self, name, build_fn=None, order=None):
        self.name = name
        self.build_fn = build_fn or (lambda: _TinyScratchNN())
        self.order = order
        self.calls = []

    def from_config(self, config, **kwargs):
        self.calls.append((config, kwargs))
        if self.order is not None:
            self.order.append(f"build:{self.name}")
        return self.build_fn()


def _fake_random_modules(order=None, build_fn=None, config=None):
    """Build the modules dict consumed by the random path, all fakes."""
    modules = {"AutoConfig": _FakeAutoConfig(config)}
    for name in (
        "AutoModel",
        "AutoModelForMaskedLM",
        "AutoModelForCausalLM",
        "AutoModelForSequenceClassification",
        "AutoModelForTokenClassification",
    ):
        modules[name] = _FakeAutoClass(name, build_fn=build_fn, order=order)
    fake_tokenizer_cls = Mock()
    fake_tokenizer_cls.from_pretrained = Mock(return_value=Mock(pad_token_id=0))
    modules["AutoTokenizer"] = fake_tokenizer_cls
    return modules


def _random_init_patches(modules, weight_fetch_guard=True):
    """ExitStack of patches for a mocked-boundary random_init load."""
    stack = ExitStack()
    stack.enter_context(patch("dnallm.models.model._setup_huggingface_mirror"))
    stack.enter_context(
        patch("dnallm.models.model._get_auto_modules_for_source", return_value=modules)
    )
    if weight_fetch_guard:
        stack.enter_context(
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                side_effect=AssertionError(
                    "weight-fetch path must never run under random_init"
                ),
            )
        )
    stack.enter_context(patch("dnallm.models.model._configure_model_padding"))
    stack.enter_context(
        patch("dnallm.models.model._get_device", return_value=torch.device("cpu"))
    )
    return stack


def _param_hashes_from_caplog(caplog):
    """Parse the per-tensor param hash lines out of captured random-init logs."""
    table = {}
    for record in caplog.records:
        msg = record.getMessage()
        if msg.startswith("[random-init] param "):
            rest = msg[len("[random-init] param ") :]
            name, sha_field = rest.split(" sha=")
            table[name] = sha_field.split(" ")[0]
    return table


def _buffer_rows_from_caplog(caplog):
    """Parse the buffer hash rows out of captured random-init logs."""
    rows = []
    for record in caplog.records:
        msg = record.getMessage()
        if msg.startswith("[random-init] buffer "):
            rows.append(msg)
    return rows


class TestRandomInit:
    """random_init=True from-scratch loading: tracer + allowlist + mocked proofs (BASE-01)."""

    def test_random_init_end_to_end_mocked_boundary(self, caplog):
        """Tracer: kwarg -> gate -> from_config random model, banner + per-tensor hashes logged, tokenizer returned."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        modules = _fake_random_modules()

        with _random_init_patches(modules), caplog.at_level(logging.INFO):
            model, tokenizer = load_model_and_tokenizer(
                "test-random-model", task_config, source="huggingface", random_init=True
            )

        assert isinstance(model, _TinyScratchNN)
        assert tokenizer is not None
        # Loud banner (greppable "randomly initialized") + >= 1 per-tensor hash line.
        assert "randomly initialized" in caplog.text
        param_table = _param_hashes_from_caplog(caplog)
        assert len(param_table) >= 1
        assert set(param_table) == {
            "embedding.weight",
            "head.weight",
            "head.bias",
        }

    @pytest.mark.parametrize(
        ("model_name", "family"),
        [
            ("evo2_7b", "evo2"),
            ("evo-1-142m", "evo1"),
            ("gpn-brassicales", "gpn"),
            ("megaDNA_phage_145M", "megadna"),
            ("Omni-DNA-20M", "omnidna"),
            ("enformer-191k", "enformer"),
            ("SPACE-v2", "space"),
            ("borzoi-replicate-0", "borzoi"),
            ("CrossDNA-8.1M", "crossdna"),
            ("dnabert-2-117m", "dnabert2"),
            ("DNABERT-S", "dnabert2"),
        ],
    )
    def test_random_init_offlist_special_family_raises(self, model_name, family):
        """Off-list special families raise a matchable ValueError before any handler claims the load (D-06/D-07)."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model._setup_huggingface_mirror"):
            with pytest.raises(ValueError, match="random_init=True is not supported"):
                load_model_and_tokenizer(model_name, task_config, random_init=True)

        # The message names the family and the allowlist so it is actionable.
        try:
            _gate_random_init(model_name, None, None)
        except ValueError as e:
            assert family in str(e)
            assert "RANDOM_INIT_SUPPORTED_FAMILIES" in str(e)
        else:
            raise AssertionError("expected ValueError from _gate_random_init")

    def test_random_init_quantization_config_rejected(self):
        """random_init x quantization_config has no from_config equivalent: matchable ValueError."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model._setup_huggingface_mirror"):
            with pytest.raises(
                ValueError,
                match="random_init=True cannot be combined with quantization_config",
            ):
                load_model_and_tokenizer(
                    "test-model",
                    task_config,
                    quantization_config={"load_in_4bit": True},
                    random_init=True,
                )

    def test_random_init_head_config_rejected(self):
        """random_init x custom head_config (DNALLMforSequenceClassification heads): matchable ValueError."""
        task_config = TaskConfig(
            task_type="binary", num_labels=2, head_config={"head": "mlp"}
        )

        with patch("dnallm.models.model._setup_huggingface_mirror"):
            with pytest.raises(
                ValueError,
                match="random_init=True is not supported with a custom head_config",
            ):
                load_model_and_tokenizer("test-model", task_config, random_init=True)

    def test_random_init_generic_path_from_config_never_pretrained(self):
        """AutoConfig.from_pretrained + the task-type Auto*.from_config are the only construction calls."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        modules = _fake_random_modules()

        with _random_init_patches(modules):
            load_model_and_tokenizer(
                "test-random-model", task_config, source="huggingface", random_init=True
            )

        # Config fetched via AutoConfig.from_pretrained with trust_remote_code;
        # revision=None is not forwarded (hub default).
        assert len(modules["AutoConfig"].calls) == 1
        name, config_kwargs = modules["AutoConfig"].calls[0]
        assert name == "test-random-model"
        assert config_kwargs["trust_remote_code"] is True
        assert "revision" not in config_kwargs

        # The mask task selected AutoModelForMaskedLM, via from_config only.
        assert len(modules["AutoModelForMaskedLM"].calls) == 1
        _, model_kwargs = modules["AutoModelForMaskedLM"].calls[0]
        assert model_kwargs["trust_remote_code"] is True
        for other in (
            "AutoModel",
            "AutoModelForCausalLM",
            "AutoModelForSequenceClassification",
            "AutoModelForTokenClassification",
        ):
            assert modules[other].calls == []

    def test_random_init_classification_head_shaping_on_config(self):
        """num_labels/id2label/label2id/problem_type are set as config attributes (A4)."""
        task_config = TaskConfig(
            task_type="binary", num_labels=3, label_names=["alpha", "beta", "gamma"]
        )
        config = PretrainedConfig(
            model_type="bert",
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            vocab_size=16,
        )
        modules = _fake_random_modules(config=config)

        with _random_init_patches(modules):
            load_model_and_tokenizer(
                "test-random-model", task_config, source="huggingface", random_init=True
            )

        assert config.num_labels == 3
        assert config.id2label == {0: "alpha", 1: "beta", 2: "gamma"}
        assert config.label2id == {"alpha": 0, "beta": 1, "gamma": 2}
        assert config.problem_type == "single_label_classification"
        assert len(modules["AutoModelForSequenceClassification"].calls) == 1

    @pytest.mark.parametrize(
        ("task_type", "expected_key"),
        [
            ("mask", "AutoModelForMaskedLM"),
            ("generation", "AutoModelForCausalLM"),
            ("binary", "AutoModelForSequenceClassification"),
            ("multiclass", "AutoModelForSequenceClassification"),
            ("multilabel", "AutoModelForSequenceClassification"),
            ("regression", "AutoModelForSequenceClassification"),
            ("token", "AutoModelForTokenClassification"),
            ("embedding", "AutoModel"),
        ],
    )
    def test_random_init_task_type_auto_class_selection(self, task_type, expected_key):
        """The random path selects the same Auto* class the pretrained task-type loader selects."""
        # Classification task types require num_labels; the others force it to 0.
        num_labels = None if task_type in ("mask", "generation", "embedding") else 3
        task_config = TaskConfig(task_type=task_type, num_labels=num_labels)
        modules = _fake_random_modules()

        with _random_init_patches(modules):
            load_model_and_tokenizer(
                "test-random-model", task_config, source="huggingface", random_init=True
            )

        assert len(modules[expected_key].calls) == 1


    @pytest.mark.parametrize(
        ("source", "expected_type"),
        [
            ("local", "huggingface-route"),
            ("huggingface", "huggingface-route"),
            ("modelscope", "modelscope-route"),
        ],
    )
    def test_get_auto_modules_for_source_bundle(self, source, expected_type):
        """The download-free module bundle carries the full Auto* key set per source."""
        modules = _get_auto_modules_for_source(source)
        assert set(modules) == {
            "AutoConfig",
            "AutoModel",
            "AutoModelForMaskedLM",
            "AutoModelForCausalLM",
            "AutoModelForSequenceClassification",
            "AutoModelForTokenClassification",
            "AutoTokenizer",
        }
        if expected_type == "modelscope-route":
            from transformers import AutoConfig as HFAutoConfig

            # modelscope's Auto* classes are dynamic wrappers (their
            # type's module reports 'builtins'), so assert they differ
            # from transformers' and name modelscope in their repr.
            assert modules["AutoConfig"] is not HFAutoConfig
            assert "modelscope" in repr(modules["AutoConfig"])

    def test_get_auto_modules_for_source_unsupported(self):
        """Unsupported sources raise a matchable ValueError."""
        with pytest.raises(ValueError, match="Unsupported source: bogus"):
            _get_auto_modules_for_source("bogus")

    def test_random_init_never_fetches_weights(self):
        """No-download proof: the weight-fetch seam is never invoked under random_init.

        The proof targets the weight-download path only. The config.json
        fetch through AutoConfig.from_pretrained (and the tokenizer files)
        is explicitly allowed and documented — config.json carries no weight
        values, and this exclusion is what separates "no weight download"
        from the unprovable "zero network at all".
        """
        task_config = TaskConfig(task_type="mask", num_labels=None)
        guard = MagicMock(
            side_effect=AssertionError(
                "_get_model_path_and_imports invoked under random_init"
            )
        )
        modules = _fake_random_modules()

        with (
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch(
                "dnallm.models.model._get_auto_modules_for_source",
                return_value=modules,
            ),
            patch("dnallm.models.model._get_model_path_and_imports", guard),
            patch("dnallm.models.model._configure_model_padding"),
            patch(
                "dnallm.models.model._get_device",
                return_value=torch.device("cpu"),
            ),
        ):
            model, tokenizer = load_model_and_tokenizer(
                "test-random-model", task_config, random_init=True
            )

        guard.assert_not_called()
        assert isinstance(model, _TinyScratchNN)
        assert tokenizer is not None

    def test_random_init_same_seed_reproducible(self, caplog):
        """Same-seed reproducibility: identical seeds reproduce the full per-tensor hash table; a different seed does not."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        tables = []
        for seed in (123, 123, 124):
            modules = _fake_random_modules()
            caplog.clear()
            with _random_init_patches(modules), caplog.at_level(logging.INFO):
                load_model_and_tokenizer(
                    "test-random-model",
                    task_config,
                    source="huggingface",
                    random_init=True,
                    random_init_seed=seed,
                )
            tables.append(_param_hashes_from_caplog(caplog))

        assert len(tables[0]) >= 2
        assert tables[0] == tables[1]
        assert tables[0] != tables[2]

    def test_random_init_seed_applied_before_construction(self):
        """Seed-before-init ordering: torch.manual_seed is called before the model-construction call.

        A seed applied after construction would silently break
        reproducibility — this pins the ordering edge.
        """
        order = []
        modules = _fake_random_modules(order=order)
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with _random_init_patches(modules), patch(
            "torch.manual_seed", side_effect=lambda s: order.append(("seed", s))
        ):
            load_model_and_tokenizer(
                "test-random-model",
                task_config,
                source="huggingface",
                random_init=True,
                random_init_seed=7,
            )

        assert order == [("seed", 7), "build:AutoModelForMaskedLM"]

    def test_tensor_digest_byte_level_semantics(self):
        """The digest is sha256[:10] over raw CPU tensor bytes (documented definition)."""
        a = torch.randn(3, 4)
        copy_of_a = a.clone()
        different = torch.randn(3, 4)

        expected = (
            hashlib.sha256(
                a.detach().cpu().contiguous().numpy().tobytes()
            ).hexdigest()[:10]
        )
        assert _tensor_digest(a) == expected
        # Equal values in separate storage digest identically (byte-level,
        # not storage-identity): deterministic and device-independent.
        assert _tensor_digest(a) == _tensor_digest(copy_of_a)
        assert _tensor_digest(a) != _tensor_digest(different)
        # A contiguous view sharing storage digests identically (same bytes
        # through the detach -> cpu -> contiguous path). A transposed copy
        # is a different logical tensor and correctly digests differently.
        view = a.view(12)
        assert _tensor_digest(view) == _tensor_digest(a)

    def test_tensor_digest_bfloat16_upcast_and_integer_tensors(self):
        """bfloat16 (no numpy dtype) is upcast to float32; int/bool tensors digest directly."""
        bf = torch.randn(2, 3).to(torch.bfloat16)
        expected = hashlib.sha256(
            bf.detach().cpu().contiguous().float().numpy().tobytes()
        ).hexdigest()[:10]
        assert _tensor_digest(bf) == expected

        as_int = torch.arange(6, dtype=torch.long)
        assert _tensor_digest(as_int) == hashlib.sha256(
            as_int.numpy().tobytes()
        ).hexdigest()[:10]
        assert len(_tensor_digest(torch.tensor(True))) == 10

    def test_log_random_init_fingerprint_tied_aliases_and_buffers(self, caplog):
        """Tied weights report identical hashes across the tie (shared storage); int/bool buffers get their own rows.

        Expected behavior documented per RESEARCH A-class exception handling:
        tied-weight aliases sharing storage and non-float buffers are the
        explicit exceptions to "every float parameter differs from
        pretrained"; both appear in the logged table with their own hashes.
        """
        model = _TinyScratchNN(tied=True)

        with caplog.at_level(logging.INFO):
            table = _log_random_init_fingerprint(
                model, "tiny-tied", "AutoModelForMaskedLM", 11
            )

        assert table["embedding.weight"] == table["lm_head.weight"]
        buffer_rows = _buffer_rows_from_caplog(caplog)
        assert len(buffer_rows) == 2
        assert "position_ids" in table
        assert "is_causal" in table
        assert "shape=" in buffer_rows[0]

    def test_log_random_init_fingerprint_skips_buffer_shadowing_param(self, caplog):
        """A buffer row whose name collides with a parameter is skipped (params win the table)."""
        fake_model = SimpleNamespace(
            named_parameters=lambda remove_duplicate=True: [
                ("x.weight", torch.ones(2))
            ],
            named_buffers=lambda: [("x.weight", torch.ones(2))],
        )

        with caplog.at_level(logging.INFO):
            table = _log_random_init_fingerprint(fake_model, "fake", "X", 1)

        assert set(table) == {"x.weight"}
        assert _buffer_rows_from_caplog(caplog) == []

    def test_random_init_config_without_init_semantics(self, tmp_path, caplog):
        """A minimal tmp config.json with no weight-init semantics still random-initializes (BASE-01 empty-config edge).

        Runs the REAL transformers path (AutoConfig.from_pretrained on the
        local dir + AutoModelForMaskedLM.from_config) — network-free because
        the config is local and tiny. Two differently-seeded loads must
        produce different hash tables; same seed reproduces.
        """
        minimal_config = {
            "model_type": "bert",
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "intermediate_size": 16,
            "vocab_size": 32,
        }
        model_dir = tmp_path / "tiny-bert"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps(minimal_config))

        task_config = TaskConfig(task_type="mask", num_labels=None)
        tables = []
        for seed in (5, 5, 6):
            caplog.clear()
            with (
                patch("dnallm.models.model._setup_huggingface_mirror"),
                patch(
                    "dnallm.models.model._get_device",
                    return_value=torch.device("cpu"),
                ),
                caplog.at_level(logging.INFO),
            ):
                model, tokenizer = load_model_and_tokenizer(
                    str(model_dir),
                    task_config,
                    source="local",
                    random_init=True,
                    random_init_seed=seed,
                )
            tables.append(_param_hashes_from_caplog(caplog))

        # A real tiny BertForMaskedLM: several float parameter tensors
        # (word embeddings, position/token type embeddings, encoder, tied
        # LM head alias) — all hashed per tensor.
        assert len(tables[0]) >= 5
        assert tables[0] == tables[1]
        assert tables[0] != tables[2]
        # Tied-weight alias visible with identical hash in a real model.
        assert (
            tables[0]["bert.embeddings.word_embeddings.weight"]
            == tables[0]["cls.predictions.decoder.weight"]
        )
        # The real local-route tokenizer fallback (DNAOneHotTokenizer).
        assert tokenizer is not None

    @pytest.mark.parametrize(
        ("model_name", "handler_attr"),
        [
            ("plant-mutbert-tiny", "tokenizer"),
            ("basenji2-tiny", None),
        ],
    )
    def test_random_init_preserves_tokenizer_post_processing(
        self, model_name, handler_attr
    ):
        """Tokenizer post-processing parity: mutbert/basenji2 handling still applies on the random path."""
        task_config = TaskConfig(task_type="mask", num_labels=None)
        modules = _fake_random_modules()
        # OneHotTokenizerWrapper calls len(tokenizer) — use a MagicMock.
        modules["AutoTokenizer"] = MagicMock()
        modules["AutoTokenizer"].from_pretrained.return_value = MagicMock(
            pad_token_id=0
        )

        with _random_init_patches(modules):
            _, tokenizer = load_model_and_tokenizer(
                model_name, task_config, source="huggingface", random_init=True
            )

        if handler_attr == "tokenizer":
            # MutBERT wrapping: the wrapper delegates attribute access.
            assert hasattr(tokenizer, "tokenizer")
        else:
            # Basenji2: replaced with the one-hot tokenizer.
            from dnallm.models.tokenizer import DNAOneHotTokenizer

            assert isinstance(tokenizer, DNAOneHotTokenizer)

    # ── slow lane: two-architecture acceptance (D-06) + difference proof ──

    _MS_DNABERT = "zhangtaolab/plant-dnabert-BPE"
    _MS_MAMBA = "zhangtaolab/plant-dnamamba-BPE-open_chromatin"

    @staticmethod
    def _skip_if_model_unavailable(model_id):
        """Typed slow-lane skip: run when the model is cached, else only if the hub is reachable."""
        cache_dir = Path.home() / ".cache" / "modelscope" / "hub" / "models" / model_id
        if (cache_dir / "config.json").is_file():
            return
        import socket

        try:
            socket.create_connection(("modelscope.cn", 443), timeout=5).close()
        except OSError as e:
            pytest.skip(
                f"network-unavailable: {model_id} not cached and "
                f"modelscope.cn unreachable ({type(e).__name__})"
            )

    @staticmethod
    def _digest_table(named_iter):
        """Digest every tensor yielded by ``named_iter`` as {name: (digest, dtype)}."""
        return {name: (_tensor_digest(t), t.dtype) for name, t in named_iter()}

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_random_init_generic_bert_family_modelscope(self, caplog):
        """D-06 first member: generic AutoModel (BERT-style) from_config on the modelscope route.

        Asserts banner + per-tensor hash lines, same-seed reload
        reproducibility of the full hash table, tokenizer loading, and one
        CPU forward pass.
        """
        self._skip_if_model_unavailable(self._MS_DNABERT)
        task_config = TaskConfig(task_type="mask", num_labels=None)

        tables = []
        for _ in range(2):
            caplog.clear()
            with caplog.at_level(logging.INFO):
                model, tokenizer = load_model_and_tokenizer(
                    self._MS_DNABERT,
                    task_config,
                    source="modelscope",
                    random_init=True,
                    random_init_seed=42,
                )
            tables.append(_param_hashes_from_caplog(caplog))

        assert "randomly initialized" in caplog.text
        # Full BERT parameter census per tensor (embeddings, encoder layers,
        # tied LM head alias) — a single global hash would hide leftovers.
        assert len(tables[0]) >= 100
        assert tables[0] == tables[1]
        assert tokenizer is not None

        model = model.to("cpu")
        enc = tokenizer("ACGTACGTACGTACGTACGT", return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        assert tuple(out.logits.shape[:2]) == tuple(enc["input_ids"].shape[:2])

    @pytest.mark.slow
    @pytest.mark.timeout(900)
    def test_random_init_mamba_trust_remote_code_modelscope(self, caplog):
        """D-06 second member (allowlist): the Mamba trust_remote_code from_config branch (A6).

        Plant DNAMamba loads through the generic AutoModelForCausalLM path —
        no special handler — and its remote-code config exercises
        ``from_config(trust_remote_code=True)``.
        """
        self._skip_if_model_unavailable(self._MS_MAMBA)
        task_config = TaskConfig(task_type="generation", num_labels=None)

        tables = []
        for _ in range(2):
            caplog.clear()
            with caplog.at_level(logging.INFO):
                model, tokenizer = load_model_and_tokenizer(
                    self._MS_MAMBA,
                    task_config,
                    source="modelscope",
                    random_init=True,
                    random_init_seed=42,
                )
            tables.append(_param_hashes_from_caplog(caplog))

        assert "randomly initialized" in caplog.text
        assert len(tables[0]) >= 10
        assert tables[0] == tables[1]
        assert tokenizer is not None

        model = model.to("cpu")
        enc = tokenizer("ACGTACGTACGTACGTACGT", return_tensors="pt")
        with torch.no_grad():
            out = model(input_ids=enc["input_ids"], attention_mask=enc.get("attention_mask"))
        assert tuple(out.logits.shape[:2]) == tuple(enc["input_ids"].shape[:2])

    @pytest.mark.slow
    @pytest.mark.timeout(1200)
    def test_random_init_per_tensor_difference_vs_pretrained(self):
        """Pitfall 4c: every float parameter tensor's hash differs from the pretrained load.

        Allowed exceptions, enumerated and counted (RESEARCH A-class):
        1. non-float (int/bool) buffers, which are deterministic constants
           (BERT: exactly two, ``position_ids`` and ``token_type_ids``);
        2. tied-weight aliases sharing storage, which match *within* each
           load (BERT: exactly two ties — the LM-head decoder weight
           aliasing the word-embedding weight, and the decoder bias
           aliasing the prediction bias) — all still differ from
           pretrained.
        Non-float parameters are also allowed to match; BERT has none.
        """
        self._skip_if_model_unavailable(self._MS_DNABERT)
        task_config = TaskConfig(task_type="mask", num_labels=None)

        pretrained_model, _ = load_model_and_tokenizer(
            self._MS_DNABERT, task_config, source="modelscope"
        )
        pre_params = self._digest_table(
            lambda: pretrained_model.named_parameters(remove_duplicate=False)
        )
        pre_buffers = self._digest_table(pretrained_model.named_buffers)
        del pretrained_model

        random_model, _ = load_model_and_tokenizer(
            self._MS_DNABERT,
            task_config,
            source="modelscope",
            random_init=True,
            random_init_seed=42,
        )
        rnd_params = self._digest_table(
            lambda: random_model.named_parameters(remove_duplicate=False)
        )
        rnd_buffers = self._digest_table(random_model.named_buffers)

        common = set(pre_params) & set(rnd_params)
        assert len(common) >= 100

        # Every float parameter tensor differs (per tensor, not globally).
        float_params = [n for n in common if rnd_params[n][1].is_floating_point]
        matching_float = [
            n for n in float_params if pre_params[n][0] == rnd_params[n][0]
        ]
        assert matching_float == []

        # Exception class 1: non-float parameters — none for BERT.
        non_float_params = [n for n in common if not rnd_params[n][1].is_floating_point]
        assert non_float_params == []

        # Exception class 2: tied-weight aliases sharing storage, matching
        # within each load. Storage-identity detection (data_ptr), NOT
        # digest collisions — fresh-init LayerNorm zero/one tensors
        # legitimately share digests across *different* tensors of equal
        # shape. BERT ties exactly two pairs: the LM-head decoder weight
        # aliasing the word-embedding weight, and the decoder bias
        # aliasing the prediction bias.
        ptr_map: dict[int, str] = {}
        tied_aliases = set()
        for name, p in random_model.named_parameters(remove_duplicate=False):
            ptr = p.data_ptr()
            if ptr in ptr_map:
                tied_aliases.add(ptr_map[ptr])
                tied_aliases.add(name)
            else:
                ptr_map[ptr] = name
        assert sorted(tied_aliases) == [
            "bert.embeddings.word_embeddings.weight",
            "cls.predictions.bias",
            "cls.predictions.decoder.bias",
            "cls.predictions.decoder.weight",
        ]
        assert (
            rnd_params["bert.embeddings.word_embeddings.weight"][0]
            == rnd_params["cls.predictions.decoder.weight"][0]
        )

        # Buffers: any buffer matching across loads must be non-float
        # (deterministic constants); BERT's only such buffer is
        # position_ids (long).
        common_buffers = set(pre_buffers) & set(rnd_buffers)
        matching_buffers = sorted(
            n for n in common_buffers if pre_buffers[n][0] == rnd_buffers[n][0]
        )
        for name in matching_buffers:
            assert not rnd_buffers[name][1].is_floating_point
        assert matching_buffers == [
            "bert.embeddings.position_ids",
            "bert.embeddings.token_type_ids",
        ]


class TestLoadPresetModel:
    """Test load_preset_model function."""

    def test_load_preset_model_success(self):
        """Test successful loading of preset model."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.modeling_auto.MODEL_INFO",
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

        with patch("dnallm.models.modeling_auto.MODEL_INFO", {}):
            result = load_preset_model("unknown-model", task_config)

            assert result == 0

    def test_load_preset_model_preset_name(self):
        """Test loading preset model by preset name."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch(
            "dnallm.models.modeling_auto.MODEL_INFO",
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

        with patch("dnallm.models.modeling_auto.MODEL_INFO", {}):
            result = load_preset_model("test-model", task_config)

            assert result == 0

    def test_load_preset_model_malformed_registry_returns_zero(self):
        """A malformed MODEL_INFO registry degrades to preset_models=[]."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        # A list registry makes MODEL_INFO[model] raise TypeError internally,
        # which degrades to preset_models=[]; a non-member name then misses.
        with patch("dnallm.models.modeling_auto.MODEL_INFO", ["entry"]):
            result = load_preset_model("missing", task_config)

            assert result == 0


@pytest.mark.parametrize(
    ("task_type", "expected_problem_type"),
    [
        ("binary", "single_label_classification"),
        ("multiclass", "single_label_classification"),
        ("multilabel", "multi_label_classification"),
        ("regression", "regression"),
    ],
)
def test_load_model_by_task_type_problem_types(task_type, expected_problem_type):
    """Test that correct problem types are set for different task types."""
    modules = {
        "AutoTokenizer": Mock(),
        "AutoModelForSequenceClassification": Mock(),
    }
    mock_tokenizer = Mock()
    mock_model = Mock()
    modules["AutoTokenizer"].from_pretrained.return_value = mock_tokenizer
    modules["AutoModelForSequenceClassification"].from_pretrained.return_value = mock_model

    _load_model_by_task_type(task_type, "test-model", 2, {}, {}, modules)

    call_args = modules["AutoModelForSequenceClassification"].from_pretrained.call_args
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
def test_load_model_by_task_type_attention_implementation(task_type, expected_attn_implementation):
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
        call_args = modules["AutoModelForSequenceClassification"].from_pretrained.call_args
    elif task_type == "token":
        call_args = modules["AutoModelForTokenClassification"].from_pretrained.call_args
    else:
        call_args = modules["AutoModel"].from_pretrained.call_args

    assert call_args[1]["attn_implementation"] == expected_attn_implementation


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch sentinel matrix (TEST-01): fault injection over the
# load_model_and_tokenizer chain. Handlers are patched where the dispatch
# module imports them (dnallm.models.model._handle_<family>_models).
# ─────────────────────────────────────────────────────────────────────────────

_FAMILY_HANDLERS = (
    "_handle_evo2_models",
    "_handle_evo1_models",
    "_handle_megadna_models",
    "_handle_enformer_models",
    "_handle_space_models",
    "_handle_borzoi_models",
)


def _declining_family_patchers(selected=None, selected_return=None):
    """Patchers that make every early-return family handler decline.

    When *selected* names one of them, that handler returns
    *selected_return* instead so it wins the dispatch chain.
    """
    patchers = []
    for name in _FAMILY_HANDLERS:
        if name == selected:
            patchers.append(patch(f"dnallm.models.model.{name}", return_value=selected_return))
        else:
            patchers.append(patch(f"dnallm.models.model.{name}", return_value=None))
    return patchers


def _gate_patchers():
    """Patchers that neutralize the gpn/omnidna import-availability gates."""
    return [
        patch("dnallm.models.model._handle_gpn_models", return_value=None),
        patch("dnallm.models.model._handle_omnidna_models", return_value=None),
    ]


class TestDispatchChain:
    """Sentinel fault-injection matrix over the load dispatch chain."""

    @pytest.mark.parametrize("selected_handler", _FAMILY_HANDLERS)
    def test_early_return_family_wins_dispatch(self, selected_handler):
        """The one resolved early-return family returns before source resolution."""
        sentinel_model = Mock()
        # load_model_and_tokenizer rebinds the model via .to(device); a plain
        # Mock would return a fresh child mock and break the identity check.
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        task_config = TaskConfig(task_type="binary", num_labels=2)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(selected_handler, (sentinel_model, sentinel_tokenizer)),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                side_effect=AssertionError("source resolution must not run"),
            ),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                side_effect=AssertionError("generic loader must not run"),
            ),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("some-model", task_config)

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer

    def test_partial_handler_result_survives_the_chain(self):
        """A handler's resolved half is never discarded by a later stage (WR-04).

        The chain's documented invariant says a handler's result is never
        overwritten; the pair-reassignment form violated it -- a dnabert2
        partial ``(model, None)`` was silently dropped when the generic
        loader returned ``(None, tokenizer)``, crashing on
        ``model._model_path``. The per-half merge preserves both halves.
        """
        sentinel_model = Mock()
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        task_config = TaskConfig(task_type="binary", num_labels=2)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/some-model", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch(
                "dnallm.models.model._handle_dnabert2_models",
                return_value=(sentinel_model, None),
            ),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(None, sentinel_tokenizer),
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("some-model", task_config)

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer

    def test_gpn_gate_import_error_propagates(self):
        """A gpn-named model surfaces the availability gate's ImportError."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model._setup_huggingface_mirror"):
            with pytest.raises(ImportError, match="gpn package is required"):
                load_model_and_tokenizer("gpn-brassicales-checkpoint", task_config)

    def test_gpn_gate_falls_through_when_available(self, monkeypatch):
        """With gpn importable the gate returns a str and the generic loader serves."""
        fake_model_module = types.ModuleType("gpn.model")
        fake_gpn = types.ModuleType("gpn")
        fake_gpn.model = fake_model_module
        monkeypatch.setitem(sys.modules, "gpn", fake_gpn)
        monkeypatch.setitem(sys.modules, "gpn.model", fake_model_module)

        sentinel_model = Mock()
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch("dnallm.models.model._handle_omnidna_models", return_value=None),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/gpn-brassicales", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(sentinel_model, sentinel_tokenizer),
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("gpn-brassicales-checkpoint", task_config)

        # The gate's str return is discarded: the generic loader served.
        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer

    def test_omnidna_gate_import_error_propagates(self):
        """An Omni-DNA-named model surfaces the olmo availability gate's ImportError."""
        task_config = TaskConfig(task_type="mask", num_labels=None)

        with patch("dnallm.models.model._setup_huggingface_mirror"):
            with pytest.raises(ImportError, match="ai2-olmo package is required"):
                load_model_and_tokenizer("Omni-DNA-20M", task_config)

    def test_omnidna_gate_falls_through_when_available(self, monkeypatch):
        """With olmo importable the gate returns a str and the generic loader serves."""
        fake_version = types.ModuleType("olmo.version")
        fake_version.VERSION = "1.0"
        fake_olmo = types.ModuleType("olmo")
        fake_olmo.version = fake_version
        monkeypatch.setitem(sys.modules, "olmo", fake_olmo)
        monkeypatch.setitem(sys.modules, "olmo.version", fake_version)

        sentinel_model = Mock()
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            patch("dnallm.models.model._handle_gpn_models", return_value=None),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/Omni-DNA-20M", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(sentinel_model, sentinel_tokenizer),
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("Omni-DNA-20M", task_config)

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer

    def test_generic_loader_serves_when_all_families_decline(self):
        """When every family handler declines, the generic loader's result survives."""
        sentinel_model = Mock()
        sentinel_model.to = Mock(return_value=sentinel_model)
        sentinel_tokenizer = Mock()
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/plain-model", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch(
                "dnallm.models.model._handle_crossdna_models",
                side_effect=AssertionError("crossdna handler must not run"),
            ),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(sentinel_model, sentinel_tokenizer),
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("plain-model", task_config, source="local")

        assert model is sentinel_model
        assert tokenizer is sentinel_tokenizer
        assert sentinel_model._model_path == "/models/plain-model"
        assert sentinel_model.source == "local"

    def test_mutbert_tokenizer_post_processing(self):
        """mutbert paths replace the loaded tokenizer via the post-processor."""
        generic_model = Mock()
        generic_model.to = Mock(return_value=generic_model)
        raw_tokenizer, mutbert_tokenizer = Mock(), Mock()
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/mutbert-3m", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(generic_model, raw_tokenizer),
            ),
            patch(
                "dnallm.models.model._handle_mutbert_tokenizer",
                return_value=mutbert_tokenizer,
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("mutbert-3m", task_config, source="local")

        assert model is generic_model
        # The post-processed tokenizer is the one returned, not the raw one.
        assert tokenizer is mutbert_tokenizer
        assert tokenizer is not raw_tokenizer

    def test_basenji2_tokenizer_post_processing(self):
        """basenji2 paths replace the loaded tokenizer via the post-processor."""
        generic_model = Mock()
        generic_model.to = Mock(return_value=generic_model)
        raw_tokenizer, basenji2_tokenizer = Mock(), Mock()
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/basenji2", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(generic_model, raw_tokenizer),
            ),
            patch(
                "dnallm.models.model._handle_basenji2_tokenizer",
                return_value=basenji2_tokenizer,
            ),
            patch("dnallm.models.model._configure_model_padding"),
            patch("dnallm.models.model._get_device", return_value=torch.device("cpu")),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            model, tokenizer = load_model_and_tokenizer("basenji2", task_config, source="local")

        assert model is generic_model
        assert tokenizer is basenji2_tokenizer
        assert tokenizer is not raw_tokenizer

    def test_quantization_config_skips_device_placement(self):
        """A quantization_config skips .to(device) and runs the bnb fix instead."""
        generic_model = Mock()
        generic_model.to = Mock(return_value=generic_model)
        task_config = TaskConfig(task_type="mask", num_labels=None)

        patchers = [
            patch("dnallm.models.model._setup_huggingface_mirror"),
            *_gate_patchers(),
            *_declining_family_patchers(),
            patch(
                "dnallm.models.model._get_model_path_and_imports",
                return_value=("/models/plain-model", {"AutoTokenizer": Mock()}),
            ),
            patch("dnallm.models.model._create_label_mappings", return_value=({}, {})),
            patch("dnallm.models.model._handle_dnabert2_models", return_value=(None, None)),
            patch(
                "dnallm.models.model._load_model_by_task_type",
                return_value=(generic_model, Mock()),
            ),
            patch("dnallm.models.model._configure_model_padding"),
        ]
        with ExitStack() as stack:
            for patcher in patchers:
                stack.enter_context(patcher)
            mock_fix = stack.enter_context(patch("dnallm.models.model._fix_bnb_quantized_layers"))
            _model, _tokenizer = load_model_and_tokenizer(
                "plain-model",
                task_config,
                source="local",
                quantization_config={"load_in_4bit": True},
            )

        mock_fix.assert_called_once_with(generic_model)
        # device_map="auto" handles placement: .to() must not run
        generic_model.to.assert_not_called()


class TestSafeNumLabels:
    """Test _safe_num_labels task-type normalization rules."""

    @pytest.mark.parametrize(
        "task_type",
        ["binary", "multiclass", "multilabel", "regression", "token"],
    )
    def test_none_num_labels_classification_raises(self, task_type):
        """Classification tasks reject a None num_labels."""
        with pytest.raises(ValueError, match="num_labels is required"):
            _safe_num_labels(None, task_type)

    def test_none_num_labels_mask_defaults_to_zero(self):
        """Mask tasks default a None num_labels to 0."""
        assert _safe_num_labels(None, "mask") == 0

    def test_generation_num_labels_forced_to_zero(self):
        """Generation tasks force num_labels to 0."""
        assert _safe_num_labels(2, "generation") == 0

    def test_mask_num_labels_forced_to_zero(self):
        """Mask tasks force a non-zero num_labels to 0."""
        assert _safe_num_labels(5, "mask") == 0

    def test_embedding_num_labels_forced_to_zero(self):
        """Embedding tasks force a non-zero num_labels to 0."""
        assert _safe_num_labels(3, "embedding") == 0

    def test_regression_multi_labels_warns_but_keeps(self):
        """Multi-output regression keeps its num_labels."""
        assert _safe_num_labels(3, "regression") == 3

    def test_multiclass_below_two_raises(self):
        """Non-binary classification requires at least two labels."""
        with pytest.raises(ValueError, match="num_labels should be at least 2"):
            _safe_num_labels(1, "multiclass")

    def test_valid_classification_passthrough(self):
        """A valid classification num_labels passes through unchanged."""
        assert _safe_num_labels(4, "multiclass") == 4


class TestGetDevice:
    """Test _get_device automatic device selection order."""

    def test_cuda_preferred(self):
        """CUDA wins when available."""
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            assert _get_device() == torch.device("cuda")

    def test_mps_when_no_cuda(self):
        """MPS is chosen when CUDA is unavailable."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=True),
        ):
            assert _get_device() == torch.device("mps")

    def test_xpu_when_no_cuda_or_mps(self, monkeypatch):
        """XPU is chosen when CUDA and MPS are unavailable."""
        monkeypatch.setattr(torch, "xpu", SimpleNamespace(is_available=lambda: True), raising=False)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            assert _get_device() == torch.device("xpu")

    def test_cpu_fallback(self, monkeypatch):
        """CPU is the final fallback."""
        if hasattr(torch, "xpu"):
            monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch.object(torch.backends.mps, "is_available", return_value=False),
        ):
            assert _get_device() == torch.device("cpu")


class TestFixBnbQuantizedLayers:
    """Test _fix_bnb_quantized_layers replacement of unpacked quantized layers."""

    def test_unpacked_linear4bit_replaced_with_linear(self):
        """A 2D-weight Linear4bit without quant_state is replaced by nn.Linear."""

        class FakeLinear4bit(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8))

        container = nn.Module()
        container.add_module("score", FakeLinear4bit())

        _fix_bnb_quantized_layers(container)

        replaced = container.score
        assert isinstance(replaced, nn.Linear)
        assert replaced.in_features == 8
        assert replaced.out_features == 4

    def test_packed_linear4bit_shape_untouched(self):
        """Properly quantized [N, 1] weights keep their module."""

        class FakeLinear4bit(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randint(0, 255, (4, 1)).float())

        container = nn.Module()
        original = FakeLinear4bit()
        container.add_module("quant", original)

        _fix_bnb_quantized_layers(container)

        assert container.quant is original

    def test_quant_state_present_skips_replacement(self):
        """A 2D weight with quant_state set is left alone."""

        class FakeLinear4bit(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8))
                self.quant_state = Mock()

        container = nn.Module()
        original = FakeLinear4bit()
        container.add_module("quant", original)

        _fix_bnb_quantized_layers(container)

        assert container.quant is original

    def test_linear8bitlt_name_also_matched(self):
        """Linear8bitLt module names take the same replacement path."""

        class FakeLinear8bitLt(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(6, 3))

        container = nn.Module()
        container.add_module("eight_bit", FakeLinear8bitLt())

        _fix_bnb_quantized_layers(container)

        assert isinstance(container.eight_bit, nn.Linear)
        assert container.eight_bit.in_features == 3

    def test_uncopyable_weight_still_replaces(self):
        """A weight that cannot be copied still yields the nn.Linear replacement."""

        class FakeLinear4bit(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(4, 8, dtype=torch.complex64))

        container = nn.Module()
        container.add_module("weird", FakeLinear4bit())

        with patch("dnallm.models.model.logger"):
            _fix_bnb_quantized_layers(container)

        assert isinstance(container.weird, nn.Linear)


class TestClearModelCache:
    """Test clear_model_cache cache directory handling."""

    def test_clear_huggingface_cache_removes_entries(self, tmp_path):
        """Files and directories inside the HF cache are removed."""
        cache = tmp_path / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True)
        model_dir = cache / "model--x"
        model_dir.mkdir()
        (model_dir / "blob").write_text("data")
        leftover_file = cache / "stale.lock"
        leftover_file.write_text("x")

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            clear_model_cache("huggingface")

        assert not model_dir.exists()
        assert not leftover_file.exists()

    def test_clear_modelscope_cache_removes_entries(self, tmp_path):
        """The ModelScope cache directory is cleaned as well."""
        cache = tmp_path / ".cache" / "modelscope" / "hub"
        cache.mkdir(parents=True)
        stale = cache / "model--y"
        stale.mkdir()

        with patch("os.path.expanduser", return_value=str(tmp_path)):
            clear_model_cache("modelscope")

        assert not stale.exists()

    def test_clear_missing_cache_is_a_noop(self, tmp_path):
        """A missing cache directory logs and returns without raising."""
        with patch("os.path.expanduser", return_value=str(tmp_path / "empty")):
            clear_model_cache("huggingface")  # must not raise

    def test_unsupported_source_warns_and_returns(self):
        """An unsupported source logs a warning and does nothing."""
        with patch("dnallm.models.model.logger") as mock_logger:
            clear_model_cache("ftp")

        mock_logger.warning.assert_called_once()

    def test_removal_failure_logs_warning(self, tmp_path):
        """Entries that fail to remove produce a warning, not an exception."""
        cache = tmp_path / ".cache" / "huggingface" / "hub"
        cache.mkdir(parents=True)
        (cache / "model--z").mkdir()

        with (
            patch("os.path.expanduser", return_value=str(tmp_path)),
            patch("shutil.rmtree", side_effect=OSError("permission denied")),
            patch("dnallm.models.model.logger") as mock_logger,
        ):
            clear_model_cache("huggingface")

        assert mock_logger.warning.called


class TestPeftForwardCompatible:
    """Test peft_forward_compatiable kwarg filtering."""

    def test_unsupported_kwargs_are_dropped(self):
        """kwargs outside the original signature are filtered out."""

        class FakeModel(nn.Module):
            def forward(self, input_ids=None, attention_mask=None):
                return (input_ids, attention_mask)

        model = FakeModel()
        wrapped = peft_forward_compatiable(model)

        input_ids, attention_mask = wrapped(
            input_ids=torch.tensor([1]), attention_mask=torch.tensor([1]), token_type_ids="dropped"
        )

        assert input_ids is not None
        assert attention_mask is not None


# ─────────────────────────────────────────────────────────────────────────────
# DNALLMforSequenceClassification: real-torch construction/forward coverage.
# The backbone is a tiny real nn.Module so autograd and pooling semantics
# are the real behavior under test (never a Mock).
# ─────────────────────────────────────────────────────────────────────────────


class TinyBackbone(nn.Module):
    """Minimal real backbone standing in for AutoModel.from_config output."""

    def __init__(self, hidden=8, vocab=6, with_pad_token_id=False, rich_outputs=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)
        if with_pad_token_id:
            self.config.pad_token_id = 0
        self.rich_outputs = rich_outputs

    def forward(self, input_ids=None, **kwargs):
        emb = self.embedding(input_ids)
        outputs = SimpleNamespace(last_hidden_state=emb)
        if self.rich_outputs:
            outputs.hidden_states = emb
            outputs.attentions = emb
        return outputs


class TinyTupleBackbone(nn.Module):
    """Backbone returning a tuple whose first element is itself a tuple."""

    def __init__(self, hidden=8, vocab=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.config = SimpleNamespace(hidden_size=hidden)

    def forward(self, input_ids=None, **kwargs):
        emb = self.embedding(input_ids)
        aux = (torch.zeros(1),)
        # The wrapper's else-arm takes outputs[0] and, when that is itself a
        # tuple, its last element — mimicking backbones with auxiliary outputs.
        return ((aux, emb),)


class TinyMegaDNABackbone(nn.Module):
    """Backbone returning the MegaDNA multi-scale embedding list."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2, 2)
        self.config = SimpleNamespace()

    def forward(self, input_ids, return_value=None, **kwargs):
        batch = input_ids.shape[0]
        return [
            torch.randn(batch, 3, 4),
            torch.randn(batch, 5, 4),
            torch.randn(batch, 6, 4),
        ]


class TinyEvoInner(nn.Module):
    """Inner module exposing blocks.* parameter names for EVO layer selection."""

    def __init__(self, hidden=8, n_blocks=4):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_blocks)])


class TinyEvoModel(nn.Module):
    """Backbone mimicking the EVO return_embeddings contract."""

    def __init__(self, hidden=8, n_blocks=4):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden)
        self.model = TinyEvoInner(hidden, n_blocks)
        self.embedding = nn.Embedding(6, hidden)

    def forward(self, input_ids, return_embeddings=False, layer_names=None, **kwargs):
        emb = self.embedding(input_ids)
        hidden = self.model.blocks[0](emb)
        embeddings = dict.fromkeys(layer_names or [], hidden)
        logits = hidden.mean(dim=1)
        return (logits, embeddings)


def _make_wrapper_config(head="basic-mlp", num_labels=2, **extra_head):
    """Build a PretrainedConfig the wrapper accepts."""
    config = PretrainedConfig()
    config.num_labels = num_labels
    config.head_config = {
        "head": head,
        "num_classes": num_labels,
        "task_type": "binary",
        **extra_head,
    }
    return config


def _build_wrapper(head="basic-mlp", num_labels=2, backbone=None, custom_model=None, **extra_head):
    """Construct the wrapper with a tiny real backbone behind AutoModel.from_config."""
    torch.manual_seed(0)
    backbone = backbone if backbone is not None else TinyBackbone(hidden=8)
    config = _make_wrapper_config(head=head, num_labels=num_labels, **extra_head)
    with patch("transformers.AutoModel.from_config", return_value=backbone):
        return DNALLMforSequenceClassification(config, custom_model=custom_model)


class _SigmoidScore(nn.Module):
    """Score wrapper squashing logits into [0, 1] while keeping the head contract."""

    def __init__(self, score):
        super().__init__()
        self.score = score
        self.task_type = score.task_type
        self.num_classes = getattr(score, "num_classes", None)

    def forward(self, x):
        return torch.sigmoid(self.score(x))


class TestDNALLMforSequenceClassificationInit:
    """Construction-path coverage for the classification wrapper."""

    def test_default_branch_builds_mlp_head(self):
        """The generic branch derives input_dim from the backbone config."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)

        assert isinstance(model.backbone, TinyBackbone)
        from dnallm.models.head import BasicMLPHead

        assert isinstance(model.score, BasicMLPHead)
        assert model.score.input_dim == 8

    @pytest.mark.parametrize(
        ("head_suffix", "expected_head_name"),
        [
            ("basic-cnn", "BasicCNNHead"),
            ("basic-lstm", "BasicLSTMHead"),
            ("basic-unet", "BasicUNet1DHead"),
        ],
    )
    def test_classifier_selection_by_head_suffix(self, head_suffix, expected_head_name):
        """Head class selection follows the head name suffix."""
        import dnallm.models.head as head_module

        model = _build_wrapper(head=head_suffix, num_labels=2)

        assert type(model.score).__name__ == expected_head_name
        assert isinstance(model.score, getattr(head_module, expected_head_name))

    def test_unknown_head_name_raises_value_error(self):
        """An unrecognized head name raises the convention ValueError (WR-06).

        UnboundLocalError used to escape ``_determine_classifier`` for head
        names with no custom_head and no mlp/cnn/lstm/unet suffix.
        """
        with pytest.raises(ValueError, match=r"Unknown head type.*'attention'"):
            _build_wrapper(head="attention", num_labels=2)

    def test_megadna_branch_uses_custom_model(self):
        """head='megadna' wires the custom model as backbone with a multi-scale head."""
        custom = TinyMegaDNABackbone()
        model = _build_wrapper(
            head="megadna",
            num_labels=2,
            custom_model=custom,
            embedding_dims=[4, 4, 4],
        )

        assert model.backbone is custom
        from dnallm.models.head import MegaDNAMultiScaleHead

        assert isinstance(model.score, MegaDNAMultiScaleHead)

    def test_evo_branch_uses_custom_model(self):
        """head='evo*' wires the custom model with an EVO layer head."""
        custom = TinyEvoModel(hidden=8)
        model = _build_wrapper(
            head="evo", num_labels=2, custom_model=custom, target_layer="blocks.1"
        )

        assert model.backbone is custom
        from dnallm.models.head import EVOForSeqClsHead

        assert isinstance(model.score, EVOForSeqClsHead)
        assert model.score.target_layers == ["blocks.1"]

    def test_lucaone_branch_imports_backbone(self, monkeypatch):
        """head containing 'lucaone' builds the LucaGPLM backbone."""

        class TinyLucaGPLM(nn.Module):
            def __init__(self, config):
                super().__init__()
                self.proj = nn.Linear(8, 8)
                self.config = SimpleNamespace()

        fake_lucagplm = types.ModuleType("lucagplm")
        fake_lucagplm.LucaGPLMModel = TinyLucaGPLM
        monkeypatch.setitem(sys.modules, "lucagplm", fake_lucagplm)

        config = _make_wrapper_config(head="lucaone-mlp", num_labels=2)
        config.hidden_size = 8

        model = DNALLMforSequenceClassification(config)

        assert isinstance(model.backbone, TinyLucaGPLM)
        assert model.score.input_dim == 8

    def test_missing_hidden_dim_raises(self):
        """Without hidden_size/d_model the default branch raises ValueError."""
        config = _make_wrapper_config(head="basic-mlp", num_labels=2)
        weird_backbone = TinyBackbone(hidden=8)
        weird_backbone.config = SimpleNamespace()  # no hidden_size, no d_model

        with patch("transformers.AutoModel.from_config", return_value=weird_backbone):
            with pytest.raises(ValueError, match="Cannot determine transformer output dimension"):
                DNALLMforSequenceClassification(config)

    def test_d_model_fallback_for_input_dim(self):
        """A backbone exposing only d_model still sets the head input dim."""
        backbone = TinyBackbone(hidden=6)
        backbone.config = SimpleNamespace(d_model=6)

        model = _build_wrapper(head="basic-mlp", num_labels=2, backbone=backbone)

        assert model.score.input_dim == 6

    def test_frozen_backbone_disables_grad(self):
        """head_config frozen=True freezes every backbone parameter."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, frozen=True)

        assert all(not p.requires_grad for p in model.backbone.parameters())

    def test_from_base_model_loads_backbone_weights(self):
        """from_base_model copies the pretrained weights into the wrapper backbone."""
        backbone = TinyBackbone(hidden=8)
        base = TinyBackbone(hidden=8)
        config = _make_wrapper_config(head="basic-mlp", num_labels=2)

        with (
            patch("transformers.AutoModel.from_config", return_value=backbone),
            patch(
                "transformers.AutoModel.from_pretrained", return_value=base
            ) as mock_from_pretrained,
        ):
            model = DNALLMforSequenceClassification.from_base_model("some-model", config=config)

        mock_from_pretrained.assert_called_once_with("some-model", trust_remote_code=True)
        for (name, param), (_, base_param) in zip(
            model.backbone.state_dict().items(), base.state_dict().items(), strict=True
        ):
            assert torch.equal(param, base_param), f"weight mismatch for {name}"

    def test_from_base_model_with_explicit_module(self):
        """An explicit module kwarg bypasses AutoModel.from_pretrained."""
        backbone = TinyBackbone(hidden=8)
        base = TinyBackbone(hidden=8)
        module = Mock()
        module.from_pretrained.return_value = base
        config = _make_wrapper_config(head="basic-mlp", num_labels=2)

        with patch("transformers.AutoModel.from_config", return_value=backbone):
            model = DNALLMforSequenceClassification.from_base_model(
                "some-model", config=config, module=module
            )

        module.from_pretrained.assert_called_once_with("some-model", trust_remote_code=True)
        assert model.backbone is backbone

    def test_from_base_model_quantization_kwargs(self):
        """A quantization_config adds quantization/device_map load kwargs."""
        backbone = TinyBackbone(hidden=8)
        module = Mock()
        module.from_pretrained.return_value = TinyBackbone(hidden=8)
        config = _make_wrapper_config(head="basic-mlp", num_labels=2)
        quantization_config = Mock()

        with patch("transformers.AutoModel.from_config", return_value=backbone):
            DNALLMforSequenceClassification.from_base_model(
                "some-model",
                config=config,
                module=module,
                quantization_config=quantization_config,
            )

        call_kwargs = module.from_pretrained.call_args.kwargs
        assert call_kwargs["quantization_config"] is quantization_config
        assert call_kwargs["device_map"] == "auto"


class TestPoolingStrategy:
    """_determine_pooling_strategy and _get_sentence_embedding coverage."""

    def test_explicit_pooling_strategy_wins(self):
        """A configured pooling_strategy is used verbatim."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy="max")

        assert model.pooling_strategy == "max"

    def test_decoder_backbone_selects_last(self):
        """A decoder backbone selects 'last' pooling."""
        backbone = TinyBackbone(hidden=8)
        backbone.config.is_decoder = True

        model = _build_wrapper(head="basic-mlp", num_labels=2, backbone=backbone)

        assert model.pooling_strategy == "last"

    def test_cls_token_id_selects_cls(self):
        """A config cls_token_id selects 'cls' pooling."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)
        model.config.cls_token_id = 5

        assert model._determine_pooling_strategy() == "cls"

    def test_cls_idx_selects_cls(self):
        """A config cls_idx selects 'cls' pooling when cls_token_id is None."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)
        model.config.cls_token_id = None
        model.config.cls_idx = 0

        assert model._determine_pooling_strategy() == "cls"

    def test_no_signal_falls_back_to_mean(self):
        """Without any signal the strategy falls back to 'mean' with a warning."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)
        model.config.cls_token_id = None

        assert model._determine_pooling_strategy() == "mean"

    @pytest.mark.parametrize(
        ("strategy", "expected_shape"),
        [
            ("cls", (2, 8)),
            ("mean", (2, 8)),
            ("max", (2, 8)),
            ("last", (2, 8)),
            ("first", (2, 8)),
        ],
    )
    def test_sentence_embedding_strategies(self, strategy, expected_shape):
        """Every pooling strategy returns the (batch, hidden) sentence embedding."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy=strategy)
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]])

        embedding = model._get_sentence_embedding(hidden, mask)

        assert embedding.shape == expected_shape

    def test_mean_pooling_respects_mask(self):
        """Mean pooling averages only over unmasked positions."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy="mean")
        hidden = torch.ones(1, 4, 8)
        mask = torch.tensor([[1, 1, 0, 0]])

        embedding = model._get_sentence_embedding(hidden, mask)

        assert torch.allclose(embedding, torch.ones(1, 8))

    def test_cls_pooling_takes_first_token(self):
        """CLS pooling returns position 0 regardless of the mask."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy="cls")
        hidden = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        mask = torch.ones(2, 3)

        embedding = model._get_sentence_embedding(hidden, mask)

        assert torch.equal(embedding, hidden[:, 0, :])

    def test_last_pooling_takes_last_unmasked(self):
        """'last' pooling picks each row's last unmasked position."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy="last")
        hidden = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        embedding = model._get_sentence_embedding(hidden, mask)

        assert torch.equal(embedding[0], hidden[0, 1, :])
        assert torch.equal(embedding[1], hidden[1, 2, :])

    def test_unsupported_pooling_strategy_raises(self):
        """An unknown pooling strategy raises ValueError."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, pooling_strategy="bogus")

        with pytest.raises(ValueError, match="Unsupported pooling strategy"):
            model._get_sentence_embedding(torch.randn(2, 3, 8), torch.ones(2, 3))


class TestWrapperForward:
    """Real-torch forward, attention-mask and loss-selection coverage."""

    def _input_ids(self, batch=2, length=5):
        return torch.randint(0, 6, (batch, length))

    def test_forward_shapes_and_gradient(self):
        """A plain forward produces (batch, num_classes) logits with real gradients."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)

        output = model(input_ids=self._input_ids())

        assert output.logits.shape == (2, 2)
        assert output.loss is None
        output.logits.sum().backward()
        grads = [p.grad for p in model.score.parameters() if p.grad is not None]
        assert grads, "head parameters must receive gradients"

    def test_forward_attention_mask_from_pad_token_id(self):
        """Without an explicit mask, padding positions are derived from pad_token_id."""
        backbone = TinyBackbone(hidden=8, with_pad_token_id=True)
        model = _build_wrapper(head="basic-mlp", num_labels=2, backbone=backbone)

        input_ids = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
        output = model(input_ids=input_ids, labels=torch.tensor([0, 1]))

        # The loss must compute: the derived mask kept every forward alive.
        assert output.loss is not None
        assert torch.isfinite(output.loss)

    def test_forward_attention_mask_ones_fallback(self):
        """Without any pad_token_id the mask defaults to all-ones."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)

        output = model(input_ids=self._input_ids(), labels=torch.tensor([0, 1]))

        assert output.loss is not None

    def test_forward_explicit_attention_mask(self):
        """An explicitly passed attention_mask is honored."""
        model = _build_wrapper(head="basic-mlp", num_labels=2)
        input_ids = self._input_ids()
        mask = torch.ones_like(input_ids)

        output = model(input_ids=input_ids, attention_mask=mask, labels=torch.tensor([1, 0]))

        assert output.loss is not None

    def test_forward_tuple_backbone_output(self):
        """Backbones returning nested tuples still yield the hidden states."""
        backbone = TinyTupleBackbone(hidden=8)
        model = _build_wrapper(head="basic-mlp", num_labels=2, backbone=backbone)

        output = model(input_ids=self._input_ids())

        assert output.logits.shape == (2, 2)

    def test_forward_hidden_states_and_attentions_outputs(self):
        """output_hidden_states/output_attentions populate the output object."""
        backbone = TinyBackbone(hidden=8, rich_outputs=True)
        model = _build_wrapper(head="basic-mlp", num_labels=2, backbone=backbone)

        output = model(
            input_ids=self._input_ids(),
            output_hidden_states=True,
            output_attentions=True,
        )

        assert output.hidden_states is not None
        assert output.attentions is not None

    def test_forward_hidden_states_none_without_attr(self):
        """Without backbone support the hidden/attention outputs are None."""
        # A bare (emb,) tuple: no hidden_states attr anywhere and len(outputs)==1.
        model = _build_wrapper(head="basic-mlp", num_labels=2)

        class BareTupleBackbone(TinyBackbone):
            def forward(self, input_ids=None, **kwargs):
                return (self.embedding(input_ids),)

        model.backbone = BareTupleBackbone(hidden=8)
        output = model(
            input_ids=self._input_ids(),
            output_hidden_states=True,
            output_attentions=True,
        )

        assert output.hidden_states is None
        assert output.attentions is None

    def test_forward_cnn_head_uses_raw_hidden_states(self):
        """Non-MLP heads consume the (batch, seq, hidden) states directly."""
        torch.manual_seed(0)
        model = _build_wrapper(head="basic-cnn", num_labels=2)

        output = model(input_ids=self._input_ids())

        assert output.logits.shape == (2, 2)

    def test_forward_num_labels_rebound_to_num_classes(self):
        """A logits/num_labels mismatch rebinds num_labels from head_config."""
        model = _build_wrapper(head="basic-mlp", num_labels=3)
        # config.num_labels=3 but the head emits num_classes=3 by default; force a
        # mismatch by asking the head for 2 classes while config says 3.
        config = _make_wrapper_config(head="basic-mlp", num_labels=3)
        config.head_config["num_classes"] = 2
        with patch("transformers.AutoModel.from_config", return_value=TinyBackbone(hidden=8)):
            model = DNALLMforSequenceClassification(config)

        output = model(input_ids=self._input_ids())

        assert output.logits.shape == (2, 2)
        assert model.num_labels == 2

    def test_forward_megadna_branch(self):
        """The megadna branch pools the multi-scale embedding list through its head."""
        model = _build_wrapper(
            head="megadna",
            num_labels=2,
            custom_model=TinyMegaDNABackbone(),
            embedding_dims=[4, 4, 4],
        )

        output = model(input_ids=self._input_ids().long(), labels=torch.tensor([0, 1]))

        assert output.logits.shape == (2, 2)
        assert output.loss is not None

    def test_forward_evo_branch(self):
        """The evo branch extracts layer embeddings and classifies them."""
        model = _build_wrapper(
            head="evo",
            num_labels=2,
            custom_model=TinyEvoModel(hidden=8),
            target_layer=["blocks.0", "blocks.1"],
        )

        output = model(input_ids=self._input_ids().long(), labels=torch.tensor([1, 0]))

        assert output.logits.shape == (2, 2)
        assert output.loss is not None

    def test_forward_regression_single_label_mse(self):
        """Single-label regression squeezes both logits and labels for MSE."""
        model = _build_wrapper(head="basic-mlp", num_labels=1, task_type="regression")
        input_ids = self._input_ids()
        labels = torch.rand(2, 1)

        output = model(input_ids=input_ids, labels=labels)
        expected = nn.MSELoss()(output.logits.squeeze(), labels.squeeze())

        assert torch.allclose(output.loss, expected)

    def test_forward_multi_regression_mse(self):
        """Multi-output regression applies MSE without squeezing."""
        model = _build_wrapper(head="basic-mlp", num_labels=2, task_type="regression")
        input_ids = self._input_ids()
        labels = torch.rand(2, 2)

        output = model(input_ids=input_ids, labels=labels)
        expected = nn.MSELoss()(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    @pytest.mark.parametrize("task_type", ["binary", "multiclass"])
    def test_forward_classification_cross_entropy(self, task_type):
        """Single-label classification uses cross entropy against int labels."""
        model = _build_wrapper(head="basic-mlp", num_labels=3, task_type=task_type)
        input_ids = self._input_ids()
        labels = torch.tensor([0, 2])

        output = model(input_ids=input_ids, labels=labels)
        expected = nn.CrossEntropyLoss()(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    def test_forward_multilabel_bce_with_logits(self):
        """Multilabel classification uses BCEWithLogitsLoss against float labels."""
        model = _build_wrapper(head="basic-mlp", num_labels=3, task_type="multilabel")
        input_ids = self._input_ids()
        labels = torch.rand(2, 3)

        output = model(input_ids=input_ids, labels=labels)
        expected = nn.BCEWithLogitsLoss()(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    @pytest.mark.parametrize(
        ("loss_name", "reference_factory"),
        [
            ("mse", nn.MSELoss),
            ("crossentropy", nn.CrossEntropyLoss),
            ("bcewithlogits", nn.BCEWithLogitsLoss),
            ("focal", FocalLoss),
            ("poisson", nn.PoissonNLLLoss),
        ],
    )
    def test_forward_loss_function_strings(self, loss_name, reference_factory):
        """Named loss functions replace the task-type default."""
        model = _build_wrapper(
            head="basic-mlp", num_labels=2, task_type="regression", loss_function=loss_name
        )
        input_ids = self._input_ids()
        labels = torch.rand(2, 2)

        output = model(input_ids=input_ids, labels=labels)
        expected = reference_factory()(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    def test_forward_loss_function_bce_on_squashed_logits(self):
        """'bce' applies BCELoss to the (sigmoid-squashed) logits."""
        model = _build_wrapper(
            head="basic-mlp", num_labels=2, task_type="regression", loss_function="bce"
        )
        # BCELoss requires inputs in [0, 1]; squash the head output to make the
        # real production path (raw head logits) reachable in a controlled way.
        model.score = _SigmoidScore(model.score)
        input_ids = self._input_ids()
        labels = torch.rand(2, 2)

        output = model(input_ids=input_ids, labels=labels)
        expected = nn.BCELoss()(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    def test_forward_loss_function_module_instance(self):
        """An nn.Module loss instance is used as-is."""
        focal = FocalLoss()
        model = _build_wrapper(
            head="basic-mlp", num_labels=2, task_type="regression", loss_function=focal
        )
        input_ids = self._input_ids()
        labels = torch.rand(2, 2)

        output = model(input_ids=input_ids, labels=labels)
        expected = focal(output.logits, labels)

        assert torch.allclose(output.loss, expected)

    def test_forward_loss_function_unsupported_string_raises(self):
        """An unknown loss name raises ValueError."""
        model = _build_wrapper(
            head="basic-mlp", num_labels=2, task_type="regression", loss_function="bogus"
        )

        with pytest.raises(ValueError, match="Unsupported loss function"):
            model(input_ids=self._input_ids(), labels=torch.rand(2, 2))

    def test_forward_loss_function_invalid_type_raises(self):
        """A non-string non-module loss raises ValueError."""
        model = _build_wrapper(
            head="basic-mlp", num_labels=2, task_type="regression", loss_function=42
        )

        with pytest.raises(ValueError, match=r"Loss function must be a string or an nn\.Module"):
            model(input_ids=self._input_ids(), labels=torch.rand(2, 2))

    def test_forward_cosine_similarity_loss_crashes(self):
        """cosine_similarity builds CosineEmbeddingLoss which needs a target arg.

        The handler passes only (logits, labels), so the call raises TypeError —
        a latent bug recorded here rather than silently skipped.
        """
        model = _build_wrapper(
            head="basic-mlp",
            num_labels=2,
            task_type="regression",
            loss_function="cosine_similarity",
        )

        with pytest.raises(TypeError):
            model(input_ids=self._input_ids(), labels=torch.rand(2, 2))
