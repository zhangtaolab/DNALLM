"""Test suite for configuration module.

This module contains comprehensive tests for all configuration classes
in the dnallm.configuration.configs module, including validation,
error handling, and edge cases.
"""

import os
import tempfile
import typing
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import BaseModel, ValidationError

from dnallm.configuration.configs import (
    BenchmarkConfig,
    BenchmarkInfoConfig,
    DatasetConfig,
    EarlyStoppingConfig,
    EvaluationConfig,
    Ia3Config,
    InferenceConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    SweepConfig,
    TaskConfig,
    TrainingConfig,
    VepConfig,
    HyperparameterSearchConfig,
    SearchSpaceDistribution,
    load_config,
)


class TestTaskConfig:
    """Test cases for TaskConfig class."""

    def test_binary_task_config_default(self):
        """Test binary task configuration with defaults."""
        config = TaskConfig(task_type="binary")

        assert config.task_type == "binary"
        assert config.num_labels == 2
        assert config.label_names == ["negative", "positive"]
        assert config.threshold == 0.5
        assert config.mlm_probability == 0.15

    def test_binary_task_config_custom(self):
        """Test binary task configuration with custom values."""
        config = TaskConfig(
            task_type="binary",
            num_labels=3,
            label_names=["neg", "pos", "neutral"],
            threshold=0.7,
            mlm_probability=0.2,
        )

        assert config.task_type == "binary"
        assert config.num_labels == 3
        assert config.label_names == ["neg", "pos", "neutral"]
        assert config.threshold == 0.7
        assert config.mlm_probability == 0.2

    def test_multiclass_task_config_default(self):
        """Test multiclass task configuration with defaults."""
        config = TaskConfig(task_type="multiclass", num_labels=3)

        assert config.task_type == "multiclass"
        assert config.num_labels == 3
        assert config.label_names == ["class_0", "class_1", "class_2"]
        assert config.threshold == 0.5

    def test_multiclass_task_config_custom_labels(self):
        """Test multiclass task configuration with custom labels."""
        config = TaskConfig(task_type="multiclass", num_labels=3, label_names=["A", "B", "C"])

        assert config.task_type == "multiclass"
        assert config.num_labels == 3
        assert config.label_names == ["A", "B", "C"]

    def test_multiclass_task_config_invalid_num_labels(self):
        """Test multiclass task configuration with invalid num_labels."""
        with pytest.raises(ValidationError, match="num_labels must be at least 2"):
            TaskConfig(task_type="multiclass", num_labels=1)

    def test_multilabel_task_config_default(self):
        """Test multilabel task configuration with defaults."""
        config = TaskConfig(task_type="multilabel", num_labels=4)

        assert config.task_type == "multilabel"
        assert config.num_labels == 4
        assert config.label_names == [
            "label_0",
            "label_1",
            "label_2",
            "label_3",
        ]

    def test_multilabel_task_config_custom_labels(self):
        """Test multilabel task configuration with custom labels."""
        config = TaskConfig(task_type="multilabel", num_labels=2, label_names=["tag1", "tag2"])

        assert config.task_type == "multilabel"
        assert config.num_labels == 2
        assert config.label_names == ["tag1", "tag2"]

    def test_regression_task_config(self):
        """Test regression task configuration."""
        config = TaskConfig(task_type="regression")

        assert config.task_type == "regression"
        assert config.num_labels == 1
        assert config.label_names == ["value"]

    def test_mask_task_config(self):
        """Test mask task configuration."""
        config = TaskConfig(task_type="mask")

        assert config.task_type == "mask"
        assert config.num_labels is None
        assert config.label_names is None

    def test_generation_task_config(self):
        """Test generation task configuration."""
        config = TaskConfig(task_type="generation")

        assert config.task_type == "generation"
        assert config.num_labels is None
        assert config.label_names is None

    def test_embedding_task_config(self):
        """Test embedding task configuration."""
        config = TaskConfig(task_type="embedding")

        assert config.task_type == "embedding"
        assert config.num_labels == 2  # default value
        assert config.label_names is None

    def test_invalid_task_type(self):
        """Test invalid task type raises ValidationError."""
        with pytest.raises(ValidationError):
            TaskConfig(task_type="invalid_task")

    def test_task_type_pattern_validation(self):
        """Test task type pattern validation."""
        valid_types = [
            "embedding",
            "mask",
            "generation",
            "binary",
            "multiclass",
            "multilabel",
            "regression",
            "token",
        ]

        for task_type in valid_types:
            config = TaskConfig(task_type=task_type)
            assert config.task_type == task_type

    def test_threshold_validation(self):
        """Test threshold field validation."""
        config = TaskConfig(task_type="binary", threshold=0.8)
        assert config.threshold == 0.8

    def test_mlm_probability_validation(self):
        """Test MLM probability field validation."""
        config = TaskConfig(task_type="mask", mlm_probability=0.25)
        assert config.mlm_probability == 0.25


class TestTrainingConfig:
    """Test cases for TrainingConfig class."""

    def test_training_config_defaults(self):
        """Test training configuration with default values."""
        config = TrainingConfig()

        assert config.num_train_epochs == 3
        assert config.per_device_train_batch_size == 8
        assert config.per_device_eval_batch_size == 16
        assert config.learning_rate == 5e-5
        assert config.weight_decay == 0.01
        assert config.seed == 42
        assert config.bf16 is False
        assert config.fp16 is False

    def test_training_config_custom_values(self):
        """Test training configuration with custom values."""
        config = TrainingConfig(
            num_train_epochs=5,
            per_device_train_batch_size=16,
            learning_rate=1e-4,
            weight_decay=0.001,
            seed=123,
            fp16=True,
        )

        assert config.num_train_epochs == 5
        assert config.per_device_train_batch_size == 16
        assert config.learning_rate == 1e-4
        assert config.weight_decay == 0.001
        assert config.seed == 123
        assert config.fp16 is True

    def test_training_config_optional_fields(self):
        """Test training configuration optional fields."""
        config = TrainingConfig(
            output_dir="/tmp/test",
            max_steps=1000,
            resume_from_checkpoint="/tmp/checkpoint",
        )

        assert config.output_dir == "/tmp/test"
        assert config.max_steps == 1000
        assert config.resume_from_checkpoint == "/tmp/checkpoint"

    def test_training_config_lr_scheduler_kwargs(self):
        """Test learning rate scheduler kwargs."""
        lr_kwargs = {"warmup_steps": 100, "num_training_steps": 1000}
        config = TrainingConfig(lr_scheduler_kwargs=lr_kwargs)

        assert config.lr_scheduler_kwargs == lr_kwargs

    def test_training_config_early_stopping_defaults(self):
        """Test early stopping configuration with defaults."""
        config = TrainingConfig()
        assert config.callbacks is not None
        assert config.callbacks.early_stopping is not None
        assert config.callbacks.early_stopping.patience is None
        assert config.callbacks.early_stopping.threshold == 0.0

    def test_training_config_early_stopping_custom(self):
        """Test early stopping configuration with custom values."""
        from dnallm.configuration.configs import CallbackConfig, EarlyStoppingConfig

        config = TrainingConfig(
            callbacks=CallbackConfig(early_stopping=EarlyStoppingConfig(patience=3, threshold=0.01))
        )
        assert config.callbacks.early_stopping.patience == 3
        assert config.callbacks.early_stopping.threshold == 0.01

    def test_training_config_early_stopping_negative_patience(self):
        """Test early stopping with negative patience raises error."""
        from dnallm.configuration.configs import CallbackConfig, EarlyStoppingConfig

        with pytest.raises(ValidationError):
            TrainingConfig(
                callbacks=CallbackConfig(early_stopping=EarlyStoppingConfig(patience=-1))
            )

    def test_training_config_max_grad_norm(self):
        """Test gradient clipping threshold propagation."""
        config = TrainingConfig(max_grad_norm=0.5)
        assert config.max_grad_norm == 0.5

    def test_training_config_max_grad_norm_propagates_to_training_arguments(self):
        """Test that max_grad_norm is passed to TrainingArguments."""
        from transformers import TrainingArguments

        config = TrainingConfig(max_grad_norm=0.5)
        args = TrainingArguments(
            output_dir="/tmp/test",
            max_grad_norm=config.max_grad_norm,
        )
        assert args.max_grad_norm == 0.5

    def test_training_config_report_to_default(self):
        """Test default report_to is tensorboard."""
        config = TrainingConfig()
        assert config.report_to == ["tensorboard"]

    def test_training_config_report_to_string_coercion(self):
        """Test that string report_to is coerced to list."""
        config = TrainingConfig(report_to="wandb")
        assert config.report_to == ["wandb"]

    def test_training_config_report_to_multiple(self):
        """Test multiple trackers."""
        config = TrainingConfig(report_to=["tensorboard", "wandb"])
        assert config.report_to == ["tensorboard", "wandb"]

    def test_training_config_report_to_none(self):
        """Test disabling all trackers."""
        config = TrainingConfig(report_to=["none"])
        assert config.report_to == ["none"]

    def test_training_config_report_to_invalid(self):
        """Test invalid tracker name raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(report_to=["invalid_tracker"])

    def test_training_config_report_to_none_with_others(self):
        """Test that 'none' combined with other trackers raises error."""
        with pytest.raises(ValidationError):
            TrainingConfig(report_to=["none", "tensorboard"])


class TestInferenceConfig:
    """Test cases for InferenceConfig class."""

    def test_inference_config_defaults(self):
        """Test inference configuration with default values."""
        config = InferenceConfig()

        assert config.batch_size == 16
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.num_workers == 4
        assert config.use_fp16 is False

    def test_inference_config_custom_values(self):
        """Test inference configuration with custom values."""
        config = InferenceConfig(
            batch_size=32,
            max_length=1024,
            device="cuda",
            num_workers=8,
            use_fp16=True,
            output_dir="/tmp/inference",
        )

        assert config.batch_size == 32
        assert config.max_length == 1024
        assert config.device == "cuda"
        assert config.num_workers == 8
        assert config.use_fp16 is True
        assert config.output_dir == "/tmp/inference"


class TestBenchmarkInfoConfig:
    """Test cases for BenchmarkInfoConfig class."""

    def test_benchmark_info_config_required_fields(self):
        """Test benchmark info configuration with required fields."""
        config = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")

        assert config.name == "Test Benchmark"
        assert config.description == "Test description"

    def test_benchmark_info_config_with_description(self):
        """Test benchmark info configuration with description."""
        config = BenchmarkInfoConfig(
            name="Test Benchmark",
            description="A test benchmark for DNA models",
        )

        assert config.name == "Test Benchmark"
        assert config.description == "A test benchmark for DNA models"


class TestModelConfig:
    """Test cases for ModelConfig class."""

    def test_model_config_required_fields(self):
        """Test model configuration with required fields."""
        config = ModelConfig(name="test_model", path="/path/to/model")

        assert config.name == "test_model"
        assert config.path == "/path/to/model"
        assert config.source == "huggingface"
        assert config.task_type == "classification"
        assert config.revision == "main"
        assert config.trust_remote_code is True
        assert config.torch_dtype == "float32"

    def test_model_config_custom_values(self):
        """Test model configuration with custom values."""
        config = ModelConfig(
            name="custom_model",
            path="huggingface/model-name",
            source="huggingface",
            task_type="regression",
            revision="v1.0",
            trust_remote_code=False,
            torch_dtype="float16",
        )

        assert config.name == "custom_model"
        assert config.path == "huggingface/model-name"
        assert config.source == "huggingface"
        assert config.task_type == "regression"
        assert config.revision == "v1.0"
        assert config.trust_remote_code is False
        assert config.torch_dtype == "float16"


class TestDatasetConfig:
    """Test cases for DatasetConfig class."""

    def test_dataset_config_required_fields(self):
        """Test dataset configuration with required fields."""
        config = DatasetConfig(
            name="test_dataset",
            path="/path/to/dataset.csv",
            task="binary_classification",
        )

        assert config.name == "test_dataset"
        assert config.path == "/path/to/dataset.csv"
        assert config.task == "binary_classification"
        assert config.format == "csv"
        assert config.text_column == "sequence"
        assert config.label_column == "label"
        assert config.max_length == 512

    def test_dataset_config_custom_values(self):
        """Test dataset configuration with custom values."""
        config = DatasetConfig(
            name="custom_dataset",
            path="/path/to/dataset.json",
            task="multiclass_classification",
            format="json",
            text_column="text",
            label_column="labels",
            max_length=1024,
            test_size=0.3,
            val_size=0.2,
            random_state=123,
            threshold=0.6,
            num_labels=5,
            label_names=["A", "B", "C", "D", "E"],
        )

        assert config.name == "custom_dataset"
        assert config.path == "/path/to/dataset.json"
        assert config.task == "multiclass_classification"
        assert config.format == "json"
        assert config.text_column == "text"
        assert config.label_column == "labels"
        assert config.max_length == 1024
        assert config.test_size == 0.3
        assert config.val_size == 0.2
        assert config.random_state == 123
        assert config.threshold == 0.6
        assert config.num_labels == 5
        assert config.label_names == ["A", "B", "C", "D", "E"]


class TestEvaluationConfig:
    """Test cases for EvaluationConfig class."""

    def test_evaluation_config_defaults(self):
        """Test evaluation configuration with default values."""
        config = EvaluationConfig()

        assert config.batch_size == 32
        assert config.max_length == 512
        assert config.device == "auto"
        assert config.num_workers == 4
        assert config.use_fp16 is False
        assert config.use_bf16 is False
        assert config.mixed_precision is True
        assert config.pin_memory is True
        assert config.memory_efficient_attention is False
        assert config.seed == 42
        assert config.deterministic is True

    def test_evaluation_config_custom_values(self):
        """Test evaluation configuration with custom values."""
        config = EvaluationConfig(
            batch_size=64,
            max_length=1024,
            device="cuda",
            num_workers=8,
            use_fp16=True,
            use_bf16=False,
            mixed_precision=False,
            pin_memory=False,
            memory_efficient_attention=True,
            seed=456,
            deterministic=False,
        )

        assert config.batch_size == 64
        assert config.max_length == 1024
        assert config.device == "cuda"
        assert config.num_workers == 8
        assert config.use_fp16 is True
        assert config.use_bf16 is False
        assert config.mixed_precision is False
        assert config.pin_memory is False
        assert config.memory_efficient_attention is True
        assert config.seed == 456
        assert config.deterministic is False


class TestOutputConfig:
    """Test cases for OutputConfig class."""

    def test_output_config_defaults(self):
        """Test output configuration with default values."""
        config = OutputConfig()

        assert config.path == "benchmark_results"
        assert config.format == "html"
        assert config.save_predictions is True
        assert config.save_embeddings is False
        assert config.save_attention_maps is False
        assert config.generate_plots is True
        assert config.report_title == "DNA Model Benchmark Report"
        assert config.include_summary is True
        assert config.include_details is True
        assert config.include_recommendations is True

    def test_output_config_custom_values(self):
        """Test output configuration with custom values."""
        config = OutputConfig(
            path="/tmp/results",
            format="json",
            save_predictions=False,
            save_embeddings=True,
            save_attention_maps=True,
            generate_plots=False,
            report_title="Custom Report",
            include_summary=False,
            include_details=False,
            include_recommendations=False,
        )

        assert config.path == "/tmp/results"
        assert config.format == "json"
        assert config.save_predictions is False
        assert config.save_embeddings is True
        assert config.save_attention_maps is True
        assert config.generate_plots is False
        assert config.report_title == "Custom Report"
        assert config.include_summary is False
        assert config.include_details is False
        assert config.include_recommendations is False


class TestBenchmarkConfig:
    """Test cases for BenchmarkConfig class."""

    def test_benchmark_config_minimal(self):
        """Test benchmark configuration with minimal required fields."""
        benchmark_info = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [DatasetConfig(name="dataset1", path="/path/to/dataset1", task="binary")]
        output = OutputConfig()

        config = BenchmarkConfig(
            benchmark=benchmark_info,
            models=models,
            datasets=datasets,
            output=output,
            metrics=None,
        )

        assert config.benchmark.name == "Test Benchmark"
        assert len(config.models) == 1
        assert len(config.datasets) == 1
        assert config.output.path == "benchmark_results"
        assert config.metrics is None

    def test_benchmark_config_with_metrics(self):
        """Test benchmark configuration with metrics."""
        benchmark_info = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [DatasetConfig(name="dataset1", path="/path/to/dataset1", task="binary")]
        output = OutputConfig()
        metrics = ["accuracy", "f1", "precision", "recall"]

        config = BenchmarkConfig(
            benchmark=benchmark_info,
            models=models,
            datasets=datasets,
            output=output,
            metrics=metrics,
        )

        assert config.metrics == metrics

    def test_benchmark_config_with_evaluation(self):
        """Test benchmark configuration with custom evaluation settings."""
        benchmark_info = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [DatasetConfig(name="dataset1", path="/path/to/dataset1", task="binary")]
        output = OutputConfig()
        evaluation = EvaluationConfig(batch_size=64, device="cuda")

        config = BenchmarkConfig(
            benchmark=benchmark_info,
            models=models,
            datasets=datasets,
            output=output,
            evaluation=evaluation,
            metrics=None,
        )

        assert config.evaluation.batch_size == 64
        assert config.evaluation.device == "cuda"


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_config_task_only(self):
        """Test loading configuration with only task config."""
        config_data = {
            "task": {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["neg", "pos"],
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            configs = load_config(config_path)

            assert "task" in configs
            assert isinstance(configs["task"], TaskConfig)
            assert configs["task"].task_type == "binary"
            assert configs["task"].num_labels == 2
            assert configs["task"].label_names == ["neg", "pos"]
            # config_path is added to config_dict but not returned in configs
            # This test documents the current behavior
        finally:
            os.unlink(config_path)

    def test_load_config_inference_only(self):
        """Test loading configuration with only inference config."""
        config_data = {
            "inference": {
                "batch_size": 32,
                "max_length": 1024,
                "device": "cuda",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            configs = load_config(config_path)

            assert "inference" in configs
            assert isinstance(configs["inference"], InferenceConfig)
            assert configs["inference"].batch_size == 32
            assert configs["inference"].max_length == 1024
            assert configs["inference"].device == "cuda"
        finally:
            os.unlink(config_path)

    def test_load_config_training_only(self):
        """Test loading configuration with only training config."""
        config_data = {
            "finetune": {
                "num_train_epochs": 5,
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 16,
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            configs = load_config(config_path)

            assert "finetune" in configs
            assert isinstance(configs["finetune"], TrainingConfig)
            assert configs["finetune"].num_train_epochs == 5
            assert configs["finetune"].learning_rate == 1e-4
            assert configs["finetune"].per_device_train_batch_size == 16
        finally:
            os.unlink(config_path)

    def test_load_config_benchmark(self):
        """Test loading configuration with benchmark config."""
        config_data = {
            "benchmark": {
                "name": "Test Benchmark",
                "description": "A test benchmark",
            },
            "models": [{"name": "model1", "path": "/path/to/model1"}],
            "datasets": [
                {
                    "name": "dataset1",
                    "path": "/path/to/dataset1",
                    "task": "binary_classification",
                }
            ],
            "output": {"path": "/tmp/results", "format": "html"},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            configs = load_config(config_path)

            assert "benchmark" in configs
            assert isinstance(configs["benchmark"], BenchmarkConfig)
            assert configs["benchmark"].benchmark.name == "Test Benchmark"
            assert len(configs["benchmark"].models) == 1
            assert len(configs["benchmark"].datasets) == 1
        finally:
            os.unlink(config_path)

    def test_load_config_mixed(self):
        """Test loading configuration with multiple config types."""
        config_data = {
            "task": {"task_type": "multiclass", "num_labels": 3},
            "inference": {"batch_size": 16, "device": "cpu"},
            "finetune": {"num_train_epochs": 3, "learning_rate": 5e-5},
            "model": {
                "model_name": "test-model",
                "model_path": "/path/to/model",
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            configs = load_config(config_path)

            assert "task" in configs
            assert "inference" in configs
            assert "finetune" in configs
            assert "model" in configs

            assert isinstance(configs["task"], TaskConfig)
            assert isinstance(configs["inference"], InferenceConfig)
            assert isinstance(configs["finetune"], TrainingConfig)
            # model stays as dict (not BaseModel)
            assert isinstance(configs["model"], dict)
        finally:
            os.unlink(config_path)

    def test_load_config_file_not_found(self):
        """Test loading configuration with non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_config("/non/existent/path.yaml")

    def test_load_config_invalid_yaml(self):
        """Test loading configuration with invalid YAML."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                load_config(config_path)
        finally:
            os.unlink(config_path)

    def test_load_config_invalid_task_type(self):
        """Test loading configuration with invalid task type."""
        config_data = {"task": {"task_type": "invalid_type"}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(config_path)
        finally:
            os.unlink(config_path)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_task_config_empty_label_names(self):
        """Test task config with empty label names list."""
        config = TaskConfig(task_type="binary", label_names=[])

        # Should use default label names
        assert config.label_names == ["negative", "positive"]

    def test_task_config_mismatched_labels_and_num_labels(self):
        """Test task config with mismatched label names and num_labels."""
        config = TaskConfig(
            task_type="multiclass",
            num_labels=3,
            label_names=["A", "B"],  # Only 2 labels for 3 classes
        )

        # Should generate default labels for missing ones
        assert config.label_names is not None
        assert len(config.label_names) == 3
        assert config.label_names == ["class_0", "class_1", "class_2"]

    def test_training_config_negative_values(self):
        """Test training config with negative values."""
        # Pydantic doesn't validate negative values by default for int fields
        # This test documents the current behavior
        config = TrainingConfig(num_train_epochs=-1)
        assert config.num_train_epochs == -1

    def test_inference_config_invalid_device(self):
        """Test inference config with invalid device."""
        # This should pass as device is just a string field
        config = InferenceConfig(device="invalid_device")
        assert config.device == "invalid_device"

    def test_benchmark_config_empty_models(self):
        """Test benchmark config with empty models list."""
        benchmark_info = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")
        datasets = [DatasetConfig(name="dataset1", path="/path/to/dataset1", task="binary")]
        output = OutputConfig()

        # Pydantic doesn't validate empty lists by default
        # This test documents the current behavior
        config = BenchmarkConfig(
            benchmark=benchmark_info,
            models=[],  # Empty models list
            datasets=datasets,
            output=output,
            metrics=None,
        )
        assert len(config.models) == 0

    def test_benchmark_config_empty_datasets(self):
        """Test benchmark config with empty datasets list."""
        benchmark_info = BenchmarkInfoConfig(name="Test Benchmark", description="Test description")
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        output = OutputConfig()

        # Pydantic doesn't validate empty lists by default
        # This test documents the current behavior
        config = BenchmarkConfig(
            benchmark=benchmark_info,
            models=models,
            datasets=[],  # Empty datasets list
            output=output,
            metrics=None,
        )
        assert len(config.datasets) == 0


class TestHyperparameterSearchConfig:
    """Test cases for HyperparameterSearchConfig and SearchSpaceDistribution."""

    def test_search_space_float_inference(self):
        """Test float type inference from value range."""
        dist = SearchSpaceDistribution(low=1e-6, high=1e-3)
        assert dist.type == "float"
        assert dist.log is True  # Auto-enabled for >10x range

    def test_search_space_int_inference(self):
        """Test int type inference from integer bounds."""
        dist = SearchSpaceDistribution(low=4, high=32, step=4)
        assert dist.type == "int"
        assert dist.log is False
        assert dist.step == 4

    def test_search_space_explicit_type(self):
        """Test explicit type override."""
        dist = SearchSpaceDistribution(low=1, high=10, type="float")
        assert dist.type == "float"
        assert dist.log is False  # 10x range, but explicit int was overridden

    def test_search_space_invalid_step_for_float(self):
        """Test that step for float raises error."""
        with pytest.raises(ValidationError):
            SearchSpaceDistribution(low=0.1, high=1.0, step=0.1)

    def test_search_space_invalid_range(self):
        """Test that low >= high raises error."""
        with pytest.raises(ValidationError):
            SearchSpaceDistribution(low=5, high=5)

    def test_hyperparameter_search_config_defaults(self):
        """Test default hyperparameter search config."""
        config = HyperparameterSearchConfig()
        assert config.n_trials == 0
        assert config.direction == "minimize"
        assert config.metric == "eval_loss"
        assert config.search_space == {}

    def test_hyperparameter_search_config_custom(self):
        """Test custom hyperparameter search config."""
        config = HyperparameterSearchConfig(
            search_space={
                "learning_rate": SearchSpaceDistribution(low=1e-6, high=1e-3),
            },
            n_trials=10,
            direction="maximize",
            metric="eval_accuracy",
        )
        assert config.n_trials == 10
        assert config.direction == "maximize"
        assert config.metric == "eval_accuracy"
        assert "learning_rate" in config.search_space

    def test_hyperparameter_search_config_invalid_direction(self):
        """Test invalid direction raises error."""
        with pytest.raises(ValidationError):
            HyperparameterSearchConfig(direction="invalid")

    def test_training_config_with_hyperparameter_search(self):
        """Test TrainingConfig integrates hyperparameter search."""
        config = TrainingConfig(
            hyperparameter_search=HyperparameterSearchConfig(
                search_space={
                    "learning_rate": SearchSpaceDistribution(low=1e-6, high=1e-3),
                },
                n_trials=3,
            )
        )
        assert config.hyperparameter_search.n_trials == 3
        assert config.hyperparameter_search.search_space["learning_rate"].type == "float"


class TestTaskConfigAliasNormalization:
    """model_post_init alias branches for verbose task_type spellings."""

    def test_binary_classification_alias_applies_binary_defaults(self):
        """'binary_classification' normalizes to the binary defaults."""
        config = TaskConfig(task_type="binary_classification")
        assert config.label_names == ["negative", "positive"]
        assert config.num_labels == 2

    def test_multi_class_classification_alias_generates_class_names(self):
        """'multi_class_classification' normalizes to multiclass name generation."""
        config = TaskConfig(task_type="multi_class_classification", num_labels=3)
        assert config.label_names == ["class_0", "class_1", "class_2"]

    def test_multi_label_classification_alias_generates_label_names(self):
        """'multi_label_classification' normalizes to multilabel name generation."""
        config = TaskConfig(task_type="multi_label_classification", num_labels=2)
        assert config.label_names == ["label_0", "label_1"]

    def test_multilabel_num_labels_below_two_rejected(self):
        """Multilabel classification requires at least two labels."""
        with pytest.raises(ValidationError, match="at least 2 for multilabel"):
            TaskConfig(task_type="multilabel", num_labels=1)

    def test_token_classification_alias_constructs_without_defaults(self):
        """'token_classification' normalizes to token, which sets no defaults."""
        config = TaskConfig(task_type="token_classification")
        assert config.task_type == "token_classification"  # stored verbatim
        assert config.label_names is None
        assert config.num_labels == 2  # untouched by the post-init chain


class TestEarlyStoppingValidation:
    """EarlyStoppingConfig field validators."""

    def test_negative_threshold_rejected(self):
        """A negative improvement threshold is rejected."""
        with pytest.raises(ValidationError, match="threshold must be non-negative"):
            EarlyStoppingConfig(threshold=-0.1)


class TestSearchSpaceDistributionEdges:
    """SearchSpaceDistribution validator branches not covered elsewhere."""

    def test_int_step_on_float_distribution_rejected(self):
        """step is only valid for int distributions (an int step on floats fails)."""
        with pytest.raises(ValidationError, match="'step' is only valid for int"):
            SearchSpaceDistribution(low=0.1, high=1.0, step=2)

    def test_log_requires_positive_low(self):
        """log=True with a non-positive low is rejected."""
        with pytest.raises(ValidationError, match="low must be positive when log=True"):
            SearchSpaceDistribution(low=0, high=1, log=True)


class TestTrainingConfigReportToEdges:
    """TrainingConfig report_to validator combinations."""

    def test_all_cannot_be_combined_with_other_trackers(self):
        """'all' mixed with a concrete tracker is rejected."""
        with pytest.raises(ValidationError, match="'all' cannot be combined"):
            TrainingConfig(report_to=["all", "wandb"])


class TestTrainingConfigScaffoldFields:
    """EVAL-01/PEFT scaffold fields on TrainingConfig (Phase 10 one-pass)."""

    def test_allow_test_as_eval_defaults_false(self):
        """The test-as-eval opt-in is off by default (leak guard)."""
        config = TrainingConfig()
        assert config.allow_test_as_eval is False

    def test_use_ia3_defaults_false(self):
        """use_ia3 lands with a False default."""
        config = TrainingConfig()
        assert config.use_ia3 is False

    def test_use_ia3_with_use_qlora_rejected_naming_both_fields(self):
        """use_ia3 x use_qlora is rejected at Pydantic time with a message
        naming both config fields (PEFT-01; peft only raises at merge time)."""
        with pytest.raises(ValidationError, match="use_ia3"):
            TrainingConfig(use_ia3=True, use_qlora=True)

        with pytest.raises(ValidationError, match="use_qlora"):
            TrainingConfig(use_ia3=True, use_qlora=True)

    def test_use_ia3_with_use_qlora_rejection_names_the_cause(self):
        """The rejection message states the 4-bit merge limitation (matchable)."""
        with pytest.raises(ValidationError, match="4-bit"):
            TrainingConfig(use_ia3=True, use_qlora=True)

    def test_use_ia3_and_use_qlora_alone_still_construct(self):
        """Each flag on its own is a valid configuration."""
        assert TrainingConfig(use_ia3=True).use_ia3 is True
        assert TrainingConfig(use_qlora=True).use_qlora is True

    def test_peft_dry_run_defaults_false(self):
        """peft_dry_run defaults to False (training runs normally)."""
        config = TrainingConfig()
        assert config.peft_dry_run is False

    def test_peft_dry_run_settable_from_plain_kwargs(self):
        """The field is constructible from kwargs as YAML section data would be."""
        config = TrainingConfig(peft_dry_run=True)
        assert config.peft_dry_run is True

    def test_allow_test_as_eval_settable_from_plain_kwargs(self):
        """The field is constructible from kwargs as YAML section data would be."""
        config = TrainingConfig(allow_test_as_eval=True)
        assert config.allow_test_as_eval is True


class TestIa3Config:
    """Ia3Config stub section: field-complete, peft-IA3Config-mirrored."""

    def test_ia3_config_defaults(self):
        """Default instantiation yields the peft-compatible field set."""
        config = Ia3Config()

        assert config.target_modules is None
        assert config.exclude_modules is None
        assert config.feedforward_modules is None
        assert config.fan_in_fan_out is False
        assert config.init_ia3_weights is True
        assert config.modules_to_save is None
        assert config.task_type == "SEQ_CLS"

    def test_ia3_config_custom_values(self):
        """All six fields accept explicit values."""
        config = Ia3Config(
            target_modules=["query", "key"],
            exclude_modules=["classifier"],
            feedforward_modules=["key"],
            fan_in_fan_out=True,
            init_ia3_weights=False,
            modules_to_save=["classifier"],
        )

        assert config.target_modules == ["query", "key"]
        assert config.exclude_modules == ["classifier"]
        assert config.feedforward_modules == ["key"]
        assert config.fan_in_fan_out is True
        assert config.init_ia3_weights is False
        assert config.modules_to_save == ["classifier"]

    def test_ia3_config_dump_fields_match_peft_surface(self):
        """Every dnallm Ia3Config field name is accepted by peft's IA3Config
        (full pass-through; nothing is dropped by the trainer's filter)."""
        from dnallm.finetune.trainer import PEFT_IA3_FIELD_NAMES

        assert set(Ia3Config().model_dump()) <= PEFT_IA3_FIELD_NAMES


class TestVepConfig:
    """VepConfig stub section for the zero-shot VEP module."""

    def test_vep_config_defaults(self):
        """Default instantiation yields mlm / 200bp window / no output dir."""
        config = VepConfig()

        assert config.paradigm == "mlm"
        assert config.context_window == 200
        assert config.output_dir is None

    def test_vep_config_custom_values(self):
        """Explicit clm paradigm and window are accepted."""
        config = VepConfig(paradigm="clm", context_window=1000, output_dir="/tmp/vep")

        assert config.paradigm == "clm"
        assert config.context_window == 1000
        assert config.output_dir == "/tmp/vep"

    def test_vep_config_invalid_paradigm_rejected(self):
        """A non-(clm|mlm) paradigm is rejected at validation time."""
        with pytest.raises(ValidationError):
            VepConfig(paradigm="bogus")

    def test_vep_config_nonpositive_window_rejected(self):
        """context_window is ge=1: zero is rejected."""
        with pytest.raises(ValidationError):
            VepConfig(context_window=0)


class TestSweepConfig:
    """SweepConfig stub section for the multi-seed sweep protocol."""

    def test_sweep_config_defaults(self):
        """Defaults: 3 seeds, 2000 bootstrap resamples, t-interval small-n policy."""
        config = SweepConfig()

        assert config.seeds == [42, 43, 44]
        assert config.out_root is None
        assert config.n_bootstrap == 2000
        assert config.bootstrap_seed == 42
        assert config.small_n_ci == "t-interval"

    def test_sweep_config_custom_values(self):
        """Explicit seeds and CI policy are accepted."""
        config = SweepConfig(
            seeds=[7, 8],
            out_root="/tmp/sweep",
            n_bootstrap=500,
            bootstrap_seed=1,
            small_n_ci="omit",
        )

        assert config.seeds == [7, 8]
        assert config.out_root == "/tmp/sweep"
        assert config.n_bootstrap == 500
        assert config.bootstrap_seed == 1
        assert config.small_n_ci == "omit"

    def test_sweep_config_empty_seeds_rejected(self):
        """min_length=1: an empty seed list is rejected."""
        with pytest.raises(ValidationError):
            SweepConfig(seeds=[])

    def test_sweep_config_invalid_small_n_ci_rejected(self):
        """A non-(t-interval|omit) CI policy is rejected."""
        with pytest.raises(ValidationError):
            SweepConfig(small_n_ci="bogus")

    def test_sweep_config_nonpositive_bootstrap_rejected(self):
        """n_bootstrap is ge=1: zero is rejected."""
        with pytest.raises(ValidationError):
            SweepConfig(n_bootstrap=0)


class TestStubSectionLoadConfig:
    """load_config registration for the ia3/vep/sweep stub sections."""

    def _write_yaml(self, tmp_path, body):
        """Write a config YAML body to tmp_path and return its path."""
        cfg_path = tmp_path / "stub_config.yaml"
        cfg_path.write_text(body, encoding="utf-8")
        return str(cfg_path)

    def test_stub_sections_roundtrip_from_yaml(self, tmp_path):
        """A YAML carrying ia3/vep/sweep produces the three typed sections."""
        config_path = self._write_yaml(
            tmp_path,
            "ia3:\n"
            "  target_modules: [query, key]\n"
            "  feedforward_modules: [key]\n"
            "vep:\n"
            "  paradigm: clm\n"
            "  context_window: 512\n"
            "sweep:\n"
            "  seeds: [1, 2, 3, 4, 5]\n"
            "  n_bootstrap: 1000\n"
            "  small_n_ci: omit\n",
        )

        configs = load_config(config_path)

        assert isinstance(configs["ia3"], Ia3Config)
        assert configs["ia3"].target_modules == ["query", "key"]
        assert isinstance(configs["vep"], VepConfig)
        assert configs["vep"].paradigm == "clm"
        assert configs["vep"].context_window == 512
        assert isinstance(configs["sweep"], SweepConfig)
        assert configs["sweep"].seeds == [1, 2, 3, 4, 5]
        assert configs["sweep"].small_n_ci == "omit"

    def test_yaml_without_stubs_omits_keys(self, tmp_path):
        """A YAML without the sections leaves the keys absent (total=False)."""
        config_path = self._write_yaml(
            tmp_path, "task:\n  task_type: binary\nmodel:\n  name: dummy\n"
        )

        configs = load_config(config_path)

        for absent in ("ia3", "vep", "sweep"):
            assert absent not in configs


class TestLoadConfigTypedDict:
    """Pin the per-key DNALLMConfig TypedDict return contract of load_config()."""

    FULL_CONFIG_YAML = """\
task:
  task_type: binary
  num_labels: 2
inference:
  batch_size: 8
model:
  name: dummy-model
  source: huggingface
finetune:
  num_train_epochs: 1
lora:
  r: 8
"""

    def test_return_annotation_is_dnallmconfig_typed_dict(self):
        """get_type_hints(load_config)['return'] is the per-key DNALLMConfig TypedDict."""
        from dnallm.configuration.configs import DNALLMConfig

        return_hint = typing.get_type_hints(load_config)["return"]
        assert return_hint is DNALLMConfig

        annotations = dict(return_hint.__annotations__)
        assert annotations["task"] is TaskConfig
        assert annotations["inference"] is InferenceConfig
        assert annotations["finetune"] is TrainingConfig
        assert annotations["lora"] is LoraConfig
        assert annotations["ia3"] is Ia3Config
        assert annotations["vep"] is VepConfig
        assert annotations["sweep"] is SweepConfig
        assert annotations["benchmark"] is BenchmarkConfig
        # model stays a plain dict (spelling-tolerant: any dict[...] annotation form)
        assert typing.get_origin(annotations["model"]) is dict

    def test_typed_dict_is_total_false(self):
        """DNALLMConfig is total=False: every key optional, none required."""
        from dnallm.configuration.configs import DNALLMConfig

        assert DNALLMConfig.__required_keys__ == frozenset()
        assert DNALLMConfig.__optional_keys__ == frozenset({
            "task",
            "inference",
            "model",
            "finetune",
            "lora",
            "ia3",
            "vep",
            "sweep",
            "benchmark",
        })

    def test_loaded_fixture_yields_per_key_instances(self, tmp_path):
        """A full fixture YAML loads to per-key config classes; model stays a dict."""
        cfg_path = tmp_path / "full_config.yaml"
        cfg_path.write_text(self.FULL_CONFIG_YAML, encoding="utf-8")

        configs = load_config(str(cfg_path))

        assert isinstance(configs["task"], TaskConfig)
        assert isinstance(configs["inference"], InferenceConfig)
        assert isinstance(configs["finetune"], TrainingConfig)
        assert isinstance(configs["lora"], LoraConfig)
        assert isinstance(configs["model"], dict)
        assert not isinstance(configs["model"], BaseModel)
        assert "benchmark" not in configs

    def test_benchmark_fixture_yields_benchmark_config(self):
        """The example benchmark YAML produces a BenchmarkConfig under 'benchmark'."""
        benchmark_yaml = (
            Path(__file__).parent.parent.parent
            / "example"
            / "notebooks"
            / "benchmark"
            / "benchmark_config.yaml"
        )

        configs = load_config(str(benchmark_yaml))

        assert isinstance(configs["benchmark"], BenchmarkConfig)

    def test_minimal_task_model_yaml_loads(self, tmp_path):
        """total=False semantics: a YAML with only task+model loads without KeyError."""
        cfg_path = tmp_path / "minimal_config.yaml"
        cfg_path.write_text(
            "task:\n  task_type: binary\nmodel:\n  name: dummy-model\n",
            encoding="utf-8",
        )

        configs = load_config(str(cfg_path))

        assert isinstance(configs["task"], TaskConfig)
        assert isinstance(configs["model"], dict)
        for absent_key in ("inference", "finetune", "lora", "ia3", "vep", "sweep", "benchmark"):
            assert absent_key not in configs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
