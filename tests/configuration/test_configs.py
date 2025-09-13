"""Test suite for configuration module.

This module contains comprehensive tests for all configuration classes
in the dnallm.configuration.configs module, including validation,
error handling, and edge cases.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from pydantic import ValidationError

from dnallm.configuration.configs import (
    BenchmarkConfig,
    BenchmarkInfoConfig,
    DatasetConfig,
    EvaluationConfig,
    InferenceConfig,
    ModelConfig,
    OutputConfig,
    TaskConfig,
    TrainingConfig,
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
        config = TaskConfig(
            task_type="multiclass", num_labels=3, label_names=["A", "B", "C"]
        )

        assert config.task_type == "multiclass"
        assert config.num_labels == 3
        assert config.label_names == ["A", "B", "C"]

    def test_multiclass_task_config_invalid_num_labels(self):
        """Test multiclass task configuration with invalid num_labels."""
        with pytest.raises(
            ValidationError, match="num_labels must be at least 2"
        ):
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
        config = TaskConfig(
            task_type="multilabel", num_labels=2, label_names=["tag1", "tag2"]
        )

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
        config = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )

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
        benchmark_info = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [
            DatasetConfig(
                name="dataset1", path="/path/to/dataset1", task="binary"
            )
        ]
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
        benchmark_info = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [
            DatasetConfig(
                name="dataset1", path="/path/to/dataset1", task="binary"
            )
        ]
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
        benchmark_info = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )
        models = [ModelConfig(name="model1", path="/path/to/model1")]
        datasets = [
            DatasetConfig(
                name="dataset1", path="/path/to/dataset1", task="binary"
            )
        ]
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
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
        benchmark_info = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )
        datasets = [
            DatasetConfig(
                name="dataset1", path="/path/to/dataset1", task="binary"
            )
        ]
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
        benchmark_info = BenchmarkInfoConfig(
            name="Test Benchmark", description="Test description"
        )
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
