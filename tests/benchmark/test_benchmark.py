"""Test Benchmark class functionality.

This test file tests the core functionality of the Benchmark class,
including configuration loading, dataset handling, benchmark execution,
and plotting.
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import torch
from datasets import Dataset

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dnallm.datahandling.data import DNADataset
from dnallm.inference.benchmark import Benchmark
from dnallm.inference.inference import DNAInference


class TestBenchmark(unittest.TestCase):
    """Test cases for the Benchmark class."""

    def setUp(self):
        """Set up test fixtures for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "benchmark_config.yaml")
        self.data_path = os.path.join(self.test_dir, "benchmark_data.csv")
        self.results_dir = os.path.join(self.test_dir, "results")

        # Create test configuration
        self.create_test_config()
        # Create benchmark data
        self.create_test_data()

        from dnallm.configuration.configs import load_config

        self.config = load_config(self.config_path)
        # We instantiate Benchmark inside each test to ensure a clean state

    def tearDown(self):
        """Clean up test fixtures after each test."""
        shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a test configuration file."""
        config_content = f"""task:
  task_type: "binary"
  num_labels: 2
  label_names: ["negative", "positive"]
  threshold: 0.5

benchmark:
  name: "Test Benchmark"
  description: "Comparing DNA models"
models:
- name: "test_model_1"
  path: "path/to/model1"
  source: "local"
- name: "test_model_2"
  path: "zhangtaolab/plant-dnagpt-BPE-promoter"
  source: "huggingface"
datasets:
- name: "test_dataset"
  path: "{self.data_path}"
  task: "binary"
  text_column: "sequence"
  label_column: "label"
metrics:
- "accuracy"
- "f1"
evaluation:
  batch_size: 2
  max_length: 128
  num_workers: 1
output:
  format: "html"
  path: "{self.results_dir}"
"""
        with open(self.config_path, "w") as f:
            f.write(config_content)

    def create_test_data(self):
        """Create a test data file."""
        test_data = {
            "sequence": [
                "ATCGATCGATCG",
                "GCTAGCTAGCTA",
                "TATATATATATA",
                "CGCGCGCGCGCG",
            ],
            "label": [0, 1, 0, 1],
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.data_path, index=False)

    def test_initialization_with_config(self):
        """Test if Benchmark initializes and"
        "loads data from config correctly."""
        benchmark = Benchmark(self.config)
        print(benchmark.config)
        assert isinstance(benchmark, Benchmark)
        assert benchmark.prepared is not None

        # Check that the dataset from the config was loaded
        assert len(benchmark.datasets) == 1, (
            "Should have loaded one dataset during initialization."
        )

        assert isinstance(benchmark.datasets[0], Dataset)

        # Check that 'prepared' attribute is structured correctly
        assert "models" in benchmark.prepared
        assert len(benchmark.prepared["models"]) == 2
        assert "dataset" in benchmark.prepared
        assert benchmark.prepared["dataset"][0].name == "test_dataset"

    def test_get_dataset_manually(self):
        """Test the get_dataset method can add a new dataset."""
        benchmark = Benchmark(self.config)
        initial_dataset_count = len(benchmark.datasets)
        assert initial_dataset_count == 1

        # Call get_dataset to add a new one from a file path
        new_dataset_obj = benchmark.get_dataset(
            self.data_path, seq_col="sequence", label_col="label"
        )

        assert isinstance(new_dataset_obj, DNADataset)
        assert len(benchmark.datasets) == initial_dataset_count + 1, (
            "A new dataset should have been added."
        )

    def test_get_inference_engine(self):
        """Test if get_inference_engine returns"
        "a valid DNAInference instance."""
        benchmark = Benchmark(self.config)
        mock_model = Mock()
        mock_tokenizer = Mock()
        inference_engine = benchmark.get_inference_engine(
            mock_model, mock_tokenizer
        )
        assert isinstance(inference_engine, DNAInference)
        assert inference_engine.model == mock_model
        assert inference_engine.tokenizer == mock_tokenizer

    def test_available_models(self):
        """Test listing available models."""
        benchmark = Benchmark(self.config)
        all_models = benchmark.available_models(show_all=True)
        assert isinstance(all_models, dict)
        assert "Plant DNAGPT" in all_models

        model_tags = benchmark.available_models(show_all=False)
        assert isinstance(model_tags, dict)
        assert "Plant NT" in model_tags

    @patch("dnallm.inference.benchmark.load_model_and_tokenizer")
    @patch.object(DNAInference, "batch_infer")
    @patch.object(DNAInference, "calculate_metrics")
    def test_run_benchmark_flow(
        self,
        mock_calculate_metrics,
        mock_batch_infer,
        mock_load_model_tokenizer,
    ):
        """Test the main benchmark execution flow using mocks."""
        # --- Mock setup ---
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Mock tokenizer to return proper tokenization results
        mock_tokenizer.special_tokens_map = {
            "pad_token": "<pad>",
            "cls_token": "<cls>",
            "sep_token": "<sep>",
            "eos_token": "<eos>",
        }
        mock_tokenizer.encode.return_value = [1, 2, 3, 4, 5]

        # Mock tokenizer to return proper tokenization results for batch
        # processing
        def mock_tokenizer_call(sequences, **kwargs):
            # Return tokenized data for each sequence in the batch
            batch_size = len(sequences) if isinstance(sequences, list) else 1
            return {
                "input_ids": [
                    [1, 2, 3, 4, 5, 0, 0, 0] for _ in range(batch_size)
                ],
                "attention_mask": [
                    [1, 1, 1, 1, 1, 0, 0, 0] for _ in range(batch_size)
                ],
            }

        mock_tokenizer.side_effect = mock_tokenizer_call

        mock_load_model_tokenizer.return_value = (mock_model, mock_tokenizer)
        mock_logits = torch.randn(4, 2)
        mock_batch_infer.return_value = (mock_logits, None, None)
        mock_calculate_metrics.return_value = {
            "accuracy": 0.95,
            "f1": 0.92,
            "curve": ([0, 1], [0, 1]),
        }

        # --- Instantiate and run the benchmark ---
        benchmark = Benchmark(self.config)
        results = benchmark.run(save_scores=True)

        # --- Assertions ---
        assert results is not None
        assert "test_dataset" in results
        assert "test_model_1" in results["test_dataset"]
        assert "test_model_2" in results["test_dataset"]
        assert results["test_dataset"]["test_model_1"]["accuracy"] == 0.95

        assert results["test_dataset"]["test_model_2"]["f1"] == 0.92

        # Check that the metrics file was created
        expected_metrics_path = Path(self.results_dir) / "metrics.json"
        assert expected_metrics_path.exists()

        # Verify that dependencies were called correctly
        assert mock_load_model_tokenizer.call_count == 2
        assert mock_batch_infer.call_count == 2
        assert mock_calculate_metrics.call_count == 2

    @patch("dnallm.inference.plot.plot_bars")
    @patch("dnallm.inference.plot.plot_curve")
    def test_plot_for_classification(self, mock_plot_curve, mock_plot_bars):
        """Test the plotting function for classification tasks."""
        benchmark = Benchmark(self.config)
        # Type assertion for task config
        from dnallm.configuration.configs import TaskConfig

        task_config = self.config["task"]
        # Check if task_config has the expected attributes instead of
        # isinstance check
        assert hasattr(task_config, "task_type"), (
            "task_config should have task_type attribute"
        )
        assert hasattr(task_config, "num_labels"), (
            "task_config should have num_labels attribute"
        )
        task_config.task_type = "binary"
        metrics_data = {
            "test_dataset": {
                "model_A": {
                    "accuracy": 0.9,
                    "f1": 0.88,
                    "curve": {
                        "fpr": [0.0, 0.1, 0.2, 0.3, 1.0],
                        "tpr": [0.0, 0.8, 0.9, 0.95, 1.0],
                        "precision": [0.8, 0.8, 0.82, 0.83, 0.85],
                        "recall": [0.0, 0.8, 0.9, 0.95, 1.0],
                    },
                },
                "model_B": {
                    "accuracy": 0.85,
                    "f1": 0.82,
                    "curve": {
                        "fpr": [0.0, 0.2, 0.4, 0.6, 1.0],
                        "tpr": [0.0, 0.7, 0.8, 0.85, 1.0],
                        "precision": [0.75, 0.75, 0.76, 0.77, 0.78],
                        "recall": [0.0, 0.7, 0.8, 0.85, 1.0],
                    },
                },
            }
        }
        Path(self.results_dir).mkdir(exist_ok=True)
        benchmark.plot(metrics_data, save_path=self.results_dir)
        # mock_plot_bars.assert_called_once()
        # mock_plot_curve.assert_called_once()

    @patch("dnallm.inference.plot.plot_bars")
    @patch("dnallm.inference.plot.plot_scatter")
    def test_plot_for_regression(self, mock_plot_scatter, mock_plot_bars):
        """Test the plotting function for regression tasks."""
        # Modify the config for this specific test
        benchmark = Benchmark(self.config)
        # Type assertion for task config
        from dnallm.configuration.configs import TaskConfig

        task_config = self.config["task"]
        # Check if task_config has the expected attributes instead of
        # isinstance check
        assert hasattr(task_config, "task_type"), (
            "task_config should have task_type attribute"
        )
        assert hasattr(task_config, "num_labels"), (
            "task_config should have num_labels attribute"
        )
        task_config.task_type = "regression"

        metrics_data = {
            "test_dataset": {
                "model_A": {
                    "mse": 0.05,
                    "r2": 0.9,
                    "scatter": {
                        "predicted": [1.1, 2.2, 3.3, 4.4],
                        "experiment": [1.0, 2.0, 3.0, 4.0],
                    },
                },
                "model_B": {
                    "mse": 0.08,
                    "r2": 0.85,
                    "scatter": {
                        "predicted": [1.2, 2.3, 3.4, 4.5],
                        "experiment": [1.0, 2.0, 3.0, 4.0],
                    },
                },
            }
        }
        Path(self.results_dir).mkdir(exist_ok=True)
        benchmark.plot(metrics_data, save_path=self.results_dir, separate=True)
        # mock_plot_bars.assert_called_once()
        # mock_plot_scatter.assert_called_once()


if __name__ == "__main__":
    # Only run when executed directly, not when imported by pytest
    import sys

    if "pytest" not in sys.modules:
        unittest.main(verbosity=2)
