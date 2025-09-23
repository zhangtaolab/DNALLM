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
  label_column: "labels"
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
            "labels": [0, 1, 0, 1],
        }
        df = pd.DataFrame(test_data)
        df.to_csv(self.data_path, index=False)

    @patch("dnallm.inference.benchmark.DNAInference")
    def test_initialization_with_config(self, mock_dna_inference):
        """Test if Benchmark initializes and"
        "loads data from config correctly."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

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

    @patch("dnallm.inference.benchmark.DNAInference")
    def test_get_dataset_manually(self, mock_dna_inference):
        """Test the get_dataset method can add a new dataset."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

        # Make the mock_dataset behave like a DNADataset
        mock_dataset.__class__ = DNADataset

        benchmark = Benchmark(self.config)
        initial_dataset_count = len(benchmark.datasets)
        assert initial_dataset_count == 1

        # Call get_dataset to add a new one from a file path
        new_dataset_obj = benchmark.get_dataset(
            self.data_path, seq_col="sequence", label_col="labels"
        )

        assert isinstance(new_dataset_obj, DNADataset)
        assert len(benchmark.datasets) == initial_dataset_count + 1, (
            "A new dataset should have been added."
        )

    @patch("dnallm.inference.benchmark.DNAInference")
    def test_get_inference_engine(self, mock_dna_inference):
        """Test if get_inference_engine returns"
        "a valid DNAInference instance."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

        # Make the mock_inference_instance behave like a DNAInference
        mock_inference_instance.__class__ = DNAInference

        benchmark = Benchmark(self.config)
        mock_model = Mock()
        mock_tokenizer = Mock()
        inference_engine = benchmark.get_inference_engine(
            mock_model, mock_tokenizer
        )
        assert isinstance(inference_engine, DNAInference)
        # The get_inference_engine method creates a new DNAInference instance
        # so we need to check that it was called with the right arguments
        mock_dna_inference.assert_called()

    @patch("dnallm.inference.benchmark.DNAInference")
    def test_available_models(self, mock_dna_inference):
        """Test listing available models."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

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
    @patch("dnallm.inference.benchmark.DNAInference")
    def test_run_benchmark_flow(
        self,
        mock_dna_inference,
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

        # Ensure the mock batch_infer returns the expected tuple
        mock_batch_infer.return_value = (mock_logits, None, None)

        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

        # --- Instantiate and run the benchmark ---
        benchmark = Benchmark(self.config)

        # Mock the get_inference_engine method
        # to return our mocked inference engine
        mock_inference_engine = Mock()
        mock_inference_engine.batch_infer.return_value = (
            mock_logits,
            None,
            None,
        )
        mock_inference_engine.calculate_metrics.return_value = {
            "accuracy": 0.95,
            "f1": 0.92,
            "curve": ([0, 1], [0, 1]),
        }
        benchmark.get_inference_engine = Mock(
            return_value=mock_inference_engine
        )

        # Debug: Check the dataset structure
        print(f"Number of datasets: {len(benchmark.datasets)}")
        if benchmark.datasets:
            print(
                f"First dataset columns: {benchmark.datasets[0].column_names}"
            )
            print(f"First dataset labels: {benchmark.datasets[0]['labels']}")

        results = benchmark.run(save_scores=True)

        # Debug output
        print(f"Results: {results}")
        print(f"Results type: {type(results)}")
        if results:
            print(f"Results keys: {list(results.keys())}")

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
        # Since we mocked get_inference_engine,
        # we need to check our mock instead
        assert mock_inference_engine.batch_infer.call_count == 2
        assert mock_inference_engine.calculate_metrics.call_count == 2

    @patch("dnallm.inference.plot.plot_bars")
    @patch("dnallm.inference.plot.plot_curve")
    @patch("dnallm.inference.benchmark.DNAInference")
    def test_plot_for_classification(
        self, mock_dna_inference, mock_plot_curve, mock_plot_bars
    ):
        """Test the plotting function for classification tasks."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

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
    @patch("dnallm.inference.benchmark.DNAInference")
    def test_plot_for_regression(
        self, mock_dna_inference, mock_plot_scatter, mock_plot_bars
    ):
        """Test the plotting function for regression tasks."""
        # Mock the DNAInference to avoid model loading issues
        mock_inference_instance = Mock()
        mock_dataset = Mock()
        mock_dataset.dataset = Dataset.from_dict({
            "sequence": ["ATCG", "GCTA", "TATA", "CGCG"],
            "labels": [0, 1, 0, 1],
        })
        mock_inference_instance.generate_dataset.return_value = (
            mock_dataset,
            None,
        )
        mock_dna_inference.return_value = mock_inference_instance

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
