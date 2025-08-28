"""Test DNATrainer class functionality.

This test file tests the core functionality of the DNATrainer class,
including model initialization, training setup, and basic operations.
"""

import os
import sys
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import torch
import numpy as np
from datasets import Dataset, DatasetDict

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from dnallm.finetune.trainer import DNATrainer
from dnallm.datahandling.data import DNADataset


class TestDNATrainer(unittest.TestCase):
    """Test cases for DNATrainer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.test_dir, "test_config.yaml")
        
        # Create test configuration
        self.create_test_config()
        
        # Mock model and tokenizer
        self.mock_model = self.create_mock_model()
        self.mock_tokenizer = self.create_mock_tokenizer()
        
        # Create test datasets
        self.mock_datasets = self.create_mock_datasets()

    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)

    def create_test_config(self):
        """Create a test configuration file."""
        config_content = """task:
  task_type: binary
  num_labels: 2
  label_names:
  - negative
  - positive
  threshold: 0.5
finetune:
  output_dir: ./test_output
  num_train_epochs: 1
  per_device_train_batch_size: 2
  per_device_eval_batch_size: 2
  learning_rate: 2.0e-05
  logging_steps: 10
  save_steps: 100
  eval_steps: 100
  save_strategy: steps
  evaluation_strategy: steps
  logging_strategy: steps
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  save_total_limit: 3
  warmup_ratio: 0.1
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-08
  report_to: none
  save_safetensors: true
"""
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def create_mock_model(self):
        """Create a mock model for testing."""
        mock_model = Mock()
        
        # Mock model config
        mock_config = Mock()
        mock_config.output_attentions = False
        mock_config.output_hidden_states = False
        mock_config.attn_implementation = "eager"
        mock_config.num_attention_heads = 12
        mock_config.num_hidden_layers = 6
        mock_config.hidden_size = 768
        mock_config.vocab_size = 1000
        mock_config.model_type = "dna_bert"
        mock_model.config = mock_config
        
        # Mock model parameters
        mock_model.parameters.return_value = iter([torch.randn(100, 100)])
        
        # Mock model methods
        mock_model.train = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        
        return mock_model

    def create_mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        mock_tokenizer = Mock()
        
        # Mock tokenizer methods
        mock_tokenizer.encode = Mock(return_value=[1, 2, 3, 4, 5])
        mock_tokenizer.encode_plus = Mock(return_value={
            'input_ids': [1, 2, 3, 4, 5],
            'attention_mask': [1, 1, 1, 1, 1],
            'token_type_ids': [0, 0, 0, 0, 0]
        })
        mock_tokenizer.save_pretrained = Mock()
        
        # Mock special tokens map
        mock_tokenizer.special_tokens_map = {
            'pad_token': '[PAD]',
            'unk_token': '[UNK]',
            'cls_token': '[CLS]',
            'sep_token': '[SEP]',
            'mask_token': '[MASK]'
        }
        
        # Mock tokenizer properties
        mock_tokenizer.pad_token = '[PAD]'
        mock_tokenizer.unk_token = '[UNK]'
        mock_tokenizer.cls_token = '[CLS]'
        mock_tokenizer.sep_token = '[SEP]'
        mock_tokenizer.mask_token = '[MASK]'
        
        return mock_tokenizer

    def create_mock_datasets(self):
        """Create mock datasets for testing."""
        # Create sample data
        sample_data = {
            "sequence": ["ATCGATCG", "GCTAGCTA", "TATATATA", "CGCGCGCG"],
            "labels": [0, 1, 0, 1]
        }
        dataset = Dataset.from_dict(sample_data)
        dataset_dict = DatasetDict({
            "train": dataset,
            "val": dataset
        })
        
        datasets = Mock(spec=DNADataset)
        datasets.dataset = dataset_dict
        datasets.tokenizer = self.mock_tokenizer
        return datasets

    def load_test_config(self):
        """Load test configuration."""
        from dnallm.configuration.configs import load_config
        return load_config(self.config_path)

    def test_init_basic(self):
        """Test basic initialization of DNATrainer."""
        config = self.load_test_config()
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        
        self.assertEqual(trainer.model, self.mock_model)
        self.assertEqual(trainer.task_config, config['task'])
        self.assertEqual(trainer.train_config, config['finetune'])
        self.assertEqual(trainer.datasets, self.mock_datasets)
        self.assertIsNone(trainer.extra_args)
        self.assertTrue(hasattr(trainer, 'trainer'))

    def test_init_with_extra_args(self):
        """Test initialization with extra arguments."""
        config = self.load_test_config()
        extra_args = {"learning_rate": 1e-4, "num_train_epochs": 5}
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets,
            extra_args=extra_args
        )
        
        self.assertEqual(trainer.extra_args, extra_args)

    @patch('dnallm.finetune.trainer.get_peft_model')
    @patch('dnallm.finetune.trainer.LoraConfig')
    def test_init_with_lora(self, mock_lora_config, mock_get_peft_model):
        """Test initialization with LoRA enabled."""
        config = self.load_test_config()
        config['lora'] = {
            "task_type": "SEQUENCE_CLASSIFICATION",
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "bias": "none"
        }
        
        mock_lora_config.return_value = Mock()
        mock_get_peft_model.return_value = self.mock_model
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets,
            use_lora=True
        )
        
        mock_lora_config.assert_called_once_with(**config["lora"])
        mock_get_peft_model.assert_called_once_with(self.mock_model, mock_lora_config.return_value)

    @patch('torch.cuda.device_count')
    def test_init_multi_gpu(self, mock_device_count):
        """Test initialization with multiple GPUs."""
        config = self.load_test_config()
        mock_device_count.return_value = 2
        
        with patch('torch.nn.DataParallel') as mock_data_parallel:
            mock_data_parallel.return_value = self.mock_model
            trainer = DNATrainer(
                model=self.mock_model,
                config=config,
                datasets=self.mock_datasets
            )
            
            mock_data_parallel.assert_called_once_with(self.mock_model)

    def test_setup_trainer_dataset_dict(self):
        """Test trainer setup with DatasetDict."""
        config = self.load_test_config()
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        
        self.assertIn("train", trainer.data_split)
        self.assertIn("val", trainer.data_split)
        self.assertTrue(hasattr(trainer, 'trainer'))

    def test_setup_trainer_single_dataset(self):
        """Test trainer setup with single dataset."""
        config = self.load_test_config()
        
        # Create single dataset
        sample_data = {
            "sequence": ["ATCGATCG", "GCTAGCTA"],
            "labels": [0, 1]
        }
        dataset = Dataset.from_dict(sample_data)
        
        datasets = Mock(spec=DNADataset)
        datasets.dataset = dataset
        datasets.tokenizer = self.mock_tokenizer
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=datasets
        )
        
        self.assertEqual(len(trainer.data_split), 0)
        self.assertTrue(hasattr(trainer, 'trainer'))

    def test_setup_trainer_no_eval_dataset(self):
        """Test trainer setup with no evaluation dataset."""
        config = self.load_test_config()
        
        # Create dataset with only train split
        sample_data = {
            "sequence": ["ATCGATCG", "GCTAGCTA"],
            "labels": [0, 1]
        }
        dataset = Dataset.from_dict(sample_data)
        dataset_dict = DatasetDict({"train": dataset})
        
        datasets = Mock(spec=DNADataset)
        datasets.dataset = dataset_dict
        datasets.tokenizer = self.mock_tokenizer
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=datasets
        )
        
        self.assertIn("train", trainer.data_split)
        self.assertEqual(trainer.training_args.eval_strategy, "no")

    def test_compute_task_metrics(self):
        """Test task-specific metrics computation."""
        config = self.load_test_config()
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        
        metrics_func = trainer.compute_task_metrics()
        self.assertTrue(callable(metrics_func))

    @patch('dnallm.finetune.trainer.compute_metrics')
    def test_compute_task_metrics_calls_compute_metrics(self, mock_compute_metrics):
        """Test that compute_task_metrics calls the compute_metrics function."""
        config = self.load_test_config()
        mock_compute_metrics.return_value = Mock()
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        
        trainer.compute_task_metrics()
        mock_compute_metrics.assert_called_once_with(config['task'])

    def test_train_basic(self):
        """Test basic training functionality."""
        config = self.load_test_config()
        
        # Mock the trainer's train method
        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5, "learning_rate": 2e-5}
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        trainer.trainer = Mock()
        trainer.trainer.train.return_value = mock_train_result
        trainer.trainer.save_model = Mock()
        
        metrics = trainer.train()
        
        self.assertEqual(metrics, mock_train_result.metrics)
        self.mock_model.train.assert_called_once()
        trainer.trainer.train.assert_called_once()
        trainer.trainer.save_model.assert_called_once()
        self.mock_datasets.tokenizer.save_pretrained.assert_called_once_with(config['finetune']['output_dir'])

    def test_train_without_save_tokenizer(self):
        """Test training without saving tokenizer."""
        config = self.load_test_config()
        
        mock_train_result = Mock()
        mock_train_result.metrics = {"train_loss": 0.5}
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        trainer.trainer = Mock()
        trainer.trainer.train.return_value = mock_train_result
        trainer.trainer.save_model = Mock()
        
        metrics = trainer.train(save_tokenizer=False)
        
        self.assertEqual(metrics, mock_train_result.metrics)
        self.mock_datasets.tokenizer.save_pretrained.assert_not_called()

    def test_evaluate(self):
        """Test evaluation functionality."""
        config = self.load_test_config()
        mock_eval_result = {"eval_loss": 0.3, "eval_accuracy": 0.85}
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        trainer.trainer = Mock()
        trainer.trainer.evaluate.return_value = mock_eval_result
        
        result = trainer.evaluate()
        
        self.assertEqual(result, mock_eval_result)
        self.mock_model.eval.assert_called_once()
        trainer.trainer.evaluate.assert_called_once()

    def test_predict_with_test_dataset(self):
        """Test prediction with test dataset available."""
        config = self.load_test_config()
        mock_predict_result = {"test_loss": 0.25, "test_accuracy": 0.9}
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        trainer.data_split = ["train", "val", "test"]
        trainer.trainer = Mock()
        trainer.trainer.predict.return_value = mock_predict_result
        
        result = trainer.predict()
        
        self.assertEqual(result, mock_predict_result)
        self.mock_model.eval.assert_called_once()
        trainer.trainer.predict.assert_called_once()

    def test_predict_without_test_dataset(self):
        """Test prediction without test dataset."""
        config = self.load_test_config()
        
        trainer = DNATrainer(
            model=self.mock_model,
            config=config,
            datasets=self.mock_datasets
        )
        trainer.data_split = ["train", "val"]  # No test split
        
        result = trainer.predict()
        
        self.assertEqual(result, {})
        self.mock_model.eval.assert_called_once()

    def test_setup_trainer_mask_task(self):
        """Test trainer setup for masked language modeling task."""
        config = self.load_test_config()
        config['task']['task_type'] = 'mask'
        config['task']['mlm_probability'] = 0.15
        
        with patch('dnallm.finetune.trainer.DataCollatorForLanguageModeling') as mock_collator:
            mock_collator.return_value = Mock()
            
            trainer = DNATrainer(
                model=self.mock_model,
                config=config,
                datasets=self.mock_datasets
            )
            
            mock_collator.assert_called_once_with(
                tokenizer=self.mock_datasets.tokenizer,
                mlm=True,
                mlm_probability=0.15
            )

    def test_setup_trainer_generation_task(self):
        """Test trainer setup for generation task."""
        config = self.load_test_config()
        config['task']['task_type'] = 'generation'
        
        with patch('dnallm.finetune.trainer.DataCollatorForLanguageModeling') as mock_collator:
            mock_collator.return_value = Mock()
            
            trainer = DNATrainer(
                model=self.mock_model,
                config=config,
                datasets=self.mock_datasets
            )
            
            mock_collator.assert_called_once_with(
                tokenizer=self.mock_datasets.tokenizer,
                mlm=False
            )

    def test_error_handling_missing_train_data(self):
        """Test error handling when train data is missing."""
        config = self.load_test_config()
        
        # Create dataset with no train split
        sample_data = {
            "sequence": ["ATCGATCG", "GCTAGCTA"],
            "labels": [0, 1]
        }
        dataset = Dataset.from_dict(sample_data)
        dataset_dict = DatasetDict({"val": dataset})  # No train split
        
        datasets = Mock(spec=DNADataset)
        datasets.dataset = dataset_dict
        datasets.tokenizer = self.mock_tokenizer
        
        with self.assertRaises(KeyError):
            DNATrainer(
                model=self.mock_model,
                config=config,
                datasets=datasets
            )

    def test_cleanup_output_directory(self):
        """Test cleanup of output directory after tests."""
        output_dir = "./test_output"
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
