#!/usr/bin/env python3
"""Test real model training with zhangtaolab/plant-dnabert-BPE.

This script demonstrates how to use the DNATrainer with a real model
for DNA sequence classification training.
"""

import os
import sys
import unittest
from pathlib import Path
from typing import Any

import pytest

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@pytest.mark.slow
class TestTrainerRealModel(unittest.TestCase):
    """Test DNATrainer with real model."""

    # Class attributes for type checking
    test_dir: str
    configs: Any

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        print("🚀 Setting up test environment...")

        # Get the directory where this test file is located
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))

        # Load the config file
        try:
            from dnallm import load_config

            cls.configs = load_config(
                os.path.join(cls.test_dir, "test_finetune_config.yaml")
            )
            print("✅ Configuration loaded")
        except Exception as e:
            print(f"❌ Failed to load config: {e}")
            cls.configs = None

    def setUp(self):
        """Set up for each test method."""
        print(f"\n🧪 Running test: {self._testMethodName}")

        # Skip if config loading failed
        if self.configs is None:
            self.skipTest("Configuration not available")

    def test_complete_training_workflow(self):
        """Test complete training workflow from start to finish."""
        try:
            print("🔧 Starting complete training workflow...")

            # Step 1: Load model and tokenizer
            print("1️⃣ Loading model and tokenizer...")
            from dnallm import load_model_and_tokenizer

            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            print(f"✅ Model loaded: {model_name}")
            print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")

            # Step 2: Load datasets
            print("2️⃣ Loading datasets...")
            from dnallm import DNADataset

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            print(f"✅ Datasets loaded: {data_name}")

            # Step 3: Encode datasets
            print("3️⃣ Encoding datasets...")
            datasets.encode_sequences()
            print("✅ Datasets encoded")

            # Step 4: Sample datasets
            print("4️⃣ Sampling datasets...")
            sampled_datasets = datasets.sampling(0.05, overwrite=True)
            print("✅ Datasets sampled")

            # Step 5: Initialize trainer
            print("5️⃣ Initializing trainer...")
            from dnallm import DNATrainer

            trainer = DNATrainer(
                model=model, config=self.configs, datasets=sampled_datasets
            )

            print("✅ Trainer initialized")

            # Step 6: Start training
            print("6️⃣ Starting training...")
            metrics = trainer.train()
            print("✅ Training completed")
            print(f"📊 Training metrics: {metrics}")

            # Step 7: Run prediction
            print("7️⃣ Running prediction...")
            trainer.infer()
            print("✅ Prediction completed")

            print("🎉 Complete training workflow test passed!")

            # Clean up resources
            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"❌ Complete training workflow failed: {e}")
            import traceback

            traceback.print_exc()
            self.fail(f"Complete training workflow failed: {e}")

    def test_load_model_and_tokenizer(self):
        """Test loading model and tokenizer independently."""
        try:
            from dnallm import load_model_and_tokenizer

            print("🔧 Loading model and tokenizer...")

            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            print(f"✅ Model loaded: {model_name}")
            print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")

            # Clean up
            del model
            del tokenizer

        except Exception as e:
            print(f"❌ Failed to load model and tokenizer: {e}")
            self.fail(f"Model and tokenizer loading failed: {e}")

    def test_load_datasets(self):
        """Test loading datasets independently."""
        try:
            from dnallm import DNADataset, load_model_and_tokenizer

            print("📊 Loading datasets...")

            # First load tokenizer
            model_name = "zhangtaolab/plant-dnabert-BPE"
            _, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            # Load the datasets
            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            print(f"✅ Datasets loaded: {data_name}")

            # Clean up
            del tokenizer
            del datasets

        except Exception as e:
            print(f"❌ Failed to load datasets: {e}")
            self.fail(f"Dataset loading failed: {e}")

    def test_encode_datasets(self):
        """Test encoding datasets independently."""
        try:
            from dnallm import DNADataset, load_model_and_tokenizer

            print("🔤 Encoding datasets...")

            # First load tokenizer and datasets
            model_name = "zhangtaolab/plant-dnabert-BPE"
            _, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            # Encode the datasets
            datasets.encode_sequences()

            print("✅ Datasets encoded")

            # Clean up
            del tokenizer
            del datasets

        except Exception as e:
            print(f"❌ Failed to encode datasets: {e}")
            self.fail(f"Dataset encoding failed: {e}")

    def test_sample_datasets(self):
        """Test sampling datasets independently."""
        try:
            from dnallm import DNADataset, load_model_and_tokenizer

            print("📝 Sampling datasets...")

            # First load tokenizer and datasets
            model_name = "zhangtaolab/plant-dnabert-BPE"
            _, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            # Encode and sample
            datasets.encode_sequences()
            sampled_datasets = datasets.sampling(0.05, overwrite=True)

            print("✅ Datasets sampled")

            # Clean up
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"❌ Failed to sample datasets: {e}")
            self.fail(f"Dataset sampling failed: {e}")

    def test_initialize_trainer(self):
        """Test trainer initialization independently."""
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer

            print("🏋️ Initializing trainer...")

            # Load all required components
            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            datasets.encode_sequences()
            sampled_datasets = datasets.sampling(0.05, overwrite=True)

            # Initialize the trainer
            trainer = DNATrainer(
                model=model, config=self.configs, datasets=sampled_datasets
            )

            print("✅ Trainer initialized")

            # Clean up
            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"❌ Failed to initialize trainer: {e}")
            self.fail(f"Trainer initialization failed: {e}")

    def test_training(self):
        """Test training process independently."""
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer

            print("🚀 Starting training...")

            # Load all required components
            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            datasets.encode_sequences()
            sampled_datasets = datasets.sampling(0.05, overwrite=True)

            trainer = DNATrainer(
                model=model, config=self.configs, datasets=sampled_datasets
            )

            # Start training
            metrics = trainer.train()

            print("✅ Training completed")
            print(f"📊 Training metrics: {metrics}")

            # Clean up
            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"❌ Training failed: {e}")
            self.fail(f"Training failed: {e}")

    def test_prediction(self):
        """Test prediction on test set independently."""
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer

            print("🔮 Running prediction...")

            # Load all required components
            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            data_name = "zhangtaolab/plant-multi-species-core-promoters"
            datasets = DNADataset.from_modelscope(
                data_name,
                seq_col="sequence",
                label_col="label",
                tokenizer=tokenizer,
                max_length=512,
            )

            datasets.encode_sequences()
            sampled_datasets = datasets.sampling(0.05, overwrite=True)

            trainer = DNATrainer(
                model=model, config=self.configs, datasets=sampled_datasets
            )

            # Do prediction on the test set
            trainer.infer()

            print("✅ Prediction completed")

            # Clean up
            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            self.fail(f"Prediction failed: {e}")

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        print("\n🧹 Cleaning up test environment...")

        # Force cleanup of any remaining processes
        import multiprocessing

        # Terminate any remaining multiprocessing processes
        try:
            for process in multiprocessing.active_children():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
        except Exception as e:
            print(f"Warning: Failed to cleanup processes: {e}")
            pass

        # Clear any remaining references
        import gc

        gc.collect()

        print("✅ Cleanup completed")


@pytest.mark.slow
def test_with_config_file():
    """Test with the provided finetune config file."""
    try:
        from dnallm import (
            DNADataset,
            DNATrainer,
            load_config,
            load_model_and_tokenizer,
        )

        print("🔧 Testing with test_finetune_config.yaml...")

        # Get the directory where this test file is located
        test_dir = os.path.dirname(os.path.abspath(__file__))

        # Load configuration
        config_path = os.path.join(test_dir, "test_finetune_config.yaml")
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False

        configs = load_config(config_path)
        print("✅ Configuration loaded from test_finetune_config.yaml")

        # Load the model and tokenizer
        model_name = "zhangtaolab/plant-dnabert-BPE"
        # Type assertion for task config
        from dnallm.configuration.configs import TaskConfig

        task_config = configs["task"]
        assert isinstance(task_config, TaskConfig)
        model, tokenizer = load_model_and_tokenizer(
            model_name, task_config=task_config, source="modelscope"
        )

        # Load the datasets
        data_name = "zhangtaolab/plant-multi-species-core-promoters"
        datasets = DNADataset.from_modelscope(
            data_name,
            seq_col="sequence",
            label_col="label",
            tokenizer=tokenizer,
            max_length=512,
        )

        # Encode the datasets
        datasets.encode_sequences()

        # sample datasets
        sampled_datasets = datasets.sampling(0.05, overwrite=True)

        # Initialize the trainer
        trainer = DNATrainer(
            model=model, config=configs, datasets=sampled_datasets
        )

        # Start training
        metrics = trainer.train()
        print(f"📊 Training metrics: {metrics}")

        # Do prediction on the test set
        trainer.infer()

        print("✅ All operations completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Error during config file testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Only run when executed directly, not when imported by pytest
    import sys

    if "pytest" not in sys.modules:
        try:
            print("🧪 DNATrainer Real Model Testing")
            print("=" * 50)

            # Test 1: Unittest framework
            print("\n1️⃣ Running unittest framework tests...")
            unittest.main(verbosity=2, exit=False)

            # Test 2: With config file (manual test)
            print("\n2️⃣ Testing with config file...")
            success = test_with_config_file()

            print("\n" + "=" * 50)
            if success:
                print("🎉 All tests completed successfully!")
            else:
                print(
                    "⚠️  Some tests failed. Check the output above\
                        for details."
                )

        except KeyboardInterrupt:
            print("\n⚠️  Test interrupted by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()
        finally:
            print("\n🧹 Cleaning up resources...")

            # Force cleanup of any remaining processes
            import multiprocessing
            import os
            import signal

            # Terminate any remaining multiprocessing processes
            try:
                for process in multiprocessing.active_children():
                    process.terminate()
                    process.join(timeout=1)
                    if process.is_alive():
                        process.kill()
            except Exception as e:
                print(f"Warning: Failed to cleanup processes: {e}")
                pass

            # Clear any remaining references
            import gc

            gc.collect()

            print("✅ Cleanup completed")
            print("👋 Exiting program...")
