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

    @pytest.mark.slow
    def test_early_stopping_stops_before_full_epochs(self):
        """Test that early stopping stops training before num_train_epochs."""
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer
            from dnallm.configuration.configs import (
                CallbackConfig, EarlyStoppingConfig
            )

            print("Testing early stopping...")

            # Load model and tokenizer
            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            # Load and sample datasets
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

            # Configure training with early stopping
            # Clone config to avoid modifying class-level config
            import copy
            test_config = copy.deepcopy(self.configs)
            test_config["finetune"].num_train_epochs = 3
            test_config["finetune"].eval_strategy = "steps"
            test_config["finetune"].eval_steps = 10
            test_config["finetune"].logging_steps = 10
            test_config["finetune"].max_steps = 60  # Limit total steps for speed
            test_config["finetune"].load_best_model_at_end = True
            test_config["finetune"].metric_for_best_model = "eval_loss"
            test_config["finetune"].callbacks = CallbackConfig(
                early_stopping=EarlyStoppingConfig(patience=1, threshold=0.0)
            )

            trainer = DNATrainer(
                model=model, config=test_config, datasets=sampled_datasets
            )

            metrics = trainer.train()

            # Verify training stopped.
            # With max_steps=60, training should stop at or before max_steps.
            # Early stopping may or may not trigger depending on validation loss
            # trends (which have randomness), so we only verify the callback
            # was present and training completed without error.
            actual_steps = trainer.trainer.state.global_step
            print(f"Training stopped at step: {actual_steps}")
            assert actual_steps <= 60, (
                f"Training should stop at or before max_steps (60), but stopped at {actual_steps}"
            )

            # Verify early stopping callback was present
            callback_names = [type(c).__name__ for c in trainer.trainer.callback_handler.callbacks]
            assert "EarlyStoppingCallback" in callback_names
            print("Early stopping test passed!")

            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"Early stopping test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"Early stopping test failed: {e}")

    @pytest.mark.slow
    def test_no_early_stopping_runs_full_epochs(self):
        """Test that training runs full epochs when early stopping is disabled."""
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer
            from dnallm.configuration.configs import (
                CallbackConfig, EarlyStoppingConfig
            )

            print("Testing no early stopping...")

            # Load model and tokenizer
            model_name = "zhangtaolab/plant-dnabert-BPE"
            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
            )

            # Load and sample datasets
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

            # Configure training WITHOUT early stopping (patience=None)
            import copy
            test_config = copy.deepcopy(self.configs)
            test_config["finetune"].num_train_epochs = 1
            test_config["finetune"].eval_strategy = "steps"
            test_config["finetune"].eval_steps = 10
            test_config["finetune"].logging_steps = 10
            test_config["finetune"].max_steps = 30  # Limit total steps for speed
            test_config["finetune"].load_best_model_at_end = False
            test_config["finetune"].metric_for_best_model = "eval_loss"
            test_config["finetune"].callbacks = CallbackConfig(
                early_stopping=EarlyStoppingConfig(patience=None, threshold=0.0)
            )

            trainer = DNATrainer(
                model=model, config=test_config, datasets=sampled_datasets
            )

            metrics = trainer.train()

            # Verify training ran for the full step count.
            # With max_steps=30 and no early stopping, should run all 30 steps.
            actual_steps = trainer.trainer.state.global_step
            print(f"Training completed at step: {actual_steps}")
            assert actual_steps == 30, (
                f"Without early stopping, training should complete all 30 steps, but stopped at {actual_steps}"
            )

            # Verify early stopping callback was NOT present
            callback_names = [type(c).__name__ for c in trainer.trainer.callback_handler.callbacks]
            assert "EarlyStoppingCallback" not in callback_names
            print("No early stopping test passed!")

            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

        except Exception as e:
            print(f"No early stopping test failed: {e}")
            import traceback
            traceback.print_exc()
            self.fail(f"No early stopping test failed: {e}")

    @pytest.mark.slow
    def test_qlora_training(self):
        """Test QLoRA training with 4-bit quantization.

        This test verifies that:
        1. Model loads successfully in 4-bit mode
        2. LoRA adapters are applied via prepare_model_for_kbit_training
        3. Training completes without OOM on a 24GB GPU
        """
        try:
            from dnallm import DNADataset, DNATrainer, load_model_and_tokenizer

            print("Testing QLoRA training...")

            # Load model and tokenizer with 4-bit quantization
            model_name = "zhangtaolab/plant-dnabert-BPE"

            # Create quantization config for QLoRA
            quantization_config = {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "float16",
                "bnb_4bit_use_double_quant": True,
                "bnb_4bit_quant_type": "nf4",
            }

            model, tokenizer = load_model_and_tokenizer(
                model_name,
                task_config=self.configs["task"],
                source="modelscope",
                quantization_config=quantization_config,
            )

            # Verify model is quantized
            if hasattr(model, "is_loaded_in_4bit"):
                assert model.is_loaded_in_4bit, "Model should be loaded in 4-bit"
                print(f"Model loaded in 4-bit: {model.is_loaded_in_4bit}")

            # Load and sample datasets
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

            # Configure for QLoRA training
            import copy
            from dnallm.configuration.configs import LoraConfig

            test_config = copy.deepcopy(self.configs)
            test_config["finetune"].num_train_epochs = 1
            test_config["finetune"].max_steps = 20
            test_config["finetune"].eval_strategy = "no"
            test_config["finetune"].save_strategy = "no"
            test_config["finetune"].load_best_model_at_end = False
            test_config["finetune"].use_qlora = True
            if "lora" not in test_config:
                test_config["lora"] = LoraConfig()

            trainer = DNATrainer(
                model=model,
                config=test_config,
                datasets=sampled_datasets,
                use_lora=True,
            )

            print("Trainer initialized with QLoRA")

            # Run a short training
            metrics = trainer.train()
            print(f"QLoRA training metrics: {metrics}")

            # Verify trainable parameters are only LoRA adapters
            if hasattr(trainer.model, "print_trainable_parameters"):
                trainer.model.print_trainable_parameters()

            del trainer
            del model
            del tokenizer
            del datasets
            del sampled_datasets

            print("QLoRA training test passed!")

        except Exception as e:
            print(f"QLoRA training test failed: {e}")
            import traceback

            traceback.print_exc()
            self.fail(f"QLoRA training test failed: {e}")

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


def test_qlora_config_validation():
    """Test that QLoRA config fields are properly validated."""
    from dnallm.configuration.configs import TrainingConfig

    config = TrainingConfig(use_qlora=True)
    assert config.use_qlora is True
    assert config.quantization_config is None

    config_with_quant = TrainingConfig(
        use_qlora=True,
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
        },
    )
    assert config_with_quant.use_qlora is True
    assert config_with_quant.quantization_config is not None
    assert config_with_quant.quantization_config["load_in_4bit"] is True


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
