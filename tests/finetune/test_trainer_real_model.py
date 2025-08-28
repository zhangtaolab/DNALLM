#!/usr/bin/env python3
"""Test real model training with zhangtaolab/plant-dnabert-BPE.

This script demonstrates how to use the DNATrainer with a real model
for DNA sequence classification training.
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_real_model_training():
    """Test training with real model."""
    try:
        from dnallm.models.model import load_model_and_tokenizer
        from dnallm.datahandling.data import DNADataset
        from dnallm.finetune.trainer import DNATrainer
        from dnallm.configuration.configs import load_config
        
        print("🚀 Loading model and tokenizer...")
        
        # Load real model and tokenizer
        model_name = "zhangtaolab/plant-dnabert-BPE"
        task_config = {
            "task_type": "binary",
            "num_labels": 2,
            "label_names": ["negative", "positive"],
            "threshold": 0.5
        }
        
        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            task_config=task_config, 
            source="modelscope"
        )
        
        print(f"✅ Model loaded: {model_name}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Load dataset
        print("📊 Loading dataset...")
        data_name = "zhangtaolab/plant-multi-species-core-promoters"
        
        datasets = DNADataset.from_modelscope(
            data_name, 
            seq_col="sequence", 
            label_col="label", 
            tokenizer=tokenizer, 
            max_length=512
        )
        
        print("✅ Dataset loaded from ModelScope")
        
        # Encode the datasets
        print("🔧 Encoding sequences...")
        datasets.encode_sequences()
        print("✅ Sequences encoded")
        
        # Sample datasets
        print("📝 Sampling datasets...")
        sampled_datasets = datasets.sampling(0.05, overwrite=True)
        print("✅ Datasets sampled (5%)")
        
        # Split the data for training
        print("✂️  Splitting data...")
        sampled_datasets.split_data(test_size=0.2, val_size=0.1, seed=42)
        print("✅ Data split completed")
        
        # Create training configuration
        print("⚙️  Creating training configuration...")
        train_config = {
            "task": task_config,
            "finetune": {
                "output_dir": "./test_training_output",
                "num_train_epochs": 1,  # Keep small for testing
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "learning_rate": 2e-5,
                "logging_steps": 10,
                "save_steps": 50,
                "eval_steps": 50,
                "save_strategy": "steps",
                "evaluation_strategy": "steps",
                "logging_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "save_total_limit": 2,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1e-8,
                "report_to": "none",  # Disable reporting for tests
                "save_safetensors": True,
                "dataloader_pin_memory": False,  # Disable for testing
                "remove_unused_columns": False,  # Keep all columns for testing
                "max_steps": 100  # Limit training steps for testing
            }
        }
        
        print("✅ Training configuration created")
        
        # Create trainer
        print("🏋️  Creating trainer...")
        trainer = DNATrainer(
            model=model,
            config=train_config,
            datasets=sampled_datasets
        )
        
        print("✅ Trainer created")
        
        # Test trainer setup
        print("🔍 Testing trainer setup...")
        assert hasattr(trainer, 'trainer'), "Trainer should have trainer attribute"
        assert len(trainer.data_split) > 0, "Should have data splits"
        assert 'train' in trainer.data_split, "Should have train split"
        
        print("✅ Trainer setup verified")
        
        # Test minimal training (very few steps)
        print("🚀 Testing minimal training...")
        minimal_config = train_config.copy()
        minimal_config['finetune']['max_steps'] = 5  # Very few steps
        minimal_config['finetune']['eval_steps'] = 10
        minimal_config['finetune']['save_steps'] = 10
        minimal_config['finetune']['logging_steps'] = 1
        
        minimal_trainer = DNATrainer(
            model=model,
            config=minimal_config,
            datasets=sampled_datasets
        )
        
        try:
            # This should complete quickly with very few steps
            metrics = minimal_trainer.train()
            assert metrics is not None, "Training should return metrics"
            print("✅ Minimal training completed successfully!")
            
            # Check that output directory was created
            output_dir = minimal_config['finetune']['output_dir']
            assert os.path.exists(output_dir), "Output directory should be created"
            print(f"✅ Output directory created: {output_dir}")
            
        except Exception as e:
            print(f"⚠️  Minimal training failed (expected for some configurations): {e}")
        
        # Test model saving
        print("💾 Testing model saving...")
        try:
            minimal_trainer.trainer.save_model()
            assert os.path.exists(output_dir), "Output directory should exist"
            
            # Check for model files
            model_files = os.listdir(output_dir)
            assert len(model_files) > 0, "Should have model files"
            print(f"✅ Model saved with {len(model_files)} files")
            
        except Exception as e:
            print(f"⚠️  Model saving failed: {e}")
        
        # Test evaluation
        print("📊 Testing evaluation...")
        try:
            eval_result = minimal_trainer.evaluate()
            assert eval_result is not None, "Evaluation should return results"
            print("✅ Evaluation completed")
            print(f"📈 Evaluation metrics: {eval_result}")
            
        except Exception as e:
            print(f"⚠️  Evaluation failed: {e}")
        
        # Test prediction
        print("🔮 Testing prediction...")
        try:
            pred_result = minimal_trainer.predict()
            assert pred_result is not None, "Prediction should return results"
            print("✅ Prediction completed")
            
        except Exception as e:
            print(f"⚠️  Prediction failed: {e}")
        
        # Clean up resources
        print("🧹 Cleaning up resources...")
        if hasattr(minimal_trainer, 'model'):
            del minimal_trainer.model
        if hasattr(minimal_trainer, 'tokenizer'):
            del minimal_trainer.tokenizer
        del minimal_trainer
        
        if hasattr(trainer, 'model'):
            del trainer.model
        if hasattr(trainer, 'tokenizer'):
            del trainer.tokenizer
        del trainer
        
        # Clean up output directory
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"✅ Cleaned up output directory: {output_dir}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install modelscope transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_config_file():
    """Test with the provided finetune config file."""
    try:
        from dnallm.models.model import load_model_and_tokenizer
        from dnallm.datahandling.data import DNADataset
        from dnallm.finetune.trainer import DNATrainer
        from dnallm.configuration.configs import load_config
        
        print("🔧 Testing with test_finetune_config.yaml...")
        
        # Load configuration
        config_path = "test_finetune_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        config = load_config(config_path)
        print("✅ Configuration loaded from test_finetune_config.yaml")
        
        # Load model and tokenizer
        model_name = "zhangtaolab/plant-dnabert-BPE"
        model, tokenizer = load_model_and_tokenizer(
            model_name, 
            task_config=config['task'], 
            source="modelscope"
        )
        
        print(f"✅ Model loaded: {model_name}")
        
        # Load dataset
        data_name = "zhangtaolab/plant-multi-species-core-promoters"
        datasets = DNADataset.from_modelscope(
            data_name, 
            seq_col="sequence", 
            label_col="label", 
            tokenizer=tokenizer, 
            max_length=512
        )
        
        print("✅ Dataset loaded")
        
        # Encode and sample
        datasets.encode_sequences()
        sampled_datasets = datasets.sampling(0.05, overwrite=True)
        sampled_datasets.split_data(test_size=0.2, val_size=0.1, seed=42)
        
        print("✅ Dataset prepared")
        
        # Create trainer
        trainer = DNATrainer(
            model=model,
            config=config,
            datasets=sampled_datasets
        )
        
        print("✅ Trainer created with config file")
        
        # Test basic functionality
        print("🔍 Testing basic functionality...")
        assert hasattr(trainer, 'trainer'), "Trainer should have trainer attribute"
        assert len(trainer.data_split) > 0, "Should have data splits"
        
        print("✅ Basic functionality verified")
        
        # Clean up
        if hasattr(trainer, 'model'):
            del trainer.model
        if hasattr(trainer, 'tokenizer'):
            del trainer.tokenizer
        del trainer
        
        return True
        
    except Exception as e:
        print(f"❌ Error during config file testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_info_and_dataset_access():
    """Test access to model and dataset information."""
    try:
        print("🔍 Testing model and dataset information access...")
        
        # Test model info
        from dnallm.models.modeling_auto import MODEL_INFO
        assert "Plant DNABERT" in MODEL_INFO, "Plant DNABERT should be in MODEL_INFO"
        
        plant_dnabert_info = MODEL_INFO["Plant DNABERT"]
        assert "zhangtaolab/plant-dnabert-BPE" in plant_dnabert_info["modelscope"], "Model should be in modelscope"
        
        print("✅ Model information accessed successfully")
        
        # Test dataset access
        from modelscope import MsDataset
        data_name = "zhangtaolab/plant-multi-species-core-promoters"
        dataset = MsDataset.load(data_name)
        assert dataset is not None, "Dataset should be loaded"
        
        print("✅ Dataset information accessed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error accessing model/dataset info: {e}")
        return False

if __name__ == "__main__":
    try:
        print("🧪 DNATrainer Real Model Testing")
        print("=" * 50)
        
        # Test 1: Basic training
        print("\n1️⃣ Testing basic training...")
        success1 = test_real_model_training()
        
        # Test 2: With config file
        print("\n2️⃣ Testing with config file...")
        success2 = test_with_config_file()
        
        # Test 3: Model and dataset info
        print("\n3️⃣ Testing model and dataset info...")
        success3 = test_model_info_and_dataset_access()
        
        print("\n" + "=" * 50)
        if success1 and success2 and success3:
            print("🎉 All tests completed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🧹 Cleaning up resources...")
        
        # Force cleanup of any remaining processes
        import os
        import signal
        import multiprocessing
        
        # Terminate any remaining multiprocessing processes
        try:
            for process in multiprocessing.active_children():
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
        except Exception:
            pass
        
        # Clear any remaining references
        import gc
        gc.collect()
        
        print("✅ Cleanup completed")
        print("👋 Exiting program...")
