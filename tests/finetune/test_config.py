#!/usr/bin/env python3
"""Test configuration loading for DNATrainer tests.

This script tests if the configuration files can be loaded correctly
without validation errors.
"""

import os
import sys
import yaml

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_config_loading():
    """Test if configuration files can be loaded correctly."""
    print("🧪 Testing configuration loading...")
    
    # Test 1: test_finetune_config.yaml
    print("\n1️⃣ Testing test_finetune_config.yaml...")
    config_path = "test_finetune_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Configuration file not found: {config_path}")
        return False
    
    try:
        # Load with yaml first
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        print("✅ YAML loading successful")
        print(f"   Task config: {yaml_config.get('task', 'NOT FOUND')}")
        print(f"   Finetune config: {yaml_config.get('finetune', 'NOT FOUND')}")
        
        # Try to load with dnallm config loader
        from dnallm.configuration.configs import load_config
        configs = load_config(config_path)
        print("✅ DNALLM config loading successful")
        
        # Check if all required configs are present
        if 'task' in configs:
            print("✅ Task config loaded successfully")
            print(f"   Task type: {configs['task'].task_type}")
            print(f"   Num labels: {configs['task'].num_labels}")
        
        if 'finetune' in configs:
            print("✅ Finetune config loaded successfully")
            print(f"   Output dir: {configs['finetune'].output_dir}")
            print(f"   Learning rate: {configs['finetune'].learning_rate}")
            print(f"   Batch size: {configs['finetune'].per_device_train_batch_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trainer_import():
    """Test if DNATrainer can be imported and configured."""
    print("\n2️⃣ Testing DNATrainer import and configuration...")
    
    try:
        from dnallm.finetune.trainer import DNATrainer
        print("✅ DNATrainer import successful")
        
        # Test creating a mock configuration
        mock_config = {
            "task": {
                "task_type": "binary",
                "num_labels": 2,
                "label_names": ["negative", "positive"],
                "threshold": 0.5
            },
            "finetune": {
                "output_dir": "./test_output",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "per_device_eval_batch_size": 2,
                "learning_rate": 2.0e-05,
                "logging_steps": 10,
                "save_steps": 100,
                "eval_steps": 100,
                "save_strategy": "steps",
                "eval_strategy": "steps",
                "logging_strategy": "steps",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "save_total_limit": 3,
                "warmup_ratio": 0.1,
                "weight_decay": 0.01,
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_epsilon": 1.0e-08,
                "report_to": "none",
                "save_safetensors": True,
                "lr_scheduler_kwargs": None,
                "resume_from_checkpoint": None,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "lr_scheduler_type": "linear",
                "seed": 42,
                "bf16": False,
                "fp16": False
            }
        }
        
        print("✅ Mock configuration created successfully")
        
        # Test if we can create a mock model and datasets
        from unittest.mock import Mock
        from datasets import Dataset, DatasetDict
        
        # Create mock model
        mock_model = Mock()
        mock_model.train = Mock()
        mock_model.eval = Mock()
        mock_model.to = Mock(return_value=mock_model)
        print("✅ Mock model created successfully")
        
        # Create mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.save_pretrained = Mock()
        print("✅ Mock tokenizer created successfully")
        
        # Create mock datasets
        sample_data = {
            "sequence": ["ATCGATCG", "GCTAGCTA"],
            "labels": [0, 1]
        }
        dataset = Dataset.from_dict(sample_data)
        dataset_dict = DatasetDict({
            "train": dataset,
            "val": dataset
        })
        
        mock_datasets = Mock()
        mock_datasets.dataset = dataset_dict
        mock_datasets.tokenizer = mock_tokenizer
        print("✅ Mock datasets created successfully")
        
        # Test DNATrainer initialization
        trainer = DNATrainer(
            model=mock_model,
            config=mock_config,
            datasets=mock_datasets
        )
        print("✅ DNATrainer initialization successful")
        
        return True
        
    except Exception as e:
        print(f"❌ DNATrainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("🚀 DNATrainer Configuration Test")
    print("=" * 50)
    
    success1 = test_config_loading()
    success2 = test_trainer_import()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("🎉 All configuration tests passed!")
        print("✅ Configuration files can be loaded correctly")
        print("✅ DNATrainer can be imported and configured")
    else:
        print("⚠️  Some configuration tests failed")
        if not success1:
            print("❌ Configuration loading failed")
        if not success2:
            print("❌ DNATrainer configuration failed")
    
    return success1 and success2

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
