#!/usr/bin/env python3
"""Test real model inference with zhangtaolab/plant-dnagpt-BPE-promoter.

This script demonstrates how to use the DNAPredictor with a real model
for DNA sequence classification.
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

def test_real_model_inference():
    """Test inference with real model."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from dnallm.inference.predictor import DNAPredictor
        from dnallm.configuration.configs import load_config
        
        print("🚀 Loading model and tokenizer...")
        
        # Load real model and tokenizer
        model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print(f"✅ Model loaded: {model_name}")
        print(f"✅ Tokenizer loaded: {type(tokenizer).__name__}")
        
        # Load configuration
        config_path = "inference_config.yaml"
        config = load_config(config_path)
        
        print("✅ Configuration loaded")
        
        # Create predictor
        predictor = DNAPredictor(model, tokenizer, config)
        
        print("✅ Predictor created")
        
        # Test with sample sequences
        test_sequences = [
            "TTGTCGAACCATTGAATCATAGCCGAACCGATGAGGAAGATGATCAAAATCATAAAATTACGAGTCGTGAGATACACAAACTATGTGGAGTAGACCATGATAGTTTGGTCAAAAAAAGTAGACCATGATAGCCACGCCGAAACGGGATGGACCCGAGAGACCATTAATCTAAGCGTCGTTGCATCTACCGTCAGGCGCCGCCATAAAAAACACACAAAAACATTAAAAAAAAGGTACTAAAACGACGTCAGATGTTGATCCGTGGTTACTCAGCTCCTGATCGCATACGTTTTTTTTTTT",
            "ATCTTGCGACACATGTATAGAACATTATAGCAAAAACTAATTACACAGTTTATCTGTAAATCATGAGACGAATCTTTTAAGCCTAATTACTTCATGATTGAACAATATTTGTTAAATAAAAATAAGAATGCTACTGTGCACAAAAATTTTTCGTGCAGGTACTAAACAAGGCCAGCGCAAATGGCCTATACTTGCTCATAAAGGATGCTTCAAGTAGGAGTACCGTACTATACAGTTAGTACAGTAGTAGTGGTATAGATGGCCATGCAGCCCGAGGCACGACGGCCCGGCCCACGGTAC"
        ]
        
        print(f"🧬 Testing with {len(test_sequences)} sequences...")
        
        # Perform prediction
        results = predictor.predict_seqs(test_sequences)
        
        print("✅ Prediction completed!")
        print("\n📊 Results:")
        print(json.dumps(results, indent=2))
        
        # Test file-based prediction
        print("\n📁 Testing file-based prediction...")
        
        # Create a temporary test file
        test_file_path = "temp_test_data.csv"
        test_data = {
            'sequence': test_sequences,
            'label': [1, 0]  # Mock labels
        }
        
        import pandas as pd
        df = pd.DataFrame(test_data)
        df.to_csv(test_file_path, index=False)
        
        # Predict from file
        file_results = predictor.predict_file(
            test_file_path,
            seq_col='sequence',
            label_col='label',
            evaluate=True
        )
        
        print("✅ File-based prediction completed!")
        
        # Clean up
        if os.path.exists(test_file_path):
            os.remove(test_file_path)
        
        # Test model information
        print("\n🔍 Model Information:")
        model_info = predictor.get_model_info()
        print(json.dumps(model_info, indent=2, default=str))
        
        # Test memory estimation
        print("\n💾 Memory Usage Estimation:")
        memory_info = predictor.estimate_memory_usage()
        print(json.dumps(memory_info, indent=2))
        
        # Clean up resources
        if hasattr(predictor, 'model'):
            del predictor.model
        if hasattr(predictor, 'tokenizer'):
            del predictor.tokenizer
        del predictor
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required packages: pip install transformers torch")
        return False
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_config_file():
    """Test with the provided inference config file."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from dnallm.inference.predictor import DNAPredictor
        from dnallm.configuration.configs import load_config
        
        print("🔧 Testing with inference_config.yaml...")
        
        # Load configuration
        config_path = "inference_config.yaml"
        if not os.path.exists(config_path):
            print(f"❌ Configuration file not found: {config_path}")
            return False
            
        config = load_config(config_path)
        print("✅ Configuration loaded from inference_config.yaml")
        
        # Load model and tokenizer
        model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Create predictor
        predictor = DNAPredictor(model, tokenizer, config)
        
        # Test with test.csv data
        if os.path.exists("test.csv"):
            print("📊 Testing with test.csv data...")
            
            try:
                results = predictor.predict_file(
                    "test.csv",
                    seq_col='sequence',
                    label_col='label',
                    evaluate=True
                )
                
                print("✅ File prediction completed!")
                
                # Save results
                if config['inference'].output_dir:
                    output_dir = Path(config['inference'].output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save results to output directory
                    with open(output_dir / "test_results.json", "w") as f:
                        json.dump(results, f, indent=2)
                    
                    print(f"💾 Results saved to {output_dir}")
                
                return True
            finally:
                # Clean up predictor resources
                if hasattr(predictor, 'model'):
                    del predictor.model
                if hasattr(predictor, 'tokenizer'):
                    del predictor.tokenizer
                del predictor
        else:
            print("❌ test.csv not found")
            return False
            
    except Exception as e:
        print(f"❌ Error during config file testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        print("🧪 DNAPredictor Real Model Testing")
        print("=" * 50)
        
        # Test 1: Basic inference
        print("\n1️⃣ Testing basic inference...")
        success1 = test_real_model_inference()
        
        # Test 2: With config file
        print("\n2️⃣ Testing with config file...")
        success2 = test_with_config_file()
        
        print("\n" + "=" * 50)
        if success1 and success2:
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
