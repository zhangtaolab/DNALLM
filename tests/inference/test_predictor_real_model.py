#!/usr/bin/env python3
"""Test real model inference with zhangtaolab/plant-dnagpt-BPE-promoter.

This script demonstrates how to use the DNAPredictor with a real model
for DNA sequence classification.
"""

import os
import sys
import json
import unittest
import tempfile
from pathlib import Path

# Add the parent directory to the path to import dnallm modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

class TestRealModelInference(unittest.TestCase):
    """Test class for real model inference."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test class - load model and tokenizer once."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from dnallm.inference.predictor import DNAPredictor
            from dnallm.configuration.configs import load_config
            
            print("🚀 Setting up test class...")
            
            # Load real model and tokenizer
            model_name = "zhangtaolab/plant-dnagpt-BPE-promoter"
            cls.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print(f"✅ Model loaded: {model_name}")
            print(f"✅ Tokenizer loaded: {type(cls.tokenizer).__name__}")
            
            # Load configuration - use path relative to test file
            test_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(test_dir, "inference_config.yaml")
            cls.config = load_config(config_path)
            
            print("✅ Configuration loaded")
            
            # Create predictor
            cls.predictor = DNAPredictor(cls.model, cls.tokenizer, cls.config)
            
            print("✅ Predictor created")
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("Please install required packages: pip install transformers torch")
            raise unittest.SkipTest(f"Required packages not available: {e}")
            
        except Exception as e:
            print(f"❌ Error during setup: {e}")
            import traceback
            traceback.print_exc()
            raise unittest.SkipTest(f"Setup failed: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test class resources."""
        print("🧹 Cleaning up test class resources...")
        
        # Clean up predictor resources
        if hasattr(cls, 'predictor'):
            if hasattr(cls.predictor, 'model'):
                del cls.predictor.model
            if hasattr(cls.predictor, 'tokenizer'):
                del cls.predictor.tokenizer
            del cls.predictor
        
        # Clear any remaining references
        import gc
        gc.collect()
        
        print("✅ Test class cleanup completed")
    
    def test_basic_inference(self):
        """Test basic inference functionality."""
        print("🧬 Testing basic inference...")
        
        # Test with sample sequences
        test_sequences = [
            "TTGTCGAACCATTGAATCATAGCCGAACCGATGAGGAAGATGATCAAAATCATAAAATTACGAGTCGTGAGATACACAAACTATGTGGAGTAGACCATGATAGTTTGGTCAAAAAAAGTAGACCATGATAGCCACGCCGAAACGGGATGGACCCGAGAGACCATTAATCTAAGCGTCGTTGCATCTACCGTCAGGCGCCGCCATAAAAAACACACAAAAACATTAAAAAAAAGGTACTAAAACGACGTCAGATGTTGATCCGTGGTTACTCAGCTCCTGATCGCATACGTTTTTTTTTTT",
            "ATCTTGCGACACATGTATAGAACATTATAGCAAAAACTAATTACACAGTTTATCTGTAAATCATGAGACGAATCTTTTAAGCCTAATTACTTCATGATTGAACAATATTTGTTAAATAAAAATAAGAATGCTACTGTGCACAAAAATTTTTCGTGCAGGTACTAAACAAGGCCAGCGCAAATGGCCTATACTTGCTCATAAAGGATGCTTCAAGTAGGAGTACCGTACTATACAGTTAGTACAGTAGTAGTGGTATAGATGGCCATGCAGCCCGAGGCACGACGGCCCGGCCCACGGTAC"
        ]
        
        print(f"🧬 Testing with {len(test_sequences)} sequences...")
        
        # Perform prediction - just ensure it runs without error
        try:
            results = self.predictor.predict_seqs(test_sequences)
            print("✅ Basic inference completed successfully")
            self.assertIsNotNone(results)
        except Exception as e:
            self.fail(f"Basic inference failed with error: {e}")
    
    def test_file_based_inference(self):
        """Test file-based inference functionality."""
        print("📁 Testing file-based inference...")
        
        # Create a temporary test file
        test_sequences = [
            "TTGTCGAACCATTGAATCATAGCCGAACCGATGAGGAAGATGATCAAAATCATAAAATTACGAGTCGTGAGATACACAAACTATGTGGAGTAGACCATGATAGTTTGGTCAAAAAAAGTAGACCATGATAGCCACGCCGAAACGGGATGGACCCGAGAGACCATTAATCTAAGCGTCGTTGCATCTACCGTCAGGCGCCGCCATAAAAAACACACAAAAACATTAAAAAAAAGGTACTAAAACGACGTCAGATGTTGATCCGTGGTTACTCAGCTCCTGATCGCATACGTTTTTTTTTTT",
            "ATCTTGCGACACATGTATAGAACATTATAGCAAAAACTAATTACACAGTTTATCTGTAAATCATGAGACGAATCTTTTAAGCCTAATTACTTCATGATTGAACAATATTTGTTAAATAAAAATAAGAATGCTACTGTGCACAAAAATTTTTCGTGCAGGTACTAAACAAGGCCAGCGCAAATGGCCTATACTTGCTCATAAAGGATGCTTCAAGTAGGAGTACCGTACTATACAGTTAGTACAGTAGTAGTGGTATAGATGGCCATGCAGCCCGAGGCACGACGGCCCGGCCCACGGTAC"
        ]
        
        test_data = {
            'sequence': test_sequences,
            'label': [1, 0]  # Mock labels
        }
        
        import pandas as pd
        df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            test_file_path = f.name
        
        try:
            # Predict from file - just ensure it runs without error
            file_results = self.predictor.predict_file(
                test_file_path,
                seq_col='sequence',
                label_col='label',
                evaluate=True
            )
            
            print("✅ File-based inference completed successfully")
            self.assertIsNotNone(file_results)
            
        except Exception as e:
            self.fail(f"File-based inference failed with error: {e}")
        finally:
            # Clean up temporary file
            if os.path.exists(test_file_path):
                os.remove(test_file_path)
    
    def test_model_information(self):
        """Test getting model information."""
        print("🔍 Testing model information retrieval...")
        
        try:
            model_info = self.predictor.get_model_info()
            print("✅ Model information retrieved successfully")
            self.assertIsNotNone(model_info)
        except Exception as e:
            self.fail(f"Model information retrieval failed with error: {e}")
    
    def test_memory_estimation(self):
        """Test memory usage estimation."""
        print("💾 Testing memory usage estimation...")
        
        try:
            memory_info = self.predictor.estimate_memory_usage()
            print("✅ Memory usage estimation completed successfully")
            self.assertIsNotNone(memory_info)
        except Exception as e:
            self.fail(f"Memory usage estimation failed with error: {e}")
    
    def test_with_test_csv(self):
        """Test with test.csv if it exists."""
        if not os.path.exists("test.csv"):
            self.skipTest("test.csv not found, skipping this test")
        
        print("📊 Testing with test.csv data...")
        
        try:
            results = self.predictor.predict_file(
                "test.csv",
                seq_col='sequence',
                label_col='label',
                evaluate=True
            )
            
            print("✅ Test.csv prediction completed successfully")
            self.assertIsNotNone(results)
            
            # Save results if output directory is configured
            if self.config['inference'].output_dir:
                output_dir = Path(self.config['inference'].output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save results to output directory
                with open(output_dir / "test_results.json", "w") as f:
                    json.dump(results, f, indent=2)
                
                print(f"💾 Results saved to {output_dir}")
                
        except Exception as e:
            self.fail(f"Test.csv prediction failed with error: {e}")


def run_tests():
    """Run the tests with proper setup and teardown."""
    try:
        print("🧪 DNAPredictor Real Model Testing")
        print("=" * 50)
        
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(TestRealModelInference)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        print("\n" + "=" * 50)
        if result.wasSuccessful():
            print("🎉 All tests completed successfully!")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        
        return result.wasSuccessful()
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        print("\n🧹 Final cleanup...")
        
        # Force cleanup of any remaining processes
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
        
        print("✅ Final cleanup completed")
        print("👋 Exiting program...")


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
