#!/usr/bin/env python3
"""
Test script to verify that the attention support checking works correctly.
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

def test_actual_model_loading():
    """Test with actual model loading to see where the error occurs."""
    
    print("\nTesting actual model loading...")
    
    try:
        # Load a small model
        model_name = "distilbert-base-uncased"
        print(f"Loading model: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        print(f"Model loaded successfully")
        print(f"Model type: {type(model).__name__}")
        print(f"Config attn_implementation: {getattr(model.config, 'attn_implementation', 'Not set')}")
        
        # Now try to set output_attentions
        try:
            model.config.output_attentions = True
            print("✅ output_attentions set successfully")
        except Exception as e:
            print(f"❌ Error setting output_attentions: {e}")
            
            if "attn_implementation" in str(e) and "sdpa" in str(e):
                print("Found the exact error! Trying our fix...")
                
                # Try switching to eager
                try:
                    model.config.attn_implementation = "eager"
                    model.config.output_attentions = True
                    print("✅ Successfully applied our fix!")
                except Exception as e2:
                    print(f"❌ Our fix failed: {e2}")
        
        # Test actual inference
        print("\nTesting actual inference...")
        inputs = tokenizer("Hello world", return_tensors="pt")
        
        try:
            with torch.no_grad():
                outputs = model(**inputs, output_attentions=True)
            print("✅ Inference with output_attentions=True successful")
            if hasattr(outputs, 'attentions'):
                print(f"✅ Attention outputs available: {len(outputs.attentions)} layers")
            else:
                print("❌ No attention outputs in results")
        except Exception as e:
            print(f"❌ Inference failed: {e}")
            
    except Exception as e:
        print(f"❌ Error in model loading test: {e}")

def test_forced_sdpa_error():
    """Test the error scenario that occurs in the actual application."""
    
    print("Testing forced SDPA error scenario...")
    
    try:
        # Create a config and force sdpa
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.attn_implementation = "sdpa"
        
        print(f"Model type: {config.model_type}")
        print(f"Attention implementation: {config.attn_implementation}")
        
        # Now try to set output_attentions - this should fail
        try:
            config.output_attentions = True
            print("❌ This should have failed but didn't!")
        except ValueError as e:
            print(f"✅ Got expected error: {e}")
            
            if "attn_implementation" in str(e) and "sdpa" in str(e):
                print("This is the exact error we're fixing!")
                
                # Try our fix - switch to eager
                try:
                    config.attn_implementation = "eager"
                    config.output_attentions = True
                    print("✅ Successfully applied our fix!")
                except Exception as e2:
                    print(f"❌ Our fix failed: {e2}")
            else:
                print("❌ Unexpected error type")
                
    except Exception as e:
        print(f"❌ Error in test: {e}")

def test_attention_support():
    """Test the attention support checking logic."""
    
    # Test with different model types
    model_names = [
        "bert-base-uncased",
        "microsoft/DialoGPT-medium",
        "gpt2",
        "facebook/opt-125m"
    ]
    
    for model_name in model_names:
        print(f"\n{'='*50}")
        print(f"Testing with: {model_name}")
        print('='*50)
        
        try:
            # Create a simple config
            config = AutoConfig.from_pretrained(model_name)
            print(f"Model type: {config.model_type}")
            print(f"Attention implementation: {getattr(config, 'attn_implementation', 'Not set')}")
            
            # Test setting output_attentions
            try:
                config.output_attentions = True
                print("✅ output_attentions can be set to True")
            except Exception as e:
                print(f"❌ Error setting output_attentions: {e}")
            
            # Test setting attn_implementation
            try:
                config.attn_implementation = "eager"
                print("✅ attn_implementation can be set to 'eager'")
            except Exception as e:
                print(f"❌ Error setting attn_implementation: {e}")
            
            # Test setting attn_implementation to sdpa
            try:
                config.attn_implementation = "sdpa"
                print("✅ attn_implementation can be set to 'sdpa'")
                
                # Now try to set output_attentions with sdpa
                try:
                    config.output_attentions = True
                    print("✅ output_attentions can be set with sdpa")
                except Exception as e:
                    print(f"❌ Error setting output_attentions with sdpa: {e}")
                    if "attn_implementation" in str(e) and "sdpa" in str(e):
                        print("This is the expected error we're fixing!")
                        
                        # Try switching back to eager
                        try:
                            config.attn_implementation = "eager"
                            config.output_attentions = True
                            print("✅ Successfully switched to eager and set output_attentions")
                        except Exception as e2:
                            print(f"❌ Error switching to eager: {e2}")
                            
            except Exception as e:
                print(f"❌ Error setting attn_implementation to sdpa: {e}")
                
        except Exception as e:
            print(f"❌ Error creating config: {e}")

if __name__ == "__main__":
    test_forced_sdpa_error()
    print("\n" + "="*80 + "\n")
    test_actual_model_loading()
    print("\n" + "="*80 + "\n")
    test_attention_support()
