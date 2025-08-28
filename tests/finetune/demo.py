#!/usr/bin/env python3
"""Demo script for DNATrainer tests.

This script demonstrates how to use the DNATrainer tests
step by step with clear explanations.
"""

import os
import sys
import time
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_step(step_num, title, description):
    """Print a step with title and description."""
    print(f"\n{step_num}️⃣ {title}")
    print("-" * 50)
    print(description)

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (NOT FOUND)")
        return False

def run_command(cmd, description, cwd=None):
    """Run a command and display the result."""
    print(f"\n🔧 {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        import subprocess
        result = subprocess.run(
            cmd, 
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Command executed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-300:] + "..." if len(result.stdout) > 300 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Command failed!")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False
    except ImportError:
        print("⚠️  subprocess module not available, skipping command execution")
        return False

def main():
    """Main demo function."""
    print_header("DNALLM DNATrainer Test Demo")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current working directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "test_trainer.py").exists():
        print("❌ Please run this script from the tests/finetune directory")
        print("   cd tests/finetune")
        print("   python demo.py")
        return
    
    print_step(1, "File Structure Check", 
               "First, let's check what test files we have available:")
    
    test_files = [
        ("test_trainer.py", "Unit tests for DNATrainer class"),
        ("test_trainer_real_model.py", "Integration tests with real models"),
        ("test_finetune_config.yaml", "Test configuration file"),
        ("test_runner.py", "Simple test runner script"),
        ("example_usage.py", "Interactive test usage example"),
        ("README.md", "Test documentation"),
        ("conftest.py", "Pytest configuration and fixtures")
    ]
    
    all_files_exist = True
    for file_path, description in test_files:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n⚠️  Some test files are missing. Please ensure all files are present.")
        return
    
    print_step(2, "Dependency Check", 
               "Let's check what dependencies are available for testing:")
    
    dependencies = {
        "torch": "PyTorch for deep learning",
        "datasets": "Hugging Face datasets library",
        "transformers": "Hugging Face transformers library",
        "modelscope": "ModelScope for model access",
        "pytest": "Testing framework"
    }
    
    available_deps = {}
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            print(f"✅ {dep}: {description}")
            available_deps[dep] = True
        except ImportError:
            print(f"❌ {dep}: {description} (NOT INSTALLED)")
            available_deps[dep] = False
    
    print_step(3, "Configuration File Analysis", 
               "Let's examine the test configuration file:")
    
    config_file = "test_finetune_config.yaml"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config_content = f.read()
            print(f"📄 Configuration file contents:")
            print(config_content)
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
    else:
        print(f"❌ Configuration file not found: {config_file}")
    
    print_step(4, "Unit Test Structure", 
               "Let's examine the unit test file structure:")
    
    try:
        with open("test_trainer.py", 'r') as f:
            content = f.read()
        
        # Count test methods
        test_methods = [line.strip() for line in content.split('\n') 
                       if line.strip().startswith('def test_')]
        
        print(f"📊 Found {len(test_methods)} test methods:")
        for method in test_methods[:10]:  # Show first 10
            print(f"   {method}")
        if len(test_methods) > 10:
            print(f"   ... and {len(test_methods) - 10} more")
            
    except Exception as e:
        print(f"❌ Error reading test file: {e}")
    
    print_step(5, "Real Model Test Structure", 
               "Let's examine the real model test file structure:")
    
    try:
        with open("test_trainer_real_model.py", 'r') as f:
            content = f.read()
        
        # Count test functions
        test_functions = [line.strip() for line in content.split('\n') 
                         if line.strip().startswith('def test_')]
        
        print(f"📊 Found {len(test_functions)} test functions:")
        for func in test_functions:
            print(f"   {func}")
            
    except Exception as e:
        print(f"❌ Error reading test file: {e}")
    
    print_step(6, "Running Unit Tests", 
               "Now let's run the unit tests to verify they work:")
    
    if available_deps.get("torch", False) and available_deps.get("datasets", False):
        success = run_command(
            [sys.executable, "-m", "unittest", "test_trainer.py", "-v"],
            "Running unit tests with unittest"
        )
        if success:
            print("🎉 Unit tests completed successfully!")
        else:
            print("⚠️  Unit tests had some issues")
    else:
        print("⚠️  Skipping unit tests - required dependencies not available")
    
    print_step(7, "Running Real Model Tests", 
               "Now let's run the real model tests (if dependencies allow):")
    
    if available_deps.get("modelscope", False):
        print("🌐 ModelScope available - running real model tests...")
        success = run_command(
            [sys.executable, "test_trainer_real_model.py"],
            "Running real model integration tests"
        )
        if success:
            print("🎉 Real model tests completed successfully!")
        else:
            print("⚠️  Real model tests had some issues")
    else:
        print("⚠️  Skipping real model tests - ModelScope not available")
        print("   To install: pip install modelscope")
    
    print_step(8, "Test Runner Script", 
               "Let's try the automated test runner:")
    
    if os.path.exists("test_runner.py"):
        success = run_command(
            [sys.executable, "test_runner.py"],
            "Running automated test runner"
        )
        if success:
            print("🎉 Test runner completed successfully!")
        else:
            print("⚠️  Test runner had some issues")
    else:
        print("❌ Test runner script not found")
    
    print_step(9, "Summary and Next Steps", 
               "Let's summarize what we've learned and what to do next:")
    
    print("\n📋 Test Summary:")
    print("   ✅ Unit tests: test_trainer.py")
    print("   ✅ Integration tests: test_trainer_real_model.py")
    print("   ✅ Configuration: test_finetune_config.yaml")
    print("   ✅ Test runner: test_runner.py")
    print("   ✅ Documentation: README.md")
    
    print("\n🚀 How to Use:")
    print("   1. Run unit tests: python -m unittest test_trainer.py -v")
    print("   2. Run real model tests: python test_trainer_real_model.py")
    print("   3. Use test runner: python test_runner.py")
    print("   4. Interactive demo: python example_usage.py")
    
    print("\n🔧 Dependencies:")
    for dep, description in dependencies.items():
        status = "✅" if available_deps.get(dep, False) else "❌"
        print(f"   {status} {dep}: {description}")
    
    print("\n💡 Tips:")
    print("   - Unit tests run quickly and don't need external resources")
    print("   - Real model tests require internet and may take longer")
    print("   - Check README.md for detailed usage instructions")
    print("   - Use test_runner.py for automated testing")
    
    print_header("Demo Complete!")
    print("🎉 You now have a complete understanding of the DNATrainer test suite!")
    print("   Feel free to explore the individual test files and run them as needed.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Demo completed. Happy testing!")
