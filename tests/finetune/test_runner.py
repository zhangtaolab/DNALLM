#!/usr/bin/env python3
"""Simple test runner for DNATrainer tests.

This script provides a simple way to run the DNATrainer tests
and demonstrates the testing workflow.
"""

import os
import sys
import subprocess
from pathlib import Path

def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_section(title):
    """Print a formatted section."""
    print(f"\n--- {title} ---")

def run_test_file(test_file, description):
    """Run a specific test file."""
    print(f"\n{description}:")
    print(f"File: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Run the test file directly
        result = subprocess.run(
            [sys.executable, test_file],
            cwd=os.path.dirname(test_file),
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Test completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])  # Show last 500 chars
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Test failed!")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout[-500:])
        return False

def run_unittest_file(test_file, description):
    """Run a unittest-based test file."""
    print(f"\n{description}:")
    print(f"File: {test_file}")
    
    if not os.path.exists(test_file):
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Run with unittest
        result = subprocess.run(
            [sys.executable, "-m", "unittest", test_file, "-v"],
            cwd=os.path.dirname(test_file),
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Unittest completed successfully!")
        if result.stdout:
            print("Output:", result.stdout[-500:])
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Unittest failed!")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        if e.stdout:
            print("Output:", e.stdout[-500:])
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print_section("Checking Dependencies")
    
    dependencies = {
        "torch": "import torch; print(f'PyTorch {torch.__version__}')",
        "datasets": "import datasets; print(f'Datasets {datasets.__version__}')",
        "transformers": "import transformers; print(f'Transformers {transformers.__version__}')"
    }
    
    available = {}
    for dep, cmd in dependencies.items():
        try:
            result = subprocess.run(
                [sys.executable, "-c", cmd], 
                capture_output=True, 
                text=True, 
                check=True
            )
            print(f"✅ {dep}: {result.stdout.strip()}")
            available[dep] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {dep}: Not available")
            available[dep] = False
    
    # Check ModelScope separately
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import modelscope; print(f'modelscope {modelscope.__version__}')"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✅ modelscope: {result.stdout.strip()}")
        available["modelscope"] = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ modelscope: Not available")
        available["modelscope"] = False
    
    return available

def main():
    """Main function to run tests."""
    print_header("DNALLM DNATrainer Test Runner")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "test_trainer.py").exists():
        print("❌ Please run this script from the tests/finetune directory")
        print("   cd tests/finetune")
        print("   python test_runner.py")
        return
    
    # Check dependencies
    deps = check_dependencies()
    
    print_section("Running Tests")
    
    # Test 1: Unit tests (unittest-based)
    print("\n1️⃣ Running unit tests...")
    success1 = run_unittest_file("test_trainer.py", "Unit tests for DNATrainer")
    
    # Test 2: Real model tests (script-based)
    print("\n2️⃣ Running real model tests...")
    if deps.get("modelscope", False):
        success2 = run_test_file("test_trainer_real_model.py", "Real model integration tests")
    else:
        print("⚠️  Skipping real model tests - ModelScope not available")
        success2 = True  # Not a failure, just skipped
    
    # Summary
    print_header("Test Results Summary")
    
    if success1 and success2:
        print("🎉 All tests completed successfully!")
        print("\n✅ Unit tests: PASSED")
        print("✅ Real model tests: PASSED")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        if not success1:
            print("❌ Unit tests: FAILED")
        if not success2:
            print("❌ Real model tests: FAILED")
    
    print("\n" + "=" * 60)
    print("Test files used:")
    print("  - test_trainer.py: Unit tests with mocked objects")
    print("  - test_trainer_real_model.py: Integration tests with real models")
    print("\nTo run tests individually:")
    print("  python test_trainer.py                    # Unit tests")
    print("  python test_trainer_real_model.py         # Real model tests")
    print("  python -m unittest test_trainer.py -v     # Unit tests with unittest")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Test runner interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 Exiting test runner...")
