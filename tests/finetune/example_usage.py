#!/usr/bin/env python3
"""Example usage of DNATrainer tests.

This script demonstrates how to use the DNATrainer tests and shows
the testing workflow for different scenarios.
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

def run_command(cmd, description, cwd=None):
    """Run a command and display the result."""
    print(f"\n{description}:")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd or os.getcwd(),
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Failed!")
        print(f"Error code: {e.returncode}")
        if e.stderr:
            print("Error:", e.stderr)
        return False

def check_dependencies():
    """Check if required dependencies are available."""
    print_section("Checking Dependencies")
    
    dependencies = {
        "pytest": "python -m pytest --version",
        "torch": "python -c 'import torch; print(f\"PyTorch {torch.__version__}\")'",
        "datasets": "python -c 'import datasets; print(f\"Datasets {datasets.__version__}\")'",
        "transformers": "python -c 'import transformers; print(f\"Transformers {transformers.__version__}\")'"
    }
    
    available = {}
    for dep, cmd in dependencies.items():
        try:
            result = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
            print(f"✅ {dep}: {result.stdout.strip()}")
            available[dep] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"❌ {dep}: Not available")
            available[dep] = False
    
    # Check ModelScope separately
    try:
        import modelscope
        print(f"✅ modelscope: {modelscope.__version__}")
        available["modelscope"] = True
    except ImportError:
        print("❌ modelscope: Not available")
        available["modelscope"] = False
    
    return available

def run_unit_tests():
    """Run unit tests."""
    print_section("Running Unit Tests")
    
    cmd = ["python", "-m", "pytest", "test_trainer.py", "-v", "--tb=short"]
    return run_command(cmd, "Running unit tests for DNATrainer")

def run_integration_tests():
    """Run integration tests."""
    print_section("Running Integration Tests")
    
    cmd = ["python", "-m", "pytest", "test_trainer_real_model.py", "-v", "--tb=short"]
    return run_command(cmd, "Running integration tests with real models")

def run_specific_test(test_name):
    """Run a specific test."""
    print_section(f"Running Specific Test: {test_name}")
    
    cmd = ["python", "-m", "pytest", f"test_trainer.py::{test_name}", "-v"]
    return run_command(cmd, f"Running test: {test_name}")

def run_tests_with_markers(marker):
    """Run tests with specific markers."""
    print_section(f"Running Tests with Marker: {marker}")
    
    cmd = ["python", "-m", "pytest", "-m", marker, "-v"]
    return run_command(cmd, f"Running tests with marker: {marker}")

def show_test_coverage():
    """Show test coverage information."""
    print_section("Test Coverage Information")
    
    # Count test functions
    test_files = ["test_trainer.py", "test_trainer_real_model.py"]
    total_tests = 0
    
    for test_file in test_files:
        if os.path.exists(test_file):
            with open(test_file, 'r') as f:
                content = f.read()
                # Count test functions (simple heuristic)
                test_count = content.count("def test_")
                print(f"📊 {test_file}: {test_count} test functions")
                total_tests += test_count
    
    print(f"📊 Total test functions: {total_tests}")
    
    # Show test categories
    print("\nTest Categories:")
    print("🔧 Unit Tests: Basic functionality with mocked objects")
    print("🌐 Integration Tests: End-to-end with real models and data")
    print("⚡ Fast Tests: Quick validation tests")
    print("🐌 Slow Tests: Longer running tests (marked with @pytest.mark.slow)")

def demonstrate_test_workflow():
    """Demonstrate the complete testing workflow."""
    print_section("Complete Testing Workflow")
    
    print("1. 🔍 Check Dependencies")
    print("   - Ensure all required packages are installed")
    print("   - Verify ModelScope access for integration tests")
    
    print("\n2. 🧪 Run Unit Tests")
    print("   - Quick validation of basic functionality")
    print("   - No external dependencies required")
    print("   - Suitable for CI/CD pipelines")
    
    print("\n3. 🌐 Run Integration Tests")
    print("   - End-to-end validation with real data")
    print("   - Requires internet connection and computational resources")
    print("   - May take several minutes to complete")
    
    print("\n4. 📊 Analyze Results")
    print("   - Review test output and coverage")
    print("   - Fix any failing tests")
    print("   - Optimize slow tests if needed")

def main():
    """Main function to demonstrate test usage."""
    print_header("DNALLM DNATrainer Test Examples")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # Check if we're in the right directory
    if not (current_dir / "test_trainer.py").exists():
        print("❌ Please run this script from the tests/finetune directory")
        print("   cd tests/finetune")
        print("   python example_usage.py")
        return
    
    # Check dependencies
    deps = check_dependencies()
    
    # Show test coverage
    show_test_coverage()
    
    # Demonstrate workflow
    demonstrate_test_workflow()
    
    # Interactive menu
    print_section("Interactive Test Runner")
    
    while True:
        print("\nChoose an option:")
        print("1. Run unit tests")
        print("2. Run integration tests")
        print("3. Run specific test")
        print("4. Run tests with markers")
        print("5. Show test help")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == "1":
            run_unit_tests()
        elif choice == "2":
            if deps.get("modelscope", False):
                run_integration_tests()
            else:
                print("❌ ModelScope not available. Install it first:")
                print("   pip install modelscope")
        elif choice == "3":
            test_name = input("Enter test name (e.g., test_init_basic): ").strip()
            if test_name:
                run_specific_test(test_name)
        elif choice == "4":
            marker = input("Enter marker (e.g., slow, integration): ").strip()
            if marker:
                run_tests_with_markers(marker)
        elif choice == "5":
            print("\nTest Help:")
            print("- Unit tests: Fast, no external dependencies")
            print("- Integration tests: Slow, require real models/data")
            print("- Markers: slow, integration, unit")
            print("- Use -v for verbose output")
            print("- Use --tb=long for detailed tracebacks")
        elif choice == "6":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()
