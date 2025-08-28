#!/usr/bin/env python3
"""Script to run DNATrainer tests.

This script provides an easy way to run the DNATrainer tests with different options.
"""

import sys
import os
import subprocess
import argparse

def run_tests(test_type="unit", verbose=False, markers=None):
    """Run the specified type of tests.
    
    Args:
        test_type: Type of tests to run ("unit", "real", "all")
        verbose: Whether to run with verbose output
        markers: Additional pytest markers to include
    """
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    # Determine which test files to run
    if test_type == "unit":
        test_files = ["test_trainer.py"]
    elif test_type == "real":
        test_files = ["test_trainer_real_model.py"]
    elif test_type == "all":
        test_files = ["test_trainer.py", "test_trainer_real_model.py"]
    else:
        print(f"Unknown test type: {test_type}")
        return False
    
    # Build pytest command
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    # Add markers if specified
    if markers:
        for marker in markers:
            cmd.extend(["-m", marker])
    
    # Add test files
    cmd.extend(test_files)
    
    # Add current directory
    cmd.append(".")
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Test type: {test_type}")
    print(f"Test files: {test_files}")
    print("-" * 50)
    
    try:
        # Run the tests
        result = subprocess.run(cmd, cwd=os.path.dirname(__file__), check=True)
        print("\n" + "=" * 50)
        print("All tests passed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nTests failed with exit code: {e.returncode}")
        return False
    except FileNotFoundError:
        print("Error: pytest not found. Please install pytest first:")
        print("pip install pytest")
        return False

def main():
    """Main function to parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run DNATrainer tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "real", "all"], 
        default="unit",
        help="Type of tests to run (default: unit)"
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--markers", 
        nargs="+",
        help="Additional pytest markers to include"
    )
    
    args = parser.parse_args()
    
    print("DNALLM DNATrainer Test Runner")
    print("=" * 40)
    
    success = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        markers=args.markers
    )
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
