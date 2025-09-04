"""Test runner for MCP server tests."""

import sys
import subprocess
from pathlib import Path


def run_tests():
    """Run all MCP server tests."""
    test_dir = Path(__file__).parent / "tests"
    
    print("Running MCP Server Tests...")
    print("=" * 50)
    
    # Run tests with pytest
    cmd = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes"
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 50)
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
