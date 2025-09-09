"""Test runner for MCP server tests."""

import sys
import subprocess
from pathlib import Path
from ..utils import get_logger

logger = get_logger("dnallm.mcp.tests")


def run_tests():
    """Run all MCP server tests."""
    test_dir = Path(__file__).parent / "tests"

    logger.info("Running MCP Server Tests...")
    logger.info("=" * 50)

    # Run tests with pytest
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(test_dir),
        "-v",
        "--tb=short",
        "--color=yes",
    ]

    try:
        subprocess.run(cmd, check=True)  # noqa: S603
        logger.info("\n" + "=" * 50)
        logger.success("All tests passed!")
        return True
    except subprocess.CalledProcessError:
        logger.info("\n" + "=" * 50)
        logger.failure("Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
