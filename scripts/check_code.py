#!/usr/bin/env python3
"""
DNALLM Code Quality Check Script

Runs the same quality checks as the GitHub Actions CI pipeline.
Use this before pushing to ensure CI will pass.

Usage:
    python scripts/check_code.py              # Standard checks (matches CI test job)
    python scripts/check_code.py --fix        # Auto-fix ruff issues where possible
    python scripts/check_code.py --all        # Include slow tests (~10min)
    python scripts/check_code.py --docs       # Also run documentation verification
    python scripts/check_code.py --ci         # Exact CI simulation (no extra checks)
    python scripts/check_code.py --verbose    # Show full command output
"""

from __future__ import annotations

import argparse
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"


def print_status(status: str, message: str) -> None:
    """Print colored status message."""
    color_map = {
        "INFO": Colors.BLUE,
        "SUCCESS": Colors.GREEN,
        "WARNING": Colors.YELLOW,
        "ERROR": Colors.RED,
    }
    color = color_map.get(status, Colors.NC)
    print(f"{color}[{status}]{Colors.NC} {message}")


def run_command(
    cmd: list[str],
    description: str,
    verbose: bool = False,
    check: bool = True,
) -> tuple[bool, str]:
    """Run a command and return success status and output.

    Args:
        cmd: Command and arguments as a list.
        description: Human-readable description of the check.
        verbose: Whether to show full command output.
        check: If False, treat failures as warnings (do not set overall_success=False).
               Used for informational checks like mypy that match CI's `|| true`.
    """
    print_status("INFO", f"Running: {description}")
    if verbose:
        print(f"Command: {' '.join(cmd)}")
        print("-" * 50)

    try:
        result = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        is_success = result.returncode == 0

        if is_success:
            print_status("SUCCESS", f"{description} passed")
        elif check:
            print_status("ERROR", f"{description} failed")
        else:
            print_status("WARNING", f"{description} found issues (informational, not blocking)")

        if verbose or not is_success:
            if result.stdout:
                print(result.stdout)
            if result.stderr and result.stderr != result.stdout:
                print("Stderr:")
                print(result.stderr)

        print()
        return is_success, result.stdout + result.stderr

    except FileNotFoundError as e:
        print_status("ERROR", f"Command not found: {e}")
        print()
        return False, str(e)
    except Exception as e:
        print_status("ERROR", f"Unexpected error: {e}")
        print()
        return False, str(e)


def check_environment() -> bool:
    """Verify we are in the project root and required tools are available."""
    if not Path("pyproject.toml").exists():
        print_status(
            "ERROR",
            "Please run this script from the DNALLM root directory "
            "(where pyproject.toml is located)",
        )
        return False

    # Check if virtual environment is activated
    if not os.environ.get("VIRTUAL_ENV"):
        print_status(
            "WARNING",
            "Virtual environment not detected. Please activate it first:",
        )
        print("  source .venv/bin/activate  # Linux/macOS")
        print("  .venv\\Scripts\\activate     # Windows")
        print()
        response = input("Continue anyway? (y/N): ").strip().lower()
        if response not in ["y", "yes"]:
            return False

    # Check required tools
    required_tools = ["ruff", "pytest", "mypy"]
    missing = []
    for tool in required_tools:
        try:
            subprocess.run(  # noqa: S603
                [tool, "--version"],
                capture_output=True,
                check=True,
                timeout=10,
            )
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            missing.append(tool)

    if missing:
        print_status("ERROR", f"Missing required tools: {', '.join(missing)}")
        print("Install with: uv pip install -e '.[test,dev]'")
        return False

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="DNALLM Code Quality Check — matches GitHub Actions CI pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_code.py              # Standard checks (fast, matches CI)
  python scripts/check_code.py --fix        # Auto-fix ruff formatting/linting
  python scripts/check_code.py --all        # Include slow tests (~10 minutes)
  python scripts/check_code.py --docs       # Also verify documentation code
  python scripts/check_code.py --ci         # Exact CI simulation (no extras)
  python scripts/check_code.py --verbose    # Show full output from all checks

CI-Matching Checks:
  1. ruff format --check .
  2. ruff check . --statistics
  3. pytest tests/ -m "not slow" --cov=dnallm --cov-report=term-missing
  4. mypy dnallm/ --show-error-codes --pretty (informational, non-blocking)
        """,
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Auto-fix ruff issues where possible (format + check --fix)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Include slow tests (finetune/inference real model tests)",
    )
    parser.add_argument(
        "--docs",
        action="store_true",
        help="Also run documentation code verification (verify_docs.py)",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="Exact CI simulation: no extra checks, no coverage",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed command output",
    )

    args = parser.parse_args()

    print_status("INFO", "Starting DNALLM code quality checks...")
    print("=" * 50)

    if not check_environment():
        return 1

    overall_success = True
    step = 0

    def run_step(
        cmd: list[str],
        description: str,
        check: bool = True,
    ) -> bool:
        nonlocal step
        step += 1
        print_status("INFO", f"Step {step}: {description}")
        success, _ = run_command(cmd, description, args.verbose, check=check)
        if check and not success:
            nonlocal overall_success
            overall_success = False
        return success

    # --- Step 1: Ruff Format ---
    if args.fix:
        run_step(["ruff", "format", "."], "Ruff code formatting (auto-fix)")
    else:
        run_step(["ruff", "format", "--check", "."], "Ruff code formatting check")

    # --- Step 2: Ruff Lint ---
    ruff_cmd = ["ruff", "check", ".", "--statistics"]
    if args.fix:
        ruff_cmd.insert(3, "--fix")
    run_step(ruff_cmd, "Ruff linting" + (" (auto-fix)" if args.fix else ""))

    # --- Step 3: Tests ---
    test_cmd = [
        "pytest",
        "tests/",
        "-v",
        "--tb=short",
    ]
    test_description = "Test suite"

    if args.all:
        test_description += " (all tests including slow)"
    else:
        test_cmd.extend(["-m", "not slow"])
        test_description += " (fast tests only, excludes slow)"

    if not args.ci:
        # Local mode: include coverage report
        test_cmd.extend([
            "--cov=dnallm",
            "--cov-report=term-missing",
        ])
        test_description += " with coverage"

    run_step(test_cmd, test_description)

    # --- Step 4: MyPy (informational, matches CI's `|| true`) ---
    mypy_cmd = [
        "mypy",
        "dnallm/",
        "--show-error-codes",
        "--pretty",
        "--exclude=dnallm/tasks/metrics/",
    ]
    # Non-blocking: CI uses `|| true`, so we don't fail the script on mypy errors
    run_step(mypy_cmd, "MyPy type checking (informational)", check=False)

    # --- Step 5: Documentation Verification (optional) ---
    if args.docs:
        verify_script = Path(__file__).parent / "verify_docs.py"
        if verify_script.exists():
            doc_cmd = ["python", str(verify_script)]
            if args.verbose:
                doc_cmd.append("--verbose")
            run_step(doc_cmd, "Documentation code verification")
        else:
            print_status("WARNING", f"verify_docs.py not found at {verify_script}")

    # --- Results ---
    print("=" * 50)
    if overall_success:
        print_status("SUCCESS", "All required checks passed! Your code is ready for commit.")
        return 0
    else:
        print_status("ERROR", "Some required checks failed. Please fix before committing.")
        if not args.fix:
            print_status(
                "INFO",
                "Run with --fix to auto-fix ruff issues: python scripts/check_code.py --fix",
            )
        return 1


if __name__ == "__main__":
    sys.exit(main())
