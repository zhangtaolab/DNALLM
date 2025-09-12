#!/usr/bin/env python3
"""
DNALLM Code Quality Check Script
This script runs all code quality checks required before committing.
It includes both CI-matching checks and additional strict checks to prevent CI
failures.

Strategy:
1. First runs exact same checks as CI workflow to ensure compatibility
2. Then runs additional strict checks (E501, E203, E402, E266) on entire
codebase
3. This ensures local checks are stricter than CI, preventing CI failures

Usage: python scripts/check_code.py [--fix] [--verbose] [--with-tests]
"""

import argparse
import os
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import NamedTuple


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    NC = "\033[0m"  # No Color


class CheckConfig(NamedTuple):
    """Configuration for a code quality check."""

    name: str
    cmd: list[str]
    description: str
    auto_fixable: bool = False


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


def _show_output(result, verbose: bool, is_success: bool) -> None:
    """Show command output based on success status and verbosity."""
    if not result.stdout:
        return

    if is_success and not verbose:
        # Show only summary lines for successful commands
        lines = result.stdout.strip().split("\n")
        for line in lines:
            if any(
                keyword in line.lower()
                for keyword in [
                    "files",
                    "issues",
                    "errors",
                    "warnings",
                    "passed",
                    "failed",
                ]
            ):
                print(f"  {line}")
    elif verbose or not is_success:
        print("Output:" if not is_success else "")
        print(result.stdout)


def _extract_error_files(output: str) -> list[str]:
    """Extract file paths from error output."""
    error_files = []
    for line in output.strip().split("\n"):
        if ":" in line and any(
            keyword in line.lower()
            for keyword in ["error", "warning", "would reformat"]
        ):
            if "Would reformat:" in line:
                error_files.append(line.replace("Would reformat:", "").strip())
            elif not line.startswith("  "):
                file_part = line.split(":")[0]
                if file_part and not file_part.startswith(("Found", "[")):
                    error_files.append(file_part)
    return error_files


def run_command(
    cmd: list[str], description: str, verbose: bool = False
) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    try:
        print_status("INFO", f"Running: {description}")
        if verbose:
            print(f"Command: {' '.join(cmd)}")
            print("-" * 40)

        result = subprocess.run(  # noqa: S603
            cmd, capture_output=True, text=True, check=False
        )
        is_success = result.returncode == 0

        if is_success:
            print_status("SUCCESS", f"{description} completed successfully")
        else:
            print_status("ERROR", f"{description} failed")
            if result.stderr:
                print("Error details:")
                print(result.stderr)

        _show_output(result, verbose, is_success)

        if not is_success and result.stdout:
            error_files = _extract_error_files(result.stdout)
            if error_files:
                print("\nFiles with issues:")
                for file_path in set(error_files):
                    print(f"  - {file_path}")

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
    """Check if we're in the right environment."""
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print_status(
            "ERROR", "Please run this script from the DNALLM root directory"
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

    # Check if required tools are available
    required_tools = ["ruff", "flake8", "mypy"]
    missing_tools = []

    for tool in required_tools:
        try:
            subprocess.run(  # noqa: S603
                [tool, "--version"],
                capture_output=True,
                check=True,
                timeout=10,
                shell=False,
            )
        except (
            subprocess.CalledProcessError,
            FileNotFoundError,
            subprocess.TimeoutExpired,
        ):
            missing_tools.append(tool)

    if missing_tools:
        print_status(
            "ERROR", f"Missing required tools: {', '.join(missing_tools)}"
        )
        print("Please install them with: uv pip install ruff flake8 mypy")
        return False

    return True


def create_check_configs(args) -> list[CheckConfig]:
    """Create check configurations based on command line arguments."""
    base_checks = [
        # CI Checks (must match exactly)
        CheckConfig(
            "Ruff Format Check (CI)",
            ["ruff", "format", "."]
            if args.fix
            else ["ruff", "format", "--check", "."],
            "Ruff code formatting check (matches CI)"
            + (" (auto-fix)" if args.fix else ""),
            auto_fixable=True,
        ),
        CheckConfig(
            "Ruff Linting Check (CI)",
            ["ruff", "check", ".", "--fix", "--statistics"]
            if args.fix
            else ["ruff", "check", ".", "--statistics"],
            "Ruff code quality check (matches CI)"
            + (" (auto-fix)" if args.fix else ""),
            auto_fixable=True,
        ),
        CheckConfig(
            "Flake8 MCP Module (CI)",
            [
                "flake8",
                "dnallm/mcp/",
                "--max-line-length=79",
                "--extend-ignore=E203,W503,C901,E402",
            ],
            "Flake8 check for MCP module (matches CI)",
        ),
        CheckConfig(
            "Flake8 Other Modules (CI)",
            [
                "flake8",
                "dnallm/",
                "--max-line-length=79",
                "--extend-ignore=E203,W503,C901,E402",
                "--exclude=dnallm/tasks/metrics/",
            ],
            "Flake8 check for other modules excluding metrics (matches CI)",
        ),
        CheckConfig(
            "MyPy Type Checking (CI)",
            [
                "mypy",
                "dnallm/",
                "--ignore-missing-imports",
                "--no-strict-optional",
                "--disable-error-code=var-annotated",
                "--disable-error-code=assignment",
                "--disable-error-code=return-value",
                "--disable-error-code=arg-type",
                "--disable-error-code=index",
                "--disable-error-code=attr-defined",
                "--disable-error-code=operator",
                "--disable-error-code=call-overload",
                "--disable-error-code=valid-type",
                "--disable-error-code=no-redef",
                "--disable-error-code=dict-item",
                "--disable-error-code=return",
                "--disable-error-code=unreachable",
                "--disable-error-code=misc",
                "--disable-error-code=import-untyped",
            ],
            "MyPy type checking (matches CI)",
        ),
        # Additional Strict Checks (more strict than CI)
        CheckConfig(
            "Line Length Check (E501) - Strict",
            ["flake8", ".", "--select=E501", "--max-line-length=79"],
            "Line length check (E501) for entire codebase - stricter than CI",
        ),
        CheckConfig(
            "Whitespace Check (E203) - Strict",
            ["flake8", ".", "--select=E203", "--max-line-length=79"],
            "Whitespace before ':' check (E203) for entire codebase - stricter\
            than CI",
        ),
        CheckConfig(
            "Import Check (E402) - Strict",
            ["flake8", ".", "--select=E402", "--max-line-length=79"],
            "Module level import not at top of file check (E402) for entire\
            codebase - stricter than CI",
        ),
        CheckConfig(
            "Block Comment Check (E266) - Strict",
            ["flake8", ".", "--select=E266", "--max-line-length=79"],
            "Too many leading '#' for block comment check (E266) for entire\
            codebase - stricter than CI",
        ),
    ]

    # Add test checks if requested
    if args.with_tests:
        base_checks.append(
            CheckConfig(
                "Test Suite",
                ["pytest", "tests/", "-v", "--tb=short"],
                "Test suite execution",
            )
        )

        if not args.fix:
            base_checks.append(
                CheckConfig(
                    "Test Coverage",
                    [
                        "pytest",
                        "tests/",
                        "--cov=dnallm",
                        "--cov-report=term-missing",
                        "--cov-report=xml",
                    ],
                    "Test coverage analysis",
                )
            )

    return base_checks


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="DNALLM Code Quality Check Script - Stricter than CI to\
                     prevent failures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Strategy:
  This script runs CI-matching checks first, then additional strict checks.
  This ensures local development catches issues before CI fails.

Examples:
  python scripts/check_code.py              # Run all checks (CI + strict)
  python scripts/check_code.py --fix        # Auto-fix issues where possible
  python scripts/check_code.py --verbose    # Show detailed output
  python scripts/check_code.py --with-tests # Include test suite execution

CI-Matching Checks:
  - Ruff format check
  - Ruff linting check
  - Flake8 MCP module check
  - Flake8 other modules check (excluding metrics)
  - MyPy type checking

Additional Strict Checks:
  - E501: Line length check (entire codebase)
  - E203: Whitespace before ':' check (entire codebase)
  - E402: Import placement check (entire codebase)
  - E266: Block comment check (entire codebase)
        """,
    )
    parser.add_argument(
        "--fix", action="store_true", help="Auto-fix issues where possible"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--with-tests",
        action="store_true",
        help="Include test suite execution",
    )

    args = parser.parse_args()

    print_status("INFO", "Starting DNALLM code quality checks...")
    print("=" * 42)

    # Check environment
    if not check_environment():
        sys.exit(1)

    # Get check configurations
    checks = create_check_configs(args)
    overall_success = True

    # Run all checks
    for i, check in enumerate(checks, 1):
        print_status("INFO", f"{i}. {check.name}...")
        success, _output = run_command(
            check.cmd, check.description, args.verbose
        )

        if not success:
            overall_success = False
            if not args.fix and check.auto_fixable:
                check_type = (
                    "formatting"
                    if "formatting" in check.name.lower()
                    else "quality"
                )
                print_status(
                    "WARNING",
                    f"Code {check_type} issues found. "
                    "Run with --fix to auto-fix.",
                )

    # Final results
    print("=" * 42)
    if overall_success:
        print_status("SUCCESS", "All checks passed! ✅")
        print_status("INFO", "Your code is ready for commit.")
        sys.exit(0)
    else:
        print_status("ERROR", "Some checks failed! ❌")
        print_status("INFO", "Please fix the issues above before committing.")
        if not args.fix:
            print_status(
                "INFO",
                "Run with --fix to auto-fix some issues: "
                "python scripts/check_code.py --fix",
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
