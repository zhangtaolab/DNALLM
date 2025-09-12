#!/usr/bin/env python3
"""
DNALLM Code Quality Check Script
This script runs all code quality checks required before committing.
Usage: python scripts/check_code.py [--fix] [--verbose]
"""

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
    NC = "\033[0m"  # No Color


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


def _show_success_output(result, verbose: bool) -> None:
    """Show output for successful commands."""
    if result.stdout and not verbose:
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
    elif verbose and result.stdout:
        print(result.stdout)


def _extract_error_files(output: str) -> list[str]:
    """Extract file paths from error output."""
    lines = output.strip().split("\n")
    error_files = []
    for line in lines:
        if ":" in line and (
            "error" in line.lower()
            or "warning" in line.lower()
            or "would reformat" in line.lower()
        ):
            if "Would reformat:" in line:
                file_path = line.replace("Would reformat:", "").strip()
                error_files.append(file_path)
            elif ":" in line and not line.startswith("  "):
                file_part = line.split(":")[0]
                if (
                    file_part
                    and not file_part.startswith("Found")
                    and not file_part.startswith("[")
                ):
                    error_files.append(file_part)
    return error_files


def _show_error_output(result) -> None:
    """Show output for failed commands."""
    if result.stderr:
        print("Error details:")
        print(result.stderr)
    if result.stdout:
        print("Output:")
        print(result.stdout)
        error_files = _extract_error_files(result.stdout)
        if error_files:
            print("\nFiles with issues:")
            for file_path in set(error_files):
                print(f"  - {file_path}")


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

        if result.returncode == 0:
            print_status("SUCCESS", f"{description} completed successfully")
            _show_success_output(result, verbose)
        else:
            print_status("ERROR", f"{description} failed")
            _show_error_output(result)

        print()
        return result.returncode == 0, result.stdout + result.stderr

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


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="DNALLM Code Quality Check Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/check_code.py              # Run code quality checks only
  python scripts/check_code.py --fix        # Auto-fix issues where possible
  python scripts/check_code.py --verbose    # Show detailed output
  python scripts/check_code.py --with-tests # Include test suite execution
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

    # Track overall success
    overall_success = True

    # Define all checks
    checks = [
        {
            "name": "Code Formatting",
            "cmd": ["ruff", "format", "."]
            if args.fix
            else ["ruff", "format", "--check", "."],
            "description": "Code formatting check"
            + (" (auto-fix)" if args.fix else ""),
        },
        {
            "name": "Code Quality (Ruff)",
            "cmd": ["ruff", "check", ".", "--fix", "--statistics"]
            if args.fix
            else ["ruff", "check", ".", "--statistics"],
            "description": "Code quality check"
            + (" (auto-fix)" if args.fix else ""),
        },
        {
            "name": "Flake8 (MCP Module)",
            "cmd": [
                "flake8",
                "dnallm/mcp/",
                "--max-line-length=79",
                "--extend-ignore=E203,W503,C901,E402",
            ],
            "description": "Flake8 check for MCP module",
        },
        {
            "name": "Type Checking (MyPy)",
            "cmd": [
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
            "description": "Type checking with MyPy",
        },
    ]

    # Add test checks only if explicitly requested
    if args.with_tests:
        checks.append({
            "name": "Test Suite",
            "cmd": ["pytest", "tests/", "-v", "--tb=short"],
            "description": "Test suite execution",
        })

        # Add coverage check if not in fix mode
        if not args.fix:
            checks.append({
                "name": "Test Coverage",
                "cmd": [
                    "pytest",
                    "tests/",
                    "--cov=dnallm",
                    "--cov-report=term-missing",
                    "--cov-report=xml",
                ],
                "description": "Test coverage analysis",
            })

    # Run all checks
    for i, check in enumerate(checks, 1):
        print_status("INFO", f"{i}. {check['name']}...")
        success, _output = run_command(
            check["cmd"], check["description"], args.verbose
        )

        if not success:
            overall_success = False
            if not args.fix and "formatting" in check["name"].lower():
                print_status(
                    "WARNING",
                    "Code formatting issues found. Run with --fix to auto-fix.",
                )
            elif not args.fix and "quality" in check["name"].lower():
                print_status(
                    "WARNING",
                    "Code quality issues found. Run with --fix to auto-fix.",
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
                "Run with --fix to auto-fix some issues: python scripts/check_code.py --fix",
            )
        sys.exit(1)


if __name__ == "__main__":
    main()
