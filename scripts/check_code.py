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


def _show_output(
    result, verbose: bool, is_success: bool, description: str = ""
) -> None:
    """Show command output based on success status and verbosity."""
    if not result.stdout:
        return

    # Special handling for test suite - always show progress
    is_test_suite = "test" in description.lower() and (
        "pytest" in result.stdout or "test session" in result.stdout
    )

    if is_test_suite:
        # For test suite, show all output to display progress
        print(result.stdout)
        if result.stderr and result.stderr != result.stdout:
            print(result.stderr)
    elif is_success and not verbose:
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

        # For failed commands, also show stderr if it contains useful info
        if not is_success and result.stderr and result.stderr != result.stdout:
            print("Error details:")
            print(result.stderr)


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


def _extract_detailed_errors(output: str) -> list[str]:
    """Extract detailed error information including file,
    line, and error type.
    """
    detailed_errors = []
    lines = output.strip().split("\n")

    for i, line in enumerate(lines):
        # Look for error patterns like "F841 Local variable `x` is assigned to
        # but never used"
        error_codes = ["F401", "F841", "E501", "E203", "E402", "E266"]
        if any(code in line for code in error_codes):
            # This is an error line, get the file and line info
            if i + 1 < len(lines) and "-->" in lines[i + 1]:
                file_line = lines[i + 1]
                if "--> " in file_line:
                    file_path = file_line.split("--> ")[1].split(":")[0]
                    line_num = (
                        file_line.split(":")[1].split(":")[0]
                        if ":" in file_line
                        else "?"
                    )
                    detailed_errors.append(f"{file_path}:{line_num} - {line}")
                else:
                    detailed_errors.append(line)
            else:
                detailed_errors.append(line)

    return detailed_errors


def colorize_pytest_line(line: str) -> str:
    """Add colors to pytest output lines."""
    line = line.rstrip()

    # Color patterns for pytest output
    if "PASSED" in line:
        return f"{Colors.GREEN}{line}{Colors.NC}"
    elif "FAILED" in line:
        return f"{Colors.RED}{line}{Colors.NC}"
    elif "ERROR" in line:
        return f"{Colors.RED}{line}{Colors.NC}"
    elif "SKIPPED" in line:
        return f"{Colors.YELLOW}{line}{Colors.NC}"
    elif "WARNING" in line or "WARN" in line:
        return f"{Colors.YELLOW}{line}{Colors.NC}"
    elif "XFAIL" in line:
        return f"{Colors.YELLOW}{line}{Colors.NC}"
    elif "XPASS" in line:
        return f"{Colors.YELLOW}{line}{Colors.NC}"
    elif "FAILURES" in line or "ERRORS" in line:
        return f"{Colors.RED}{line}{Colors.NC}"
    elif "short test summary info" in line:
        return f"{Colors.RED}{line}{Colors.NC}"
    elif "passed" in line and "failed" in line and "warnings" in line:
        # Summary line like "2 passed, 1 failed, 1 warning in 5.80s"
        if "failed" in line and int(line.split()[0]) > 0:
            return f"{Colors.RED}{line}{Colors.NC}"
        elif "passed" in line and "failed" not in line:
            return f"{Colors.GREEN}{line}{Colors.NC}"
        else:
            return f"{Colors.YELLOW}{line}{Colors.NC}"
    else:
        return line


def run_command_realtime(
    cmd: list[str], description: str, verbose: bool = False
) -> tuple[bool, str]:
    """Run a command with real-time output for test suites."""
    try:
        print_status("INFO", f"Running: {description}")
        if verbose:
            print(f"Command: {' '.join(cmd)}")
            print("-" * 40)

        # Start the process without capturing output
        process = subprocess.Popen(  # noqa: S603
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        output_lines = []

        # Read output line by line and print immediately with colors
        for line in iter(process.stdout.readline, ""):
            if line:
                # Colorize and print line without extra newline
                colored_line = colorize_pytest_line(line)
                print(colored_line)
                output_lines.append(line)

        # Wait for process to complete
        process.wait()
        is_success = process.returncode == 0

        # Combine all output
        full_output = "".join(output_lines)

        # Only show status if verbose or failed
        if verbose or not is_success:
            if is_success:
                print_status(
                    "SUCCESS", f"{description} completed successfully"
                )
            else:
                print_status("ERROR", f"{description} failed")
        elif is_success:
            # For successful test runs, just show a brief success indicator
            print()

        return is_success, full_output

    except FileNotFoundError as e:
        print_status("ERROR", f"Command not found: {e}")
        print()
        return False, str(e)
    except Exception as e:
        print_status("ERROR", f"Unexpected error: {e}")
        print()
        return False, str(e)


def run_command(
    cmd: list[str], description: str, verbose: bool = False
) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    # Check if this is a test suite command
    is_test_suite = "pytest" in cmd[0] and any(
        "test" in arg.lower() for arg in cmd
    )

    if is_test_suite:
        return run_command_realtime(cmd, description, verbose)

    # Use original method for other commands
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

        _show_output(result, verbose, is_success, description)

        if not is_success and result.stdout:
            error_files = _extract_error_files(result.stdout)
            detailed_errors = _extract_detailed_errors(result.stdout)

            if detailed_errors:
                print("\nDetailed errors:")
                for error in detailed_errors:
                    print(f"  {error}")
            elif error_files:
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
                "--exclude=dnallm/tasks/metrics/",
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
        # Determine test selection based on arguments
        test_args = ["pytest", "tests/", "-v", "--tb=short", "--durations=10"]
        test_description = "Test suite execution"

        # Handle slow test selection
        if args.slow and args.not_slow:
            print_status(
                "WARNING",
                "Both --slow and --not-slow specified. Using --slow.",
            )
            test_args.extend(["-m", "slow"])
            test_description += " (slow tests only)"
        elif args.slow:
            test_args.extend(["-m", "slow"])
            test_description += " (slow tests only)"
        elif args.not_slow:
            test_args.extend(["-m", "not slow"])
            test_description += " (excluding slow tests)"
        else:
            # Default behavior: exclude slow tests
            test_args.extend(["-m", "not slow"])
            test_description += " (excluding slow tests - default)"
            print_status(
                "INFO", "Running tests excluding slow ones by default (~24s)."
            )
            print_status(
                "INFO",
                "Use --slow to include all tests (~10min) or --not-slow to be\
                    explicit.",
            )

        base_checks.append(
            CheckConfig(
                "Test Suite",
                test_args,
                test_description,
            )
        )

        if not args.fix:
            base_checks.append(
                CheckConfig(
                    "Test Coverage",
                    [
                        "pytest",
                        "tests/",
                        "-v",
                        "--durations=10",
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
  python scripts/check_code.py                    # Run all checks (CI + strict)  # noqa: E501
  python scripts/check_code.py --fix              # Auto-fix issues where possible  # noqa: E501
  python scripts/check_code.py --verbose          # Show detailed output  # noqa: E501
  python scripts/check_code.py --with-tests       # Include tests (excludes slow by default)  # noqa: E501
  python scripts/check_code.py --with-tests --slow # Include all tests including slow ones  # noqa: E501
  python scripts/check_code.py --with-tests --not-slow # Explicitly exclude slow tests  # noqa: E501

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
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests (finetune and inference real model tests)",
    )
    parser.add_argument(
        "--not-slow",
        action="store_true",
        help="Exclude slow tests (default behavior when --with-tests is used)",
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
