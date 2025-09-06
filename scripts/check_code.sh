#!/bin/bash

# DNALLM Code Quality Check Script
# This script runs all code quality checks required before committing
# Usage: ./scripts/check_code.sh [--fix] [--verbose]

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
FIX_MODE=false
VERBOSE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fix)
            FIX_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--fix] [--verbose]"
            echo "  --fix     Auto-fix issues where possible"
            echo "  --verbose Show detailed output"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}[INFO]${NC} $message"
            ;;
        "SUCCESS")
            echo -e "${GREEN}[SUCCESS]${NC} $message"
            ;;
        "WARNING")
            echo -e "${YELLOW}[WARNING]${NC} $message"
            ;;
        "ERROR")
            echo -e "${RED}[ERROR]${NC} $message"
            ;;
    esac
}

# Function to run command with optional verbose output
run_command() {
    local cmd="$1"
    local description="$2"
    
    print_status "INFO" "Running: $description"
    
    if [ "$VERBOSE" = true ]; then
        echo "Command: $cmd"
        echo "----------------------------------------"
    fi
    
    if eval "$cmd"; then
        print_status "SUCCESS" "$description completed successfully"
    else
        print_status "ERROR" "$description failed"
        return 1
    fi
    
    echo ""
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_status "ERROR" "Please run this script from the DNALLM root directory"
    exit 1
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    print_status "WARNING" "Virtual environment not detected. Please activate it first:"
    echo "  source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "INFO" "Starting DNALLM code quality checks..."
echo "=========================================="

# Track overall success
OVERALL_SUCCESS=true

# 1. Code Formatting Check
print_status "INFO" "1. Checking code formatting with Ruff..."
if [ "$FIX_MODE" = true ]; then
    run_command "ruff format ." "Code formatting (auto-fix)"
else
    run_command "ruff format --check ." "Code formatting check"
fi
if [ $? -ne 0 ]; then
    OVERALL_SUCCESS=false
    if [ "$FIX_MODE" = false ]; then
        print_status "WARNING" "Code formatting issues found. Run with --fix to auto-fix."
    fi
fi

# 2. Code Quality Check with Ruff
print_status "INFO" "2. Running code quality checks with Ruff..."
if [ "$FIX_MODE" = true ]; then
    run_command "ruff check . --fix --statistics" "Code quality check (auto-fix)"
else
    run_command "ruff check . --statistics" "Code quality check"
fi
if [ $? -ne 0 ]; then
    OVERALL_SUCCESS=false
    if [ "$FIX_MODE" = false ]; then
        print_status "WARNING" "Code quality issues found. Run with --fix to auto-fix."
    fi
fi

# 3. Flake8 Check for MCP Module
print_status "INFO" "3. Running flake8 check for MCP module..."
run_command "flake8 dnallm/mcp/ --max-line-length=79 --extend-ignore=E203,W503,C901,E402" "Flake8 check for MCP module"
if [ $? -ne 0 ]; then
    OVERALL_SUCCESS=false
fi

# 4. Type Checking with MyPy
print_status "INFO" "4. Running type checking with MyPy..."
run_command "mypy dnallm/ --ignore-missing-imports --no-strict-optional --disable-error-code=var-annotated --disable-error-code=assignment --disable-error-code=return-value --disable-error-code=arg-type --disable-error-code=index --disable-error-code=attr-defined --disable-error-code=operator --disable-error-code=call-overload --disable-error-code=valid-type --disable-error-code=no-redef --disable-error-code=dict-item --disable-error-code=return --disable-error-code=unreachable --disable-error-code=misc --disable-error-code=import-untyped" "Type checking with MyPy"
if [ $? -ne 0 ]; then
    OVERALL_SUCCESS=false
fi

# 5. Test Suite
print_status "INFO" "5. Running test suite..."
run_command "pytest tests/ -v --tb=short" "Test suite execution"
if [ $? -ne 0 ]; then
    OVERALL_SUCCESS=false
fi

# 6. Test Coverage (optional, only if not in fix mode)
if [ "$FIX_MODE" = false ]; then
    print_status "INFO" "6. Running test coverage analysis..."
    run_command "pytest tests/ --cov=dnallm --cov-report=term-missing --cov-report=xml" "Test coverage analysis"
    if [ $? -ne 0 ]; then
        OVERALL_SUCCESS=false
    fi
fi

# Final Results
echo "=========================================="
if [ "$OVERALL_SUCCESS" = true ]; then
    print_status "SUCCESS" "All checks passed! ✅"
    print_status "INFO" "Your code is ready for commit."
    exit 0
else
    print_status "ERROR" "Some checks failed! ❌"
    print_status "INFO" "Please fix the issues above before committing."
    if [ "$FIX_MODE" = false ]; then
        print_status "INFO" "Run with --fix to auto-fix some issues: ./scripts/check_code.sh --fix"
    fi
    exit 1
fi
