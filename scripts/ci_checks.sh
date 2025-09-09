#!/bin/bash

# DNALLM Local CI Checks Script
# This script runs the same quality checks and tests that the GitHub Actions CI pipeline runs
# Run this script before pushing code to ensure it will pass CI

set -e  # Exit on any error

echo "🚀 Starting DNALLM Local CI Checks..."
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}ℹ️  ${message}${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ ${message}${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  ${message}${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ ${message}${NC}"
            ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_status "ERROR" "This script must be run from the project root directory (where pyproject.toml is located)"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_status "WARNING" "Virtual environment not found. Creating one..."
    if command_exists uv; then
        uv venv
    else
        print_status "ERROR" "UV not found. Please install UV first: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    fi
fi

# Activate virtual environment
print_status "INFO" "Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if needed
print_status "INFO" "Installing/updating dependencies..."
uv pip install -e ".[test,dev]"

echo ""
print_status "INFO" "Running Code Quality Checks..."
echo "======================================"

# 1. Black formatting check
print_status "INFO" "Checking code formatting with Black..."
if black --check --diff .; then
    print_status "SUCCESS" "Black formatting check passed"
else
    print_status "ERROR" "Black formatting check failed. Run 'black .' to auto-format your code"
    exit 1
fi

# 2. isort import sorting check
print_status "INFO" "Checking import sorting with isort..."
if isort --check-only --diff .; then
    print_status "SUCCESS" "isort import sorting check passed"
else
    print_status "ERROR" "isort import sorting check failed. Run 'isort .' to organize your imports"
    exit 1
fi

# 3. Flake8 linting check
print_status "INFO" "Running Flake8 linting checks..."
if flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics; then
    print_status "SUCCESS" "Flake8 critical errors check passed"
else
    print_status "ERROR" "Flake8 critical errors found. Please fix these issues"
    exit 1
fi

if flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics; then
    print_status "SUCCESS" "Flake8 style and complexity check passed"
else
    print_status "WARNING" "Flake8 style and complexity issues found. Consider fixing these for better code quality"
fi

# 4. MyPy type checking
print_status "INFO" "Running MyPy type checking..."
if mypy dnallm/ --ignore-missing-imports --no-strict-optional; then
    print_status "SUCCESS" "MyPy type checking passed"
else
    print_status "WARNING" "MyPy type checking found issues. Consider adding type annotations for better code quality"
fi

echo ""
print_status "INFO" "Running Tests..."
echo "======================================"

# 5. Run tests with coverage
print_status "INFO" "Running pytest with coverage..."
if pytest tests/ -v --cov=dnallm --cov-report=term-missing; then
    print_status "SUCCESS" "All tests passed"
else
    print_status "ERROR" "Some tests failed. Please fix the failing tests before pushing"
    exit 1
fi

echo ""
print_status "INFO" "Running Additional Checks..."
echo "======================================"

# 6. Check for common issues
print_status "INFO" "Checking for common issues..."

# Check for TODO/FIXME comments
todo_count=$(grep -r "TODO\|FIXME" dnallm/ --exclude-dir=__pycache__ --exclude-dir=.git | wc -l)
if [ "$todo_count" -gt 0 ]; then
    print_status "WARNING" "Found $todo_count TODO/FIXME comments. Consider addressing these before pushing"
else
    print_status "SUCCESS" "No TODO/FIXME comments found"
fi

# Check for print statements (should use logging instead)
print_count=$(grep -r "print(" dnallm/ --exclude-dir=__pycache__ --exclude-dir=.git | wc -l)
if [ "$print_count" -gt 0 ]; then
    print_status "WARNING" "Found $print_count print statements. Consider using logging instead for production code"
else
    print_status "SUCCESS" "No print statements found"
fi

# Check for hardcoded paths
hardcoded_paths=$(grep -r "/home\|/Users\|C:\\" dnallm/ --exclude-dir=__pycache__ --exclude-dir=.git | wc -l)
if [ "$hardcoded_paths" -gt 0 ]; then
    print_status "WARNING" "Found $hardcoded_paths potentially hardcoded paths. Use relative paths or environment variables"
else
    print_status "SUCCESS" "No hardcoded paths found"
fi

echo ""
print_status "SUCCESS" "🎉 All CI checks completed successfully!"
print_status "INFO" "Your code is ready to be pushed to GitHub"
echo ""
print_status "INFO" "Next steps:"
echo "  1. git add ."
echo "  2. git commit -m 'Your commit message'"
echo "  3. git push origin your-branch"
echo ""
print_status "INFO" "The GitHub Actions CI will run these same checks automatically"
