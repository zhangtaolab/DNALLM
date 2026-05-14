#!/bin/bash
# DNALLM Local CI Simulation
# Runs the exact same checks as the GitHub Actions CI pipeline.
# Use this before pushing to verify CI will pass.
#
# This script handles environment setup automatically:
#   - Creates .venv if missing
#   - Installs dependencies via uv
#   - Activates the virtual environment
#
# Usage: ./scripts/ci_checks.sh [--all]
#   --all    Include slow tests (finetune/inference real model tests)

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")    echo -e "${BLUE}ℹ️  ${message}${NC}" ;;
        "SUCCESS") echo -e "${GREEN}✅ ${message}${NC}" ;;
        "WARNING") echo -e "${YELLOW}⚠️  ${message}${NC}" ;;
        "ERROR")   echo -e "${RED}❌ ${message}${NC}" ;;
    esac
}

# Parse arguments
INCLUDE_SLOW=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            INCLUDE_SLOW=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--all]"
            echo "  --all  Include slow tests (~10 minutes)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check project root
if [ ! -f "pyproject.toml" ]; then
    print_status "ERROR" "Must run from project root (where pyproject.toml is located)"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    print_status "INFO" "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv if missing
if [ ! -d ".venv" ]; then
    print_status "INFO" "Creating virtual environment..."
    uv venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies
print_status "INFO" "Installing dependencies..."
uv pip install -e ".[test,dev]"

echo ""
print_status "INFO" "Running CI checks..."
echo "======================================"

# 1. Ruff formatting (matches CI step "Run linting and code quality checks with Ruff")
print_status "INFO" "1/4: Ruff formatting check..."
if ruff format --check .; then
    print_status "SUCCESS" "Ruff formatting check passed"
else
    print_status "ERROR" "Ruff formatting check failed. Run 'ruff format .' to fix"
    exit 1
fi

# 2. Ruff linting
print_status "INFO" "2/4: Ruff linting check..."
if ruff check . --statistics; then
    print_status "SUCCESS" "Ruff linting check passed"
else
    print_status "ERROR" "Ruff linting check failed. Run 'ruff check . --fix' to auto-fix"
    exit 1
fi

# 3. Tests with coverage (matches CI step "Run fast tests")
echo ""
if [ "$INCLUDE_SLOW" = true ]; then
    print_status "INFO" "3/4: Running full test suite (including slow tests)..."
    pytest tests/ -v --cov=dnallm --cov-report=term-missing --cov-report=xml --tb=short
else
    print_status "INFO" "3/4: Running fast tests (excludes slow)..."
    pytest tests/ -v -m "not slow" --cov=dnallm --cov-report=term-missing --cov-report=xml --tb=short
fi
print_status "SUCCESS" "Tests passed"

# 4. MyPy type checking (informational, matches CI's `|| true`)
echo ""
print_status "INFO" "4/4: MyPy type checking (informational)..."
if mypy dnallm/ --show-error-codes --pretty --exclude=dnallm/tasks/metrics/; then
    print_status "SUCCESS" "MyPy passed"
else
    print_status "WARNING" "MyPy found issues (informational, does not block CI)"
fi

echo ""
print_status "SUCCESS" "🎉 All CI checks completed successfully!"
print_status "INFO" "Your code is ready to push."
