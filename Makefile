.PHONY: help format lint check test clean install-dev

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install-dev: ## Install development dependencies
	pip install -e ".[dev,test]"
	pre-commit install

format: ## Format code with Ruff
	ruff format .

lint: ## Lint code with Ruff
	ruff check . --statistics

check: ## Check code formatting and linting
	ruff format --check .
	ruff check . --statistics

fix: ## Auto-fix linting issues
	ruff check . --fix

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=dnallm --cov-report=html --cov-report=term-missing

clean: ## Clean up temporary files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/

ci: check test ## Run CI checks (format + lint + test)

all: format fix check test ## Format, fix, check, and test everything
