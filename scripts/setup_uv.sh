#!/bin/bash

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create virtual environment
uv venv

# Install dependencies with development extras
uv pip install -e ".[dev,notebook,mcp]"

# Lock dependencies
uv lock

echo "DNALLM environment setup complete!"
echo "Activate with: source .venv/bin/activate" 