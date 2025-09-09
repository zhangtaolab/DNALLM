#!/bin/bash
# Script to run MkDocs with the virtual environment
# Usage: ./mkdocs_venv.sh [mkdocs-command]

# Change to the project directory
cd "$(dirname "$0")"

# Activate the virtual environment
source .venv/bin/activate

# Run MkDocs with the provided arguments
mkdocs "$@"
