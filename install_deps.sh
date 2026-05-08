#!/bin/bash
set -e

echo "=== DNALLM Dependencies Install (RTX 5090, CUDA 12.8) ==="
echo ""

# Step 1: Install everything except mamba
echo "[1/3] Installing core dependencies + torch (CUDA 12.8)..."
.venv/bin/python -m pip install \
    -e ".[dev,test,cuda128]" \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128

echo ""
echo "[2/3] Verifying torch CUDA support..."
.venv/bin/python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "[3/3] Installing mamba (native CUDA compilation, no build isolation)..."
.venv/bin/python -m pip install \
    -e ".[mamba]" \
    -i https://mirrors.aliyun.com/pypi/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    --no-build-isolation \
    --no-cache-dir

echo ""
echo "=== Installation complete ==="
