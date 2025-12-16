#!/bin/bash
# ==============================================================================
# NBH Environment Setup Script
# ==============================================================================
# This script sets up the complete NBH development environment.
# Run from the repository root: ./scripts/setup_env.sh
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "  NBH (Next Best Hallucination) Setup"
echo "=============================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda not found. Please install Miniconda or Anaconda first."
    exit 1
fi

# Environment name
ENV_NAME="nbh"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "⚠️  Environment '${ENV_NAME}' already exists."
    read -p "Do you want to remove and recreate it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "Activating existing environment..."
        eval "$(conda shell.bash hook)"
        conda activate ${ENV_NAME}
        echo "✓ Environment '${ENV_NAME}' activated"
        exit 0
    fi
fi

echo ""
echo "Step 1/6: Creating conda environment..."
conda create -n ${ENV_NAME} python=3.11 -y

# Activate environment
eval "$(conda shell.bash hook)"
conda activate ${ENV_NAME}

echo ""
echo "Step 2/6: Installing core dependencies (numpy, scipy)..."
pip install "numpy>=1.24.0,<2.0" "scipy>=1.10.0,<1.14"

echo ""
echo "Step 3/6: Installing main requirements..."
pip install -r requirements.txt

echo ""
echo "Step 4/6: Installing PyTorch with CUDA..."
# Detect CUDA version
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release //' | sed 's/,.*//')
    echo "Detected CUDA version: ${CUDA_VERSION}"
    
    if [[ ${CUDA_VERSION} == 12.* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    elif [[ ${CUDA_VERSION} == 11.* ]]; then
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "⚠️  Unknown CUDA version, installing CPU-only PyTorch"
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
else
    echo "⚠️  CUDA not detected, installing CPU-only PyTorch"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi

echo ""
echo "Step 5/6: Installing pyastar2d..."
pip install setuptools wheel cython
pip install --no-build-isolation pyastar2d

echo ""
echo "Step 6/6: Building range_libc..."
cd external/range_libc/pywrapper
rm -rf build *.so 2>/dev/null || true
python setup.py install
cd ../../..

echo ""
echo "=============================================="
echo "  Verifying installation..."
echo "=============================================="

python -c "
import sys
print(f'Python: {sys.version}')

import numpy as np
print(f'NumPy: {np.__version__}')

import scipy
print(f'SciPy: {scipy.__version__}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import pyastar2d
print('pyastar2d: OK')

import range_libc
print('range_libc: OK')

print()
print('✓ All dependencies installed successfully!')
"

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run NBH exploration:"
echo "  python -m nbh.explore --mode nbh --local_planner astar"
echo ""

