# NBH Installation Guide

This guide provides step-by-step instructions for setting up the NBH (Next Best Hallucination) environment.

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- CUDA 11.8+ (for GPU support)
- Conda (Miniconda or Anaconda)

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/nbh.git
cd nbh

# Run the setup script
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

## Manual Installation

If the setup script fails, follow these manual steps:

### Step 1: Create Conda Environment

```bash
conda create -n nbh python=3.11 -y
conda activate nbh
```

### Step 2: Install Core Dependencies (Order Matters!)

⚠️ **IMPORTANT**: NumPy must be installed first with version < 2.0 to ensure compatibility with range_libc.

```bash
# Install numpy and scipy with version constraints
pip install "numpy>=1.24.0,<2.0" "scipy>=1.10.0,<1.14"
```

### Step 3: Install Main Requirements

```bash
pip install -r requirements.txt
```

### Step 4: Install PyTorch (with CUDA)

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Step 5: Install pyastar2d

⚠️ **IMPORTANT**: pyastar2d requires `--no-build-isolation` flag.

```bash
pip install setuptools wheel cython
pip install --no-build-isolation pyastar2d
```

### Step 6: Build range_libc

```bash
cd external/range_libc/pywrapper

# Clean any previous builds
rm -rf build *.so

# Build and install
python setup.py install

# Return to project root
cd ../../..
```

### Step 7: Verify Installation

```bash
python -c "
import numpy as np
import scipy
import torch
import pyastar2d
import range_libc
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ All dependencies installed successfully!')
"
```

## Download Data & Models

### Test Maps (KTH Dataset)

```bash
# Download test maps
./scripts/download_data.sh
# Or manually download from: [URL]
```

### Pretrained Models

```bash
# Download LAMA weights
./scripts/download_models.sh
# Or manually download from: [URL]

# The models should be placed in:
# - models/pretrained/weights/big_lama/
# - models/pretrained/weights/lama_ensemble/
```

## Troubleshooting

### Issue: `ImportError: cannot import name 'range_libc'`

**Solution**: Rebuild range_libc after numpy installation:
```bash
cd external/range_libc/pywrapper
rm -rf build *.so
python setup.py install
```

### Issue: `numpy.dtype size changed` error

**Solution**: Your numpy version is incompatible. Ensure numpy < 2.0:
```bash
pip install "numpy>=1.24.0,<2.0"
# Then rebuild range_libc
```

### Issue: `pyastar2d` build fails

**Solution**: Use `--no-build-isolation`:
```bash
pip install setuptools wheel cython
pip install --no-build-isolation pyastar2d
```

### Issue: CUDA out of memory

**Solution**: Reduce batch size in `configs/base.yaml` or use CPU:
```yaml
lama_device: 'cpu'
```

## Development Setup

For development, install additional tools:

```bash
pip install pytest black isort flake8 mypy
```

## Environment Variables

Add to your `~/.bashrc` or activate script:

```bash
export NBH_ROOT=/path/to/nbh
export PYTHONPATH=$NBH_ROOT:$PYTHONPATH
```

