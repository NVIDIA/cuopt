---
name: cuopt-installation
description: Install and set up NVIDIA cuOpt including pip, conda, Docker, and GPU requirements. Use when the user asks about installation, setup, environments, CUDA versions, GPU requirements, or getting started.
---

# cuOpt Installation Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Set up NVIDIA cuOpt for GPU-accelerated optimization.

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **What's your environment?**
   - Local machine with NVIDIA GPU?
   - Cloud instance (AWS, GCP, Azure)?
   - Docker/Kubernetes?
   - No GPU (need cloud solution)?

2. **What's your CUDA version?**
   ```bash
   nvcc --version
   # or
   nvidia-smi
   ```

3. **What do you need?**
   - Python API only?
   - REST Server for production?
   - C API for embedding?

4. **Package manager preference?**
   - pip
   - conda
   - Docker

## System Requirements

### GPU Requirements
- NVIDIA GPU with Compute Capability >= 7.0 (Volta or newer)
- Supported: V100, A100, H100, RTX 20xx/30xx/40xx, etc.
- NOT supported: GTX 10xx series (Pascal)

### CUDA Requirements
- CUDA 12.x or CUDA 13.x (match package suffix)
- Compatible NVIDIA driver

### Check Your System

```bash
# Check GPU
nvidia-smi

# Check CUDA version
nvcc --version

# Check compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
```

## Installation Methods

### pip (Recommended for Python)

```bash
# For CUDA 13
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13

# For CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12

# With version pinning (recommended for reproducibility)
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'
```

### pip: Server + Client

```bash
# CUDA 12 example
pip install --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu12 cuopt-sh-client

# With version pinning
pip install --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.*
```

### conda

```bash
# Python API
conda install -c rapidsai -c conda-forge -c nvidia cuopt

# Server + client
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server cuopt-sh-client

# With version pinning
conda install -c rapidsai -c conda-forge -c nvidia cuopt=26.02.*
```

### Docker (Recommended for Server)

```bash
# Pull image
docker pull nvidia/cuopt:latest-cuda12.9-py3.13

# Run server
docker run --gpus all -it --rm \
  -p 8000:8000 \
  -e CUOPT_SERVER_PORT=8000 \
  nvidia/cuopt:latest-cuda12.9-py3.13

# Verify
curl http://localhost:8000/cuopt/health
```

### Docker: Interactive Python

```bash
docker run --gpus all -it --rm nvidia/cuopt:latest-cuda12.9-py3.13 python
```

## Verification

### Verify Python Installation

```python
# Test import
import cuopt
print(f"cuOpt version: {cuopt.__version__}")

# Test GPU access
from cuopt import routing
dm = routing.DataModel(n_locations=3, n_fleet=1, n_orders=2)
print("GPU access OK")
```

### Verify Server Installation

```bash
# Start server
python -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &

# Wait and test
sleep 5
curl -s http://localhost:8000/cuopt/health | jq .
```

### Verify C API Installation

```bash
# Find header
find $CONDA_PREFIX -name "cuopt_c.h"

# Find library
find $CONDA_PREFIX -name "libcuopt.so"
```

## Common Installation Issues

### "No module named 'cuopt'"

```bash
# Check if installed
pip list | grep cuopt

# Check Python environment
which python
echo $CONDA_PREFIX

# Reinstall
pip uninstall cuopt-cu12 cuopt-cu13
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12
```

### "CUDA not available" / GPU not detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA toolkit
nvcc --version

# In Python
import torch  # if using PyTorch
print(torch.cuda.is_available())
```

### Version mismatch (CUDA 12 vs 13)

```bash
# Check installed CUDA
nvcc --version

# Install matching package
# For CUDA 12.x
pip install cuopt-cu12

# For CUDA 13.x
pip install cuopt-cu13
```

### Docker: "could not select device driver"

```bash
# Install NVIDIA Container Toolkit
# Ubuntu:
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Environment Setup

### Create Clean Environment (conda)

```bash
conda create -n cuopt-env python=3.11
conda activate cuopt-env
conda install -c rapidsai -c conda-forge -c nvidia cuopt
```

### Create Clean Environment (pip/venv)

```bash
python -m venv cuopt-env
source cuopt-env/bin/activate  # Linux/Mac
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12
```

## Cloud Deployment

### AWS

- Use p4d.24xlarge (A100) or p3.2xlarge (V100)
- Deep Learning AMI has CUDA pre-installed
- Or use provided Docker image

### GCP

- Use a2-highgpu-1g (A100) or n1-standard with T4
- Deep Learning VM has CUDA pre-installed

### Azure

- Use NC-series (T4, A100)
- Data Science VM has CUDA pre-installed

## Offline Installation

```bash
# Download wheels on connected machine
pip download --extra-index-url=https://pypi.nvidia.com cuopt-cu12 -d ./wheels

# Transfer wheels directory to offline machine

# Install from local wheels
pip install --no-index --find-links=./wheels cuopt-cu12
```

## Upgrade

```bash
# pip
pip install --upgrade --extra-index-url=https://pypi.nvidia.com cuopt-cu12

# conda
conda update -c rapidsai -c conda-forge -c nvidia cuopt

# Docker
docker pull nvidia/cuopt:latest-cuda12.9-py3.13
```

## Verification Examples

See [resources/verification_examples.md](resources/verification_examples.md) for:
- Python installation verification
- LP/MILP verification
- Server verification
- C API verification
- System requirements check
- Docker verification

## Additional Resources

- Full installation docs: `docs/cuopt/source/cuopt-python/quick-start.rst`
- Server setup: `docs/cuopt/source/cuopt-server/quick-start.rst`
- [NVIDIA cuOpt Documentation](https://docs.nvidia.com/cuopt/user-guide/latest/)
