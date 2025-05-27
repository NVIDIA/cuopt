# cuOpt - GPU accelerated Optimization Engine

[![Build Status](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml/badge.svg)](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml)

NVIDIA® cuOpt™ is a GPU-accelerated optimization engine that excels in mixed integer programming (MIP), linear programming (LP), and vehicle routing problems (VRP). It enables near real-time solutions for large-scale challenges with millions of variables and constraints, offering easy integration into existing solvers and seamless deployment across hybrid and multi-cloud environments.

For the latest stable version ensure you are on the `main` branch.

## Build from Source

Please see our [guide for building cuOpt from source](CONTRIBUTING.md#build-nvidia-cuopt-from-source)

## Contributing Guide

Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for information on how to contribute code and issues to the project.

## Resources

- [cuopt (Python) documentation](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [libcuopt (C++/CUDA) documentation](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [Examples and Notebooks](https://github.com/NVIDIA/cuopt-examples)
- [Test cuopt with Brev](https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ): Examples notebooks are pulled and hosted on [Brev](https://docs.nvidia.com/brev/latest/).

## Installation

### CUDA/GPU requirements

* CUDA 12.0+
* NVIDIA driver >= 525.60.13 (Linux) and >= 527.41 (Windows)
* Volta architecture or better (Compute Capability >=7.0)

### Pip

cuOpt can be installed via `pip` from the NVIDIA Python Package Index.
Be sure to select the appropriate cuOpt package depending
on the major version of CUDA available in your environment:

For CUDA 12.x:

```bash
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12
```

### Conda

cuOpt can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)) from the `nvidia` channel:


For CUDA 12.x:
```bash
conda install -c rapidsai -c conda-forge -c nvidia \
    cuopt=25.05 python=3.12 cuda-version=12.8
```

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

Note: cuOpt is supported only on Linux, and with Python versions 3.10 and later.