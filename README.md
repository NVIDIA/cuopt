# cuOpt - GPU accelerated Optimization Engine

[![Build Status](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml/badge.svg)](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml)

NVIDIA® cuOpt™ is a GPU-accelerated optimization engine that excels in mixed integer programming (MIP), linear programming (LP), and vehicle routing problems (VRP). It enables near real-time solutions for large-scale challenges with millions of variables and constraints, offering easy integration into existing solvers and seamless deployment across hybrid and multi-cloud environments.

For the latest stable version ensure you are on the `main` branch.

## Supported APIs

cuOpt supports the following APIs:

- C API support
    - Linear Programming (LP)
    - Mixed Integer Linear Programming (MILP)
- C++ API support
    - cuOpt is written in C++ and includes a native C++ API. However, we do not provide documentation for the C++ API at this time. We anticipate that the C++ API will change significantly in the future. Use it at your own risk.
- Python support
    - Routing (TSP, VRP, and PDP)
    - Linear Programming (LP) and Mixed Integer Linear Programming (MILP)
        - cuOpt includes a Python API that is used as the backend of the cuOpt server. However, we do not provide documentation for the Python API at this time. We suggest using cuOpt server to access cuOpt via Python. We anticipate that the Python API will change significantly in the future. Use it at your own risk.
- Server support
    - Linear Programming (LP)
    - Mixed Integer Linear Programming (MILP)
    - Routing (TSP, VRP, and PDP)


## Installation

### CUDA/GPU requirements

* CUDA 12.0+
* NVIDIA driver >= 525.60.13 (Linux) and >= 527.41 (Windows)
* Volta architecture or better (Compute Capability >=7.0)

### Python requirements

* Python >=3.10.x, <= 3.12.x

### OS requirements

* Only Linux is supported and Windows via WSL2
* x86_64 (64-bit)
* aarch64 (64-bit)

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

### Container 

Users can pull the cuOpt container from the NVIDIA container registry

We also provide [nightly Conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch.

```bash
docker pull nvidia/cuopt:25.5.0-cuda12.8-py312 
```
More information about the cuOpt container can be found [here](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/quick-start.html#container-from-docker-hub)


## Build from Source and Test

Please see our [guide for building cuOpt from source](CONTRIBUTING.md#setting-up-your-build-environment)

## Contributing Guide

Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for information on how to contribute code and issues to the project.

## Resources

- [libcuopt (C) documentation](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-c/index.html)
- [cuopt (Python) documentation](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/index.html)
- [cuopt (Server) documentation](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/index.html)
- [Examples and Notebooks](https://github.com/NVIDIA/cuopt-examples)
- [Test cuopt with Brev](https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ): Examples notebooks are pulled and hosted on [Brev](https://docs.nvidia.com/brev/latest/).