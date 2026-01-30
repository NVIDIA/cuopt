# cuOpt - GPU accelerated Optimization Engine

[![Build Status](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml/badge.svg)](https://github.com/NVIDIA/cuopt/actions/workflows/build.yaml)
[![Version](https://img.shields.io/badge/version-26.04.00-blue)](https://github.com/NVIDIA/cuopt/releases)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen)](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
[![Docker Hub](https://img.shields.io/badge/docker-nvidia%2Fcuopt-blue?logo=docker)](https://hub.docker.com/r/nvidia/cuopt)
[![Examples](https://img.shields.io/badge/examples-cuopt--examples-orange)](https://github.com/NVIDIA/cuopt-examples)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NVIDIA/cuopt-examples/blob/cuopt_examples_launcher/cuopt_examples_launcher.ipynb)
[![NVIDIA Launchable](https://img.shields.io/badge/NVIDIA-Launchable-76b900?logo=nvidia)](https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ)
[![Videos and Tutorials](https://img.shields.io/badge/Videos_and_Tutorials-red?logo=youtube)](https://docs.nvidia.com/cuopt/user-guide/latest/resources.html#cuopt-examples-and-tutorials-videos)



NVIDIA® cuOpt™ is a GPU-accelerated optimization engine that excels in mixed integer linear programming (MILP), linear programming (LP), quadratic programming (QP) and vehicle routing problems (VRP). It enables near real-time solutions for large-scale challenges with millions of variables and constraints, offering
easy integration into existing solvers and seamless deployment across hybrid and multi-cloud environments.

The core engine is written in C++ and wrapped with a C API, Python API and Server API.

For the latest stable version ensure you are on the `main` branch.

## Supported APIs

cuOpt supports the following APIs:

- C API support
    - Linear Programming (LP)
    - Mixed Integer Linear Programming (MILP)
    - Quadratic Programming (QP)
- C++ API support
    - cuOpt is written in C++ and includes a native C++ API. However, we do not provide documentation for the C++ API at this time. We anticipate that the C++ API will change significantly in the future. Use it at your own risk.
- Python support
    - Routing (TSP, VRP, and PDP)
    - Linear Programming (LP), Mixed Integer Linear Programming (MILP) and Quadratic Programming (QP)
        - Algebraic modeling Python API allows users to easily build constraints and objectives
- Server support
    - Linear Programming (LP)
    - Mixed Integer Linear Programming (MILP)
    - Routing (TSP, VRP, and PDP)

This repo is also hosted as a [COIN-OR](http://github.com/coin-or/cuopt/) project.

## Latest Release Notes:

[RELEASE-NOTES.md](RELEASE-NOTES.md)

## Installation

### CUDA/GPU requirements

* CUDA 12.0+ or CUDA 13.0+
* NVIDIA driver >= 525.60.13 (Linux) and >= 527.41 (Windows)
* Volta architecture or better (Compute Capability >=7.0)

### Python requirements

* Python >=3.11, <=3.13

### OS requirements

* Only Linux is supported and Windows via WSL2
    * x86_64 (64-bit)
    * aarch64 (64-bit)

Note: WSL2 is tested to run cuOpt, but not for building.

More details on system requirements can be found [here](https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html)

### Pip

Pip wheels are easy to install and easy to configure. Users with existing workflows who uses pip as base to build their workflows can use pip to install cuOpt.

cuOpt can be installed via `pip` from the NVIDIA Python Package Index.
Be sure to select the appropriate cuOpt package depending
on the major version of CUDA available in your environment:

For CUDA 12.x:

```bash
pip install \
  --extra-index-url=https://pypi.nvidia.com \
  nvidia-cuda-runtime-cu12==12.9.* \
  cuopt-server-cu12==26.04.* cuopt-sh-client==26.04.*
```

Development wheels are available as nightlies, please update `--extra-index-url` to `https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/` to install latest nightly packages.
```bash
pip install --pre \
  --extra-index-url=https://pypi.nvidia.com \
  --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
  cuopt-server-cu12==26.04.* cuopt-sh-client==26.04.*
```

For CUDA 13.x:

```bash
pip install \
  --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu13==26.04.* cuopt-sh-client==26.04.*
```

Development wheels are available as nightlies, please update `--extra-index-url` to `https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/` to install latest nightly packages.
```bash
pip install --pre \
  --extra-index-url=https://pypi.nvidia.com \
  --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
  cuopt-server-cu13==26.04.* cuopt-sh-client==26.04.*
```


### Conda

cuOpt can be installed with conda (via [miniforge](https://github.com/conda-forge/miniforge)):

All other dependencies are installed automatically when `cuopt-server` and `cuopt-sh-client` are installed.

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server=26.04.* cuopt-sh-client=26.04.*
```

We also provide [nightly conda packages](https://anaconda.org/rapidsai-nightly) built from the HEAD
of our latest development branch. Just replace `-c rapidsai` with `-c rapidsai-nightly`.

### Container

Users can pull the cuOpt container from the NVIDIA container registry.

```bash
# For CUDA 12.x
docker pull nvidia/cuopt:latest-cuda12.9-py3.13

# For CUDA 13.x
docker pull nvidia/cuopt:latest-cuda13.0-py3.13
```

Note: The ``latest`` tag is the latest stable release of cuOpt. If you want to use a specific version, you can use the ``<version>-cuda12.9-py3.13`` or ``<version>-cuda13.0-py3.13`` tag. For example, to use cuOpt 25.10.0, you can use the ``25.10.0-cuda12.9-py3.13`` or ``25.10.0-cuda13.0-py3.13`` tag. Please refer to `cuOpt dockerhub page <https://hub.docker.com/r/nvidia/cuopt/tags>`_ for the list of available tags.

More information about the cuOpt container can be found [here](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/quick-start.html#container-from-docker-hub).

Users who are using cuOpt for quick testing or research can use the cuOpt container. Alternatively, users who are planning to plug cuOpt as a service in their workflow can quickly start with the cuOpt container. But users are required to build security layers around the service to safeguard the service from untrusted users.

## Build from Source and Test

Please see our [guide for building cuOpt from source](CONTRIBUTING.md#setting-up-your-build-environment). This will be helpful if users want to add new features or fix bugs for cuOpt. This would also be very helpful in case users want to customize cuOpt for their own use cases which require changes to the cuOpt source code.

## Release Timeline

cuOpt follows the RAPIDS release schedule and is part of the **"others"** category in the release timeline. The release cycle consists of:

- **Development**: Active feature development and bug fixes targeting `main`
- **Burn Down**: Focus shifts to stabilization; new features should target the next release
- **Code Freeze**: Only critical bug fixes allowed; PRs require admin approval
- **Release**: Final testing, tagging, and official release

For current release timelines and dates, refer to the [RAPIDS Maintainers Docs](https://docs.rapids.ai/maintainers/).

## Contributing Guide

Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for information on how to contribute code and issues to the project.
