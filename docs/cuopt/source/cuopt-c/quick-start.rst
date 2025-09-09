=================
Quickstart Guide
=================

NVIDIA cuOpt provides C API for LP and MILP. This section will show you how to install cuOpt C API and how to use it to solve LP and MILP problems.


Installation
============

pip
---

This wheel is a Python wrapper around the C++ library and eases installation and access to libcuopt. This also helps in the pip environment to load libraries dynamically while using the Python SDK.

.. code-block:: bash

    # This is a deprecated module and no longer used, but it shares the same name for the CLI, so we need to uninstall it first if it exists.
    pip uninstall cuopt-thin-client

    # CUDA 13
    pip install --extra-index-url=https://pypi.nvidia.com \
      'nvidia-cuda-runtime-cu13==13.0.*' \
      'libcuopt-cu13==25.10.*'

    # CUDA 12
    pip install --extra-index-url=https://pypi.nvidia.com \
      'nvidia-cuda-runtime-cu12==12.9.*' \
      'libcuopt-cu12==25.10.*'


.. note::
    For development wheels which are available as nightlies, please update `--extra-index-url` to `https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/`.

.. code-block:: bash

    # CUDA 13
    pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
      'nvidia-cuda-runtime-cu13==13.0.*' \
      'libcuopt-cu13==25.10.*'

    # CUDA 12
    pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
      'nvidia-cuda-runtime-cu12==12.9.*' \
      'libcuopt-cu12==25.10.*'

Conda
-----

NVIDIA cuOpt can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_) from the ``nvidia`` channel:

.. code-block:: bash

    # This is a deprecated module and no longer used, but it shares the same name for the CLI, so we need to uninstall it first if it exists.
    conda remove cuopt-thin-client

    # CUDA 13
    conda install -c rapidsai -c conda-forge -c nvidia libcuopt=25.10.* cuda-version=13.0

    # CUDA 12
    conda install -c rapidsai -c conda-forge -c nvidia libcuopt=25.10.* cuda-version=12.9

Please visit examples under each section to learn how to use the cuOpt C API.

.. note::
    For development conda packages which are available as nightlies, please update `-c rapidsai` to `-c rapidsai-nightly`.
