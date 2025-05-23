=================
Quickstart Guide
=================

NVIDIA cuOpt provides C API for LP and MILP. This section will show you how to install cuOpt C API and how to use it to solve LP and MILP problems.


Installation
============

pip
---

For CUDA 12.x:

.. code-block:: bash

    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    pip uninstall cuopt-thin-client
    pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu12==25.5.*


Conda
-----

NVIDIA cuOpt can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_) from the ``nvidia`` channel:

For CUDA 12.x:

.. code-block:: bash
    
    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    conda remove cuopt-thin-client
    conda install -c rapidsai -c conda-forge -c nvidia \
        libcuopt=25.5.* python=3.12 cuda-version=12.8


Please visit examples under each section to learn how to use the cuOpt C API.