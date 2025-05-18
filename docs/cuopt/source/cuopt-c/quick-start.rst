=================
Quickstart Guide
=================

NVIDIA cuOpt provides C API for LP and MILP. This section will show you how to install cuOpt C API and how to use it to solve LP and MILP problems.


Installation
============

pip
---

For CUDA 11.x:

.. code-block:: bash

    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    pip uninstall cuopt-thin-client
    pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu11==25.5.*

For CUDA 12.x:

.. code-block:: bash

    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    pip uninstall cuopt-thin-client
    pip install --extra-index-url=https://pypi.nvidia.com libcuopt-cu12==25.5.*


Conda
-----

NVIDIA cuOpt can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_) from the ``nvidia`` channel:

For CUDA 11.x:

.. code-block:: bash

    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    conda remove cuopt-thin-client
    conda install -c rapidsai -c conda-forge -c nvidia \
        libcuopt=25.5.* python=3.12 cuda-version=11.8

For CUDA 12.x:

.. code-block:: bash
    
    # This is deprecated module and not longer used, but share same name for the CLI, so we need to uninstall it first if it exists.
    conda remove cuopt-thin-client
    conda install -c rapidsai -c conda-forge -c nvidia \
        libcuopt=25.5.* python=3.12 cuda-version=12.8


Please visit examples under each section to learn how to use the cuOpt C API.


Smoke Test
==========

To test the installation, you can run the following command:

.. code-block:: bash

    cuopt_cli --help

This will print the help message for the cuOpt CLI. If the installation is successful, you should see the help message.

Lets try to solve a simple LP problem:

.. code-block:: bash

    echo "* optimize
   *  cost = 0.2 * VAR1 + 0.1 * VAR2
   * subject to
   *  3 * VAR1 + 4 * VAR2 <= 5.4
   *  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
   NAME   good-1
   ROWS
    N  COST
    L  ROW1
    L  ROW2
   COLUMNS
      VAR1      COST      0.2
      VAR1      ROW1      3              ROW2      2.7
      VAR2      COST      0.1
      VAR2      ROW1      4              ROW2      10.1
   RHS
      RHS1      ROW1      5.4            ROW2      4.9
   ENDATA" > sample.mps

    cuopt_cli sample.mps

This will print the solution to the console.

.. code-block:: text
    
    [2025-05-17 12:33:16:228750] [CUOPT] [info  ] Running file sample.mps
    Solving a problem with 2 constraints 2 variables (0 integers) and 4 nonzeros
    Objective offset 0.000000 scaling_factor 1.000000
    Running concurrent

    Dual simplex finished in 0.00 seconds
       Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time
          0 +0.00000000e+00 +0.00000000e+00  0.00e+00   0.00e+00     2.00e-01   0.024s
    PDLP finished
    Concurrent time:  0.026s
    Solved with dual simplex
    Status: Optimal   Objective: -3.60000000e-01  Iterations: 1  Time: 0.026s







