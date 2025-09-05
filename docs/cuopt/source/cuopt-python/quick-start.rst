=================
Quickstart Guide
=================

NVIDIA cuOpt provides a Python API for routing optimization that enables users to solve complex optimization problems efficiently. For now, cuOpt python API supports routing optimization only.

Installation
============

pip
---

.. code-block:: bash

    # CUDA 13
    pip install --extra-index-url=https://pypi.nvidia.com \
      'nvidia-cuda-runtime-cu13==13.0.*' \
      'cuopt-cu13==25.10.*'

    # CUDA 12
    pip install --extra-index-url=https://pypi.nvidia.com \
      'nvidia-cuda-runtime-cu12==12.9.*' \
      'cuopt-cu12==25.10.*'


.. note::
   For development wheels which are available as nightlies, please update `--extra-index-url` to `https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/`.

.. code-block:: bash

    # CUDA 13
    pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
      'nvidia-cuda-runtime-cu13==13.0.*' \
      'cuopt-cu13==25.10.*'

    # CUDA 12
    pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
      'nvidia-cuda-runtime-cu12==12.9.*' \
      'cuopt-cu12==25.10.*'


Conda
-----

NVIDIA cuOpt can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_) from the ``nvidia`` channel:

.. code-block:: bash

    # CUDA 13
    conda install -c rapidsai -c conda-forge -c nvidia cuopt=25.10.* cuda-version=13.0

    # CUDA 12
    conda install -c rapidsai -c conda-forge -c nvidia cuopt=25.10.* cuda-version=12.9

.. note::
   For development conda packages which are available as nightlies, please update `-c rapidsai` to `-c rapidsai-nightly`.


Container
---------

NVIDIA cuOpt is also available as a container from Docker Hub:

.. code-block:: bash

    docker pull nvidia/cuopt:latest-cuda12.9-py3.12

.. note::
   The ``latest`` tag is the latest stable release of cuOpt. If you want to use a specific version, you can use the ``<version>-cuda12.9-py3.12`` tag. For example, to use cuOpt 25.5.0, you can use the ``25.5.0-cuda12.9-py3.12`` tag. Please refer to `cuOpt dockerhub page <https://hub.docker.com/r/nvidia/cuopt>`_ for the list of available tags.

.. note::
   The nightly version of cuOpt is available as ``[VERSION]a-cuda12.9-py3.12`` tag. For example, to use cuOpt 25.8.0a, you can use the ``25.8.0a-cuda12.9-py3.12`` tag.

The container includes both the Python API and self-hosted server components. To run the container:

.. code-block:: bash

    docker run --gpus all -it --rm nvidia/cuopt:latest-cuda12.9-py3.12 /bin/bash

This will start an interactive session with cuOpt pre-installed and ready to use.

.. note::
   Make sure you have the NVIDIA Container Toolkit installed on your system to enable GPU support in containers. See the `installation guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ for details.


NVIDIA Launchable
-------------------

NVIDIA cuOpt can be tested with `NVIDIA Launchable <https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ>`_ with `example notebooks <https://github.com/NVIDIA/cuopt-examples/>`_. For more details, please refer to the `NVIDIA Launchable documentation <https://docs.nvidia.com/brev/latest/>`_.

Smoke Test
----------

After installation, you can verify that NVIDIA cuOpt is working correctly by running a simple test:

.. code-block:: bash

   python -c "
   import cudf
   from cuopt import routing
   cost_matrix = cudf.DataFrame([[0,2,2,2],[2,0,2,2],[2,2,0,2],[2,2,2,0]], dtype='float32')
   task_locations = cudf.Series([1,2,3])
   n_vehicles = 2
   dm = routing.DataModel(cost_matrix.shape[0], n_vehicles, len(task_locations))
   dm.add_cost_matrix(cost_matrix)
   dm.add_transit_time_matrix(cost_matrix.copy(deep=True))
   ss = routing.SolverSettings()
   sol = routing.Solve(dm, ss)
   print(sol.get_route())
   print('\n\n****************** Display Routes *************************')
   sol.display_routes()
   "


Example Response:

.. code-block:: text

        route  arrival_stamp  truck_id  location      type
           0            0.0         0         0     Depot
           2            2.0         0         2  Delivery
           1            4.0         0         1  Delivery
           0            6.0         0         0     Depot


      ****************** Display Routes *************************
      Vehicle-0 starts at: 0.0, completes at: 6.0, travel time: 6.0,  Route :
        0(Dpt)->2(D)->1(D)->0(Dpt)

      This results in a travel time of 6.0 to deliver all routes
