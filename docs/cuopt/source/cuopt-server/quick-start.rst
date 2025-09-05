=================
Quickstart Guide
=================

Installation
============

pip
---

For CUDA 12.x:

.. code-block:: bash

    pip install --extra-index-url=https://pypi.nvidia.com \
      nvidia-cuda-runtime-cu12==12.9.* \
      cuopt-server-cu12==25.10.* \
      cuopt-sh-client==25.10.*

.. note::
   For development wheels which are available as nightlies, please update `--extra-index-url` to `https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/`.

.. code-block:: bash

    pip install --pre --extra-index-url=https://pypi.nvidia.com --extra-index-url=https://pypi.anaconda.org/rapidsai-wheels-nightly/simple/ \
      nvidia-cuda-runtime-cu12==12.9.* \
      cuopt-server-cu12==25.10.* \
      cuopt-sh-client==25.10.*

Conda
-----

cuOpt Server can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_ from the ``nvidia`` channel:

.. code-block:: bash

    conda install -c rapidsai -c conda-forge -c nvidia cuopt-server=25.10.* cuopt-sh-client=25.10.*

.. note::
   For development conda packages which are available as nightlies, please update `-c rapidsai` to `-c rapidsai-nightly`.


Container from Docker Hub
-------------------------

NVIDIA cuOpt is also available as a container from Docker Hub:

.. code-block:: bash

    docker pull nvidia/cuopt:latest-cuda12.9-py3.12

.. note::
   The ``latest`` tag is the latest stable release of cuOpt. If you want to use a specific version, you can use the ``<version>-cuda12.9-py3.12`` tag. For example, to use cuOpt 25.5.0, you can use the ``25.5.0-cuda12.9-py3.12`` tag. Please refer to `cuOpt dockerhub page <https://hub.docker.com/r/nvidia/cuopt>`_ for the list of available tags.

The container includes both the Python API and self-hosted server components. To run the container:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cuda12.9-py3.12

.. note::
   The nightly version of cuOpt is available as ``[VERSION]a-cuda12.9-py3.12`` tag. For example, to use cuOpt 25.8.0a, you can use the ``25.8.0a-cuda12.9-py3.12`` tag.

.. note::
   Make sure you have the NVIDIA Container Toolkit installed on your system to enable GPU support in containers. See the `installation guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ for details.

.. _container-from-nvidia-ngc:

Container from NVIDIA NGC
-------------------------

Step 1: Get a subscription for `NVIDIA AI Enterprise (NVAIE) <https://www.nvidia.com/en-us/data-center/products/ai-enterprise/>`_ to get the cuOpt container to host in your cloud.

Step 2: Once given access, users can find `cuOpt container <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cuopt/containers/cuopt>`_ in the NGC catalog.

Step 3: Access NGC registry:

* Log into NGC using the invite and choose the appropriate NGC org.
* Generate an NGC API key from settings. If you have not generated an API Key, you can generate it by going to the Setup option in your profile and choose Get API Key. Store this or generate a new one next time. More information can be found `here <https://docs.nvidia.com/ngc/ngc-private-registry-user-guide/index.html#generating-api-key>`_.

Step 4: Pull the container:

* Go to the container section for cuOpt and copy the pull tag for the latest image.
* Log into the nvcr.io container registry in your cluster setup, using the NGC API key as shown below.

    .. code-block:: bash

        docker login nvcr.io
        Username: $oauthtoken
        Password: <your_api_key>

* Pull the container

    .. code-block:: bash

        docker pull CONTAINER_IMAGE_PATH


The container includes both the Python API and self-hosted server components. To run the container:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 <CONTAINER_IMAGE_PATH>

NVIDIA Launchable
-------------------

NVIDIA cuOpt can be tested with `NVIDIA Launchable <https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ>`_ with `example notebooks <https://github.com/NVIDIA/cuopt-examples/>`_. For more details, please refer to the `NVIDIA Launchable documentation <https://docs.nvidia.com/brev/latest/>`_.

Smoke Test
----------

After installation, you can verify that cuOpt Server is working correctly by running a simple test.

.. note::

   The following example is for running the server locally. If you are using the container approach, you should comment out the server start and kill commands in the script below since the server is already running in the container.

The following example is testing with a simple routing problem constuctured as Json request and sent over HTTP to the server using ``curl``.This example is running server with few configuration options such as ``--ip`` and ``--port``.
Additional configuration options for server can be found in :doc:`Server CLI <server-api/server-cli>`.


Install jq and curl for basic HTTP requests and parsing JSON responses

.. code-block:: bash

    sudo apt install jq curl

Run the server and test

.. code-block:: bash

    # Set the server IP and port to be used
    SERVER_IP=0.0.0.0
    SERVER_PORT=8000

    # Start server and store PID
    python3 -m cuopt_server.cuopt_service --ip $SERVER_IP --port $SERVER_PORT > cuopt_server.log 2>&1 &
    SERVER_PID=$!

    # Check if cuOpt server is ready
    for i in {1..5}; do
        if [ "$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_IP}:${SERVER_PORT}/cuopt/health)" = "200" ]; then
            echo "cuOpt server is ready"
            break
        fi
        if [ $i -eq 5 ]; then
            echo "Error: cuOpt server failed to start"
            exit 1
        fi
        sleep 1
    done

    # Test the server with sample routing problem
    # Use /cuopt/request to submit a request to the server
    REQID=$(curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/request" \
        --header 'Content-Type: application/json' \
        --header "CLIENT-VERSION: custom" \
        -d '{
            "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
            "task_data": {"task_locations": [1], "demand": [[1]], "task_time_windows": [[0, 10]], "service_times": [1]},
            "fleet_data": {"vehicle_locations":[[0, 0]], "capacities": [[2]], "vehicle_time_windows":[[0, 20]] },
            "solver_config": {"time_limit": 2}
        }' | jq -r '.reqId')

    # Verify we got a 200 response and reqId
    if [ -z "$REQID" ]; then
        echo "Error: Failed to get reqId from server"
        exit 1
    else
        echo "Successfully submitted request with ID: $REQID"
    fi

    # Poll for results
    # Use /cuopt/solution/${REQID} to poll for results
    for i in {1..5}; do
        RESPONSE=$(curl --location "http://${SERVER_IP}:${SERVER_PORT}/cuopt/solution/${REQID}" \
            --header 'Content-Type: application/json' \
            --header "CLIENT-VERSION: custom")

        if echo "$RESPONSE" | jq -e 'has("response")' > /dev/null 2>&1; then
            echo "Got solution response:"
            echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
            break
        else
            echo "Response status:"
            echo "$RESPONSE" | jq '.' 2>/dev/null || echo "$RESPONSE"
        fi

        if [ $i -eq 5 ]; then
            echo "Error: Timed out waiting for solution"
            exit 1
        fi

        echo "Waiting for solution..."
        sleep 1
    done

    # Shutdown the server
    kill $SERVER_PID

The Open API specification for the server is available in :doc:`open-api spec <../open-api>`.

Example Response:

.. code-block:: json

    {
        "response": {
            "solver_response": {
                "status": 0,
                "num_vehicles": 1,
                "solution_cost": 2,
                "objective_values": {
                    "cost": 2
                },
                "vehicle_data": {
                    "0": {
                        "task_id": [
                            "Depot",
                            "0",
                            "Depot"
                        ],
                        "arrival_stamp": [
                            0,
                            1,
                            3
                        ],
                        "type": [
                            "Depot",
                            "Delivery",
                            "Depot"
                        ],
                        "route": [
                            0,
                            1,
                            0
                        ]
                    }
                },
                "initial_solutions": [],
                "dropped_tasks": {
                    "task_id": [],
                    "task_index": []
                }
            },
            "total_solve_time": 0.10999655723571777
        },
        "reqId": "afea72c2-6c76-45ce-bcf7-0d55049f32e4"
    }
