=================
Quickstart Guide
=================

Installation
============

pip
---

For CUDA 11.x:

.. code-block:: bash

    pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu11==25.5.* cuopt-sh==25.5.*

For CUDA 12.x:

.. code-block:: bash

    pip install --extra-index-url=https://pypi.nvidia.com cuopt-server-cu12==25.5.* cuopt-sh==25.5.*


Conda
-----

cuOpt Server can be installed with Conda (via `miniforge <https://github.com/conda-forge/miniforge>`_) from the ``nvidia`` channel:

For CUDA 11.x:

.. code-block:: bash

    conda install -c rapidsai -c conda-forge -c nvidia \
        cuopt-server=25.5.* cuopt-sh=25.5.* python=3.12 cuda-version=11.8

For CUDA 12.x:

.. code-block:: bash

    conda install -c rapidsai -c conda-forge -c nvidia \
        cuopt-server=25.5.* cuopt-sh=25.5.* python=3.12 cuda-version=12.8


Container from Docker Hub
-------------------------

NVIDIA cuOpt is also available as a container from Docker Hub:

.. code-block:: bash

    docker pull nvidia/cuopt:25.5.0

The container includes both the Python API and self-hosted server components. To run the container:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:25.5.0 /bin/bash -c "python3 -m cuopt_server.cuopt_service"

.. note::
   Make sure you have the NVIDIA Container Toolkit installed on your system to enable GPU support in containers. See the `installation guide <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ for details.

Container from NVIDIA NGC
-------------------------

Step 1: Get a subscription for `NVIDIA AI Enterprise (NVAIE) <https://www.nvidia.com/en-us/ai-enterprise/products/cuopt/>`_ to get the cuOpt container to host in your cloud.

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

        docker pull nvcr.io/nvidia/cuopt:25.5.0


The container includes both the Python API and self-hosted server components. To run the container:

.. code-block:: bash

    docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvcr.io/nvidia/cuopt/cuopt:25.5.0 /bin/bash -c "python3 -m cuopt_server.cuopt_service"


Smoke Test
----------

After installation, you can verify that cuOpt Server is working correctly by running a simple test.

.. note::

   The following example is for running the server locally. If you are using the container approach, you should comment out the server start and kill commands in the script below since the server is already running in the container.

The following example is testing with a simple routing problem constuctured as Json request and sent over HTTP to the server using ``curl``.This example is running server with few configuration options such as ``--ip`` and ``--port``.
Additional configuration options for server can be found in `Server CLI <server-api/server-cli.html>`_


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

The Open API specification for the server is available in `open-api spec <../open-api.html>`_.

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