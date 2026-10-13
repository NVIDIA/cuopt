=================
Quickstart Guide
=================

Installation
============

Choose your install method below; the selector is pre-set for the HTTP Server. Copy the command and run it in your environment. See :doc:`../install` for all interfaces and options.

The HTTP server package depends on ``libcuopt``, so this install also provides ``cuopt_grpc_server``. The proxy forwards jobs to that binary. Installing ``libcuopt`` alone does not install the HTTP proxy.

.. install-selector::
   :default-iface: server

.. _container-from-nvidia-ngc:

Container from NVIDIA NGC
-------------------------

Step 1: Get a subscription for `NVIDIA AI Enterprise (NVAIE) <https://www.nvidia.com/en-us/data-center/products/ai-enterprise/>`_ to get the cuOpt container to host in your cloud.

Step 2: Once given access, users can find `cuOpt container <https://catalog.ngc.nvidia.com/orgs/nvidia/cuopt/containers/cuopt>`_ in the NGC catalog.

Step 3: Access NGC registry:

* Log into NGC using the invite and choose the appropriate NGC org.
* Generate an NGC API key from settings. If you have not generated an API Key, you can generate it by going to the Setup option in your profile and choose Get API Key. Store this or generate a new one next time. More information can be found `here <https://docs.nvidia.com/ngc/latest/ngc-private-registry-user-guide.html#generating-a-personal-api-key>`_.

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


The container includes the HTTP proxy and ``cuopt_grpc_server``. Start commands are in :ref:`starting-the-server` below. The :doc:`legacy HTTP server <legacy-http-server>` is deprecated and is not the default container command.

.. _starting-the-server:

Starting the Server
===================

The HTTP proxy accepts REST requests and forwards each solve to ``cuopt_grpc_server``. Start the gRPC server first so the proxy has a server to dial, then start the proxy.

From the command line, ``cuopt_grpc_server`` listens on port 5001. The proxy listens on port 5000 and forwards requests to that gRPC port:

.. code-block:: bash

    cuopt_grpc_server --port 5001 &
    python -m cuopt_server.cuopt_proxy --ip 0.0.0.0 --port 5000 --grpc-port 5001

The container starts both processes. With no command, it runs ``cuopt_grpc_server`` on port 5001 inside the container and the HTTP proxy on port 5000. Publishing port 5000 exposes the proxy:

.. code-block:: bash

    docker run --gpus all -it --rm -p 5000:5000 <CONTAINER_IMAGE_PATH>

``python -m cuopt_server.cuopt_proxy --help`` lists the HTTP proxy server flags. For further startup options, including how to specify the container mode in a Kubernetes environment, see :doc:`advanced`.

Smoke Test
----------

After installation, you can verify that the cuOpt server is working correctly by running a simple test.

The following example sends a simple routing problem as JSON over HTTP with ``curl``. It starts ``cuopt_grpc_server`` and the HTTP proxy with ``--ip`` and ``--port``. ``python -m cuopt_server.cuopt_proxy --help`` lists the other proxy flags.


Install jq and curl for basic HTTP requests and parsing JSON responses

.. code-block:: bash

    sudo apt install jq curl

Run the server and test

.. code-block:: bash

    # Set the server IP and port to be used
    SERVER_IP=0.0.0.0
    SERVER_PORT=5000

    # Start gRPC, then the HTTP proxy, and store PIDs
    cuopt_grpc_server --port 5001 > cuopt_grpc.log 2>&1 &
    GRPC_PID=$!
    python3 -m cuopt_server.cuopt_proxy --ip $SERVER_IP --port $SERVER_PORT --grpc-port 5001 > cuopt_server.log 2>&1 &
    SERVER_PID=$!
    trap 'kill $SERVER_PID $GRPC_PID 2>/dev/null || true' EXIT

    # The proxy health check waits until it can reach gRPC
    for i in {1..30}; do
        if [ "$(curl -s -o /dev/null -w "%{http_code}" http://${SERVER_IP}:${SERVER_PORT}/cuopt/health)" = "200" ]; then
            echo "cuOpt server is ready"
            break
        fi
        if [ $i -eq 30 ]; then
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
