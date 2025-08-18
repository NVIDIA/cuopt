========================================
Routing Python Examples
========================================

The following example showcases how to use the ``CuOptServiceSelfHostClient`` to solve a simple routing problem.

The OpenAPI specification for the server is available in :doc:`open-api spec <../../open-api>`. The example data is structured as per the OpenAPI specification for the server, please refer :doc:`OptimizeRoutingData under "POST /cuopt/request" <../../open-api>` under schema section.

Generic Example
---------------

.. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient
    import json
    import time

    # Example data for routing problem
    # The data is structured as per the OpenAPI specification for the server, please refer /cuopt/request -> schema -> OptimizeRoutingData
    data = {"cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
            "task_data": {"task_locations": [0,1]},
            "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}}

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        polling_timeout=25,
        timeout_exception=False
    )

    def repoll(solution, repoll_tries):
        # If solver is still busy solving, the job will be assigned a request id and response is sent back in the
        # following format {"reqId": <REQUEST-ID>}.
        # Solver needs to be re-polled for response using this <REQUEST-ID>.

        if "reqId" in solution and "response" not in solution:
            req_id = solution["reqId"]
            for i in range(repoll_tries):
                solution = cuopt_service_client.repoll(req_id, response_type="dict")
                if "reqId" in solution and "response" in solution:
                    break;

                # Sleep for a second before requesting
                time.sleep(1)

        return solution

    solution = cuopt_service_client.get_optimized_routes(data)

    # Number of repoll requests to be carried out for a successful response
    repoll_tries = 500

    solution = repoll(solution, repoll_tries)

    print(json.dumps(solution, indent=4))

The response would be as follows:

  .. code-block:: json
    :linenos:

    {
    "response": {
        "solver_response": {
            "status": 0,
            "num_vehicles": 1,
            "solution_cost": 2.0,
            "objective_values": {
                "cost": 2.0
            },
            "vehicle_data": {
                "0": {
                    "task_id": [
                        "Depot",
                        "0",
                        "1",
                        "Depot"
                    ],
                    "arrival_stamp": [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "type": [
                        "Depot",
                        "Delivery",
                        "Delivery",
                        "Depot"
                    ],
                    "route": [
                        0,
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
        "total_solve_time": 0.1120915412902832
    },
    "reqId": "ebd378a3-c02a-47f3-b0a1-adec81be7cdd"
    }


.. _initial-solution-in-python:

Initial Solution
----------------

Previously run solutions or uploaded solutions can be used as the initial solution for new requests using previously run reqIds as follows:

.. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient
    import json
    import time

    data = {"cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
            "task_data": {"task_locations": [0,1]},
            "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}}

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        timeout_exception=False
    )

    # Get initial solution
    # Set delete_solution to false so it can be used in next request
    initial_solution = cuopt_service_client.get_optimized_routes(
        data, delete_solution=False
    )


    # Upload a solution returned/saved from previous request as initial solution
    initial_solution_3 = cuopt_service_client.upload_solution(initial_solution)

    # Use previous solution saved in server as initial solution to this request.
    # That solution is referenced with previous request id.
    solution = cuopt_service_client.get_optimized_routes(
        data,
        initial_ids=[
            initial_solution["reqId"],
            initial_solution_3["reqId"]
        ]
    )

    print(json.dumps(solution, indent=4))

    # Delete saved solution if not required to save space
    cuopt_service_client.delete(initial_solution["reqId"])
    cuopt_service_client.delete(initial_solution_3["reqId"])

    # Another option is to add a solution that was generated
    # to data model option as follows
    initial_solution_2 = [
        {
            "0": {
                "task_id": ["Depot", "0", "1", "Depot"],
                "type": ["Depot", "Delivery", "Delivery", "Depot"]
            }
        }
    ]

    data["initial_solution"] = initial_solution_2
    solution = cuopt_service_client.get_optimized_routes(data)

    print(json.dumps(solution, indent=4))


The initial solution in the response is ``not accepted``, because the problem is too small, and the optimal solution is found even before cuOpt could use an initial solution.

The response would be as follows:

  .. code-block:: json
    :linenos:

    {
    "response": {
        "solver_response": {
            "status": 0,
            "num_vehicles": 1,
            "solution_cost": 2.0,
            "objective_values": {
                "cost": 2.0
            },
            "vehicle_data": {
                "0": {
                    "task_id": [
                        "Depot",
                        "0",
                        "1",
                        "Depot"
                    ],
                    "arrival_stamp": [
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    "type": [
                        "Depot",
                        "Delivery",
                        "Delivery",
                        "Depot"
                    ],
                    "route": [
                        0,
                        0,
                        1,
                        0
                    ]
                }
            },
            "initial_solutions": [
                "not accepted",
            ],
            "dropped_tasks": {
                "task_id": [],
                "task_index": []
            }
        },
        "total_solve_time": 0.06160402297973633
    },
    "reqId": "ebd378a3-c02a-47f3-b0a1-adec81be7cdd"
    }

The ``data`` argument to ``get_optimized_routes`` may be a dictionary of the format shown in :doc:`Get Routes Open-API spec <../../open-api>`.
It may also be the path of a file containing such a dictionary as JSON or written using the Python *msgpack* module.
A JSON file may optionally be compressed with zlib. More details on the responses can be found under the responses schema in :doc:`"get /cuopt/request" and "get /cuopt/solution" API spec <../../open-api>`.


To enable HTTPS:

* In the case of the server using public certificates, simply enable https.

  .. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient

    data = {"cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
            "task_data": {"task_locations": [0,1]},
            "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}}

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        use_https=True
    )

* In the case of a self-signed certificate, provide the complete path to the certificate.

  .. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient

    data = {"cost_matrix_data": {"data": {"0": [[0,1],[1,0]]}},
            "task_data": {"task_locations": [0,1]},
            "fleet_data": {"vehicle_locations": [[0,0],[0,0]]}}

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        use_https=True,
        self_signed_cert=/complete/path/to/certificate
    )


  You can generate a self-signed certificate easily as follows:

  .. code-block:: shell

     openssl genrsa -out ca.key 2048
     openssl req -new -x509 -days 365 -key ca.key -subj "/C=CN/ST=GD/L=SZ/O=Acme, Inc./CN=Acme Root CA" -out ca.crt

     openssl req -newkey rsa:2048 -nodes -keyout server.key -subj "/C=CN/ST=GD/L=SZ/O=Acme, Inc./CN=*.example.com" -out server.csr
     openssl x509 -req -extfile <(printf "subjectAltName=DNS:example.com,DNS:www.example.com") -days 365 -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out server.crt


  ``server.crt`` and ``server.key`` are meant for server, ``ca.crt`` is meant for client.


More examples are available in the `Examples Notebooks Repository <https://github.com/NVIDIA/cuopt-examples>`_.

Aborting a Running Job in Thin Client
-------------------------------------

Please refer to the :ref:`aborting-thin-client` for more details.

========================================
Routing CLI Examples
========================================

Create a ``data.json`` file containing this sample data:

Routing Example
---------------

.. code-block:: shell

    echo '{"cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
     "task_data": {"task_locations": [0, 1]},
     "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]}}' > data.json

Invoke the CLI.

.. code-block:: shell

   # client's default ip address for cuOpt is localhost:5000 if ip/port are not specified
   export ip="localhost"
   export port=5000
   cuopt_sh data.json -i $ip -p $port

.. _initial-solution-in-cli:

Initial Solution in CLI
-----------------------

To use a previous solution as an initial solution for a new request ID, you are required to save the previous solution, which can be accomplished use option ``-k``. Use the previous reqId in the next request as follows:

.. code-block:: shell

   # Please update ip and port if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   reqId=$(cuopt_sh data.json -i $ip -p $port -k | sed "s/'/\"/g" | jq -r '.reqId')

   cuopt_sh data.json -i $ip -p $port -id $reqId

   # delete previous saved solutions using follwing command
   cuopt_sh -i $ip -p $port -d $reqId


Uploading a Solution
--------------------

Users can also upload a solution which might have been saved for later runs.

.. code-block:: shell

   # Please update ip and port if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000

   # Save solution to a file
   cuopt_sh data.json -i $ip -p $port | sed "s/'/\"/g" > solution.json

   # Upload the solution and get request-id generated for that
   reqId=$(cuopt_sh solution.json -us -i $ip -p $port | sed "s/'/\"/g" | jq -r '.reqId')

   # Use this request id for initial solution
   cuopt_sh data.json -i $ip -p $port -id $reqId

   # delete previous saved solutions using follwing command
   cuopt_sh -i $ip -p $port -ds $reqId


Aborting a Running Job In CLI
-----------------------------

Please refer to the :ref:`aborting-cli` for more in MILP Example.

.. note::
   Please use solver settings while using .mps files.

To enable HTTPS
----------------

* In the case of the server using public certificates, simply enable https.

  .. code-block:: shell

   cuopt_sh data.json -s -i $ip -p $port

* In the case of a self-signed certificate, provide the complete path to the certificate.

  .. code-block:: shell

   cuopt_sh data.json -s -c /complete/path/to/certificate -i $ip -p $port
