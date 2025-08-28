======================================
Build Your Own Self-Hosted Thin Client
======================================


Overview
----------------

The thin client and CLI provide ease of access to the cuOpt service, but these are just references on how to communicate with the cuOpt service. Underlying details of the API have been discussed below if you want to create your own thin client.

.. note::
   Please don't use ``cuopt_self_host_client.py`` or ``cuopt_sh_client.py`` as your thin client name, just to avoid confusion in case if you had installed cuopt thin client from pypi.

Invoking cuOpt Service
-------------------------

Sending Request
###############

The ``requestBody`` contains a JSON object which will be processed by cuOpt. ``/cuopt/request`` endpoint is used to send the request.

Run server with following command in a terminal:

Here we are sharing current directory as both data and result directory.

.. code-block:: bash
   :linenos:

   ip="0.0.0.0"
   port=5000
   python -m cuopt_server.cuopt_service -d="$PWD" -r="$PWD" -i="$ip" -p="$port"

Example with JSON data as a direct string:

.. code-block:: bash
    :linenos:

    ip="0.0.0.0"
    port=5000

    curl --location "http://$ip:$port/cuopt/request" --header 'Content-Type: application/json' -d '{
        "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
        "task_data": {"task_locations": [1], "demand": [[1]], "task_time_windows": [[0, 10]], "service_times": [1]},
        "fleet_data": {"vehicle_locations":[[0, 0]], "capacities": [[2]], "vehicle_time_windows":[[0, 20]] },
        "solver_config": {"time_limit": 2}
    }'

Example with JSON data as a file:

.. code-block:: bash
    :linenos:

    ip="0.0.0.0"
    port=5000

    echo '{
         "cost_matrix_data": {"data": {"0": [[0, 1], [1, 0]]}},
         "task_data": {"task_locations": [0, 1]},
         "fleet_data": {"vehicle_locations": [[0, 0], [0, 0]]},
         "solver_config": {"time_limit": 2}
    }' > data.json

    curl --location "http://$ip:$port/cuopt/request" \
    --header 'Content-Type: application/json' \
    --header 'CUOPT-DATA-FILE: data.json' \
    -d '{}'


Success Response:

.. code-block:: json

   {"reqId":"1df28c33-8b8c-4bb7-9ff9-1e19929094c6"}


When sending files to the server, the server must be configured with appropriate data and result directories to temporarily store these files. These directories can be set using the ``-d`` and ``-r`` options when starting the server. Please refer to the :doc:`Server CLI documentation <../server-api/server-cli>` for more details on configuring these directories.

``JSON_DATA`` should follow the :doc:`spec under "POST /cuopt/request" schema <../../open-api>` described for cuOpt input.

Polling for Request Status:
---------------------------

The cuOpt service employs an asynchronous interface for invocation and result retrieval.
When you make an invocation request, the system will submit your problem to the solver and return a request id.

Users can poll the request id for status with the help of ``/cuopt/request/{request-id}`` endpoint.

.. code-block:: bash
   :linenos:

   curl --location "http://$ip:$port/cuopt/request/{request-id}"

In case the solver has completed the job, the response will be "completed".

Please refer to the :doc:`Solver status in spec using "GET /cuopt/request/{request-id}" <../../open-api>` for more details on responses.


cuOpt Result Retrieval
------------------------

Once you have received successful response from solver with status "completed", you can retrieve the result with the help of ``/cuopt/solution/{request-id}`` endpoint.

.. code-block:: bash
   :linenos:

   curl --location "http://$ip:$port/cuopt/solution/{request-id}"


This would fetch the result in JSON format. Please refer to the :doc:`Response structure in spec using "GET /cuopt/solution/{request-id}" <../../open-api>` for more details on responses.


.. important::
   It is user's responsibility to delete the request and solution files from the data and result directories respectively after retrieving the result. Please refer to the API spec for more details on deletion.
