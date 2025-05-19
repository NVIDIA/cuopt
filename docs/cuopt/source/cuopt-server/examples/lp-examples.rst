===============================
LP Python Examples
===============================

The following example showcases how to use the ``CuOptServiceSelfHostClient`` to solve a simple LP problem in normal mode and batch mode (where multiple problems are solved at once).

The OpenAPI specification for the server is available in `open-api spec <../../open-api.html>`_. The example data is structured as per the OpenAPI specification for the server, please refer `LPData <../../open-api.html#/default/postrequest_cuopt_request_post>`_ under schema section. LP and MILP share same spec.

If you want to run server locally, please run the following command in a terminal or tmux session so you can test examples in another terminal.

.. code-block:: bash
    :linenos:

    export ip="localhost"
    export port=5000
    python -m cuopt_server.cuopt_service --ip $ip --port $port

Genric Example With Normal Mode and Batch Mode
------------------------------------------------

.. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient
    import json
    import time

    # Example data for LP problem
    # The data is structured as per the OpenAPI specification for the server, please refer /cuopt/request -> schema -> LPData
    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2, 4],
            "indices": [0, 1, 0, 1],
            "values": [3.0, 4.0, 2.7, 10.1]
        },
        "constraint_bounds": {
            "upper_bounds": [5.4, 4.9],
            "lower_bounds": ["ninf", "ninf"]
        },
        "objective_data": {
            "coefficients": [-0.2, 0.1],
            "scalability_factor": 1.0,
            "offset": 0.0
        },
        "variable_bounds": {
            "upper_bounds": ["inf", "inf"],
            "lower_bounds": [0.0, 0.0]
        },
        "maximize": False,
        "solver_config": {
            "tolerances": {
                "optimality": 0.0001
            }
        }
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        polling_timeout=25,
        timeout_exception=False
    )

    # Number of repoll requests to be carried out for a successful response
    repoll_tries = 500
    
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

    # Logging callback
    def log_callback(log):
        for i in log:
            print("server-log: ", log)

    solution = cuopt_service_client.get_LP_solve(
        data, response_type="dict", logging_callback=log_callback
    )

    solution = repoll(solution, repoll_tries)

    print("---------- Normal mode ---------------  \n", json.dumps(solution, indent=4))

    # For batch mode send list of mps/dict/DataModel

    solution = cuopt_service_client.get_LP_solve(
        [data, data], response_type="dict", logging_callback=log_callback
    )
    solution = repoll(solution, repoll_tries)

    print("---------- Batch mode -----------------  \n", json.dumps(solution, indent=4))

  
The response would be as follows:

Normal mode response:

.. code-block:: json 
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": 1,
                "solution": {
                    "problem_category": 0,
                    "primal_solution": [
                        1.8,
                        0.0
                    ],
                    "dual_solution": [
                        -0.06666666666666668,
                        0.0
                    ],
                    "primal_objective": -0.36000000000000004,
                    "dual_objective": 6.92188481708744e-310,
                    "solver_time": 0.006462812423706055,
                    "vars": {},
                    "lp_statistics": {
                        "primal_residual": 6.92114652678267e-310,
                        "dual_residual": 6.9218848170975e-310,
                        "gap": 6.92114652686054e-310,
                        "nb_iterations": 1
                    },
                    "reduced_cost": [
                        0.0,
                        0.0031070813207920247
                    ],
                    "milp_statistics": {}
                }
            },
            "total_solve_time": 0.013341188430786133
        },
        "reqId": "c7f2e5a1-d210-4e2e-9308-4257d0a86c4a"
    }



Batch mode response:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": [
                {
                    "status": 1,
                    "solution": {
                        "problem_category": 0,
                        "primal_solution": [
                            1.8,
                            0.0
                        ],
                        "dual_solution": [
                            -0.06666666666666668,
                            0.0
                        ],
                        "primal_objective": -0.36000000000000004,
                        "dual_objective": 6.92188481708744e-310,
                        "solver_time": 0.005717039108276367,
                        "vars": {},
                        "lp_statistics": {
                            "primal_residual": 6.92114652678267e-310,
                            "dual_residual": 6.9218848170975e-310,
                            "gap": 6.92114652686054e-310,
                            "nb_iterations": 1
                        },
                        "reduced_cost": [
                            0.0,
                            0.0031070813207920247
                        ],
                        "milp_statistics": {}
                    }
                },
                {
                    "status": 1,
                    "solution": {
                        "problem_category": 0,
                        "primal_solution": [
                            1.8,
                            0.0
                        ],
                        "dual_solution": [
                            -0.06666666666666668,
                            0.0
                        ],
                        "primal_objective": -0.36000000000000004,
                        "dual_objective": 6.92188481708744e-310,
                        "solver_time": 0.007481813430786133,
                        "vars": {},
                        "lp_statistics": {
                            "primal_residual": 6.921146112128e-310,
                            "dual_residual": 6.9218848170975e-310,
                            "gap": 6.92114611220587e-310,
                            "nb_iterations": 1
                        },
                        "reduced_cost": [
                            0.0,
                            0.0031070813207920247
                        ],
                        "milp_statistics": {}
                    }
                }
            ],
            "total_solve_time": 0.013
        },
        "reqId": "69dc8f36-16c3-4e28-8fb9-3977eb92b480"
    }

.. note::
    Warm start is only applicable to LP and not for MILP.

Warm Start
----------

Previously run solutions can be saved and be used as warm start for new requests using previously run reqIds as follows:

.. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient
    import json

    data = {
        "csr_constraint_matrix": {
            "offsets": [0, 2, 4],
            "indices": [0, 1, 0, 1],
            "values": [3.0, 4.0, 2.7, 10.1]
        },
        "constraint_bounds": {
            "upper_bounds": [5.4, 4.9],
            "lower_bounds": ["ninf", "ninf"]
        },
        "objective_data": {
            "coefficients": [-0.2, 0.1],
            "scalability_factor": 1.0,
            "offset": 0.0
        },
        "variable_bounds": {
            "upper_bounds": ["inf", "inf"],
            "lower_bounds": [0.0, 0.0]
        },
        "maximize": False,
        "solver_config": {
            "tolerances": {
                "optimality": 0.0001
            }
        }
    }

    # If cuOpt is not running on localhost:5000, edit ip and port parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        timeout_exception=False
    )

    # Set delete_solution to false so it can be used in next request
    initial_solution = cuopt_service_client.get_LP_solve(
        data, delete_solution=False, response_type="dict"
    )

    # Use previous solution saved in server as initial solution to this request.
    # That solution is referenced with previous request id.
    solution = cuopt_service_client.get_LP_solve(
        data, warmstart_id=initial_solution["reqId"], response_type="dict"
    )

    print(json.dumps(solution, indent=4))
    
    # Delete saved solution if not required to save space
    cuopt_service_client.delete(initial_solution["reqId"])

The response would be as follows:

.. code-block:: json 
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": 1,
                "solution": {
                    "problem_category": 0,
                    "primal_solution": [
                        1.8,
                        0.0
                    ],
                    "dual_solution": [
                        -0.06666666666666668,
                        0.0
                    ],
                    "primal_objective": -0.36000000000000004,
                    "dual_objective": 6.92188481708744e-310,
                    "solver_time": 0.006613016128540039,
                    "vars": {},
                    "lp_statistics": {
                        "primal_residual": 6.921146112128e-310,
                        "dual_residual": 6.9218848170975e-310,
                        "gap": 6.92114611220587e-310,
                        "nb_iterations": 1
                    },
                    "reduced_cost": [
                        0.0,
                        0.0031070813207920247
                    ],
                    "milp_statistics": {}
                }
            },
            "total_solve_time": 0.013310909271240234
        },
        "reqId": "6d1e278f-5505-4bcc-8a33-2f7f7d6f8a30"
    }


Using MPS file directly
-----------------------

An example on using .mps files as input is shown below:

.. code-block:: python
    :linenos:

    from cuopt_sh_client import CuOptServiceSelfHostClient, ThinClientSolverSettings
    import json

    data = "sample.mps"

    mps_data = """* optimize
    *  cost = -0.2 * VAR1 + 0.1 * VAR2
    * subject to
    *  3 * VAR1 + 4 * VAR2 <= 5.4
    *  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
    NAME   good-1
    ROWS
     N  COST
     L  ROW1
     L  ROW2
    COLUMNS
        VAR1      COST      -0.2
        VAR1      ROW1      3              ROW2      2.7
        VAR2      COST      0.1
        VAR2      ROW1      4              ROW2      10.1
    RHS
        RHS1      ROW1      5.4            ROW2      4.9
    ENDATA
    """

    with open(data, "w") as file:
        file.write(mps_data)

    # If cuOpt is not running on localhost:5000, edit `ip` and `port` parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        timeout_exception=False
    )

    ss = ThinClientSolverSettings()
 
    ss.set_parameter("time_limit", 5)
    ss.set_optimality_tolerance(0.00001)
 
    solution = cuopt_service_client.get_LP_solve(data, solver_config=ss, response_type="dict")
 
    print(json.dumps(solution, indent=4))

The response is:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": 1,
                "solution": {
                    "problem_category": 0,
                    "primal_solution": [
                        1.8,
                        0.0
                    ],
                    "dual_solution": [
                        -0.06666666666666668,
                        0.0
                    ],
                    "primal_objective": -0.36000000000000004,
                    "dual_objective": 6.92188481708744e-310,
                    "solver_time": 0.008397102355957031,
                    "vars": {
                        "VAR1": 1.8,
                        "VAR2": 0.0
                    },
                    "lp_statistics": {
                        "primal_residual": 6.921146112128e-310,
                        "dual_residual": 6.9218848170975e-310,
                        "gap": 6.92114611220587e-310,
                        "nb_iterations": 1
                    },
                    "reduced_cost": [
                        0.0,
                        0.0031070813207920247
                    ],
                    "milp_statistics": {}
                }
            },
            "total_solve_time": 0.014980316162109375
        },
        "reqId": "3f36bad7-6135-4ffd-915b-858c449c7cbb"
    }


Generate Datamodel from MPS Parser
----------------------------------

Use a datamodel generated from mps file as input; this yields a solution object in response. For more details please refer to `LP/MILP parameters <../../lp-milp-settings.html>`_. 

.. code-block:: python
    :linenos:

    from cuopt_sh_client import (
        CuOptServiceSelfHostClient,
        ThinClientSolverSettings,
        PDLPSolverMode
    )
    import cuopt_mps_parser
    import json
    import time

    # -- Parse the MPS file --

    data = "sample.mps"

    mps_data = """* optimize
    *  cost = -0.2 * VAR1 + 0.1 * VAR2
    * subject to
    *  3 * VAR1 + 4 * VAR2 <= 5.4
    *  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
    NAME   good-1
    ROWS
     N  COST
     L  ROW1
     L  ROW2
    COLUMNS
        VAR1      COST      -0.2
        VAR1      ROW1      3              ROW2      2.7
        VAR2      COST      0.1
        VAR2      ROW1      4              ROW2      10.1
    RHS
        RHS1      ROW1      5.4            ROW2      4.9
    ENDATA
    """

    with open(data, "w") as file:
        file.write(mps_data)

    # Parse the MPS file and measure the time spent
    parse_start = time.time()
    data_model = cuopt_mps_parser.ParseMps(data)
    parse_time = time.time() - parse_start

    # -- Build the client object --

    # If cuOpt is not running on localhost:5000, edit `ip` and `port` parameters
    cuopt_service_client = CuOptServiceSelfHostClient(
        ip="localhost",
        port=5000,
        timeout_exception=False
    )

    # -- Set the solver settings --

    ss = ThinClientSolverSettings()

    # Set the solver mode to the same of the blogpost, Fast1.
    # Stable1 could also be used.
    ss.set_parameter("pdlp_solver_mode", PDLPSolverMode.Fast1)

    # Set the general tolerance to 1e-4 which is already the default value.
    # For more detail on optimality checkout `SolverSettings.set_optimality_tolerance()`
    ss.set_optimality_tolerance(1e-4)

    # Here you could set an iteration limit to 1000 and time limit to 10 seconds
    # By default there is no iteration limit and the max time limit is 10 minutes
    # Any problem taking more than 10 minutes to solve will stop and the current solution will be returned
    # For this example, no limit is set
    # settings.set_iteration_limit(1000)
    # settings.set_time_limit(10)
    ss.set_parameter("time_limit", 5)

    # -- Call solve --

    network_time = time.time()
    solution = cuopt_service_client.get_LP_solve(data_model, ss)
    network_time = time.time() - network_time

    # -- Retrieve the solution object and print the details --

    solution_status = solution["response"]["solver_response"]["status"]
    solution_obj = solution["response"]["solver_response"]["solution"]

    # Check Termination Reason
    # For more detail on termination reasons: checkout `Solution.get_termination_reason()`
    print("Termination Reason: (1 is Optimal)")
    print(solution_status)

    # Check found objective value
    print("Objective Value:")
    print(solution_obj["primal_objective"])

    # Check the MPS parse time
    print(f"Mps Parse time: {parse_time:.3f} sec")

    # Check network time (client call - solve time)
    network_time = network_time - (solution_obj["solver_time"])
    print(f"Network time: {network_time:.3f} sec")

    # Check solver time
    solve_time = solution_obj["solver_time"]
    print(f"Engine Solve time: {solve_time:.3f} sec")

    # Check the total end to end time (mps parsing + network + solve time)
    end_to_end_time = parse_time + network_time + solve_time
    print(f"Total end to end time: {end_to_end_time:.3f} sec")

    # Print the found decision variables
    print("Variables Values:")
    print(solution_obj["vars"])


The response would be as follows:

.. code-block:: text
   :linenos:

    Termination Reason: (1 is Optimal)
    1
    Objective Value:
    -0.36000000000000004
    Mps Parse time: 0.000 sec
    Network time: 1.062 sec
    Engine Solve time: 0.004 sec
    Total end to end time: 1.066 sec
    Variables Values:
    {'VAR1': 1.8, 'VAR2': 0.0}

Example with DataModel is available in the `Examples Notebooks Repository <https://github.com/NVIDIA/cuopt-examples>`_.

The ``data`` argument to ``get_LP_solve`` may be a dictionary of the format shown in `LP Open-API spec <../../open-api.html#operation/postrequest_cuopt_request_post>`_. More details on the response can be found under the responses schema `request and solution API spec <../../open-api.html#/default/getrequest_cuopt_request__id__get>`_.


Aborting a Running Job in Thin Client 
-------------------------------------

Please refer to the `MILP Example on Aborting a Running Job in Thin Client <milp-examples.html#aborting-a-running-job-in-thin-client>`_ for more details.


=================================================
LP CLI Examples
=================================================

Generic Example
---------------

The following examples showcase how to use the ``cuopt_sh`` CLI to solve a simple LP problem.

.. code-block:: shell

    echo '{
        "csr_constraint_matrix": {
            "offsets": [0, 2, 4],
            "indices": [0, 1, 0, 1],
            "values": [3.0, 4.0, 2.7, 10.1]
        },
        "constraint_bounds": {
            "upper_bounds": [5.4, 4.9],
            "lower_bounds": ["ninf", "ninf"]
        },
        "objective_data": {
            "coefficients": [0.2, 0.1],
            "scalability_factor": 1.0,
            "offset": 0.0
        },
        "variable_bounds": {
            "upper_bounds": ["inf", "inf"],
            "lower_bounds": [0.0, 0.0]
        },
        "maximize": "False",
        "solver_config": {
            "tolerances": {
                "optimality": 0.0001
            }
        }
     }' > data.json

Invoke the CLI.

.. code-block:: shell

   # Please update these values if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh data.json -t LP -i $ip -p $port -sl

Response is as follows:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": 1,
                "solution": {
                    "problem_category": 0,
                    "primal_solution": [1.8, 0.0],
                    "dual_solution": [-0.06666666666666668, 0.0],
                    "primal_objective": -0.36000000000000004,
                    "dual_objective": 6.92188481708744e-310,
                    "solver_time": 0.007324934005737305,
                    "vars": {},
                    "lp_statistics": {
                        "primal_residual": 6.921146112128e-310,
                        "dual_residual": 6.9218848170975e-310,
                        "gap": 6.92114611220587e-310,
                        "nb_iterations": 1
                    },
                    "reduced_cost": [0.0, 0.0031070813207920247],
                    "milp_statistics": {}
                }
            },
            "total_solve_time": 0.014164209365844727
        },
        "reqId": "4665e513-341e-483b-85eb-bced04ba598c"
    }

Warm Start in CLI
-----------------

To use a previous solution as the initial/warm start solution for a new request ID, you are required to save the previous solution, which can be accomplished use option ``-k``. Use the previous reqId in the next request as follows:

.. note::
    Warm start is only applicable to LP and not for MILP.

.. code-block:: shell

   # Please update these values if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   reqId=$(cuopt_sh -t LP data.json -i $ip -p $port -k | sed "s/'/\"/g" | jq -r '.reqId')

   cuopt_sh data.json -t LP -i $ip -p $port -wid $reqId

In case the user needs to update solver settings through CLI, the option ``-ss`` can be used as follows:

.. code-block:: shell

   # Please update these values if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh data.json -t LP -i $ip -p $port -ss '{"tolerances": {"optimality": 0.0001}, "time_limit": 5}'

In the case of batch mode, you can send a bunch of ``mps`` files at once, and acquire results. The batch mode works only for ``mps`` in the case of CLI:

.. note::
   Batch mode is not available for MILP problems.

.. code-block:: shell

    echo "* optimize
   *  cost = -0.2 * VAR1 + 0.1 * VAR2
   * subject to
   *  3 * VAR1 + 4 * VAR2 <= 5.4
   *  2.7 * VAR1 + 10.1 * VAR2 <= 4.9
   NAME   good-1
   ROWS
    N  COST
    L  ROW1
    L  ROW2
   COLUMNS
      VAR1      COST      -0.2
      VAR1      ROW1      3              ROW2      2.7
      VAR2      COST      0.1
      VAR2      ROW1      4              ROW2      10.1
   RHS
      RHS1      ROW1      5.4            ROW2      4.9
   ENDATA" > sample.mps

   # Please update these values if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh sample.mps sample.mps sample.mps -t LP -i $ip -p $port -ss '{"tolerances": {"optimality": 0.0001}, "time_limit": 5}'


Aborting a Running Job In CLI
-----------------------------

Please refer to the `MILP Example <milp-examples.html#aborting-a-running-job-in-cli>`_ for more details.

.. note::
   Please use solver settings while using .mps files.