========================================
MILP Python Examples
========================================

The major difference between this example and the LP example is that some of the variables are integers, so ``variable_types`` need to be shared.
The OpenAPI specification for the server is available in :doc:`open-api spec <../../open-api>`. The example data is structured as per the OpenAPI specification for the server, please refer :doc:`LPData data under "POST /cuopt/request" <../../open-api>` under schema section. LP and MILP share same spec.

Generic Example
---------------

:download:`basic_milp_example.py <milp/examples/basic_milp_example.py>`

.. literalinclude:: milp/examples/basic_milp_example.py
   :language: python
   :linenos:

The response would be as follows:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "problem_category": "MIP",
                    "primal_solution": [
                        0.0,
                        5000.0
                    ],
                    "dual_solution": null,
                    "primal_objective": 8500.0,
                    "dual_objective": null,
                    "solver_time": 0.0,
                    "vars": {
                        "x": 0.0,
                        "y": 5000.0
                    },
                    "lp_statistics": {},
                    "reduced_cost": null,
                    "milp_statistics": {
                        "mip_gap": 0.0,
                        "solution_bound": 8500.0,
                        "presolve_time": 0.007354775,
                        "max_constraint_violation": 0.0,
                        "max_int_violation": 0.0,
                        "max_variable_bound_violation": 0.0,
                        "num_nodes": 1999468624,
                        "num_simplex_iterations": 21951
                    }
                }
            },
            "total_solve_time": 0.08600544929504395
        },
        "reqId": "524e2e37-3494-4c16-bd06-2a9bfd768f76"
    }

.. _incumbent-and-logging-callback:

Incumbent and Logging Callback
------------------------------

The incumbent solution can be retrieved using a callback function as follows:

.. note::
    Incumbent solution callback is only applicable to MILP.

:download:`incumbent_callback_example.py <milp/examples/incumbent_callback_example.py>`

.. literalinclude:: milp/examples/incumbent_callback_example.py
   :language: python
   :linenos:

Log the callback response:

.. code-block:: text
   :linenos:

   server-log:  Solving a problem with 1 constraints 2 variables (2 integers) and 2 nonzeros
   server-log:  Objective offset 0.000000 scaling_factor -1.000000
   server-log:  After trivial presolve updated 1 constraints 2 variables
   server-log:  Running presolve!
   server-log:  Solving LP root relaxation
   .....

Incumbent callback response:

.. code-block:: text
   :linenos:

    Solution : [0.0, 5000.0] cost : 8500.0

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "problem_category": "MIP",
                    "primal_solution": [
                        0.0,
                        5000.0
                    ],
                    "dual_solution": null,
                    "primal_objective": 8500.0,
                    "dual_objective": null,
                    "solver_time": 0.0,
                    "vars": {
                        "x": 0.0,
                        "y": 5000.0
                    },
                    "lp_statistics": {},
                    "reduced_cost": null,
                    "milp_statistics": {
                        "mip_gap": 0.0,
                        "solution_bound": 8500.0,
                        "presolve_time": 0.001391178,
                        "max_constraint_violation": 0.0,
                        "max_int_violation": 0.0,
                        "max_variable_bound_violation": 0.0,
                        "num_nodes": 1999468624,
                        "num_simplex_iterations": 21951
                    }
                }
            },
            "total_solve_time": 0.025009632110595703
        },
        "reqId": "eb753ac0-c6a2-4fda-9ad4-ee595cddf0ec"
    }


An example with DataModel is available in the `Examples Notebooks Repository <https://github.com/NVIDIA/cuopt-examples>`_.

The ``data`` argument to ``get_LP_solve`` may be a dictionary of the format shown in :doc:`MILP Open-API spec <../../open-api>`. More details on the response can be found under responses schema in :doc:`"/cuopt/request" and "/cuopt/solution" API spec <../../open-api>`.
They can be of different format as well, please check the documentation.


.. _aborting-thin-client:

Aborting a Running Job in Thin Client
-------------------------------------

:download:`abort_job_example.py <milp/examples/abort_job_example.py>`

.. literalinclude:: milp/examples/abort_job_example.py
   :language: python
   :linenos:

========================================
MILP CLI Examples
========================================

Generic MILP Example
---------------------

The only difference between this example and the prior LP example would be the variable types provided in data.

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
        "variable_names": ["x", "y"],
        "variable_types": ["I", "I"],
        "maximize": "False",
        "solver_config": {
            "time_limit": 30
        }
     }' > data.json

Invoke the CLI:

.. code-block:: shell

   # Please update ip and port if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh data.json -t LP -i $ip -p $port -sl -il

In case the user needs to update solver settings through CLI, the option ``-ss`` can be used as follows:

.. code-block:: shell

   # Please update ip and port if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh data.json -t LP -i $ip -p $port -ss '{"time_limit": 5}'

.. note::
   Batch mode is not supported for MILP.

.. _aborting-cli:

Aborting a Running Job In CLI
-----------------------------

UUID that is returned by the solver while the solver is trying to find a solution so users can come back and check the status or query for results.

This aborts a job with UUID if it's in running state.

.. code-block:: bash

   # Please update ip and port if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh -d -r -q <UUID> -i $ip -p $port
