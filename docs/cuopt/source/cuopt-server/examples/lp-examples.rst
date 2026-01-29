===============================
LP Python Examples
===============================

The following example showcases how to use the ``CuOptServiceSelfHostClient`` to solve a simple LP problem in normal mode and batch mode (where multiple problems are solved at once).

The OpenAPI specification for the server is available in :doc:`open-api spec <../../open-api>`. The example data is structured as per the OpenAPI specification for the server, please refer :doc:`LPData under "POST /cuopt/request" <../../open-api>` under schema section. LP and MILP share same spec.

If you want to run server locally, please run the following command in a terminal or tmux session so you can test examples in another terminal.

.. code-block:: bash
    :linenos:

    export ip="localhost"
    export port=5000
    python -m cuopt_server.cuopt_service --ip $ip --port $port

.. _generic-example-with-normal-and-batch-mode:

Genric Example With Normal Mode and Batch Mode
------------------------------------------------

:download:`basic_lp_example.py <lp/examples/basic_lp_example.py>`

.. literalinclude:: lp/examples/basic_lp_example.py
   :language: python
   :linenos:


The response would be as follows:

Normal mode response:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "problem_category": "LP",
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
                    "status": "Optimal",
                    "solution": {
                        "problem_category": "LP",
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
                    "status": "Optimal",
                    "solution": {
                        "problem_category": "LP",
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

.. _warm-start:

Warm Start
----------

Previously run solutions can be saved and be used as warm start for new requests using previously run reqIds as follows:

:download:`warmstart_example.py <lp/examples/warmstart_example.py>`

.. literalinclude:: lp/examples/warmstart_example.py
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
                    "problem_category": "LP",
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

:download:`mps_file_example.py <lp/examples/mps_file_example.py>`

.. literalinclude:: lp/examples/mps_file_example.py
   :language: python
   :linenos:

The response is:

.. code-block:: json
    :linenos:

    {
        "response": {
            "solver_response": {
                "status": "Optimal",
                "solution": {
                    "problem_category": "LP",
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

Use a datamodel generated from mps file as input; this yields a solution object in response. For more details please refer to :doc:`LP/QP/MILP parameters <../../lp-qp-milp-settings>`.

:download:`mps_datamodel_example.py <lp/examples/mps_datamodel_example.py>`

.. literalinclude:: lp/examples/mps_datamodel_example.py
   :language: python
   :linenos:


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

The ``data`` argument to ``get_LP_solve`` may be a dictionary of the format shown in :doc:`LP Open-API spec <../../open-api>`. More details on the response can be found under the responses schema :doc:`"get /cuopt/request" and "get /cuopt/solution" API spec <../../open-api>`.


Aborting a Running Job in Thin Client
-------------------------------------

Please refer to the :ref:`aborting-thin-client` in the MILP Example for more details.


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
                "status": "Optimal",
                "solution": {
                    "problem_category": "LP",
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
   reqId=$(cuopt_sh -t LP data.json -i $ip -p $port -k | sed "s/'/\"/g" | sed 's/False/false/g' | jq -r '.reqId')

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

A sample MPS file (:download:`sample.mps <lp/examples/sample.mps>`):

.. literalinclude:: lp/examples/sample.mps
   :language: text
   :linenos:

Run the example:

.. code-block:: bash

   # Please update these values if the server is running on a different IP address or port
   export ip="localhost"
   export port=5000
   cuopt_sh sample.mps sample.mps sample.mps -t LP -i $ip -p $port -ss '{"tolerances": {"optimality": 0.0001}, "time_limit": 5}'


Aborting a Running Job In CLI
-----------------------------

Please refer to the :ref:`aborting-cli` in the MILP Example for more details.

.. note::
   Please use solver settings while using .mps files.
