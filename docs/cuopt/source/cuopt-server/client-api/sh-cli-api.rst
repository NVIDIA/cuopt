=========================================
Self-Hosted Service Client API Reference
=========================================

Client
---------

Service Client
###############

.. autoclass:: cuopt_sh_client.CuOptServiceSelfHostClient
    :members:
    :undoc-members:

LP Supporting Classes
---------------------

.. autoclass:: cuopt_sh_client.PDLPSolverMode
    :members:
    :undoc-members:
    :no-inherited-members:

.. autoclass:: cuopt_sh_client.SolverMethod
    :members:
    :undoc-members:
    :no-inherited-members:

.. autoclass:: solution.solution.PDLPWarmStartData
    :members:
    :undoc-members:
    :no-inherited-members:

.. autoclass:: data_model.DataModel
    :members:
    :undoc-members:

.. autoclass:: cuopt_sh_client.ThinClientSolverSettings
    :members:
    :undoc-members:

.. autoclass:: solution.Solution
    :members:
    :undoc-members:


Self-Hosted Client CLI
----------------------

CLI provides an option for users to interact with the cuOpt on C level using mps files as input.

.. literalinclude:: sh-cli-help.txt
    :language: shell
    :linenos:
