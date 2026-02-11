========================================
Routing Examples
========================================

This section contains examples for the cuOpt routing Python API.

TSP Batch Mode
--------------

The routing Python API supports **batch mode** for solving many TSP (or routing) instances in a single call. Instead of calling :func:`cuopt.routing.Solve` repeatedly, you build a list of :class:`cuopt.routing.DataModel` objects and call :func:`cuopt.routing.BatchSolve`. The solver runs the problems in parallel to improve throughput.

**When to use batch mode:**

- You have **many similar routing problems** (e.g., dozens or hundreds of small TSPs).
- You want to **maximize throughput** by utilizing the GPU across multiple problems at once.
- Problem sizes and structure are compatible with the same :class:`cuopt.routing.SolverSettings` (e.g., same time limit).

**Returns:** A list of :class:`cuopt.routing.Assignment` objects, one per input data model, in the same order as ``data_model_list``. Use :meth:`cuopt.routing.Assignment.get_status` and other assignment methods to inspect each solution.

The following example builds several TSPs of different sizes, solves them in one batch, and prints a short summary per solution.

:download:`tsp_batch_example.py <examples/tsp_batch_example.py>`

.. literalinclude:: examples/tsp_batch_example.py
   :language: python
   :linenos:

Sample output:

.. code-block:: text

   Solved 6 TSPs in batch.
     TSP 0 (size 5): status=SUCCESS, vehicles=1
     TSP 1 (size 8): status=SUCCESS, vehicles=1
     TSP 2 (size 10): status=SUCCESS, vehicles=1
     ...

**Notes:**

- All problems in the batch use the **same** :class:`cuopt.routing.SolverSettings` (e.g., time limit, solver options).
- Callbacks are not supported in batch mode.
- For best practices when batching many instances, see the *Add best practices for batch solving* note in the release documentation.
