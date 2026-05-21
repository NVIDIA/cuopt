~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cuOpt problem file parser example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Example
-------

Read MPS, QPS, or LP files (including ``.gz`` / ``.bz2`` compressed variants)
with :func:`~cuopt.linear_programming.ParseProblem`:

.. code-block:: python
    :linenos:

    from cuopt.linear_programming import ParseProblem
    from cuopt.linear_programming.problem import Problem

    # MPS / QPS
    mps_model = ParseProblem("good-mps-1.mps")

    # LP (plain or compressed)
    lp_model = ParseProblem("good-mps-1.lp")
    lp_gz = ParseProblem("good-mps-1.lp.gz")

    # High-level API
    problem = Problem.read("good-mps-1.lp")
