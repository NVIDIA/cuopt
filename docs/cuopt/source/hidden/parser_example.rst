~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
cuOpt problem file parser example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Example
-------

Read MPS, QPS, or LP files (including ``.gz`` / ``.bz2`` compressed variants)
with :func:`~cuopt.linear_programming.Read`:

.. code-block:: python
    :linenos:

    from cuopt.linear_programming import Read
    from cuopt.linear_programming.problem import Problem

    # MPS / QPS
    mps_model = Read("good-mps-1.mps")

    # LP (plain or compressed)
    lp_model = Read("good-mps-1.lp")
    lp_gz = Read("good-mps-1.lp.gz")

    # High-level API
    problem = Problem.read("good-mps-1.lp")
