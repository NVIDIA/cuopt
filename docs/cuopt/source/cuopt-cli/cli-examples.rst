Examples
========

Basic Usage
###########

To solve a simple LP problem using cuopt_cli:

:download:`basic_lp_example.sh <examples/lp/examples/basic_lp_example.sh>`

.. literalinclude:: examples/lp/examples/basic_lp_example.sh
   :language: bash
   :linenos:

This should give you the following output:

.. code-block:: bash
   :caption: Output

   Running file sample.mps
   Solving a problem with 2 constraints 2 variables (0 integers) and 4 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   Running concurrent

   Dual simplex finished in 0.00 seconds
      Iter    Primal Obj.      Dual Obj.    Gap        Primal Res.  Dual Res.   Time
         0 +0.00000000e+00 +0.00000000e+00  0.00e+00   0.00e+00     2.00e-01   0.033s
   PDLP finished
   Concurrent time:  0.036s
   Solved with dual simplex
   Status: Optimal   Objective: -3.60000000e-01  Iterations: 1  Time: 0.036s


Mixed Integer Programming Example
#################################

Here's an example of solving a Mixed Integer Programming (MIP) problem using the CLI:

:download:`basic_milp_example.sh <examples/milp/examples/basic_milp_example.sh>`

.. literalinclude:: examples/milp/examples/basic_milp_example.sh
   :language: bash
   :linenos:

This should produce output similar to:

.. code-block:: bash
   :caption: Output

   Running file mip_sample.mps
   Solving a problem with 3 constraints 2 variables (2 integers) and 6 nonzeros
   Objective offset 0.000000 scaling_factor 1.000000
   After trivial presolve updated 3 constraints 2 variables
   Running presolve!
   After trivial presolve updated 3 constraints 2 variables
   Solving LP root relaxation
   Scaling matrix. Maximum column norm 1.225464e+00
   Dual Simplex Phase 1
   Dual feasible solution found.
   Dual Simplex Phase 2
   Iter     Objective   Primal Infeas  Perturb  Time
      1 -3.04000000e+01 7.57868205e+00 0.00e+00 0.00

   Root relaxation solution found in 3 iterations and 0.00s
   Root relaxation objective -3.01818182e+01

   Strong branching on 2 fractional variables
   | Explored | Unexplored | Objective   |    Bound    | Depth | Iter/Node |  Gap   |    Time
         0        1                +inf  -3.018182e+01      1   0.0e+00       -        0.00
   B       3        1       -2.700000e+01  -2.980000e+01      2   6.7e-01     10.4%      0.00
   B&B added a solution to population, solution queue size 0 with objective -27
   B       4        0       -2.800000e+01  -2.980000e+01      2   7.5e-01      6.4%      0.00
   B&B added a solution to population, solution queue size 1 with objective -28
   Explored 4 nodes in 0.00s.
   Absolute Gap 0.000000e+00 Objective -2.8000000000000004e+01 Lower Bound -2.8000000000000004e+01
   Optimal solution found.
   Consuming B&B solutions, solution queue size 2
   Solution objective: -28.000000 , relative_mip_gap 0.000000 solution_bound -28.000000 presolve_time 0.227418 total_solve_time 0.000000 max constraint violation 0.000000 max int violation 0.000000 max var bounds violation 0.000000 nodes 4 simplex_iterations 3


Using Solver Parameters
#######################

You can customize the solver behavior using various command line parameters. Here's a comprehensive example showing different parameter options:

:download:`solver_parameters_example.sh <examples/lp/examples/solver_parameters_example.sh>`

.. literalinclude:: examples/lp/examples/solver_parameters_example.sh
   :language: bash
   :linenos:
