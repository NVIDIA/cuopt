#!/usr/bin/env python3
"""Minimal reproducer: barrier hard failure on a portfolio SOCP read from MPS.

    python repro_hardfail_mps.py data/portfolio_socp_n5000_k50_mu_t12_hardfail.mps

Expected output is a barrier log ending in

    Numerical error in objective
    ... nan residuals ...
    Search direction computation failed

With cudss_deterministic on, the failure is bit-reproducible: same iteration
count and same iterate sequence on every run and in every process.

Two further things this shows, both independent of the failure itself:

  * get_termination_reason() is not reached, because retrieving the solution
    raises first. On a non-optimal exit cuOpt leaves the primal solution in
    expanded cone coordinates (10102 entries for this model) rather than user
    space (5050), and the Python wrapper refuses to pair that with the 5050
    variable names the MPS supplied. A model built through DataModel has no
    names and so returns normally.
  * The failure needs both halves of the configuration below: step scale 0.99
    alone, with the default automatic initial point, converges on these steps.
"""

import sys

from cuopt.linear_programming import ParseMps, Solve, SolverMethod, SolverSettings
from cuopt.linear_programming.solver import solver_parameters as PAR

path = sys.argv[1] if len(sys.argv) > 1 else "data/portfolio_socp_n5000_k50_mu_t12_hardfail.mps"

s = SolverSettings()
s.set_parameter(PAR.CUOPT_METHOD, SolverMethod.Barrier)
s.set_parameter(PAR.CUOPT_AUGMENTED, 1)
s.set_parameter(PAR.CUOPT_BARRIER_ITERATIVE_REFINEMENT, 0)  # 0 = off; the default 1 converges
s.set_parameter(PAR.CUOPT_BARRIER_DUAL_INITIAL_POINT, 1)  # 1 = DualLeastSquares
s.set_parameter(PAR.CUOPT_BARRIER_STEP_SCALE, 0.99)  # default 0.9
s.set_parameter(PAR.CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES, 0)
s.set_parameter(PAR.CUOPT_CUDSS_DETERMINISTIC, True)  # only for reproducibility
s.set_optimality_tolerance(1e-8)  # default 1e-4

try:
    sol = Solve(ParseMps(path), s)
    print(f"termination: {sol.get_termination_reason()}  objective {sol.get_primal_objective()}")
except ValueError as exc:
    print(f"solution retrieval raised: {exc}")
