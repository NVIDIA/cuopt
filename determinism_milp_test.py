# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap

WORKER = textwrap.dedent("""
    import sys
    import numpy as np
    from cuopt.linear_programming.problem import Problem, INTEGER, MINIMIZE
    from cuopt.linear_programming import SolverSettings
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_MIP_DETERMINISM_MODE, CUOPT_TIME_LIMIT)

    mode = sys.argv[1]
    rng = np.random.default_rng(0)
    n = 120
    a = rng.uniform(1.0, 10.0, n)
    b = a + rng.uniform(-0.5, 0.5, n)            # correlated second weight vector
    need_a = 0.80 * a.sum()                       # capture >=80% of a-value ...
    cap_b  = 0.30 * b.sum()                       # ... using <=30% of b-value -> infeasible (a~b)

    p = Problem("infeasible_milp")
    x = [p.addVariable(lb=0.0, ub=1.0, vtype=INTEGER, name=f"x{i}") for i in range(n)]
    p.setObjective(sum(x), sense=MINIMIZE)
    p.addConstraint(sum(float(a[i]) * x[i] for i in range(n)) >= float(need_a), name="need_a")
    p.addConstraint(sum(float(b[i]) * x[i] for i in range(n)) <= float(cap_b),  name="cap_b")

    s = SolverSettings()
    s.set_parameter(CUOPT_TIME_LIMIT, 15.0)
    if mode == "deterministic":
        s.set_parameter(CUOPT_MIP_DETERMINISM_MODE, 1)
    p.solve(s)
    print("STATUS=" + str(getattr(p.Status, "name", p.Status)))
""")

for mode in ["deterministic", "opportunistic"]:
    r = subprocess.run(
        [sys.executable, "-c", WORKER, mode], capture_output=True, text=True
    )
    out = r.stdout.strip() or (r.stderr.strip().splitlines() or [""])[-1]
    print(f"{mode:>13} -> exit {r.returncode:<4} | {out}")
