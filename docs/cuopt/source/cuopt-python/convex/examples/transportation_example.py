# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Transportation Example (model built over index sets)

Most examples in this guide declare a handful of variables by name. Real models
are indexed over sets and built in loops. This example shows that shape:

- variables held in a dict keyed by the index tuple
- constraints accumulated with ``sum(...)`` over a generator, then compared

Problem:
    Ship goods from 2 plants to 3 warehouses at least cost.

    - Plant supply:      [20, 30]
    - Warehouse demand:  [10, 25, 15]
    - Unit shipping cost:
          to W0   W1   W2
      P0   2.0  3.0  1.0
      P1   5.0  4.0  8.0

Expected Output:
    Status: Optimal
    Total shipping cost: 150.0
      plant 0 -> warehouse 0: 5.0
      plant 0 -> warehouse 2: 15.0
      plant 1 -> warehouse 0: 5.0
      plant 1 -> warehouse 1: 25.0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

supply = [20.0, 30.0]
demand = [10.0, 25.0, 15.0]
cost = [[2.0, 3.0, 1.0],
        [5.0, 4.0, 8.0]]

plants = range(len(supply))
warehouses = range(len(demand))

problem = Problem("transportation")

# Index the variables with a dict keyed by the index tuple.
ship = {
    (p, w): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"ship_{p}_{w}")
    for p in plants
    for w in warehouses
}

# addConstraint needs a comparison. Accumulate the row expression with sum(),
# then compare it in the call -- passing a bare expression raises ValueError.
for p in plants:
    problem.addConstraint(
        sum(ship[p, w] for w in warehouses) <= supply[p], name=f"supply_{p}"
    )

for w in warehouses:
    problem.addConstraint(
        sum(ship[p, w] for p in plants) >= demand[w], name=f"demand_{w}"
    )

problem.setObjective(
    sum(cost[p][w] * ship[p, w] for p in plants for w in warehouses),
    sense=MINIMIZE,
)

settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Status values are PascalCase; "OPTIMAL" would never match.
print(f"Status: {problem.Status.name}")
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Total shipping cost: {problem.ObjValue}")
    for (p, w), var in sorted(ship.items()):
        if var.getValue() > 1e-6:
            print(f"  plant {p} -> warehouse {w}: {var.getValue()}")
