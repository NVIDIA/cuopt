# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Facility Location Example (binary variables with linking constraints)

Shows the two things most MILP models need beyond a simple integer variable:

- a binary variable, expressed as INTEGER with lb=0 and ub=1
- a linking constraint tying a continuous quantity to a binary decision, so
  that shipping from a facility forces that facility to be open

Problem:
    Choose which facilities to open and how much to ship from each.

    - Fixed cost to open:  [100, 80]
    - Capacity:            [50, 50]
    - Customer demand:     [30, 20]
    - Unit shipping cost:
          to C0   C1
      F0   4.0  6.0
      F1   5.0  3.0

Expected Output:
    Status: Optimal
    Total cost: 290.0
    Facility 0: closed
    Facility 1: open
"""

from cuopt.linear_programming.problem import (
    Problem,
    CONTINUOUS,
    INTEGER,
    MINIMIZE,
)
from cuopt.linear_programming.solver_settings import SolverSettings

fixed_cost = [100.0, 80.0]
capacity = [50.0, 50.0]
demand = [30.0, 20.0]
ship_cost = [[4.0, 6.0], [5.0, 3.0]]

facilities = range(len(fixed_cost))
customers = range(len(demand))

problem = Problem("facility_location")

# A binary variable is INTEGER with lb=0 and ub=1.
is_open = {
    f: problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"open_{f}")
    for f in facilities
}
ship = {
    (f, c): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"ship_{f}_{c}")
    for f in facilities
    for c in customers
}

for c in customers:
    problem.addConstraint(
        sum(ship[f, c] for f in facilities) >= demand[c], name=f"demand_{c}"
    )

# Linking: a facility can ship only up to its capacity, and only if open.
for f in facilities:
    problem.addConstraint(
        sum(ship[f, c] for c in customers) <= capacity[f] * is_open[f],
        name=f"link_{f}",
    )

problem.setObjective(
    sum(fixed_cost[f] * is_open[f] for f in facilities)
    + sum(ship_cost[f][c] * ship[f, c] for f in facilities for c in customers),
    sense=MINIMIZE,
)

settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)
problem.solve(settings)

print(f"Status: {problem.Status.name}")
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Total cost: {problem.ObjValue}")
    for f in facilities:
        state = "open" if is_open[f].getValue() > 0.5 else "closed"
        print(f"Facility {f}: {state}")
