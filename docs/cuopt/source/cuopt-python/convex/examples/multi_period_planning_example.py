# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-Period Planning Example (inventory carried between periods)

Shows a model whose constraints link one period to the next. The balance
constraint refers to the previous period's inventory, so the first period
needs a boundary value -- a plain Python number mixes with variables in an
expression, so no special casing is needed beyond a conditional.

Problem:
    Meet demand over 3 periods by producing and carrying inventory.

    - Demand per period:  [10, 20, 15]
    - Production capacity per period: 18
    - Production cost: 4.0 per unit
    - Holding cost:    1.0 per unit carried

Expected Output:
    Status: Optimal
    Total cost: 182.0
    Period 0: make 12.0, inventory 2.0
    Period 1: make 18.0, inventory 0.0
    Period 2: make 15.0, inventory 0.0
"""

from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

demand = [10.0, 20.0, 15.0]
production_capacity = 18.0
production_cost = 4.0
holding_cost = 1.0

periods = range(len(demand))

problem = Problem("multi_period_planning")

make = {
    t: problem.addVariable(
        lb=0, ub=production_capacity, vtype=CONTINUOUS, name=f"make_{t}"
    )
    for t in periods
}
inventory = {
    t: problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"inventory_{t}")
    for t in periods
}

# Balance: previous inventory + production - demand == inventory.
# Opening stock is zero, so period 0 uses the plain number 0.0.
for t in periods:
    previous = inventory[t - 1] if t > 0 else 0.0
    problem.addConstraint(
        previous + make[t] - demand[t] == inventory[t], name=f"balance_{t}"
    )

problem.setObjective(
    sum(
        production_cost * make[t] + holding_cost * inventory[t]
        for t in periods
    ),
    sense=MINIMIZE,
)

problem.solve(SolverSettings())

print(f"Status: {problem.Status.name}")
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Total cost: {problem.ObjValue}")
    for t in periods:
        # Adding 0.0 normalises a -0.0 the solver may return at a bound.
        made = round(make[t].getValue(), 6) + 0.0
        held = round(inventory[t].getValue(), 6) + 0.0
        print(f"Period {t}: make {made}, inventory {held}")
