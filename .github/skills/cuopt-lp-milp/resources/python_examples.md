# LP/MILP: Python API Examples

## Linear Programming (LP)

```python
"""
Production Planning LP:
    maximize    40*chairs + 30*tables  (profit)
    subject to  2*chairs + 3*tables <= 240  (wood constraint)
                4*chairs + 2*tables <= 200  (labor constraint)
                chairs, tables >= 0
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("ProductionPlanning")

# Decision variables (continuous, non-negative)
chairs = problem.addVariable(lb=0, vtype=CONTINUOUS, name="chairs")
tables = problem.addVariable(lb=0, vtype=CONTINUOUS, name="tables")

# Constraints
problem.addConstraint(2 * chairs + 3 * tables <= 240, name="wood")
problem.addConstraint(4 * chairs + 2 * tables <= 200, name="labor")

# Objective: maximize profit
problem.setObjective(40 * chairs + 30 * tables, sense=MAXIMIZE)

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
settings.set_parameter("log_to_console", 1)

# Solve
problem.solve(settings)

# Check status and extract results
status = problem.Status.name
print(f"Status: {status}")

if status in ["Optimal", "PrimalFeasible"]:
    print(f"Optimal profit: ${problem.ObjValue:.2f}")
    print(f"Chairs to produce: {chairs.getValue():.1f}")
    print(f"Tables to produce: {tables.getValue():.1f}")

    # Get dual values (shadow prices)
    wood_constraint = problem.getConstraint("wood")
    labor_constraint = problem.getConstraint("labor")
    print(f"\nShadow price (wood): ${wood_constraint.DualValue:.2f} per unit")
    print(f"Shadow price (labor): ${labor_constraint.DualValue:.2f} per unit")
else:
    print(f"No optimal solution found. Status: {status}")
```

## Mixed-Integer Linear Programming (MILP)

```python
"""
Facility Location MILP:
- Decide which warehouses to open (binary)
- Assign customers to open warehouses
- Minimize fixed costs + transportation costs
"""
from cuopt.linear_programming.problem import (
    Problem, CONTINUOUS, INTEGER, MINIMIZE
)
from cuopt.linear_programming.solver_settings import SolverSettings

# Problem data
warehouses = ["W1", "W2", "W3"]
customers = ["C1", "C2", "C3", "C4"]

fixed_costs = {"W1": 100, "W2": 150, "W3": 120}
capacities = {"W1": 50, "W2": 70, "W3": 60}
demands = {"C1": 20, "C2": 25, "C3": 15, "C4": 30}

transport_cost = {
    ("W1", "C1"): 5, ("W1", "C2"): 8, ("W1", "C3"): 6, ("W1", "C4"): 10,
    ("W2", "C1"): 7, ("W2", "C2"): 4, ("W2", "C3"): 9, ("W2", "C4"): 5,
    ("W3", "C1"): 6, ("W3", "C2"): 7, ("W3", "C3"): 4, ("W3", "C4"): 8,
}

# Create problem
problem = Problem("FacilityLocation")

# Binary variables: y[w] = 1 if warehouse w is open
y = {w: problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"open_{w}")
     for w in warehouses}

# Continuous variables: x[w,c] = units shipped from w to c
x = {(w, c): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"ship_{w}_{c}")
     for w in warehouses for c in customers}

# Objective: minimize fixed + transportation costs
problem.setObjective(
    sum(fixed_costs[w] * y[w] for w in warehouses) +
    sum(transport_cost[w, c] * x[w, c] for w in warehouses for c in customers),
    sense=MINIMIZE
)

# Constraints: meet customer demand
for c in customers:
    problem.addConstraint(
        sum(x[w, c] for w in warehouses) == demands[c],
        name=f"demand_{c}"
    )

# Constraints: respect warehouse capacity (only if open)
for w in warehouses:
    problem.addConstraint(
        sum(x[w, c] for c in customers) <= capacities[w] * y[w],
        name=f"capacity_{w}"
    )

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)

# Solve
problem.solve(settings)

# Results
status = problem.Status.name
print(f"Status: {status}")

if status in ["Optimal", "FeasibleFound"]:
    print(f"Total cost: ${problem.ObjValue:.2f}")
    print("\nOpen warehouses:")
    for w in warehouses:
        if y[w].getValue() > 0.5:
            print(f"  {w} (fixed cost: ${fixed_costs[w]})")

    print("\nShipments:")
    for w in warehouses:
        for c in customers:
            shipped = x[w, c].getValue()
            if shipped > 0.01:
                print(f"  {w} -> {c}: {shipped:.1f} units")
```

## Knapsack Problem (MILP)

```python
"""
0/1 Knapsack: select items to maximize value within weight limit
"""
from cuopt.linear_programming.problem import Problem, INTEGER, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

items = ["laptop", "camera", "phone", "tablet", "headphones"]
values = [1000, 500, 300, 600, 150]
weights = [3, 1, 0.5, 1.5, 0.3]
max_weight = 5

problem = Problem("Knapsack")

# Binary variables: x[i] = 1 if item i is selected
x = [problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=items[i])
     for i in range(len(items))]

# Objective: maximize total value
problem.setObjective(sum(values[i] * x[i] for i in range(len(items))), sense=MAXIMIZE)

# Constraint: weight limit
problem.addConstraint(sum(weights[i] * x[i] for i in range(len(items))) <= max_weight)

problem.solve(SolverSettings())

if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Total value: ${problem.ObjValue:.0f}")
    print("Selected items:")
    for i, item in enumerate(items):
        if x[i].getValue() > 0.5:
            print(f"  {item}: value=${values[i]}, weight={weights[i]}")
```

## Transportation Problem (LP)

```python
"""
Minimize shipping cost from suppliers to customers
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE

suppliers = ["S1", "S2"]
customers = ["C1", "C2", "C3"]
supply = {"S1": 100, "S2": 150}
demand = {"C1": 80, "C2": 70, "C3": 100}
cost = {
    ("S1", "C1"): 4, ("S1", "C2"): 6, ("S1", "C3"): 8,
    ("S2", "C1"): 5, ("S2", "C2"): 3, ("S2", "C3"): 7,
}

problem = Problem("Transportation")

x = {(s, c): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"x_{s}_{c}")
     for s in suppliers for c in customers}

# Minimize total shipping cost
problem.setObjective(sum(cost[s,c] * x[s,c] for s in suppliers for c in customers),
                     sense=MINIMIZE)

# Supply constraints
for s in suppliers:
    problem.addConstraint(sum(x[s,c] for c in customers) <= supply[s])

# Demand constraints
for c in customers:
    problem.addConstraint(sum(x[s,c] for s in suppliers) >= demand[c])

problem.solve()

if problem.Status.name in ("Optimal", "PrimalFeasible"):
    print(f"Total cost: ${problem.ObjValue:.2f}")
    for s in suppliers:
        for c in customers:
            val = x[s,c].getValue()
            if val > 0.01:
                print(f"  {s} -> {c}: {val:.0f} units")
```

## Status Checking (Critical)

```python
# ✅ CORRECT - use PascalCase
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)

# ❌ WRONG - will silently fail!
if problem.Status.name == "OPTIMAL":  # Never matches!
    print(problem.ObjValue)

# LP status values: Optimal, PrimalFeasible, PrimalInfeasible,
#                   DualInfeasible, TimeLimit, NumericalError
# MILP status values: Optimal, FeasibleFound, Infeasible,
#                     Unbounded, TimeLimit, NoTermination
```

---

## Additional References (tested in CI)

For more complete examples, read these files:

| Example | File | Description |
|---------|------|-------------|
| Simple LP | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/simple_lp_example.py` | Basic LP setup |
| Simple MILP | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/simple_milp_example.py` | Integer variables |
| Production Planning | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/production_planning_example.py` | Real-world LP |
| Expressions | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/expressions_constraints_example.py` | Advanced constraint syntax |
| Incumbent Solutions | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/incumbent_solutions_example.py` | Tracking MIP progress |
| Warmstart | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/pdlp_warmstart_example.py` | Warm starting LP |
| Solution Handling | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/solution_example.py` | Working with results |

These examples are tested by CI (`ci/test_doc_examples.sh`) and represent canonical usage.
