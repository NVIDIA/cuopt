---
name: cuopt-lp-milp-api-python
description: LP and MILP with cuOpt — Python API only. Use with cuopt-lp-milp-common for concepts. Use when the user is building or solving LP/MILP in Python.
---

# cuOpt LP/MILP — Python API

**Concepts:** Read `cuopt-lp-milp-common/SKILL.md` for problem type, when to use, when to escalate.

This skill is **Python only**. For C, use `cuopt-lp-milp-api-c`.

## Quick Reference: Python API

### LP Example

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("MyLP")
x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")
problem.addConstraint(2*x + 3*y <= 120, name="resource_a")
problem.addConstraint(4*x + 2*y <= 100, name="resource_b")
problem.setObjective(40*x + 30*y, sense=MAXIMIZE)
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Objective: {problem.ObjValue}")
    print(f"x = {x.getValue()}, y = {y.getValue()}")
```

### MILP Example (integer variables)

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, INTEGER, MINIMIZE

problem = Problem("FacilityLocation")
open_facility = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name="open")
production = problem.addVariable(lb=0, vtype=CONTINUOUS, name="production")
problem.addConstraint(production <= 1000 * open_facility, name="link")
problem.setObjective(500*open_facility + 2*production, sense=MINIMIZE)
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)
problem.solve(settings)

if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Open: {open_facility.getValue() > 0.5}, Production: {production.getValue()}")
```

## CRITICAL: Status Checking (PascalCase)

```python
# ✅ CORRECT
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)

# ❌ WRONG — never matches
if problem.Status.name == "OPTIMAL":
    print(problem.ObjValue)
```

**LP status:** `Optimal`, `PrimalFeasible`, `PrimalInfeasible`, `TimeLimit`, etc.  
**MILP status:** `Optimal`, `FeasibleFound`, `Infeasible`, `TimeLimit`, etc.

## Solver Settings

```python
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
settings.set_parameter("mip_relative_gap", 0.01)
settings.set_parameter("log_to_console", 1)
```

## Common Issues

| Problem | Fix |
|---------|-----|
| Status never "Optimal" | Use `"Optimal"` (PascalCase), not `"OPTIMAL"` |
| Integer var fractional | Use `vtype=INTEGER` |
| Infeasible | Check constraint logic and bounds |
| Slow solve | Set time_limit, mip_relative_gap |

## Debugging

**Diagnostic — status:** `print(f"Actual status: '{problem.Status.name}'")`

**Infeasible inspection:** List constraints and check for conflicts:
```python
if problem.Status.name == "Infeasible":
    for name in constraint_names:
        c = problem.getConstraint(name)
        print(f"{name}: {c}")
```

**Wrong objective:** Print variable values: `print(f"{var.name}: {var.getValue()}")` and `problem.ObjValue`.

## Dual Values (LP only)

```python
if problem.Status.name == "Optimal":
    c = problem.getConstraint("resource_a")
    print(f"Shadow price: {c.DualValue}")
```

## Examples

- [examples.md](resources/examples.md) — LP, MILP, knapsack, transportation
- [server_examples.md](resources/server_examples.md) — REST from Python

## Escalate

See `cuopt-lp-milp-common` for when to use cuopt-qp-common or cuopt-developer.
