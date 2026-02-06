---
name: cuopt-lp-milp
description: Solve Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) with NVIDIA cuOpt. Use when the user asks about optimization with linear constraints, integer variables, scheduling, resource allocation, facility location, or production planning.
---

# cuOpt LP/MILP Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Model and solve linear and mixed-integer linear programs using NVIDIA cuOpt's GPU-accelerated solver.

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **Problem formulation?**
   - What are the decision variables?
   - What is the objective (minimize/maximize what)?
   - What are the constraints?

2. **Variable types?**
   - All continuous (LP)?
   - Some integer/binary (MILP)?

3. **Interface preference?**
   - Python API (recommended for modeling)
   - C API (native embedding)
   - CLI (quick solve from MPS file)
   - REST Server (production deployment)

4. **Do you have an MPS file?**
   - If yes, CLI or C API may be simpler
   - If no, Python API is best for building the model

## Interface Support

| Interface | LP | MILP |
|-----------|:--:|:----:|
| Python    | ✓  | ✓    |
| C API     | ✓  | ✓    |
| CLI       | ✓  | ✓    |
| REST      | ✓  | ✓    |

## Quick Reference: Python API

### LP Example

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("MyLP")

# Decision variables
x = problem.addVariable(lb=0, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=0, vtype=CONTINUOUS, name="y")

# Constraints
problem.addConstraint(2*x + 3*y <= 120, name="resource_a")
problem.addConstraint(4*x + 2*y <= 100, name="resource_b")

# Objective
problem.setObjective(40*x + 30*y, sense=MAXIMIZE)

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Check status (CRITICAL: use PascalCase!)
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Objective: {problem.ObjValue}")
    print(f"x = {x.getValue()}")
    print(f"y = {y.getValue()}")
```

### MILP Example (with integer variables)

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, INTEGER, MINIMIZE

problem = Problem("FacilityLocation")

# Binary variable (integer with bounds 0-1)
open_facility = problem.addVariable(lb=0, ub=1, vtype=INTEGER, name="open")

# Continuous variable
production = problem.addVariable(lb=0, vtype=CONTINUOUS, name="production")

# Linking constraint: can only produce if facility is open
problem.addConstraint(production <= 1000 * open_facility, name="link")

# Objective: fixed cost + variable cost
problem.setObjective(500*open_facility + 2*production, sense=MINIMIZE)

# MILP-specific settings
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)  # 1% optimality gap

problem.solve(settings)

# Check status
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(f"Open facility: {open_facility.getValue() > 0.5}")
    print(f"Production: {production.getValue()}")
```

## CRITICAL: Status Checking

**Status values use PascalCase, NOT ALL_CAPS:**

```python
# ✅ CORRECT
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)

# ❌ WRONG - will silently fail!
if problem.Status.name == "OPTIMAL":  # Never matches!
    print(problem.ObjValue)
```

**LP Status Values:** `Optimal`, `NoTermination`, `NumericalError`, `PrimalInfeasible`, `DualInfeasible`, `IterationLimit`, `TimeLimit`, `PrimalFeasible`

**MILP Status Values:** `Optimal`, `FeasibleFound`, `Infeasible`, `Unbounded`, `TimeLimit`, `NoTermination`

## Quick Reference: C API

```c
#include <cuopt/linear_programming/cuopt_c.h>

// CSR format for constraints
cuopt_int_t row_offsets[] = {0, 2, 4};
cuopt_int_t col_indices[] = {0, 1, 0, 1};
cuopt_float_t values[] = {2.0, 3.0, 4.0, 2.0};

// Variable types
char var_types[] = {CUOPT_CONTINUOUS, CUOPT_INTEGER};

cuOptCreateRangedProblem(
    num_constraints, num_variables, CUOPT_MINIMIZE,
    0.0,  // objective offset
    objective_coefficients,
    row_offsets, col_indices, values,
    constraint_lower, constraint_upper,
    var_lower, var_upper,
    var_types,
    &problem
);

cuOptSolve(problem, settings, &solution);
cuOptGetObjectiveValue(solution, &obj_value);
```

## Quick Reference: CLI

```bash
# Solve LP from MPS file
cuopt_cli problem.mps

# With options
cuopt_cli problem.mps --time-limit 120 --mip-relative-tolerance 0.01
```

## Common Modeling Patterns

### Binary Selection
```python
# Select exactly k items from n
items = [problem.addVariable(lb=0, ub=1, vtype=INTEGER) for _ in range(n)]
problem.addConstraint(sum(items) == k)
```

### Big-M Linking
```python
# If y=1, then x <= 100; if y=0, x can be anything up to M
M = 10000
problem.addConstraint(x <= 100 + M*(1 - y))
```

### Piecewise Linear (SOS2)
```python
# Approximate nonlinear function with breakpoints
# Use lambda variables that sum to 1, at most 2 adjacent non-zero
```

## Solver Settings

```python
settings = SolverSettings()

# Time limit
settings.set_parameter("time_limit", 60)

# MILP gap tolerance (stop when within X% of optimal)
settings.set_parameter("mip_relative_gap", 0.01)

# Logging
settings.set_parameter("log_to_console", 1)
```

## Common Issues

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Status never "OPTIMAL" | Using wrong case | Use `"Optimal"` not `"OPTIMAL"` |
| Integer var has fractional value | Defined as CONTINUOUS | Use `vtype=INTEGER` |
| Infeasible | Conflicting constraints | Check constraint logic |
| Unbounded | Missing bounds | Add variable bounds |
| Slow solve | Large problem | Set time limit, increase gap tolerance |

## Getting Dual Values (LP only)

```python
if problem.Status.name == "Optimal":
    constraint = problem.getConstraint("resource_a")
    shadow_price = constraint.DualValue
    print(f"Shadow price: {shadow_price}")
```

## Examples

See `resources/` for complete examples:
- [Python API](resources/python_examples.md) — LP, MILP, knapsack, transportation
- [C API](resources/c_api_examples.md) — LP/MILP with build instructions
- [CLI](resources/cli_examples.md) — MPS file format and commands
- [REST Server](resources/server_examples.md) — curl and Python requests

## When to Escalate

Switch to **cuopt-qp** if:
- Objective has quadratic terms (x², x*y)

Switch to **cuopt-debugging** if:
- Infeasible and can't determine why
- Numerical issues

Switch to **cuopt-developer** if:
- User wants to modify solver internals
