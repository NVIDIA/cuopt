---
name: cuopt-qp-api-python
description: Quadratic Programming (QP) with cuOpt — Python API only (beta). Use with cuopt-qp-formulation for concepts. Use when the user is building or solving QP in Python.
---

# cuOpt QP — Python API (beta)

**Concepts:** Read `cuopt-qp-formulation/SKILL.md` for when QP applies, minimize-only, when to escalate.

This skill is **Python only**. For C, use `cuopt-qp-api-c`. **QP is beta.**

## CRITICAL: MINIMIZE only

```python
# ❌ WRONG
problem.setObjective(x*x + y*y, sense=MAXIMIZE)

# ✅ CORRECT — negate for maximization
problem.setObjective(-(x*x + y*y), sense=MINIMIZE)
```

## Portfolio Example

```python
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("Portfolio")
x1 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_c")
problem.setObjective(
    0.04*x1*x1 + 0.02*x2*x2 + 0.01*x3*x3 + 0.02*x1*x2 + 0.01*x1*x3 + 0.016*x2*x3,
    sense=MINIMIZE
)
problem.addConstraint(x1 + x2 + x3 == 1, name="budget")
problem.addConstraint(r1*x1 + r2*x2 + r3*x3 >= 0.08, name="min_return")
problem.solve(SolverSettings())
```

## Status (PascalCase)

```python
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(problem.ObjValue)
```

## Debugging

**Diagnostic:** `print(f"Actual status: '{problem.Status.name}'")`. For numerical issues, check Q is PSD and variables are scaled.

## Examples

- [examples.md](resources/examples.md) — portfolio, least squares, maximization workaround
- **Reference models:** This skill's `assets/` — [portfolio](assets/portfolio/), [least_squares](assets/least_squares/), [maximization_workaround](assets/maximization_workaround/). See [assets/README.md](assets/README.md).

## Escalate

See `cuopt-qp-formulation` for when to use cuopt-lp-milp-formulation or cuopt-developer.
