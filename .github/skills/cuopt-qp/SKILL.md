---
name: cuopt-qp
description: Solve Quadratic Programming (QP) with NVIDIA cuOpt. Use when the user asks about quadratic objectives, portfolio optimization, variance minimization, or least squares problems. Note that QP support is currently in beta.
---

# cuOpt QP Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Model and solve quadratic programs using NVIDIA cuOpt. **QP support is currently in beta.**

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **Is this actually QP?**
   - Does the objective have x² or x*y terms?
   - Are constraints still linear?
   - (If constraints are also quadratic, cuOpt doesn't support that)

2. **Minimize or maximize?**
   - ⚠️ **QP only supports MINIMIZE**
   - For maximization, negate the objective

3. **Is the quadratic form convex?**
   - Q matrix should be positive semi-definite for minimization
   - Non-convex QP may not solve correctly

## Critical Constraints

### ⚠️ MINIMIZE ONLY

**QP objectives MUST be minimization.** The solver rejects maximize for QP.

```python
# ❌ WRONG - will fail
problem.setObjective(x*x + y*y, sense=MAXIMIZE)

# ✅ CORRECT - minimize instead
# To maximize f(x), minimize -f(x)
problem.setObjective(-(x*x + y*y), sense=MINIMIZE)
```

### Interface Support

| Interface | QP Support |
|-----------|:----------:|
| Python    | ✓ (beta)   |
| C API     | ✓          |
| REST      | ✗          |
| CLI       | ✗          |

## Quick Reference: Python API

### Portfolio Optimization Example

```python
"""
Minimize portfolio variance (risk):
    minimize    x^T * Q * x
    subject to  sum(x) = 1         (fully invested)
                r^T * x >= target  (minimum return)
                x >= 0             (no short selling)
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

problem = Problem("Portfolio")

# Portfolio weights (decision variables)
x1 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0, ub=1, vtype=CONTINUOUS, name="stock_c")

# Expected returns
r1, r2, r3 = 0.12, 0.08, 0.05

# Quadratic objective: variance = x^T * Q * x
# Q = [[0.04, 0.01, 0.005],
#      [0.01, 0.02, 0.008],
#      [0.005, 0.008, 0.01]]
# Expanded: 0.04*x1² + 0.02*x2² + 0.01*x3² + 2*0.01*x1*x2 + ...
problem.setObjective(
    0.04*x1*x1 + 0.02*x2*x2 + 0.01*x3*x3 +
    0.02*x1*x2 + 0.01*x1*x3 + 0.016*x2*x3,
    sense=MINIMIZE  # MUST be minimize for QP!
)

# Linear constraints
problem.addConstraint(x1 + x2 + x3 == 1, name="budget")
problem.addConstraint(r1*x1 + r2*x2 + r3*x3 >= 0.08, name="min_return")

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Check results
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Variance: {problem.ObjValue:.6f}")
    print(f"Std Dev: {problem.ObjValue**0.5:.4f}")
    print(f"Allocation: A={x1.getValue():.2%}, B={x2.getValue():.2%}, C={x3.getValue():.2%}")
```

### Least Squares Example

```python
"""
Minimize ||Ax - b||² = x^T*A^T*A*x - 2*b^T*A*x + b^T*b
"""
problem = Problem("LeastSquares")

x = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="x")
y = problem.addVariable(lb=-100, ub=100, vtype=CONTINUOUS, name="y")

# Quadratic objective: (x-3)² + (y-4)² = x² + y² - 6x - 8y + 25
problem.setObjective(
    x*x + y*y - 6*x - 8*y + 25,
    sense=MINIMIZE
)

problem.solve(SolverSettings())

print(f"x = {x.getValue()}")  # Should be ~3
print(f"y = {y.getValue()}")  # Should be ~4
```

## Formulating Quadratic Objectives

### From Covariance Matrix

Given covariance matrix Q and weights x:
```
variance = x^T * Q * x = Σᵢ Σⱼ Qᵢⱼ * xᵢ * xⱼ
```

Expand manually:
```python
# Q = [[a, b], [b, c]]
# x^T * Q * x = a*x1² + 2b*x1*x2 + c*x2²
objective = a*x1*x1 + 2*b*x1*x2 + c*x2*x2
```

### Maximization Workaround

To maximize `f(x) = -x² + 4x`:
```python
# maximize -x² + 4x
# = minimize -(-x² + 4x)
# = minimize x² - 4x
problem.setObjective(x*x - 4*x, sense=MINIMIZE)
# Then negate the objective value for the true maximum
true_max = -problem.ObjValue
```

## Status Checking

Same as LP/MILP - use PascalCase:

```python
if problem.Status.name in ["Optimal", "PrimalFeasible"]:
    print(f"Optimal variance: {problem.ObjValue}")
```

## Common Issues

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| "Quadratic problems must be minimized" | Using MAXIMIZE | Use MINIMIZE, negate objective |
| Poor convergence | Non-convex Q | Ensure Q is positive semi-definite |
| NumericalError | Ill-conditioned Q | Scale variables, regularize |
| Slow solve | Large dense Q | Check if problem can be simplified |

## Solver Notes

- QP uses **Barrier method** internally (different from LP/MILP defaults)
- May be more sensitive to numerical issues than LP
- Beta status means API may change in future versions

## Examples

See `resources/` for complete examples:
- [Python API](resources/python_examples.md) — portfolio, least squares, maximization workaround

## When to Escalate

Switch to **cuopt-lp-milp** if:
- Objective is actually linear (no x² or x*y terms)

Switch to **cuopt-debugging** if:
- Numerical errors
- Unexpected results

Switch to **cuopt-developer** if:
- Need features not in beta QP
