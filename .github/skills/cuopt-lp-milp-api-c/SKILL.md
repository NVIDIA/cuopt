---
name: cuopt-lp-milp-api-c
description: LP and MILP with cuOpt — C API only. Use with cuopt-lp-milp-formulation for concepts. Use when the user is embedding LP/MILP in C/C++.
---

# cuOpt LP/MILP — C API

**Concepts:** Read `cuopt-lp-milp-formulation/SKILL.md` for problem type and formulation.

This skill is **C only**. For Python, use `cuopt-lp-milp-api-python`.

## Quick Reference: C API

```c
#include <cuopt/linear_programming/cuopt_c.h>

// CSR format for constraints
cuopt_int_t row_offsets[] = {0, 2, 4};
cuopt_int_t col_indices[] = {0, 1, 0, 1};
cuopt_float_t values[] = {2.0, 3.0, 4.0, 2.0};
char var_types[] = {CUOPT_CONTINUOUS, CUOPT_INTEGER};

cuOptCreateRangedProblem(
    num_constraints, num_variables, CUOPT_MINIMIZE,
    0.0, objective_coefficients,
    row_offsets, col_indices, values,
    constraint_lower, constraint_upper,
    var_lower, var_upper, var_types,
    &problem
);
cuOptSolve(problem, settings, &solution);
cuOptGetObjectiveValue(solution, &obj_value);
```

## Debugging (MPS / C)

**MPS parsing:** Required sections in order: NAME, ROWS, COLUMNS, RHS, (optional) BOUNDS, ENDATA. Integer markers: `'MARKER'`, `'INTORG'`, `'INTEND'`.

**OOM or slow:** Check problem size (variables, constraints); use sparse matrix; set time limit and gap tolerance.

## Examples

- [examples.md](resources/examples.md) — LP/MILP with build instructions

For **CLI** (MPS files), use `cuopt-lp-milp-api-cli`.

## Escalate

See `cuopt-lp-milp-formulation` for when to use cuopt-qp-formulation or cuopt-developer.
