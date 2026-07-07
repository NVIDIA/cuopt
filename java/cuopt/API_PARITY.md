<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Java/Python cuOpt binding parity plan

This document tracks what it would mean for the Java bindings to be at parity
with the Python bindings and for tests to check behavioral parity.

## Scope choices

There are two materially different interpretations of "Python binding parity":

1. **Full `cuopt` Python package parity.** This includes:
   - `cuopt.linear_programming`
   - `cuopt.routing`
   - `cuopt.distance_engine`
   - public helper functions exported by those packages

   The current Java source tree only has `com.nvidia.cuopt.linearprogramming`.
   Full package parity would require new Java routing and distance-engine
   packages plus new JNI/native wrappers for those C++ APIs.

2. **Linear-programming package parity.** This covers the APIs currently closest
   to the Java module:
   - `cuopt.linear_programming.DataModel`
   - `cuopt.linear_programming.Problem`
   - `cuopt.linear_programming.SolverSettings`
   - `cuopt.linear_programming.Solve`
   - `cuopt.linear_programming.BatchSolve`
   - `cuopt.linear_programming.Solution`
   - `cuopt.linear_programming.Read`
   - `cuopt.linear_programming.ParseMps`

   This is the realistic first parity target for the current Java module.

## Current Java binding scope

The Java module currently exposes LP/MILP/QP functionality through:

- `DataModel`
- `Problem`
- `Variable`
- `Constraint`
- `LinearExpression`
- `QuadraticExpression`
- `SolverSettings`
- `Solution`
- `LPStats`
- `MIPStats`
- `PDLPWarmStartData`
- MIP callback interfaces

It is backed by `java/cuopt/src/main/native/cuopt_jni.cpp` and the C LP API in
`cpp/include/cuopt/linear_programming/cuopt_c.h`.

## Existing parity tests

`PythonParityTest` currently verifies a representative set of LP/MILP solve
flows against Python cuOpt:

- `lp_min_ge_unique_solution`
- `lp_max_le_unique_solution`
- `lp_equal_with_offset`
- `lp_ranged_bounds`
- `lp_mixed_bounds_negative_coefficients`
- `lp_max_ranged_bounds`
- `milp_integer_unique_solution`
- `milp_mixed_integer_continuous_max`
- `qp_diagonal_objective`
- `lp_infeasible_status`

These are useful regression tests with model, solve, solution, LP-stat,
MIP-stat, and QP coverage, but they do not prove every Python binding function
has a Java equivalent or matching behavior.

## Linear-programming API gap matrix

### `DataModel`

| Python API | Java status | Test status |
| --- | --- | --- |
| `__init__` plus incremental setters | Supported by the public no-argument `DataModel` constructor and fluent setters. | Native contract coverage. |
| `set_maximize` / `get_sense` | Supported via `ObjectiveSense` at creation and `getObjectiveSense`. | Covered by parity cases. |
| `set_csr_constraint_matrix` / matrix getters | Supported through `CsrMatrix` at creation and `getConstraintMatrix`. | Covered by parity cases. |
| `set_constraint_bounds`, `set_row_types`, RHS/row-type getters | Supported at creation plus `getConstraintRhs`, `getConstraintSense`. | Covered by parity cases. |
| `set_constraint_lower_bounds`, `set_constraint_upper_bounds`, bound getters | Supported through ranged factory plus getters. | Covered by ranged parity case. |
| `set_objective_coefficients`, `set_objective_offset`, getters | Supported at creation plus getters. | Covered by parity cases. |
| `set_objective_scaling_factor`, `get_objective_scaling_factor` | Supported by mutable Java `DataModel`. | Python oracle compares the scaling factor. |
| `set_quadratic_objective_matrix`, quadratic objective matrix getters | Supported alongside the expression-based quadratic setter. | Python oracle compares Q values, indices, and offsets. |
| `add_quadratic_constraint`, `clear_quadratic_constraints`, `get_quadratic_constraints` | Supported with a public `QuadraticConstraint` host representation. | Native contract coverage includes add/get/clear and row names. |
| `set_variable_lower_bounds`, `set_variable_upper_bounds`, bound getters | Supported at creation plus getters. | Covered by parity cases. |
| `set_variable_types`, `get_variable_types` | Supported at creation plus getter. | Covered by parity cases. |
| `set_variable_names`, `get_variable_names` | Supported by Java `DataModel`. | Python oracle compares names, including multi-character names. |
| `set_row_names`, `get_row_names` | Supported by Java `DataModel`. | Python oracle compares names. |
| `set_objective_name`, `get_objective_name` | Supported by Java `DataModel`. | Python oracle compares the objective name. |
| `set_problem_name`, `get_problem_name` | Supported by Java `DataModel`. | Python oracle compares the problem name. |
| `set_initial_primal_solution`, `set_initial_dual_solution`, getters | Supported on both `DataModel` and `SolverSettings`. | Native contract coverage verifies defensive copies and dimensions. |
| `get_ascii_row_types` | Supported as `getAsciiRowTypes`; `getRowTypes` and `getSense` aliases are also available. | Native contract coverage. |
| `writeMPS` | Supported as `writeMPS`. | Native MPS write/read/ParseMps/Problem.read round-trip coverage. |

### `SolverSettings`

| Python API | Java status | Test status |
| --- | --- | --- |
| `get_solver_parameter_names` | Supported. | Smoke-tested. |
| `get_solver_setting` | Supported as `SolverSettings.getSolverSetting`; Java also exposes a typed `getParameter`. | Native settings coverage. |
| `set_parameter`, `get_parameter` | Supported. | Lightly covered; not full parameter sweep. |
| `set_optimality_tolerance` | Supported as `setOptimalityTolerance`. | Native settings contract coverage. |
| `set_pdlp_warm_start_data` | Supported. | Not parity-tested end-to-end. |
| `get_pdlp_warm_start_data` on settings | Supported. | Java contract coverage; solution warm-start extraction is exercised by LP smoke tests. |
| `set_mip_callback` | Supported with Java callback interfaces. | Native registration/list contract coverage; callback invocation remains runtime-only. |
| `get_mip_callbacks` | Supported as an immutable Java view. | Java contract coverage through callback registration state. |
| `dump_parameters_to_file`, `load_parameters_from_file`, `toDict` | Supported through JNI/C API parameter-file bindings. | Native settings round-trip coverage. |

### `Solution`

| Python API | Java status | Test status |
| --- | --- | --- |
| `get_primal_solution` | Supported. | Covered by parity cases except infeasible solution values. |
| `get_dual_solution` | Supported for LP. | Covered for LP availability and values. |
| `get_reduced_cost` | Supported for LP. | Covered for LP availability and values. |
| `get_primal_objective` | Supported as `getPrimalObjective`. | Covered by parity cases. |
| `get_dual_objective` | Supported as `getDualObjective`. | Covered for non-MIP parity cases. |
| `get_termination_status` | Supported. | Covered by parity cases. |
| `get_termination_reason` | Supported as `getTerminationReason`. | Covered by the parity status assertions. |
| `get_error_status`, `get_error_message` | Supported. | Covered by parity cases. |
| `get_solve_time` | Supported. | Covered as non-negative when available, with `NaN` allowed when unavailable. Exact parity is not expected because Java and Python solve independently. |
| `get_solved_by`, `get_solved_by_pdlp` | Supported as `getSolvedBy` and deprecated-compatible `getSolvedByPdlp`. | LP parity compares solved-by method; helper is covered by API compilation. |
| `get_vars` | Supported as an immutable name-to-primal-value map when variable names are present. | Metadata/QP parity case covers the name-preserving path. |
| `get_lp_stats` | Supported through `LPStats`. | Availability and field parity are covered for LP/QP-style parity cases. |
| `get_milp_stats` | Supported through `MIPStats`, plus `getMipGap` and `getSolutionBound`. | Availability and field parity are covered, except elapsed presolve time is checked as non-negative when available, with `NaN` allowed when unavailable. |
| `get_pdlp_warm_start_data` | Supported. | LP smoke coverage plus field accessors; full field-by-field cross-language comparison remains runtime-only. |
| `get_problem_category` | Supported as `ProblemCategory`, while retaining `isMip`. | LP/MIP/QP parity compares category behavior. |

### `Problem`, `Variable`, `LinearExpression`, `QuadraticExpression`, `Constraint`

| Python API area | Java status | Test status |
| --- | --- | --- |
| Variable bounds, objective coefficient, type, name, value, MIP start | Mostly supported. | Lightly covered by modeling tests; no full parity oracle. |
| Linear expression arithmetic and comparisons | Java uses fluent methods instead of Python operator overloads. | Lightly covered. |
| Quadratic expression arithmetic and comparisons | Java has a smaller fluent API. Python operator coverage is broader. | QP smoke and Java-vs-Python diagonal-QP parity coverage exist. Broader quadratic expression/operator behavior remains open. |
| Constraint sense, RHS, coefficient, slack | Supported, including quadratic slack evaluation. | Pure-Java modeling coverage. |
| Problem add variable/constraint/objective, solve, read/write, getters | Supported for the LP/MILP/QP surface listed here. | Pure-Java, native, and parity coverage. |
| Problem update APIs: `update`, `updateConstraint`, `updateObjective`, `reset_solved_values` | Supported with fluent Java modeling equivalents. | Pure-Java modeling coverage. |
| Problem incumbent/warm-start helpers | Supported, including incumbent lookup and warm-start aliases. | Pure-Java/API contract coverage. |
| Problem `getQCSR`, `getQcsr`, `getQuadraticConstraints` | Supported. | Pure-Java quadratic modeling coverage. |
| Problem `relax` | Supported and preserves names/bounds/objective while converting variables to continuous. | Pure-Java modeling coverage. |
| Python operator overloads | Not one-to-one in Java by language design. | Should be mapped to equivalent fluent-method behavior, not copied literally. |

### Top-level solver and parser APIs

| Python API | Java status | Test status |
| --- | --- | --- |
| `Solve` | Supported as `DataModel.solve` and `Problem.solve`. | Covered by smoke and parity cases. |
| `BatchSolve` | Supported as a sequential Java compatibility entry point returning `BatchSolveResult`. | Native integration coverage. |
| `Read` | Supported as `DataModel.read` and `Problem.read`, including fixed-format dispatch. | Missing-file and generated-MPS round-trip coverage; successful execution requires native build/runtime. |
| `ParseMps` with fixed-MPS flag | Supported as `DataModel.parseMps` and `Problem.readMPS`. | Generated-MPS round-trip coverage; successful execution requires native build/runtime. |
| `toDict` | Supported as `DataModel.toDict`. | Native metadata/data-model coverage. |

## Non-linear-programming package gaps

### Routing

Python exposes routing APIs through `cuopt.routing`, including:

- `DataModel`
- `SolverSettings`
- `Solve`
- `BatchSolve`
- `Assignment`
- `SolutionStatus`
- `Objective`
- `DatasetDistribution`
- routing utility functions

There is no Java routing package in the current source tree. Full parity here
requires new Java classes, JNI wrappers, and routing-specific parity tests.

### Distance engine

Python exposes `cuopt.distance_engine.WaypointMatrix` with:

- `compute_cost_matrix`
- `compute_waypoint_sequence`
- `compute_shortest_path_costs`

There is no Java distance-engine package in the current source tree. Full
parity here requires new Java classes, JNI wrappers, and distance-engine parity
tests.

## Recommended implementation phases

### Phase 1: Exhaustive tests for existing Java LP APIs

No new Java binding APIs. Expand tests so every currently exposed Java
LP/MILP/QP binding method has either:

- a Python behavior comparison, when Python has a true equivalent; or
- a Java-only contract test, when the method is Java-specific plumbing such as
  lifecycle closing.

Remaining targets:

- parity-test `writeMPS`/`read` round trips on generated tiny models;
- parity-test `SolverSettings.getParameter` after string/int/float/bool set;
- parity-test PDLP warm-start data presence and field shapes;
- parity-test MIP callbacks with a tiny deterministic MILP if stable enough.

Already covered by the expanded parity harness:

- `DataModel.getNumNonZeros`;
- `Solution.getDualObjective`;
- `Solution.getErrorStatus` and `getErrorMessage`;
- `getSolveTime` behavior on both sides, allowing `NaN` when unavailable;
- deterministic `LPStats` fields, with `NaN` parity handled explicitly;
- deterministic `MIPStats` fields, with presolve time checked as non-negative
  when available;
- a diagonal QP solve path.

### Phase 2: Fill missing LP/MILP/QP binding APIs — implemented

The current Java module now includes the following APIs mapped to the current
C API or existing JNI layer:

- mutable `DataModel` construction and all LP/MIP/QP setters/getters;
- quadratic objective matrix and quadratic-constraint accessors;
- `BatchSolve` compatibility entry point;
- `Read`/`ParseMps` fixed-format entry points;
- solution category, termination-reason, solved-by, and variable-map helpers;
- problem update, relaxation, incumbent, QCSR, and warm-start helpers;
- parameter dump/load/to-map equivalents.

The remaining validation work is successful parser round-trip execution and
stable callback/warm-start end-to-end checks on a CUDA-enabled build.

### Phase 3: Decide how to handle Python-only modeling conveniences

Python operator overloads do not translate directly to Java. The parity target
should be equivalent behavior through Java idioms:

- fluent `plus`, `minus`, `times`, `le`, `ge`, `eq` methods;
- builder/factory APIs instead of Python's mutable empty `DataModel`, if that
  remains the chosen Java style.

Each intentional Java design difference should be documented in the matrix and
covered by behavior tests.

### Phase 4: Routing and distance-engine parity, if full package parity is required

This is a separate project-sized effort:

- design Java package names and API style;
- add native wrappers for routing and distance-engine C++ APIs;
- port Python routing/distance model construction tests to Java;
- add Java-vs-Python oracle tests like `PythonParityTest`;
- decide which Python utility functions are public API versus examples/helpers.

## Definition of done for "every binding function"

A binding function is considered covered only when one of these is true:

1. **Parity-covered:** Java exposes equivalent behavior and a test compares it
   against Python on the same input.
2. **Contract-covered:** Java exposes behavior that has no Python equivalent,
   and a Java-only test checks the Java contract.
3. **Intentionally unsupported:** The function is listed in this matrix with a
   reason, and there is no silent gap.

The LP/MILP/QP API surface is substantially covered, but the suite is not at
the full-package definition of done until parser round trips and the optional
callback/warm-start runtime checks run in a CUDA-enabled environment.
