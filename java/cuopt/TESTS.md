<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cuOpt Java binding tests

This module currently has three Java test classes under
`src/test/java/com/nvidia/cuopt/linearprogramming`:

- `ProblemModelingTest`: pure Java modeling tests.
- `NativeIntegrationTest`: JNI/native cuOpt smoke tests.
- `PythonParityTest`: Java-vs-Python cuOpt parity tests.

The full suite has 22 test executions: 3 modeling tests, 9 native integration
tests, and 10 Python parity dynamic tests.

For the broader API coverage and Java/Python parity backlog, see
`API_PARITY.md`.

## How to run

Run these commands from a shell that can see the GPU. A good quick check is:

```bash
nvidia-smi
```

The Java module has its own native CMake project. It does not add a target to
the repository-level `build.sh` and does not modify the main cuOpt C/C++ API.
Build the JNI library first:

```bash
cd /home/cbrissette/cuopt/java/cuopt

CUOPT_PREFIX=/home/cbrissette/cuopt/.cuopt_env \
bash scripts/build_native.sh
```

The native helper preserves the compiler recorded in an existing Java CMake
cache and clears inherited conda `CFLAGS`, `CXXFLAGS`, `CPPFLAGS`, and
`LDFLAGS` while configuring/building. This prevents conda's host compiler flags
from exposing incompatible CUDA FP4/FP6 headers in the standalone JNI build.

Then run only the Python parity suite:

```bash
cd /home/cbrissette/cuopt/java/cuopt

JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/home/cbrissette/cuopt/.cuopt_env/targets/x86_64-linux/lib:/home/cbrissette/cuopt/.cuopt_env/lib:/home/cbrissette/cuopt/java/cuopt/build/native \
mvn clean test -Dtest=PythonParityTest \
  -Dcuopt.native.dir=/home/cbrissette/cuopt/java/cuopt/build/native \
  -Dcuopt.python=/home/cbrissette/cuopt/.cuopt_env/bin/python
```

To run the full Java binding suite:

```bash
cd /home/cbrissette/cuopt/java/cuopt

JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64 \
LD_LIBRARY_PATH=/home/cbrissette/cuopt/.cuopt_env/targets/x86_64-linux/lib:/home/cbrissette/cuopt/.cuopt_env/lib:/home/cbrissette/cuopt/java/cuopt/build/native \
mvn test \
  -Dcuopt.native.dir=/home/cbrissette/cuopt/java/cuopt/build/native \
  -Dcuopt.python=/home/cbrissette/cuopt/.cuopt_env/bin/python
```

The helper script combines both steps and accepts Maven arguments:

```bash
cd /home/cbrissette/cuopt/java/cuopt
CUOPT_PREFIX=/home/cbrissette/cuopt/.cuopt_env bash scripts/test.sh -Dtest=PythonParityTest
```

Important runtime inputs:

- `cuopt.native.dir` must point at the Java module's native build directory,
  containing `libcuopt_jni`.
- `cuopt.python` should point at a Python environment with Python cuOpt
  installed.
- `LD_LIBRARY_PATH` needs the CUDA runtime directory, the conda environment
  libraries, and the local cuOpt native build directory when invoking Maven
  directly. `scripts/test.sh` configures these paths automatically.

## `ProblemModelingTest`

These tests do not require the native library, Python cuOpt, or a GPU. They
exercise the Java modeling layer before anything crosses into JNI.

| Test | What it verifies |
| --- | --- |
| `buildsLinearModelAndCsr` | Builds a small two-variable MIP through the high-level `Problem` API. It checks variable indices, MIP detection, CSR row offsets, column indices, matrix values, objective setup, constraint count, and RHS adjustment when a constraint expression has a constant term. |
| `duplicateLinearTermsAreMergedForSlack` | Adds the same variable twice to one linear expression and verifies the coefficients are merged. It also checks slack computation and confirms the model is not marked as MIP when all variables are continuous. |

## `NativeIntegrationTest`

These tests verify that the Java bindings can load and call the JNI/native cuOpt
library. Tests that solve models also require a visible CUDA driver.

| Test | What it verifies |
| --- | --- |
| `solverParameterNamesAreAvailable` | Loads the native library and confirms solver parameter names can be queried through `SolverSettings`. This is a low-cost JNI smoke test. |
| `settingsExposeTypedValuesAndParameterFileRoundTrip` | Checks typed parameter reads, optimality-tolerance propagation, callback registration state, the settings map view, and native parameter dump/load bindings. |
| `emptyDataModelCanBeClosed` | Verifies a JNI-created empty `DataModel` can be closed without crossing the native ownership boundary incorrectly. |
| `mutableDataModelExposesPythonMetadataAndQuadraticFields` | Exercises mutable DataModel setters, names, objective scaling, initial vectors, Q-CSR getters, quadratic-constraint add/get/clear, and `toDict`. |
| `solvesSmallLpAndReportsStats` | Builds a tiny LP directly as a `DataModel`, solves it with PDLP, checks the termination status and primal objective, verifies the primal solution satisfies the single constraint, and confirms LP-only stats are available while MIP stats are rejected. |
| `solvesProblemApiMilpAndLifecycleCloseIsIdempotent` | Builds a one-variable integer problem through the high-level `Problem` API, solves it, checks the integer solution value, verifies MIP stats are available and dual values are rejected, and confirms calling `Solution.close()` twice is safe. |
| `solvesSmallQp` | Starts from the tiny LP model, adds a quadratic objective, solves it, and verifies the Java binding can retrieve a primal solution for a QP-shaped model. |
| `rejectsMissingFileThroughCuOptException` | Calls `DataModel.read` on a missing MPS file and verifies the Java API raises `CuOptException` with the expected MPS file error status. |
| `writesAndReadsMpsThroughReadAndParseMps` | Writes a generated MPS model and reads it through both extension-dispatch and direct MPS parser APIs, plus the high-level `Problem.read` path. |
| `batchSolveCompatibilityReturnsAllSolutions` | Verifies the Java BatchSolve compatibility entry point returns one solution per model and reports elapsed time. |

## `PythonParityTest`

`PythonParityTest` is a dynamic JUnit test factory. For each case below, Java
constructs a `DataModel`, writes the same model data to a temporary JSON file,
and invokes `src/test/resources/python_binding_parity.py` as the Python oracle.
The Python helper builds the same model with Python cuOpt, solves it with the
same deterministic settings, and emits `CUOPT_COMPARE key=value` lines for Java
to compare.

Each parity case compares both model data and solve behavior:

- model dimensions and nonzero count;
- objective sense, offset, scaling factor, and coefficients;
- CSR matrix values, column indices, and row offsets;
- variable bounds, variable types, variable names, row names, and model names;
- quadratic objective values, column indices, and row offsets for the QP case;
- row sense/RHS for row-sense models, or constraint lower/upper bounds for
  ranged-bound models;
- solution category, termination status, error status/message, and solve time
  that is either non-negative or `NaN` when unavailable;
- primal objective and primal solution for cases expected to have solution
  values;
- dual, dual-objective, and reduced-cost availability and values for LP/QP
  solutions;
- LP-stat availability, residuals, gap, iteration count, and solved-by method;
- MIP-stat availability, MIP gap, solution bound, presolve time that is either
  non-negative or `NaN` when unavailable, violation fields, node count, and
  simplex-iteration count.

The infeasible case only compares status and availability-style behavior, not
solution vectors.

| Dynamic test | What it verifies |
| --- | --- |
| `lp_min_ge_unique_solution` | Compares a small continuous minimization LP with one greater-than-or-equal constraint, bounded variables, and a nonzero objective offset. |
| `lp_max_le_unique_solution` | Compares a continuous maximization LP with three less-than-or-equal constraints. This exercises maximize sense handling and a denser multi-row CSR matrix. |
| `lp_equal_with_offset` | Compares a continuous LP with an equality constraint and a nonzero objective offset. This checks equality row type handling and objective-offset parity. |
| `lp_ranged_bounds` | Compares a continuous LP created through the ranged-constraint API using explicit constraint lower and upper bounds instead of row sense plus RHS. |
| `lp_mixed_bounds_negative_coefficients` | Compares an LP with negative variable lower bounds, negative objective coefficients, equality constraints, and a nonzero negative objective offset. |
| `lp_max_ranged_bounds` | Compares a maximization LP built with ranged constraints to cover maximize sense and ranged-bound handling together. |
| `milp_integer_unique_solution` | Compares a small integer minimization model. This checks integer variable type propagation, MIP solve behavior, and MIP-specific stat availability. |
| `milp_mixed_integer_continuous_max` | Compares a maximization MILP with one integer and one continuous variable, exercising mixed variable types and fractional continuous values in a MIP solution. |
| `qp_diagonal_objective` | Compares a convex QP with a diagonal quadratic objective matrix, one quadratic constraint, linear objective terms, named variables/rows, objective scaling, ranged constraints, and bounded continuous variables. |
| `lp_infeasible_status` | Compares a contradictory one-variable LP with lower and upper row requirements that cannot both hold. This verifies Java and Python agree on infeasible termination behavior. |

## Python oracle helper

`src/test/resources/python_binding_parity.py` is not a standalone JUnit test, but
it is part of the parity suite.

It supports two modes:

- `--probe`: imports Python cuOpt and emits `CUOPT_COMPARE probe=ok`; the Java
  tests use this to skip cleanly if Python cuOpt is unavailable.
- `case_file`: reads a JSON model spec, builds the matching Python cuOpt
  `DataModel`, solves it, and emits model/solution fields for Java assertions.

You can run the probe by hand:

```bash
/home/cbrissette/cuopt/.cuopt_env/bin/python \
  /home/cbrissette/cuopt/java/cuopt/src/test/resources/python_binding_parity.py --probe
```

## Pass, fail, and skip behavior

The expected successful Maven result is `BUILD SUCCESS`.

Some tests intentionally skip instead of fail when prerequisites are missing:

- native tests skip when `cuopt.native.dir` is unset or `libcuopt_jni` is not
  present;
- GPU solve tests skip when `nvidia-smi` cannot see a CUDA driver;
- Python parity tests skip when Python cuOpt cannot be imported.

If a test fails or the forked JVM crashes, inspect:

```text
/home/cbrissette/cuopt/java/cuopt/target/surefire-reports
```
