<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# cuOpt Java Binding Tests

The Java module has three test classes under
`src/test/java/com/nvidia/cuopt/mathematicaloptimization`:

- `ProblemModelingTest`: five pure Java modeling tests.
- `NativeIntegrationTest`: seven JNI/native cuOpt smoke tests.
- `ProblemIntegrationTest`: ten standalone Java problem and solve tests.

The suite has no dependency on the cuOpt Python interface.

## How to Run

Build the JNI library and run all Java tests with:

```bash
cd /path/to/cuopt/java/cuopt
export JAVA_HOME=/path/to/jdk-11
export CUOPT_PREFIX=/path/to/cuopt/conda/environment
bash scripts/test.sh
```

To run one test class:

```bash
bash scripts/test.sh -Dtest=ProblemIntegrationTest
```

When invoking Maven directly, build the JNI library first and provide the
native directory:

```bash
bash scripts/build_native.sh
mvn test -Dcuopt.native.dir=build/native
```

`scripts/test.sh` configures `LD_LIBRARY_PATH` for the selected cuOpt and CUDA
runtime libraries.

## Coverage

`ProblemModelingTest` exercises generated enum constants, legacy problem
category mapping, expression construction, CSR generation, duplicate-term
merging, problem updates, relaxation, and quadratic inspection without loading
the native library.

`NativeIntegrationTest` covers settings, setting-file round trips, LP/MILP/QP
solves, solution statistics, error propagation, and MPS read/write paths.

`ProblemIntegrationTest` constructs ten LP, MILP, and QP cases entirely in
Java. Each dynamic test verifies the public `Problem` model built from the
case data. The LP/MILP cases also check solve status, variable bounds,
integrality, constraint feasibility, objective values, and type-specific
solution behavior. The QP case verifies quadratic-objective and
quadratic-constraint construction; QP solve callability is covered by
`NativeIntegrationTest`. The cases cover minimization and maximization,
equality and ranged constraints, mixed bounds, mixed integer/continuous
variables, metadata, and infeasibility.

## Prerequisite Behavior

- Pure modeling tests run without JNI or a GPU.
- Native tests skip when `cuopt.native.dir` is unset or `libcuopt_jni` is not
  present.
- Solve tests skip when a CUDA driver is unavailable.

The expected successful Maven result is `BUILD SUCCESS`. If a forked JVM
crashes, inspect `target/surefire-reports`.
