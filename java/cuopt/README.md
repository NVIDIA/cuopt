# cuOpt Java bindings

This directory contains a source module for cuOpt LP, MILP, QP,
QCQP, and SOCP Java bindings. The repository CI and release workflows build
and test the module against the matching `libcuopt` conda artifact and retain
the Maven `target/` output as a workflow artifact. Publication to a supported
Maven repository has not been defined.

The module is not connected to the repository-level `build.sh` or main CMake
targets.

For local development, compile the Java module against an existing cuOpt
installation:

```bash
cd java/cuopt
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/build_native.sh
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/test.sh
```

`build_native.sh` builds `libcuopt_jni.so` in `build/native`. `test.sh` builds
that library and runs the Maven tests. Java 11 or newer and a C++20 compiler
are required. Native solve tests require a CUDA driver and skip automatically
when one is unavailable.

The standalone native project links to `${CUOPT_PREFIX}/lib/libcuopt.so`. No
Java-specific symbol or source file is required by the main cuOpt build.

The seven entry points the bindings once declared in a Java-local shim are now
part of the public C API (`cuOptLoadParametersFromFile`,
`cuOptDumpParametersToFile`, `cuOptGetNumSolverParameters`,
`cuOptGetSolverParameterName`, `cuOptSolutionIsMIP`, `cuOptGetLPSolverStats`,
and `cuOptGetMIPSolverStats`), so nothing in the settings or solution path
depends on private headers any more.

The problem path still does. `cuopt_jni.cpp` includes
`pdlp/cuopt_c_internal.hpp` from the checkout for the operations the C API does
not yet cover:

- setting the problem, variable, and row names (the C API only reads them),
- reading the quadratic objective matrix and the quadratic constraint rows
  (see the `TODO` in `cuopt_c.h`),
- reading variable and row names when the problem has none, which the C API
  string-array getter rejects rather than reporting as empty,
- reading the problem category,
- reading the dual solution and reduced costs. These are empty when the solve
  did not produce them (an infeasible LP, for instance), and the Java API
  reports that as an empty array. The C API getters are copy-out into a
  caller-sized buffer and report no length, so switching to them would turn
  "unavailable" into a buffer of zeros.

Closing those gaps in the C API is the remaining prerequisite for shipping this
module as a standalone binary distribution.

## Generated constants

Maven generates `CuOptConstants.java` under
`target/generated-sources/cuopt/com/nvidia/cuopt/mathematicalprogramming/`
from `cpp/include/cuopt/mathematical_optimization/constants.h`. Do not edit the
generated file. Regenerate it after changing the C++ constants header with:

```bash
cd java/cuopt
mvn generate-sources
```
