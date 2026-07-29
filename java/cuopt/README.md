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

The standalone native project links to `${CUOPT_PREFIX}/lib/libcuopt.so` and
uses private cuOpt headers from the checkout to implement the Java-local
bridge. The Java-required native extensions need C API review before this
module can become a supported binary distribution. No Java-specific symbol or
source file is currently required by the main cuOpt build.

## Generated constants

Maven generates `CuOptConstants.java` under
`target/generated-sources/cuopt/com/nvidia/cuopt/mathematicalprogramming/`
from `cpp/include/cuopt/mathematical_optimization/constants.h`. Do not edit the
generated file. Regenerate it after changing the C++ constants header with:

```bash
cd java/cuopt
mvn generate-sources
```
