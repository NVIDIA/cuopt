# cuOpt Java bindings (beta)

This directory is an isolated, customer-specific beta module for the cuOpt
LP, MILP, and QP Java bindings. It is intentionally not connected to the
repository-level `build.sh`, CMake targets, public C/C++ headers, dependency
manifest, or Python build.

The Java module must be compiled separately against an existing cuOpt
installation:

```bash
cd java/cuopt
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/build_native.sh
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/test.sh
```

`build_native.sh` builds `libcuopt_jni.so` in `build/native`. `test.sh` builds
that library and runs the Maven tests. Java 11 or newer and a C++20 compiler
are required. The native solve and Python-parity tests require a CUDA driver;
they skip automatically when one is unavailable.

The standalone native project links to `${CUOPT_PREFIX}/lib/libcuopt.so` and
uses private cuOpt headers from the checkout only to implement the Java-local
bridge. No Java-specific symbol or source file is required by the main cuOpt
build.
