# cuOpt Java Bindings

This directory contains a source module for cuOpt LP, MIP, QP,
QCQP, and SOCP Java bindings. The repository CI and release workflows build
and test the module against the matching `libcuopt` conda artifact and retain
the Maven `target/` output as a workflow artifact. Publication to a supported
Maven repository has not been defined.

## Building

The module is an opt-in target of the repository-level `build.sh`. It is not
part of a default build, and it is not wired into the main CMake targets.

```bash
./build.sh libcuopt          # once, to produce cpp/build/libcuopt.so
./build.sh java              # build libcuopt_jni.so and package the jar
./build.sh java --run-java-tests   # the same, then run the test suite
```

`./build.sh java` prefers the `cpp/build` tree, so it works without
`--install`. It falls back to the active conda prefix when no build tree is
present, which is what CI does with the prebuilt `libcuopt` conda artifact.

The module can also be driven directly against an existing cuOpt installation:

```bash
cd java/cuopt
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/build_native.sh
CUOPT_PREFIX=/path/to/cuopt/conda/environment bash scripts/test.sh
```

`build_native.sh` builds `libcuopt_jni.so` in `build/native`. `test.sh` builds
that library and runs the Maven tests. Java 11 or newer and a C++20 compiler
are required. Native solve tests require a CUDA driver and skip automatically
when one is unavailable.

`CUOPT_LIBRARY`, `CUOPT_EXTRA_INCLUDE_DIRS`, `CUOPT_EXTRA_LIBRARY_DIRS`, and
`CUOPT_PRELOAD_LIBS` override where the scripts look for `libcuopt` and its
dependencies; `build.sh` sets them when it targets a build tree. The rmm and
raft headers must be exactly the ones `libcuopt` was compiled against, because
rmm encodes its version in an inline namespace (`rmm::_RMM_26_10`) — mixing a
different copy links cleanly and then fails at `dlopen` with an undefined
symbol.

The standalone native project links to `${CUOPT_PREFIX}/lib/libcuopt.so`. No
Java-specific symbol or source file is required by the main cuOpt build.

## JNI Symbol Check

The bindings are hand-written, so every `static native` method in
`NativeCuOpt.java` needs a matching `Java_com_nvidia_..._name` function in
`cuopt_jni.cpp`. Nothing in the compiler enforces that pairing: a missing entry
point compiles cleanly and fails at run time with `UnsatisfiedLinkError`, and a
renamed one leaves dead code behind in the library.

`scripts/check_jni_symbols.sh` compares the prototypes `javac -h` derives from
the Java sources against the symbols the built library actually exports, and
fails on a mismatch in either direction. It reads the built library rather than
parsing the source, so the macro-generated entry points need no special casing.

`build_native.sh` runs it after every native build, so `./build.sh java` and
both CI jobs cover it. It takes about a second. It is not a pre-commit hook,
because it needs a built `libcuopt_jni.so` and therefore a full `libcuopt`
build, which the other hooks do not require.

To skip it while iterating — say, after adding a `native` declaration but
before writing its entry point — set `CUOPT_SKIP_JNI_SYMBOL_CHECK=1`. It can
also be run on its own once the library exists:

```bash
cd java/cuopt
bash scripts/check_jni_symbols.sh
```

## Generated Constants

Maven generates `CuOptConstants.java` under
`target/generated-sources/cuopt/com/nvidia/cuopt/mathematicaloptimization/`
from `cpp/include/cuopt/mathematical_optimization/constants.h`. Do not edit the
generated file. Regenerate it after changing the C++ constants header with:

```bash
cd java/cuopt
mvn generate-sources
```
