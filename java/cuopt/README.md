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
that library and runs the Maven tests. Java 17 or newer and a C++20 compiler
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

## Remote gRPC Client (Experimental)

`GrpcClient` (`com.nvidia.cuopt.mathematicaloptimization.GrpcClient`) talks to a
remote `cuopt_grpc_server` instead of solving in-process. It wraps
`cuopt::cython::grpc_python_client_t` — the same C++ class the Python bindings
wrap — via `src/main/native/cuopt_grpc_jni.cpp`, so it inherits that class's
automatic chunked upload/download for problems too large for a single protobuf
message.

Covers LP and MIP submit/status/wait/cancel/delete/result, plaintext or TLS.
Not covered yet: QP (the wire protocol supports it, this client doesn't wire
it through yet), log streaming, and incumbent callbacks. Not wired into the
Maven-packaged classifier jar or CI yet.

`submit()` takes a `Problem` directly, matching the Python client's own
`submit(problem, settings)` — no raw arrays needed for the common case:

```java
try (Problem problem = Problem.read("model.mps");
    GrpcClient client = new GrpcClient("localhost", 5001)) {
  client.connect();

  String jobId = client.submit(problem, /* timeLimitSeconds= */ 30.0);
  GrpcJobStatus status = client.waitForCompletion(jobId, /* timeoutSeconds= */ 60);

  if (status == GrpcJobStatus.COMPLETED) {
    GrpcMipResult result = client.getMipResult(jobId); // or getLpResult(jobId)
    System.out.println("objective: " + result.getObjective());
  }
}
```

A lower-level `submit(RawProblem, ...)` overload also exists, taking the same
CSR-array shape `NativeCuOpt.createProblem` uses, for callers that already
have raw arrays and don't want to build a `Problem`.

### Installing and trying it

There's no published artifact yet (see the top of this README) — trying it
means building from source. From the repo root:

```bash
mamba env create -p ./.cuopt_env --file conda/environments/all_cuda-133_arch-$(uname -m).yaml
mamba activate ./.cuopt_env
./build.sh libcuopt java          # produces cpp/build/libcuopt.so, cpp/build/cuopt_grpc_server,
                                   # and java/cuopt/build/native/libcuopt_jni.so
```

Start a server and run the checked-in example, which submits the same small
problem two ways — built via the regular API, and read from an MPS file —
against a real server:

```bash
cpp/build/cuopt_grpc_server --port 5001 &

cd java/cuopt
mvn -q test-compile
mvn -q exec:java -Dexec.mainClass=com.nvidia.cuopt.mathematicaloptimization.GrpcClientExample \
    -Dexec.classpathScope=test -Dcuopt.native.dir=build/native -Dexec.args="localhost 5001"
```

If the server or the example fails to load a native library with an
`UnsatisfiedLinkError` naming an `rmm::` symbol, the conda environment's
`librmm.so` doesn't export something the current build needs (a version-pin
gap seen locally, not confirmed as a general issue) — work around it with
`LD_PRELOAD=$(pwd)/../../.cuopt_env/lib/librmm.so` on the failing command.
