# cuopt-java

Java interface to NVIDIA cuOpt for Linear Programming (LP), Mixed-Integer
Linear Programming (MILP), and Quadratic Programming (QP).

This directory contains the source for the `com.nvidia.cuopt:cuopt-java`
Maven artifact.

## Status

This is the initial skeleton. The first end-to-end demo (`Solver.getVersion()`)
proves the FFM bridge from Java to `libcuopt.so` works. The full LP / MILP / QP
public API will land in subsequent PRs.

## Architecture (5 layers)

```
Layer 5  Public API (Java 21)         src/main/java/com/nvidia/cuopt/
            ▼  ServiceLoader / SPI
Layer 4  CuOptProvider (sealed)       src/main/java/.../spi/
            ▼  resolved to java22 impl on Java 22+ JVMs
Layer 3  FFM Implementation           src/main/java22/.../internal/
         (Java 22)                    Owns Arena.ofConfined(), MemorySegment
            ▼
Layer 2  Panama Bindings              src/main/java22/.../internal/panama/
         (jextract output)            Generated; never edited by hand
            ▼  MethodHandle.invokeExact (no JNI)
Layer 1  Native library               libcuopt.so (cpp/build/)
                                      cpp/include/cuopt/linear_programming/cuopt_c.h
```

The public API (Layer 5) compiles to **Java 21 release** so the JAR can be
loaded on Java 21 classpaths at compile time. The FFM implementation (Layer 3)
uses `java.lang.foreign.*` types which require **Java 22 runtime**. The
multi-release JAR resolves the right classes at runtime via the JVM's
`META-INF/versions/22/` lookup.

The SPI bridge (Layer 4) exists because Layer 5 cannot reference `MemorySegment`
or other Java 22 types in its method signatures (those classes don't exist in
Java 21). `CuOptProvider` is a Java-21-compatible interface that
`ServiceLoader` resolves to the Java 22 implementation at runtime.

## Layout

```
java/
├── README.md                       (this file)
├── build.sh                        Top-level entry point — auto-downloads
│                                   jextract, regenerates bindings, runs mvn.
├── license-header.txt              Apache 2.0 SPDX header for source files.
├── cuopt-java/                     The published Maven artifact.
│   ├── pom.xml                     Java 21 + Java 22 MR-JAR config.
│   └── src/
│       ├── main/java/              Java 21 public API.
│       ├── main/java22/            Java 22 FFM implementation.
│       ├── main/resources/         ServiceLoader registration; platform table.
│       └── test/java/              JUnit 5 tests.
└── panama-bindings/                jextract binding generation.
    ├── headers.h                   Umbrella include of cuopt_c.h.
    └── generate-bindings.sh        jextract invocation.
```

## Building (developer workstation)

Prerequisites:

- JDK 22+ (`conda install -c conda-forge openjdk=22`)
- Maven 3.9.6+ (`conda install -c conda-forge maven`)
- A built `libcuopt.so` (from `../cpp/build/`)

**jextract** is not available on conda-forge — `panama-bindings/generate-bindings.sh`
auto-downloads `openjdk-22-jextract+6-47` from `download.java.net` on first
run and extracts it to `panama-bindings/jextract-22/` (gitignored). Subsequent
builds reuse the local copy. Set `JEXTRACT=/path/to/jextract` to override.

```bash
# Build libcuopt.so first if not already built
cd /path/to/cuopt
./build.sh libcuopt

# Then build the Java side (auto-downloads jextract on first run)
./java/build.sh
```

This regenerates panama bindings from `cpp/include/cuopt/linear_programming/cuopt_c.h`,
then runs `mvn clean verify`. Output JARs land in `java/cuopt-java/target/`.

## Per-folder design notes

Each directory under `java/` has its own `README.md` explaining the role
of that layer in the architecture. Read them in this order if you want
the full picture:

1. `cuopt-java/README.md` — the published artifact
2. `cuopt-java/src/main/java/com/nvidia/cuopt/README.md` — Layer 5 public API
3. `cuopt-java/src/main/java22/com/nvidia/cuopt/README.md` — Layers 3 + 2 (FFM impl + panama)
4. `panama-bindings/README.md` — binding generation pipeline

## References

- C API header: `cpp/include/cuopt/linear_programming/cuopt_c.h`
- cuvs-java reference: https://github.com/rapidsai/cuvs/tree/main/java
- jextract: https://github.com/openjdk/jextract
- JEP 454 (FFM Final): https://openjdk.org/jeps/454
