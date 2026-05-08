# cuopt-java (Maven artifact)

Source for the published `com.nvidia.cuopt:cuopt-java` artifact.

## Build setup highlights

The `pom.xml` encodes three lessons from cuvs-java:

- **Java 21 base + Java 22 multi-release JAR** (cuvs Issue #1066): the
  default-compile execution targets `<release>21</release>`; a separate
  `compile-java-22` execution targets `<release>22</release>` and writes
  to the `META-INF/versions/22/` section of the JAR via
  `multiReleaseOutput=true`.

- **`maven-compiler-plugin` 3.11.0+ with explicit phase bindings**
  (cuvs Issue #1293): the two compile executions are bound to the
  `compile` and `process-classes` lifecycle phases respectively, so they
  run **serially** rather than in parallel. Without this, ~50% of builds
  fail with phantom "cannot find symbol" errors as the java22 compile
  starts before the java21 compile finishes.

- **Spotless in `<build><plugins>` only** (cuvs Issue #1090): never list
  spotless under `<dependencies>` — that pulls spotless code into a
  fat JAR with multiple `module-info.class` files, breaking modularized
  consumers.

## Layout

```
cuopt-java/
├── pom.xml
└── src/
    ├── main/
    │   ├── java/                            ← Java 21 public API (Layer 5)
    │   ├── java22/                          ← Java 22 FFM impl (Layers 3 + 2)
    │   └── resources/META-INF/
    │       ├── cuopt/supported-platforms.properties   ← data-driven platform check
    │       └── services/                              ← ServiceLoader registration
    └── test/
        └── java/                            ← JUnit 5
```
