# Layer 5 — Public API (Java 21)

This package compiles to **Java 21 release**. Code here cannot reference
`java.lang.foreign.*` types (`MemorySegment`, `Arena`, etc.) because those
classes don't exist in Java 21.

## What lives here

- The user-facing API: `Solver` (currently exposes `getVersion()` only;
  `Problem`, `Variable`, `LinearExpr`, etc. land in subsequent PRs).
- Public exceptions: `cuOptException`.
- The SPI interface (`spi/cuOptProvider.java`) — sealed; permits only the
  Java 22 implementation in `internal/`.

## How calls reach the FFM implementation

Layer 5 calls the SPI:

```java
public final class Solver {
    public static String getVersion() {
        return cuOptProvider.instance().getVersion();
    }
}
```

`cuOptProvider.instance()` uses `java.util.ServiceLoader` to find the
implementation registered in `META-INF/services/`. On a Java 22+ JVM,
the registered class lives in `META-INF/versions/22/com/nvidia/cuopt/internal/`
(the multi-release JAR layer), so the JVM picks it up automatically.

On a Java 21 JVM the implementation class is missing entirely, and
`ServiceLoader.findFirst()` returns empty. We throw a clear error in
that case — the JAR loads at compile time but cannot run.
