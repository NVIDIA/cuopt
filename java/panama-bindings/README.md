# panama-bindings — jextract binding generation

This directory invokes [`jextract`](https://github.com/openjdk/jextract)
to generate Java bindings for cuOpt's C API.

## Files

- `headers.h` — umbrella include listing the C headers to bind. Currently
  pulls in `cpp/include/cuopt/linear_programming/cuopt_c.h` and
  `constants.h`.
- `generate-bindings.sh` — invokes `jextract` with the right flags.
  Output goes to
  `cuopt-java/src/main/java22/com/nvidia/cuopt/internal/panama/`.

## When this runs

`build.sh` (one level up) runs this script on every build. The generated
files are committed to the repo so reviewers can see the binding surface
and IntelliJ can navigate to symbols.

## Drift detection

After regeneration, `build.sh` runs `git diff --exit-code` on the
generated directory. If `jextract` produces output that differs from
what's committed (e.g., because `cuopt_c.h` changed), the build fails
with a clear message asking the developer to commit the regenerated
files. This catches additive C API changes that the upstream
`check-c-abi` gate (planned in a follow-up PR) accepts as non-breaking.

## Why we don't use a Maven plugin

Running `jextract` from `mvn` (e.g., via `exec-maven-plugin`) would tie
binding regeneration to `mvn verify`, which developers may skip locally.
Running it from `build.sh` keeps it as a deliberate step.
