---
name: cuopt-installation-developer
version: "26.06.00"
description: First-time dev env setup for cuOpt (CUDA check, conda env, initial build). Hand off to cuopt-developer for ongoing build/test/contribute work.
---

# cuOpt Installation — Developer

**First-time dev environment setup**: verify CUDA/driver compatibility, create the conda env, clone, complete the first build. Once the env works, switch to `cuopt-developer` for ongoing build/test/contribute work.

## When to use this skill

- User is **setting up the dev environment for the first time** — clone, conda env, build deps, initial build.
- Once the env is working, switch to `cuopt-developer` for ongoing build, test, debug, and contribute work.
- Not for *using* cuOpt (pip/conda) — use the user installation skill instead.

## Required questions (environment)

Ask these if not already clear:

1. **OS and GPU** — Linux? Which CUDA version (e.g. 12.x)?
2. **Goal** — Contributing upstream, or local fork/modification?
3. **Component** — C++/CUDA core, Python bindings, server, docs, or CI?

<!-- skill-evolution:start — driver/toolkit CUDA mismatch surfaced at runtime -->
## Validate CUDA/driver compatibility before building

Before creating the conda env or running `./build.sh`, check that the conda env's
CUDA toolkit **major** version matches what the installed driver supports. CUDA
guarantees minor-version compatibility within a major (e.g. CUDA 12.9 runtime
works on a driver that tops out at CUDA 12.8), but a major-version jump does
not (e.g. CUDA 13.x runtime on a CUDA-12-only driver). A major mismatch builds
successfully but fails at runtime inside RMM with:

```
RMM failure ... cudaMallocAsync not supported with this CUDA driver/runtime version
```

Steps:

1. Query the driver's max CUDA: `nvidia-smi` → top-right "CUDA Version:" field.
   Note the **major** version (e.g. `12.8` → major 12).
2. List available env files: `ls conda/environments/all_cuda-*_arch-$(uname -m).yaml`.
   Each filename encodes the CUDA version (e.g. `all_cuda-129_...` = CUDA 12.9,
   `all_cuda-131_...` = CUDA 13.1).
3. Pick an env whose CUDA **major** is ≤ the driver's max CUDA major. The env's
   minor version may exceed the driver's minor version — that's supported.
4. If a `.cuopt_env*` was already built against an incompatible major CUDA,
   create a new env against a compatible toolkit and `./build.sh clean` before
   rebuilding — do not reuse cached build artifacts across CUDA major versions.

Do this check before starting the build — a full build takes tens of minutes
and the failure only appears when tests run.
<!-- skill-evolution:end -->

## Typical setup (conceptual)

1. **Clone** the cuOpt repo (and submodules if any).
2. **Build dependencies** — CUDA toolkit, compiler, CMake; see repo docs for the canonical list.
3. **Configure and build** — e.g. top-level `build.sh` or CMake; Debug/Release.
4. **Run tests** — e.g. `pytest` for Python, `ctest` or project test runner for C++.
5. **Optional** — Python env for bindings; pre-commit or style checks.

Use the repository’s own documentation (README, CONTRIBUTING, or docs/) for exact commands and versions.

## After setup

Once the developer can build and run tests, use **cuopt-developer** for behavior rules, code patterns, and contribution workflow (DCO, PRs).
