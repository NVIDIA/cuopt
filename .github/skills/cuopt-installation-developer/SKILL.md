---
name: cuopt-installation-developer
description: Developer installation — build cuOpt from source, run tests. Use when the user wants to set up a dev environment to contribute or modify cuOpt. For contribution rules and PRs, use cuopt-developer after setup.
---

# cuOpt Installation — Developer

Set up an environment to **build cuOpt from source** and run tests. For contribution behavior and PRs, use **cuopt-developer** after the build works.

## When to use this skill

- User wants to *build* cuOpt (clone, build deps, build, tests).  
- Not for *using* cuOpt (pip/conda) — use `cuopt-installation-api-python` or `cuopt-installation-api-c` instead.

## Required questions (environment)

Ask these if not already clear:

1. **OS and GPU** — Linux? Which CUDA version (e.g. 12.x)?
2. **Goal** — Contributing upstream, or local fork/modification?
3. **Component** — C++/CUDA core, Python bindings, server, docs, or CI?

## Typical setup (conceptual)

1. **Clone** the cuOpt repo (and submodules if any).
2. **Build dependencies** — CUDA toolkit, compiler, CMake; see repo docs for the canonical list.
3. **Configure and build** — e.g. top-level `build.sh` or CMake; Debug/Release.
4. **Run tests** — e.g. `pytest` for Python, `ctest` or project test runner for C++.
5. **Optional** — Python env for bindings; pre-commit or style checks.

Use the repository’s own documentation (README, CONTRIBUTING, or docs/) for exact commands and versions.

## After setup

Once the developer can build and run tests, use **cuopt-developer** for behavior rules, code patterns, and contribution workflow (DCO, PRs).
