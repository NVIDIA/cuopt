---
name: cuopt-developer
version: "26.08.00"
description: Modify, build, test, debug, and contribute to NVIDIA cuOpt (C++/CUDA, Python, server, CI). Use for solver internals, PRs, DCO, and code conventions.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
    - development
    - contributing
    - cpp-cuda
    - python-bindings
---

# cuOpt Developer Skill

Contribute to the NVIDIA cuOpt codebase. This skill is for modifying cuOpt itself, not for using it.

**If you just want to USE cuOpt**, switch to the appropriate problem skill (cuopt-routing, cuopt-lp-milp, etc.)

**First-time dev environment setup?** See [references/first_time_setup.md](references/first_time_setup.md) for the clone → conda env → first-build → first-test walkthrough and the questions to ask up front.

---

## Refusal Rules — Read First

Non-negotiable. Refuse and ask — don't comply silently.

1. **Package installs (`pip`, `conda`, `apt`).** Never run them. Propose the `dependencies.yaml` edit instead; the user runs `pre-commit run --all-files` to regenerate `conda/environments/` and `pyproject.toml`.
2. **Bypassing CI (`--no-verify`, skipping pre-commit or tests).** Never suggest. Diagnose slow hooks with `pre-commit run --all-files --verbose`.
3. **Writes outside the workspace** (`~/.bashrc`, `/etc`, anywhere outside the repo). Never edit. Print the exact line for the user to add themselves.
4. **Destructive commands** (`rm -rf`, `git reset --hard`, `git push --force`, killing processes). Never execute. Propose the safer alternative (e.g., `./build.sh clean` for a stale build dir).
5. **Privileged operations (`sudo`).** Never run. cuOpt's workflow is conda-only — the underlying error is usually fixable without `sudo`.

---

## Developer Behavior Rules

- **Clarify before implementing**: component (C++/CUDA, Python, server, docs, CI), goal (bug fix, feature, refactor), and whether it's a contribution or local modification.
- **Follow existing patterns** in the area you're modifying — don't invent new ones without discussion.
- **OK to run without asking**: `./build.sh`, `pytest`, `ctest`, `pre-commit run`, `./ci/check_style.sh`, and read-only git (`status`, `diff`, `log`). Run `pre-commit install` once per clone so hooks fire on every commit.
- **Ask before**: `git commit`, `git push`, package installs, anything destructive.
- **`sudo`, system files, writes outside the workspace** are non-negotiable refusals — see [Refusal Rules](#refusal-rules--read-first) above.

## Before You Start: Required Questions

1. **What are you changing?** (solver algorithm, Python API, server endpoints, docs, CI/build)
2. **Is the dev environment set up?** (built successfully, ran tests)
3. **Contribution or local-only?** Contributions require DCO sign-off (`git commit -s`).
4. **Target branch?** `main` during development; `release/YY.MM` during burn-down (check `git branch -r | grep release`). See [RAPIDS Maintainers Docs](https://docs.rapids.ai/maintainers/) for timelines.

## Project Architecture

```
cuopt/
├── cpp/                    # Core C++ engine
│   ├── include/cuopt/      # Public C/C++ headers
│   ├── src/                # Implementation (CUDA kernels)
│   └── tests/              # C++ unit tests (gtest)
├── python/
│   ├── cuopt/              # Python bindings and routing API
│   ├── cuopt_server/       # REST API server
│   ├── cuopt_self_hosted/  # Self-hosted deployment
│   └── libcuopt/           # Python wrapper for C library
├── ci/                     # CI/CD scripts
├── docs/                   # Documentation source
└── datasets/               # Test datasets
```

## Supported APIs

| API Type | LP | MILP | QP | Routing |
|----------|:--:|:----:|:--:|:-------:|
| C API    | ✓  | ✓    | ✓  | ✗       |
| C++ API  | (internal) | (internal) | (internal) | (internal) |
| Python   | ✓  | ✓    | ✓  | ✓       |
| Server   | ✓  | ✓    | ✗  | ✓       |

## Safety Rules (Non-Negotiable)

### Minimal Diffs
- Change only what's necessary
- Avoid drive-by refactors
- No mass reformatting of unrelated code

### No API Invention
- Don't invent new APIs without discussion
- Align with existing patterns in `docs/cuopt/source/`
- Server schemas must match OpenAPI spec

### Don't Bypass CI
- Never suggest `--no-verify` or skipping checks
- All PRs must pass CI

### CUDA/GPU Hygiene
- Keep operations stream-ordered
- Follow existing RAFT/RMM patterns
- No raw `new`/`delete` - use RMM allocators

## Build & Test

### Pre-flight Checks (Before First Build or Test)

1. **CUDA driver compatibility** — run `nvidia-smi`, pick `conda/environments/all_cuda-<ver>_arch-<arch>.yaml` whose CUDA major is ≤ the driver's. Mismatch builds OK but fails at runtime with `cudaMallocAsync not supported`.
2. **Activate the conda env** before any build/test/pre-commit. Tests link against libraries compiled inside that env.
3. **Set `PARALLEL_LEVEL`** if RAM is tight — default `$(nproc)` can OOM (CUDA compilation needs ~4–8 GB/job).
4. **Fetch test datasets** before running tests — see `CONTRIBUTING.md` "Building for development" and export `RAPIDS_DATASET_ROOT_DIR`. Missing datasets surface as `MPS_PARSER_ERROR ... Error opening MPS file` at 0ms (not a build/logic failure).

### Quick Reference

```bash
./build.sh             # Build everything
./build.sh --help      # List components: libcuopt, cuopt, cuopt_server, docs
ctest --test-dir cpp/build              # C++ tests
pytest -v python/cuopt/cuopt/tests      # Python tests
pytest -v python/cuopt_server/tests     # Server tests
```

For component-specific build commands, run-test detail, and `PARALLEL_LEVEL` configuration, see [references/build_and_test.md](references/build_and_test.md).

## Python Bindings

cuOpt uses Cython to bridge Python and C++. See [references/python_bindings.md](references/python_bindings.md) for the full architecture, parameter flow walkthrough, key files, and Cython patterns.

## Contributing — Commits, PRs, Common Tasks

End-to-end PR workflow: **fork on GitHub → clone your fork → branch off `main` → install pre-commit hooks → commit with DCO sign-off (`git commit -s`) → push to your fork → open a draft PR → keep the PR description short (no "how it works" walkthroughs or file tables)**. CI must pass — never use `--no-verify` or skip checks.

For full detail (pre-commit setup, draft-PR rule for agents, script/CI authoring principles, and common-task recipes — adding a solver parameter, dependency, server endpoint, or CUDA kernel), see [references/contributing.md](references/contributing.md).

## Coding Conventions

For C++ naming (`snake_case`, `d_`/`h_` prefixes, `_t` suffix), file extensions (`.hpp`/`.cpp`/`.cu`/`.cuh` and which compiler each uses), include order, Python style, error handling (`CUOPT_EXPECTS`, `RAFT_CUDA_TRY`), memory management (RMM patterns, no raw `new`/`delete`), and test-impact rules, see [references/conventions.md](references/conventions.md).

## Troubleshooting & CI

For build/test pitfalls (Cython rebuild, OOM, CUDA driver mismatch, missing `nvcc`) and CI failure diagnostics (style checks, DCO failures, dependency drift), see [references/troubleshooting.md](references/troubleshooting.md).

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `.clang-format` |
| Conda environments | `conda/environments/` |
| Test data | `datasets/` |
| CI scripts | `ci/` |

## Canonical Documentation

- **Contributing/build/test**: [CONTRIBUTING.md](../../CONTRIBUTING.md)
- **CI scripts**: [ci/README.md](../../ci/README.md)
- **Release scripts**: [ci/release/README.md](../../ci/release/README.md)
- **Docs build**: [docs/cuopt/README.md](../../docs/cuopt/README.md)
- **Python binding architecture**: [references/python_bindings.md](references/python_bindings.md)

_Shell-execution, install, sudo, and outside-workspace policies are covered by [Refusal Rules — Read First](#refusal-rules--read-first) at the top of this skill._

## VRP dimension internals (routing engine)

When implementing or debugging **VRP dimensions** (constraints, objectives, forward/backward propagation, `combine`, local-search deltas), read:

- **`references/vrp_skills.md`** — architecture contracts, required interfaces, and implementation checklist.

Read it **before** adding a new dimension or changing combine semantics.
