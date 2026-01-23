# cuOpt engineering contract (cuopt_developer)

You are modifying the **cuOpt codebase**. Your priorities are correctness, performance, compatibility, and minimal-risk diffs.

If you only need to **use** cuOpt (not change it), switch to `cuopt_user` (`.github/agents/cuopt-user.md`).

## Project overview (developer context)

**cuOpt** is NVIDIA's GPU-accelerated optimization engine for:

- **Mixed Integer Linear Programming (MILP)**
- **Linear Programming (LP)**
- **Quadratic Programming (QP)**
- **Vehicle Routing Problems (VRP)** including TSP and PDP

### Architecture (high level)

```
cuopt/
├── cpp/                    # Core C++ engine (libcuopt, libmps_parser)
│   ├── include/cuopt/      # Public C/C++ headers
│   ├── src/                # Implementation (CUDA kernels, algorithms)
│   └── tests/              # C++ unit tests (gtest)
├── python/
│   ├── cuopt/              # Python bindings and routing API
│   ├── cuopt_server/       # REST API server
│   ├── cuopt_self_hosted/  # Self-hosted deployment utilities
│   └── libcuopt/           # Python wrapper for C library
├── ci/                     # CI/CD scripts and Docker configurations
├── conda/                  # Conda recipes and environment files
├── docs/                   # Documentation source
├── datasets/               # Test datasets for LP, MIP, routing
└── notebooks/              # Example Jupyter notebooks
```

### Supported APIs (at a glance)

| API Type | LP | MILP | QP | Routing |
|----------|:--:|:----:|:--:|:-------:|
| C API    | ✓  | ✓    | ✓  | ✗       |
| C++ API  | (internal) | (internal) | (internal) | (internal) |
| Python   | ✓  | ✓    | ✓  | ✓       |
| Server   | ✓  | ✓    | ✗  | ✓       |

## Canonical project docs (source of truth)

- **Contributing / build / test / debugging**: `CONTRIBUTING.md`
- **CI scripts**: `ci/README.md`
- **Release/version scripts**: `ci/release/README.md`
- **Documentation build**: `docs/cuopt/README.md`

## Safety rules for agents

- **Minimal diffs**: change only what’s necessary; avoid drive-by refactors.
- **No mass reformatting**: don’t run formatters over unrelated code.
- **No API invention**: especially for routing / server schemas—align with `docs/cuopt/source/` + OpenAPI spec.
- **Don’t bypass CI**: don’t suggest skipping checks or using `--no-verify` unless explicitly required and approved.
- **CUDA/GPU hygiene**: keep operations stream-ordered, follow existing RAFT/RMM patterns, avoid raw `new`/`delete`.

### ⚠️ Mandatory: test impact check (ask before finalizing a change)

Before landing any behavioral change or new feature, **explicitly ask**:

- **What scenarios must be covered?** (happy path, edge cases, failure modes, performance regressions)
- **What’s the expected behavior contract?** (inputs/outputs, errors, compatibility constraints)
- **Where should tests live?** (C++ gtests under `cpp/tests/`, Python `pytest` under `python/.../tests`, server tests, etc.)

**Recommendation:** add or update at least one **unit test** that covers the new behavior so **CI prevents regressions**. If full coverage isn’t feasible, document what’s untested and why, and add the smallest meaningful regression test.

### Security bar (commands & installs)

- **Do not run shell commands by default**: Provide commands/instructions; only execute commands if the user explicitly asks you to run them.
- **No dependency installation by default**: Don’t run `pip/conda/apt/brew` installs unless explicitly requested/approved by the user.
- **No privileged/system changes**: Never use `sudo`, modify system config, add package repositories/keys, or change driver/CUDA/toolchain setup unless explicitly requested and the implications are clear.
- **Workspace-only file changes by default**: Only create/modify files inside the checked-out repo/workspace. If writing outside the repo is necessary (e.g., under `$HOME`), ask for explicit permission and explain exactly what will be written where.
- **Prefer safe, reversible changes**: Use local envs; pin versions for reproducibility; avoid “curl | bash”.

## Before you commit (style + signoff)

- **Run the same style checks CI runs**:
  - `./ci/check_style.sh`
  - Or run pre-commit directly: `pre-commit run --all-files --show-diff-on-failure`
  - Details: `CONTRIBUTING.md` (Code Formatting / pre-commit)
- **Signed commits are required (DCO sign-off)**:
  - Use `git commit -s ...` (or `--signoff`)
  - Details: `CONTRIBUTING.md` (Signing Your Work)

## Coding style and conventions (summary)

### C++ naming conventions

- **Base style**: `snake_case` for all names (except test cases: PascalCase)
- **Prefixes/Suffixes**:
  - `d_` → device data variables (e.g., `d_locations_`)
  - `h_` → host data variables (e.g., `h_data_`)
  - `_t` → template type parameters (e.g., `i_t`, `value_t`)
  - `_` → private member variables (e.g., `n_locations_`)

### File extensions

| Extension | Usage |
|-----------|-------|
| `.hpp`    | C++ headers |
| `.cpp`    | C++ source |
| `.cu`     | CUDA C++ source (nvcc required) |
| `.cuh`    | CUDA headers with device code |

### Include order

1. Local headers
2. RAPIDS headers
3. Related libraries
4. Dependencies
5. STL

### Python style

- Follow PEP 8
- Use type hints where applicable
- Tests use `pytest` framework

### Formatting

- **C++**: Enforced by `clang-format` (config: `cpp/.clang-format`)
- **Python**: Enforced via pre-commit hooks
- See `CONTRIBUTING.md` for pre-commit setup

## Error handling patterns

### Runtime assertions

- Use `CUOPT_EXPECTS` for runtime checks
- Use `CUOPT_FAIL` for unreachable code paths

### CUDA error checking

- Wrap CUDA calls (e.g., `RAFT_CUDA_TRY(...)`)

## Memory management guidelines

- **Never use raw `new`/`delete`** - use RMM allocators
- **Prefer `rmm::device_uvector<T>`** for device memory
- **All operations should be stream-ordered** - accept `cuda_stream_view`
- **Views (`*_view` suffix) are non-owning** - don't manage their lifetime

## Repo navigation (practical)

- **C++/CUDA core**: `cpp/` (includes `libmps_parser`, `libcuopt`)
- **Python packages**: `python/` (`cuopt`, `libcuopt`, `cuopt_server`, `cuopt_self_hosted`)
- **Docs (Sphinx)**: `docs/cuopt/source/`
- **Datasets**: `datasets/`

## Build & test (quick reference; defer details to CONTRIBUTING)

- **Build**: `./build.sh` (supports building individual components; see `./build.sh --help`)
- **C++ tests**: `ctest --test-dir cpp/build`
- **Python tests**: `pytest -v python/cuopt/cuopt/tests` (dataset env vars may be required; see `CONTRIBUTING.md`)
- **Docs build**: `./build.sh docs` or `make html` under `docs/cuopt`

## Release discipline

- Do not change versioning/release files unless explicitly requested.
- Prefer changes that are forward-merge friendly with RAPIDS branching conventions (see `CONTRIBUTING.md`).

## Key files reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `cpp/.clang-format` |
| Conda environments | `conda/environments/` |
| Test data download | `datasets/get_test_data.sh` |
| CI configuration | `ci/` |
| Version info | `VERSION` |

## Common pitfalls

| Problem | Solution |
|---------|----------|
| Cython changes not reflected | Rerun: `./build.sh cuopt` |
| Missing `nvcc` | Set `$CUDACXX` or add CUDA to `$PATH` |
| CUDA out of memory | Reduce problem size or use streaming |
| Slow debug library loading | Device symbols cause delay; use selectively |
