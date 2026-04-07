---
name: cuopt-developer
version: "26.06.00"
description: Contribute to NVIDIA cuOpt codebase including C++/CUDA, Python, server, docs, and CI. Use when the user wants to modify solver internals, add features, submit PRs, or understand the codebase architecture.
---

# cuOpt Developer Skill

Contribute to the NVIDIA cuOpt codebase. This skill is for modifying cuOpt itself, not for using it.

**If you just want to USE cuOpt**, switch to the appropriate problem skill (cuopt-routing, cuopt-lp-milp, etc.)

---

## Developer Behavior Rules

These rules are specific to development tasks. They differ from user rules.

### 1. Ask Before Assuming

Clarify before implementing:
- What component? (C++/CUDA, Python, server, docs, CI)
- What's the goal? (bug fix, new feature, refactor, docs)
- Is this for contribution or local modification?

### 2. Verify Understanding

Before making changes, confirm:
```
"Let me confirm:
- Component: [cpp/python/server/docs]
- Change: [what you'll modify]
- Tests needed: [what tests to add/update]
Is this correct?"
```

### 3. Follow Codebase Patterns

- Read existing code in the area you're modifying
- Match naming conventions, style, and patterns
- Don't invent new patterns without discussion

### 4. Ask Before Running — Modified for Dev

**OK to run without asking** (expected for dev work):
- `./build.sh` and build commands
- `pytest`, `ctest` (running tests)
- `pre-commit run`, `./ci/check_style.sh` (formatting)
- `git status`, `git diff`, `git log` (read-only git)

**Still ask before**:
- `git commit`, `git push` (write operations)
- Package installs (`pip`, `conda`, `apt`)
- Any destructive or irreversible commands

### 5. No Privileged Operations

Same as user rules — never without explicit request:
- No `sudo`
- No system file changes
- No writes outside workspace

---

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **What are you trying to change?**
   - Solver algorithm/performance?
   - Python API?
   - Server endpoints?
   - Documentation?
   - CI/build system?

2. **Do you have the development environment set up?**
   - Built the project successfully?
   - Ran tests?

3. **Is this for contribution or local modification?**
   - If contributing: will need to follow DCO signoff

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

### Build Everything

```bash
./build.sh
```

### Build Specific Components

```bash
./build.sh libcuopt    # C++ library
./build.sh cuopt       # Python package
./build.sh cuopt_server # Server
./build.sh docs        # Documentation
```

### Run Tests

```bash
# C++ tests
ctest --test-dir cpp/build

# Python tests
pytest -v python/cuopt/cuopt/tests

# Server tests
pytest -v python/cuopt_server/tests
```

## Before You Commit

### Run Style Checks

```bash
./ci/check_style.sh
# or
pre-commit run --all-files --show-diff-on-failure
```

### Sign Your Commits (DCO Required)

```bash
git commit -s -m "Your message"
```

## Coding Conventions

### C++ Naming

| Element | Convention | Example |
|---------|------------|---------|
| Variables | `snake_case` | `num_locations` |
| Functions | `snake_case` | `solve_problem()` |
| Classes | `snake_case` | `data_model` |
| Test cases | `PascalCase` | `SolverTest` |
| Device data | `d_` prefix | `d_locations_` |
| Host data | `h_` prefix | `h_data_` |
| Template params | `_t` suffix | `value_t` |
| Private members | `_` suffix | `n_locations_` |

### File Extensions

| Extension | Usage |
|-----------|-------|
| `.hpp` | C++ headers |
| `.cpp` | C++ source |
| `.cu` | CUDA source (nvcc required) |
| `.cuh` | CUDA headers with device code |

### Include Order

1. Local headers
2. RAPIDS headers
3. Related libraries
4. Dependencies
5. STL

### Python Style

- Follow PEP 8
- Use type hints
- Tests use pytest

## Error Handling

### Runtime Assertions

```cpp
CUOPT_EXPECTS(condition, "Error message");
CUOPT_FAIL("Unreachable code reached");
```

### CUDA Error Checking

```cpp
RAFT_CUDA_TRY(cudaMemcpy(...));
```

## Memory Management

```cpp
// ❌ WRONG
int* data = new int[100];

// ✅ CORRECT - use RMM
rmm::device_uvector<int> data(100, stream);
```

- All operations should accept `cuda_stream_view`
- Views (`*_view` suffix) are non-owning

## Test Impact Check

**Before any behavioral change, ask:**

1. What scenarios must be covered?
2. What's the expected behavior contract?
3. Where should tests live?
   - C++ gtests: `cpp/tests/`
   - Python pytest: `python/.../tests/`

**Add at least one regression test for new behavior.**

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `.clang-format` |
| Conda environments | `conda/environments/` |
| Test data | `datasets/` |
| CI scripts | `ci/` |

## Common Tasks

### Adding a Solver Parameter

1. Add to settings struct in `cpp/include/cuopt/`
2. Expose in Python bindings `python/cuopt/`
3. Add to server schema if applicable
4. Add tests
5. Update documentation

### Adding a Server Endpoint

1. Add route in `python/cuopt_server/cuopt_server/webserver.py`
2. Update OpenAPI spec `docs/cuopt/source/cuopt_spec.yaml`
3. Add tests in `python/cuopt_server/tests/`
4. Update documentation

### Modifying CUDA Kernels

1. Edit kernel in `cpp/src/`
2. Follow stream-ordering patterns
3. Run C++ tests: `ctest --test-dir cpp/build`
4. Run benchmarks to check performance

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Cython changes not reflected | Rerun: `./build.sh cuopt` |
| Missing `nvcc` | Set `$CUDACXX` or add CUDA to `$PATH` |
| CUDA out of memory | Reduce problem size |
| Slow debug library loading | Device symbols cause delay |

<!-- skill-evolution:start — SC reformulation implementation gotchas -->
### Semi-Continuous Reformulation Implementation Gotchas

#### 1. `bound_presolve_t` crashes with 0 constraints

`bound_presolve_t::solve()` internally calls `update_bounds_kernel`, which uses a CUB block reduce over constraint rows. When the problem has **0 constraints**, CUDA raises `cudaErrorInvalidConfiguration`.

**Guard bounds propagation before constructing `bound_presolve_t`:**

```cpp
if (op_relaxed.get_n_constraints() > 0) {
  problem_t<i_t, f_t> temp_pb(op_relaxed, settings.get_tolerances());
  mip_solver_context_t<i_t, f_t> ctx(handle_ptr, &temp_pb, settings);
  bound_presolve_t<i_t, f_t> bps(ctx, bp_settings);
  bps.resize(temp_pb);
  bps.solve(temp_pb);
  // copy tightened bounds...
}
// else: tight_ub stays at original variable upper bounds (safe fallback)
```

#### 2. `problem_t` does not understand `var_t::SEMI_CONTINUOUS`

`problem_t` is constructed from `optimization_problem_t` inside the MIP solver. It has no concept of `var_t::SEMI_CONTINUOUS` — passing a problem that still contains SC vars causes undefined behavior or assertion failures.

**When running GPU bounds propagation as part of SC reformulation, construct a relaxed copy** where SC vars are changed to `var_t::CONTINUOUS` with `lb = 0` before building `problem_t`. The real `problem_t` is created from the fully-reformulated `op_problem` (no SC vars remain) later in `solve_mip()`.

```cpp
optimization_problem_t<i_t, f_t> op_relaxed(op_problem);
// change SC vars → CONTINUOUS with lb=0 in op_relaxed before passing to problem_t
```

#### 3. Auxiliary binary variables leak into the returned solution

`reformulate_semi_continuous()` adds `n_binary` binary variables to `op_problem`. After `run_mip()` returns, the solution vector has `n_orig + n_binary` entries. The caller only knows about `n_orig` variables.

**Strip the auxiliary binaries before returning the solution:**

```cpp
const i_t n_orig_before_sc = op_problem.get_n_variables();
const bool had_sc          = detail::reformulate_semi_continuous(op_problem, settings);
// ... solve ...
if (had_sc && sol.get_solution().size() > static_cast<size_t>(n_orig_before_sc)) {
  sol.get_solution().resize(n_orig_before_sc, op_problem.get_handle_ptr()->get_stream());
}
```

Do this **after** any `.sol` file write (which can include auxiliary vars for debugging) but **before** `return sol`.
<!-- skill-evolution:end -->

## Canonical Documentation

- **Contributing/build/test**: `CONTRIBUTING.md`
- **CI scripts**: `ci/README.md`
- **Release scripts**: `ci/release/README.md`
- **Docs build**: `docs/cuopt/README.md`

## Security Rules

- **No shell commands by default** - provide instructions, only run if asked
- **No package installs by default** - ask before pip/conda/apt
- **No privileged changes** - never use sudo without explicit request
- **Workspace-only file changes** - ask for permission for writes outside repo
