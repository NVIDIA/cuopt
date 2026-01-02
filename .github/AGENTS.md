# AGENTS.md - AI Coding Agent Guidelines for cuOpt

> This file provides essential context for AI coding assistants (Codex, Cursor, GitHub Copilot, etc.) working with the NVIDIA cuOpt codebase.

> **For setup, building, testing, and contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).**

---

## Project Overview

**cuOpt** is NVIDIA's GPU-accelerated optimization engine for:
- **Mixed Integer Linear Programming (MILP)**
- **Linear Programming (LP)**
- **Quadratic Programming (QP)**
- **Vehicle Routing Problems (VRP)** including TSP and PDP

### Architecture

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

### Supported APIs

| API Type | LP | MILP | QP | Routing |
|----------|:--:|:----:|:--:|:-------:|
| C API    | ✓  | ✓    | ✓  | ✗       |
| C++ API  | ✓  | ✓    | ✓  | ✓       |
| Python   | ✓  | ✓    | ✓  | ✓       |
| Server   | ✓  | ✓    | ✗  | ✓       |

---

## Coding Style and Conventions

### C++ Naming Conventions

- **Base style**: `snake_case` for all names (except test cases: PascalCase)
- **Prefixes/Suffixes**:
  - `d_` → device data variables (e.g., `d_locations_`)
  - `h_` → host data variables (e.g., `h_data_`)
  - `_t` → template type parameters (e.g., `i_t`, `value_t`)
  - `_` → private member variables (e.g., `n_locations_`)

```cpp
// Example naming pattern
template <typename i_t>
class locations_t {
 private:
  i_t n_locations_{};
  i_t* d_locations_{};  // device pointer
  i_t* h_locations_{};  // host pointer
};
```

### File Extensions

| Extension | Usage |
|-----------|-------|
| `.hpp`    | C++ headers |
| `.cpp`    | C++ source |
| `.cu`     | CUDA C++ source (nvcc required) |
| `.cuh`    | CUDA headers with device code |

### Include Order

1. Local headers
2. RAPIDS headers
3. Related libraries
4. Dependencies
5. STL

### Python Style

- Follow PEP 8
- Use type hints where applicable
- Tests use `pytest` framework

### Formatting

- **C++**: Enforced by `clang-format` (config: `cpp/.clang-format`)
- **Python**: Enforced via pre-commit hooks
- See [CONTRIBUTING.md](../CONTRIBUTING.md) for pre-commit setup

---

## Error Handling Patterns

### Runtime Assertions

```cpp
// Use CUOPT_EXPECTS for runtime checks
CUOPT_EXPECTS(lhs.type() == rhs.type(), "Column type mismatch");

// Use CUOPT_FAIL for unreachable code paths
CUOPT_FAIL("This code path should not be reached.");
```

### CUDA Error Checking

```cpp
// Always wrap CUDA calls
RAFT_CUDA_TRY(cudaMemcpy(&dst, &src, num_bytes));
```

---

## Memory Management Guidelines

- **Never use raw `new`/`delete`** - Use RMM allocators
- **Prefer `rmm::device_uvector<T>`** for device memory
- **All operations should be stream-ordered** - Accept `cuda_stream_view`
- **Views (`*_view` suffix) are non-owning** - Don't manage their lifetime

---

## Key Files Reference

| Purpose | Location |
|---------|----------|
| Main build script | `build.sh` |
| Dependencies | `dependencies.yaml` |
| C++ formatting | `cpp/.clang-format` |
| Conda environments | `conda/environments/` |
| Test data download | `datasets/get_test_data.sh` |
| CI configuration | `ci/` |
| Version info | `VERSION` |

---

## Common Pitfalls

| Problem | Solution |
|---------|----------|
| Cython changes not reflected | Rerun: `./build.sh cuopt` |
| Missing `nvcc` | Set `$CUDACXX` or add CUDA to `$PATH` |
| CUDA out of memory | Reduce problem size or use streaming |
| Slow debug library loading | Device symbols cause delay; use selectively |

---

*For detailed setup, build instructions, testing workflows, debugging, and contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).*
