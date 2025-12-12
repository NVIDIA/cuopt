# AGENTS.md - AI Coding Agent Guidelines for cuOpt

> This file provides essential context for AI coding assistants (Codex, Cursor, GitHub Copilot, etc.) working with the NVIDIA cuOpt codebase.

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
| Server   | ✓  | ✓    | ✓  | ✓       |

---

## Setup and Installation

### System Requirements

- **CUDA**: 12.5+ or 13.0+
- **GPU**: Volta architecture or better (Compute Capability ≥7.0)
- **Driver**: ≥525.60.13 (Linux), ≥527.41 (Windows)
- **Python**: 3.10 - 3.13
- **OS**: Linux (x86_64, aarch64), Windows via WSL2

### Development Environment Setup

```bash
# Clone repository
CUOPT_HOME=$(pwd)/cuopt
git clone https://github.com/NVIDIA/cuopt.git $CUOPT_HOME
cd $CUOPT_HOME

# Create conda environment (recommended)
conda env create --name cuopt_dev --file conda/environments/all_cuda-130_arch-$(uname -m).yaml
conda activate cuopt_dev

# Build all components
./build.sh

# Build specific components
./build.sh libmps_parser libcuopt  # C++ libraries only
./build.sh libcuopt -g              # Debug build
```

### Quick Install (Users)

```bash
# Pip (CUDA 12.x)
pip install --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu12==25.12.* cuopt-sh-client==25.12.*

# Conda
conda install -c rapidsai -c conda-forge -c nvidia \
  cuopt-server=25.12.* cuopt-sh-client=25.12.*
```

---

## Testing Workflows

### Running Tests

```bash
# Download test datasets first
cd $CUOPT_HOME/datasets && ./get_test_data.sh
cd $CUOPT_HOME && datasets/linear_programming/download_pdlp_test_dataset.sh
datasets/mip/download_miplib_test_dataset.sh
export RAPIDS_DATASET_ROOT_DIR=$CUOPT_HOME/datasets/

# C++ tests (gtest)
ctest --test-dir ${CUOPT_HOME}/cpp/build

# Python tests (pytest)
pytest -v ${CUOPT_HOME}/python/cuopt/cuopt/tests
```

### CI Scripts

Located in `ci/` directory:
- `build_cpp.sh` - Build C++ components
- `build_python.sh` - Build Python packages
- `test_cpp.sh` - Run C++ tests
- `test_python.sh` - Run Python tests
- `check_style.sh` - Code style verification

---

## Coding Style and Conventions

### C++ Style

- **Naming**: `snake_case` for all names (except test cases which use PascalCase)
- **Prefixes**:
  - `d_` for device data variables
  - `h_` for host data variables
  - `_t` suffix for template type parameters
  - `_` suffix for private member variables
- **Formatting**: Enforced by `clang-format` (config: `cpp/.clang-format`)
- **Include order**: Local → RAPIDS → Related libs → Dependencies → STL

```cpp
// Example naming conventions
template <typename i_t>
class locations_t {
 private:
  i_t n_locations_{};
  i_t* d_locations_{};  // device pointer
};
```

### File Extensions

| Extension | Usage |
|-----------|-------|
| `.hpp`    | C++ headers |
| `.cpp`    | C++ source |
| `.cu`     | CUDA C++ source (nvcc required) |
| `.cuh`    | CUDA headers with device code |

### Python Style

- Follow PEP 8 guidelines
- Use type hints where applicable
- Tests use `pytest` framework

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Run all checks
pre-commit run --all-files --show-diff-on-failure

# Auto-run on commits
pre-commit install
```

---

## Pull Request Guidelines

### Workflow

1. Fork the repository and create a descriptive branch (`fix-documentation`, `add-feature-x`)
2. Implement changes with appropriate unit tests
3. Run pre-commit hooks: `pre-commit run --all-files`
4. Sign off commits: `git commit -s -m "Your message"`
5. Open PR (draft for CI testing without review)
6. Ensure all CI checks pass
7. Address reviewer feedback

### Commit Signing (Required)

All commits must be signed off to certify origin:

```bash
git commit -s -m "Add cool feature"
# Results in: Signed-off-by: Your Name <your@email.com>
```

### Code Review Focus Areas

- Correctness and performance
- Memory management (use RMM, avoid raw pointers)
- CUDA error handling (`RAFT_CUDA_TRY` macro)
- Test coverage for new functionality
- Documentation for public APIs

---

## Common Pitfalls and Troubleshooting

### Build Issues

| Problem | Solution |
|---------|----------|
| Cython changes not reflected | Rerun Python build: `./build.sh cuopt` |
| Missing `nvcc` | Ensure CUDA toolkit in PATH or set `$CUDACXX` |
| Conda environment issues | Update env: `conda env update --file conda/environments/...` |

### Runtime Issues

| Problem | Solution |
|---------|----------|
| CUDA out of memory | Reduce problem size or use streaming |
| Slow library loading (debug) | Device debug symbols cause delay; use selectively |
| Import errors | Verify `$CONDA_PREFIX` matches install location |

### Debugging

```bash
# Debug build
./build.sh libcuopt -g

# CUDA debugging
cuda-gdb -ex r --args python script.py

# Memory checking
compute-sanitizer --tool memcheck python script.py
```

### Adding Device Debug Symbols (Selectively)

```cmake
# In cpp/CMakeLists.txt - only for specific files
set_source_files_properties(src/routing/data_model_view.cu
  PROPERTIES COMPILE_OPTIONS "-G")
```

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

## Quick Command Reference

```bash
# Build everything
./build.sh

# Build help
./build.sh --help

# Run style checks
pre-commit run --all-files

# Run C++ tests
ctest --test-dir cpp/build

# Run Python tests
pytest -v python/cuopt/cuopt/tests

# Debug build
./build.sh libcuopt -g
```

---

*Last updated: December 2024 | cuOpt v25.12*
