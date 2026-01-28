# Memory Model Summary

This document describes how memory is handled for **local** vs **remote** solves.

 ## Core idea

 The solver now supports **two memory modes** for problem input and solution output:

- **Device (GPU) memory**: used for local solves.
- **Host (CPU) memory**: used for remote solves.

 A non-owning `data_model_view_t` (host or device) is the entry point that drives the path:
 - Device view → local GPU solve
 - Host view + `CUOPT_REMOTE_*` → remote solve path

## Local solve (GPU memory)

**C++ entry points** (`solve_lp`, `solve_mip` in `cpp/src/linear_programming/solve.cu` and
`cpp/src/mip/solve.cu`) behave as follows:

1. If the view is device memory, the view is converted into an `optimization_problem_t` and
   solved locally.
2. If the view is **host memory**, the data is copied **CPU → GPU** and solved locally.
   This path requires a valid `raft::handle_t`.
3. Solutions are returned in **device memory**, and wrappers expose device buffers.

**Python / Cython (`python/cuopt/.../solver_wrapper.pyx`)**:
- `DataModel` is **host-only**: the Cython wrapper accepts `np.ndarray` inputs and raises
  if GPU-backed objects are provided.
- For local solves, host data is **copied to GPU** when building the
  `optimization_problem_t` (requires a valid `raft::handle_t`).
- The solution is wrapped into `rmm::device_buffer` and converted to NumPy arrays via
  `series_from_buf`.

 **CLI (`cpp/cuopt_cli.cpp`)**:
 - Initializes CUDA/RMM for local solve paths.
 - Uses `raft::handle_t` and GPU memory as usual.

## Remote solve (CPU memory)

Remote solve is enabled when **both** `CUOPT_REMOTE_HOST` and `CUOPT_REMOTE_PORT` are set.
This is detected early in the solve path.

 **C++ entry points**:
 - `solve_lp` / `solve_mip` check `get_remote_solve_config()` first.
 - If input data is on **GPU** and remote is enabled, it is copied to CPU for serialization.
 - If input data is already on **CPU**, it is passed directly to `solve_*_remote`.
 - Remote solve returns **host vectors** and sets `is_device_memory = false`.

**Remote stub implementation** (`cpp/include/cuopt/linear_programming/utilities/remote_solve.hpp`):
- Returns **dummy host solutions** (all zeros).
- Sets termination stats to **finite values** (no NaNs) for predictable output.

**Python / Cython (`python/cuopt/.../solver_wrapper.pyx`)**:
- **Input handling**: builds a `data_model_view_t` from the Python `DataModel` before calling C++.
- **Solution handling**: for remote solves, the solution is **host memory**, so NumPy arrays
  are built directly from host vectors and **avoid `rmm::device_buffer`** (no CUDA).

 **CLI (`cpp/cuopt_cli.cpp`)**:
 - Detects remote solve **before** any CUDA initialization.
 - Skips `raft::handle_t` creation and GPU setup when remote is enabled.
- Builds the problem in **host memory** for remote solves.

 ## Batch solve

 Batch solve uses the same memory model:

 - **Local batch**: GPU memory, with CUDA resources and PDLP/dual simplex paths.
- **Remote batch**: each problem is routed through `solve_lp_remote` or `solve_mip_remote`
  and returns host data. If inputs are already on GPU, they are copied to host first.

## Expected outputs for remote stubs

 - Termination status: `Optimal`
 - Objective values: `0.0`
 - Primal/dual/reduced-cost vectors: zero-filled host arrays

This is useful for verifying the **CPU-only data path** without a remote service.
