# cuOpt Solution Memory Architecture

This document describes how cuOpt manages solution data memory for both local GPU-based solving and remote CPU-only solving.

## Overview

cuOpt solutions can exist in either GPU memory (for local high-performance workflows) or CPU memory (for remote solve and CPU-only clients). The architecture supports both use cases efficiently.

## Solution Classes

### LP Solution: `optimization_problem_solution_t<i_t, f_t>`

Located in: `cpp/include/cuopt/linear_programming/pdlp/solver_solution.hpp`

**Key Data Members:**
```cpp
// GPU memory (primary storage for local solve)
rmm::device_uvector<f_t> primal_solution_;
rmm::device_uvector<f_t> dual_solution_;
rmm::device_uvector<f_t> reduced_cost_;

// CPU memory (used for remote solve or explicit host access)
std::vector<f_t> primal_solution_host_;
std::vector<f_t> dual_solution_host_;
std::vector<f_t> reduced_cost_host_;

// Scalars (always on host)
f_t objective_value_;
f_t dual_objective_value_;
f_t l2_primal_residual_;
f_t l2_dual_residual_;
f_t gap_;
i_t nb_iterations_;
f_t solve_time_;
pdlp_termination_status_t termination_status_;
error_type_t error_status_;
```

### MIP Solution: `mip_solution_t<i_t, f_t>`

Located in: `cpp/include/cuopt/linear_programming/mip/solver_solution.hpp`

**Key Data Members:**
```cpp
// GPU memory (primary storage for local solve)
rmm::device_uvector<f_t> solution_;
std::vector<rmm::device_uvector<f_t>> solution_pool_;

// CPU memory (used for remote solve)
std::vector<f_t> solution_host_;
std::vector<std::vector<f_t>> solution_pool_host_;

// Scalars (always on host)
f_t objective_;
f_t mip_gap_;
f_t max_constraint_violation_;
f_t max_int_violation_;
f_t max_variable_bound_violation_;
mip_termination_status_t termination_status_;
error_type_t error_status_;
```

## Memory Management Strategy

### Local Solve (GPU)

When solving locally on a GPU:

1. **Solver computes** → Results in GPU memory (`device_uvector`)
2. **Solution returned** → Contains GPU buffers
3. **User accesses** → Can work directly with GPU data or copy to host as needed

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Solver    │ ──► │  Solution   │ ──► │    User     │
│   (GPU)     │     │ (GPU mem)   │     │ (GPU/CPU)   │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Remote Solve (CPU-only client)

When solving remotely from a CPU-only machine:

1. **Client sends** → Problem data serialized and sent to server
2. **Server solves** → Results computed on GPU
3. **`to_host()` called** → GPU data copied to CPU memory
4. **Solution serialized** → CPU data sent back to client
5. **Client receives** → Solution with CPU memory only

```
┌──────────┐     ┌──────────────────────────────────────────┐     ┌──────────┐
│  Client  │ ──► │              SERVER                      │ ◄── │  Client  │
│ (no GPU) │     │ GPU solve → to_host() → serialize        │     │(solution)│
└──────────┘     └──────────────────────────────────────────┘     └──────────┘
```

## The `to_host()` Method

Both solution classes provide a `to_host()` method that copies GPU data to CPU:

```cpp
// LP Solution
void optimization_problem_solution_t<i_t, f_t>::to_host(rmm::cuda_stream_view stream_view)
{
  if (primal_solution_.size() > 0) {
    primal_solution_host_.resize(primal_solution_.size());
    raft::copy(primal_solution_host_.data(), primal_solution_.data(),
               primal_solution_.size(), stream_view);
  }
  // ... similar for dual_solution_, reduced_cost_
  stream_view.synchronize();
}

// MIP Solution
void mip_solution_t<i_t, f_t>::to_host(rmm::cuda_stream_view stream_view)
{
  if (solution_.size() > 0) {
    solution_host_.resize(solution_.size());
    raft::copy(solution_host_.data(), solution_.data(),
               solution_.size(), stream_view);
  }
  // ... similar for solution_pool_
  stream_view.synchronize();
}
```

### When to Call `to_host()`

- **Server-side remote solve**: Called before serializing solution for network transmission
- **Client accessing host data**: If user needs `std::vector` access to solution data
- **Writing to files**: When saving solutions to disk

### Performance Considerations

The `to_host()` copy adds overhead, but:
- Only called when CPU access is actually needed
- GPU computation dominates solve time for non-trivial problems
- One-time cost after solve completes

**Typical overhead**: Negligible for problems with thousands of variables. For a 10,000-variable problem, copying ~80KB takes <1ms.

## Accessor Methods

### GPU Accessors (for local solve)

```cpp
// LP
const rmm::device_uvector<f_t>& get_primal_solution() const;
const rmm::device_uvector<f_t>& get_dual_solution() const;
const rmm::device_uvector<f_t>& get_reduced_cost() const;

// MIP
const rmm::device_uvector<f_t>& get_solution() const;
```

### CPU Accessors (for remote solve)

```cpp
// LP
const std::vector<f_t>& get_primal_solution_host() const;
const std::vector<f_t>& get_dual_solution_host() const;
const std::vector<f_t>& get_reduced_cost_host() const;

// MIP
const std::vector<f_t>& get_solution_host() const;
```

### Checking Memory Location

```cpp
// Returns true if solution data is on GPU
bool is_device_memory() const;
```

## Usage in Remote Solve Server

The server calls `to_host()` before serialization:

```cpp
// In cuopt_remote_server.cpp
if (is_mip) {
  mip_solution_t<i_t, f_t> solution = solve_mip(...);
  solution.to_host(stream);  // Copy GPU → CPU
  result_data = serializer->serialize_mip_solution(solution);
} else {
  optimization_problem_solution_t<i_t, f_t> solution = solve_lp(...);
  solution.to_host(stream);  // Copy GPU → CPU
  result_data = serializer->serialize_lp_solution(solution);
}
```

## Design Rationale

### Why Not Pure CPU Memory?

An earlier design considered using only `std::vector` for solutions. We chose the hybrid approach because:

1. **GPU performance**: Local solves benefit from keeping data on GPU
2. **Minimize changes**: Existing GPU-based code continues to work unchanged
3. **Flexibility**: Users can choose GPU or CPU access as needed

### Why Not Pure GPU Memory?

Pure GPU memory would fail for:

1. **Remote solve**: CPU-only clients need CPU data
2. **Serialization**: Network transmission requires CPU memory
3. **File I/O**: Writing to disk typically uses CPU

### Hybrid Approach Benefits

- ✅ Local GPU workflows remain efficient
- ✅ Remote solve works with CPU-only clients
- ✅ Minimal code changes to existing solvers
- ✅ On-demand copy (only when needed)
- ✅ Clear separation of concerns

## Files Involved

| File | Description |
|------|-------------|
| `cpp/include/cuopt/linear_programming/pdlp/solver_solution.hpp` | LP solution class declaration |
| `cpp/src/linear_programming/solver_solution.cu` | LP solution implementation + `to_host()` |
| `cpp/include/cuopt/linear_programming/mip/solver_solution.hpp` | MIP solution class declaration |
| `cpp/src/mip/solver_solution.cu` | MIP solution implementation + `to_host()` |
| `cpp/cuopt_remote_server.cpp` | Server calls `to_host()` before serialization |
| `cpp/src/linear_programming/utilities/protobuf_serializer.cu` | Uses host accessors for serialization |

## Summary

The cuOpt solution memory architecture uses a **hybrid GPU/CPU approach**:

1. **Primary storage**: GPU (`device_uvector`) for local solve performance
2. **Secondary storage**: CPU (`std::vector`) for remote solve and host access
3. **On-demand copying**: `to_host()` method copies GPU → CPU when needed
4. **Transparent to users**: Local users get GPU data, remote users get CPU data automatically
