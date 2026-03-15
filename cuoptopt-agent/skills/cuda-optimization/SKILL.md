---
name: cuda-optimization
version: "26.04.00"
description: CUDA kernel optimization — memory coalescing, shared memory, warp divergence, async copy, cooperative groups, occupancy, tensor cores.
---

# CUDA Optimization

Reference for writing and tuning high-performance CUDA kernels in the cuOpt codebase.

---

## 1. Memory Access Patterns

### Coalesced Global Memory Access
- **Rule:** Threads in a warp should access consecutive 128-byte aligned addresses.
- Misaligned or strided access serializes into multiple transactions → up to 32× bandwidth waste.
- Use `__ldg()` for read-only global data; enables L1 caching on all architectures ≥ Kepler.
- Structure-of-Arrays (SoA) > Array-of-Structures (AoS) for warp-level coalescing.

### Shared Memory
- 32/64 KB per SM (architecture-dependent); split into 32 banks of 4 bytes each.
- **Bank conflict:** Two threads in a warp access different addresses in the same bank → serialized.
- Avoid by padding arrays: `__shared__ float sdata[N + 1]` for power-of-two strides.
- Use `__syncthreads()` before any cross-thread shared memory read after a write.

### L1 / L2 Cache Tuning
- Configure L1 size with `cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, ...)`.
- On Ampere+: 128 KB unified L1/shared per SM; carve-out controls the split.
- Use `--ptxas-options=-v` to see register and shared memory usage per kernel.

---

## 2. Warp-Level Execution

### Warp Divergence
- All 32 threads in a warp execute the same instruction; divergent branches serialize.
- Minimize `if` statements whose condition varies within a warp.
- Use warp shuffle intrinsics (`__shfl_sync`, `__ballot_sync`, `__reduce_add_sync`) to avoid shared memory for intra-warp communication.

### Warp Occupancy
- Occupancy = active warps / max warps per SM.
- Low occupancy can hide instruction latency — but high register pressure reduces occupancy.
- Target ≥ 50% occupancy; check with `nvcc --ptxas-options=-v` or Nsight Compute.
- Use `__launch_bounds__(MAX_THREADS, MIN_BLOCKS)` to guide register allocation.

---

## 3. Async Data Movement

### `cp.async` (Ampere+)
```cuda
// Async copy from global to shared, 16 bytes at a time
__pipeline_memcpy_async(dst, src, 16);
__pipeline_commit();
// ... compute with previously loaded data ...
__pipeline_wait_prior(0);
```
- Hides global memory latency by overlapping compute and data movement.
- Requires `#include <cooperative_groups/memcpy_async.h>` (CUDA 11.1+).

### `cudaMemcpyAsync` + Streams
- Overlap H↔D transfers with kernel execution using multiple streams.
- Pin host memory with `cudaMallocHost` for maximum transfer bandwidth.

---

## 4. Cooperative Groups

```cuda
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void kernel() {
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // warp-level reduce
    float val = warp.shfl_down(x, 16);
}
```
- Preferred over legacy `__syncthreads()` for partial-block synchronization.
- `cg::grid_group` enables cross-block synchronization (requires cooperative launch).

---

## 5. Tensor Cores (Ampere / Hopper)

### WMMA API (all architectures with Tensor Cores)
```cuda
#include <mma.h>
using namespace nvcuda::wmma;
fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
fragment<accumulator, 16, 16, 16, float> c_frag;
fill_fragment(c_frag, 0.0f);
load_matrix_sync(a_frag, a_ptr, lda);
load_matrix_sync(b_frag, b_ptr, ldb);
mma_sync(c_frag, a_frag, b_frag, c_frag);
store_matrix_sync(c_ptr, c_frag, ldc, mem_row_major);
```
- Tile sizes: 16×16×16 (FP16), 8×32×16, 32×8×16 (FP16 Turing+), 16×16×8 (TF32 Ampere).
- Use `cublasSgemmEx` or cuBLAS for production GEMM rather than hand-rolled WMMA.

---

## 6. Kernel Launch Configuration

| Heuristic | Value |
|-----------|-------|
| Block size | 128 or 256 threads (multiple of warp size 32) |
| Grid size | `(N + block - 1) / block` |
| Registers | ≤ 64/thread for ≥ 50% occupancy (cc 8.x) |
| Shared mem | Leave headroom; max varies by carve-out |

- Prefer 2D/3D blocks for 2D/3D data to simplify index arithmetic.
- Avoid tail serialization: pad data to grid-even sizes when possible.

---

## 7. Common Pitfalls

| Pitfall | Fix |
|---------|-----|
| Uncoalesced access | Transpose data, or use shared memory staging |
| Excessive atomics | Reduce in shared memory first, single global atomic at end |
| Too many registers | Use `--maxrregcount=64` pragma or split kernel |
| Kernel launch overhead | Persistent kernels, CUDA Graphs |
| CPU-GPU sync in loop | Use streams + events; profile with Nsight Systems |

---

## 8. cuOpt-Specific Notes

- Core LP/MILP solvers live in `cpp/src/`; CUDA kernels are `.cu` files.
- Build with `cmake --build build --target cuopt` after changes.
- Profile with `nsys profile --trace=cuda python -m cuopt solve --mps <file>`.
- Always benchmark against `datasets/linear_programming/` and `datasets/mip/` after any kernel change.
