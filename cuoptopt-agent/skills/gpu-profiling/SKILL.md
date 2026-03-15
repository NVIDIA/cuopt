---
name: gpu-profiling
version: "26.04.00"
description: GPU profiling for cuOpt — Nsight Systems (nsys), Nsight Compute (ncu), key metrics, roofline model, workflow for identifying bottlenecks.
---

# GPU Profiling for cuOpt

Use profiling to identify bottlenecks before writing any optimization code.
The rule: **measure first, optimize second**.

---

## Tool Overview

| Tool | Purpose | Overhead |
|------|---------|---------|
| `nsys` (Nsight Systems) | System-level timeline: kernel launch, CPU/GPU gaps, PCIe transfers | Low (~5%) |
| `ncu` (Nsight Compute) | Per-kernel micro-architectural metrics | High (5–100×) |
| `nvtx` | Annotate code regions for nsys timeline | Minimal |
| `cuda-memcheck` | Memory error detection | Very high |

---

## Nsight Systems (`nsys`) — Timeline Profiling

### Basic Usage
```bash
# Profile a cuOpt solve
nsys profile --trace=cuda,nvtx,osrt \
  --output /tmp/cuopt_profile \
  python -m cuopt solve --mps datasets/linear_programming/afiro.mps

# Open in GUI
nsys-ui /tmp/cuopt_profile.nsys-rep
```

### Useful Options
```bash
--trace=cuda           # CUDA API calls and kernel launches
--trace=nvtx           # Custom NVTX annotations from cuOpt
--trace=osrt           # OS runtime (threads, synchronization)
--stats=true           # Print summary table to stdout
--force-overwrite=true # Overwrite existing output file
--delay=5              # Skip first 5 seconds (skip init)
--duration=30          # Profile for 30 seconds only
```

### What to Look For
- **CPU-GPU gaps**: long periods of SM idle → suspect CPU bottleneck or sync
- **Kernel occupancy**: short, frequent kernels → consider kernel fusion
- **PCIe transfers**: large H↔D copies not overlapped with compute
- **CUDA API overhead**: excessive `cudaLaunchKernel` / `cudaMalloc` calls

---

## Nsight Compute (`ncu`) — Kernel Profiling

### Basic Usage
```bash
# Profile all CUDA kernels (very slow for large problems)
ncu --output /tmp/kernel_report \
  python -m cuopt solve --mps datasets/linear_programming/afiro.mps

# Profile only specific kernels by name pattern
ncu --kernel-name-base "spgemv" \
  --output /tmp/spgemv_report \
  python -m cuopt solve --mps datasets/linear_programming/afiro.mps
```

### Key Metric Sets
```bash
# Memory throughput and efficiency
ncu --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_hit_rate.pct,lts__t_hit_rate.pct \
python ...

# Compute throughput
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum \
python ...

# Occupancy
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__occupancy_theoretical \
python ...

# Warp stalls (identify latency sources)
ncu --metrics smsp__warp_issue_stalled_long_sb_per_warp_active.pct,\
smsp__warp_issue_stalled_short_sb_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct \
python ...
```

### Interpreting Stall Reasons
| Stall Type | Cause | Fix |
|-----------|-------|-----|
| `long_sb` | Long latency (global mem) | Increase ILP, prefetch |
| `short_sb` | Dependency on recent instr | Interleave independent ops |
| `wait` | Synchronization barrier | Reduce `__syncthreads` calls |
| `not_selected` | Warp not scheduled | Increase occupancy |
| `mio_throttle` | Memory instruction queue full | Reduce memory ops frequency |

---

## Roofline Model

The roofline model identifies whether a kernel is **memory-bound** or **compute-bound**.

### Arithmetic Intensity (AI)
```
AI = FLOPs / Bytes transferred (DRAM)
```

### Ridge Point
The crossover between memory-bound and compute-bound:
```
Ridge AI = Peak TFLOPS / Peak Memory Bandwidth (TB/s)
```

| GPU | Peak FP64 (TFLOPS) | BW (TB/s) | FP64 Ridge (FLOPs/Byte) |
|-----|-------------------|-----------|------------------------|
| L40 | 0.72 | 0.864 | 0.83 |
| A100 | 9.7 | 2.04 | 4.8 |
| H100 | 33.5 | 3.35 | 10.0 |
| B200 | ~120 | ~8.0 | ~15.0 |

**If your kernel's AI < Ridge Point → memory-bound → optimize memory access.**
**If AI > Ridge Point → compute-bound → optimize arithmetic or use Tensor Cores.**

### Getting AI from ncu
```bash
ncu --metrics smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
python ...
```
Then: `AI = 2 × (fadd + fmul + 2×ffma) / (bytes_ld + bytes_st)`

---

## NVTX Annotations

Add markers to cuOpt source code for better timeline readability:

```cpp
#include <nvtx3/nvToolsExt.h>

// Range (appears as colored band in nsys timeline)
nvtxRangePush("Presolver::bound_tightening");
run_bound_tightening_kernel<<<grid, block>>>(args);
nvtxRangePop();

// Instant marker
nvtxMark("LP relaxation converged");
```

Python equivalent:
```python
import nvtx
with nvtx.annotate("presolver", color="blue"):
    cuopt_solver.presolve()
```

---

## Profiling Workflow for cuOpt Optimization

1. **Baseline nsys profile** → identify top kernels by time (95th percentile)
2. **Target top-3 kernels** → run ncu with full metrics on each
3. **Compute AI** → determine if memory or compute bound
4. **Check occupancy** → if < 50%, investigate register pressure or shared memory
5. **Identify stall reason** → target the dominant stall type
6. **Implement fix** → one change at a time
7. **Verify with ncu** → confirm metric improvement before benchmarking end-to-end

---

## Common cuOpt Bottlenecks by Solver Type

| Solver | Typical Bottleneck | Metric to Check |
|--------|-------------------|----------------|
| LP (PDLP) | SpMV memory bandwidth | `gpu__dram_throughput` |
| MILP presolver | Branch divergence in bound tightening | `smsp__warp_issue_stalled` |
| Routing VRP | Irregular memory access (graph traversal) | `l1tex__t_hit_rate.pct` |
| MIP cutting planes | Sparse tableau kernel | `lts__t_hit_rate.pct` |
