# SKILL: NVIDIA RTX A1000 Laptop GPU Architecture

## Purpose
Optimization guidance for cuOpt workloads on the NVIDIA RTX A1000 Laptop GPU
(RTX A1000 6 GB / RTX A1000 Embedded). This GPU is representative of a class
of Ampere-generation mobile and embedded devices that are **severely bandwidth-
and SM-count-constrained** relative to the data-centre GPUs cuOpt is tuned for.

---

## Hardware Specification

| Property | Value |
|---|---|
| Architecture | Ampere (GA107) |
| Compute Capability | 8.6 |
| Streaming Multiprocessors (SMs) | 16 |
| CUDA Cores | 2048 |
| Tensor Cores (3rd gen) | 64 |
| Memory Type | GDDR6 |
| Memory Capacity | 6 GB |
| Memory Bandwidth | ~128 GB/s (4× slower than A100's 2 TB/s) |
| L2 Cache | 2 MB (vs. 40 MB on A100) |
| TDP (cTGP) | 35–50 W (configurable) |
| NVLink | None |
| Max Threads / SM | 1536 |
| Max Warps / SM | 48 |
| Shared Memory / SM | 100 KB (configurable up to 100 KB) |
| Register File / SM | 256 KB |

---

## Critical Differences vs. Data-Centre Ampere (A100)

| | A1000 (GA107, cc 8.6) | A100 (GA100, cc 8.0) |
|---|---|---|
| SMs | 16 | 108 |
| Memory bandwidth | ~128 GB/s | ~2,000 GB/s |
| L2 cache | 2 MB | 40 MB |
| VRAM | 6 GB | 40/80 GB |
| NVLink | No | Yes (600 GB/s) |
| Multi-Instance GPU | No | Yes |
| Async copy (cp.async) | Yes (cc 8.0+) | Yes |
| BF16 / TF32 Tensor Cores | Yes | Yes |

**Implication:** Any kernel tuned for 108 SMs or 2 TB/s bandwidth will be
starved on the A1000. Work-per-SM must go up and memory traffic must go down.

---

## Optimization Principles for the A1000

### 1. Thread-Block Size

* Default 512-thread blocks waste occupancy on 16 SMs. Prefer 128–256 threads.
* Theoretical max occupancy with a 256-thread block and typical register usage
  (~40 regs/thread) is 4 blocks/SM × 256 = 1024 threads/SM, or 67% occupancy.
  With 512-thread blocks this often drops to 2 blocks/SM at the same register
  pressure, leaving 33% of capacity idle.
* Use `cudaOccupancyMaxPotentialBlockSize` at runtime or `nvcc --ptxas-info`
  to verify for each kernel.

### 2. Memory-Bandwidth Conservation

* The A1000's 128 GB/s budget is easily saturated. Reduce scan widths, prefer
  local reductions over global gathers, and maximise cache reuse.
* The tiny 2 MB L2 (vs. 40 MB on A100) means data sets > 2 MB cause constant
  L2 misses. Tile and block computations so working sets fit in L2.
* Prefer `__ldg` (read-only data-path cache) and `cudaFuncSetAttribute` with
  `cudaFuncAttributePreferredSharedMemoryCarveout` to trade L1/shared for L2
  effectiveness.

### 3. Load Balancing

* With only 16 SMs, any kernel with variable-width work per thread-block will
  leave SMs idle if the threshold for the load-balanced codepath is too high.
  Lower `load_balancing_codepath_min_varcount` from the default 3200 to ~800
  so the balanced path activates for medium-sized problems.

### 4. Reduction Width / Sampled Moves

* Algorithms that scan hundreds of candidates per thread (e.g., move scoring
  in Feasibility Jump) thrash the 128 GB/s bus.
* Cutting `max_sampled_moves` from 512 to 128 reduces DRAM reads per
  iteration by 4× with minimal impact on solution quality (the top-k
  distribution is concentrated in the first 128 entries anyway).

### 5. Shared Memory vs. L1

* On cc 8.6 the combined L1/shared memory is 128 KB per SM. Kernels that
  declare ≥ 64 KB of shared memory halve L1 capacity; prefer ≤ 32 KB shared
  unless the kernel is purely compute-bound.

### 6. Warp Scheduling and Latency Hiding

* With 48 warps/SM capacity, fill it: launch at least 8 warps/block so the
  scheduler can cover 200-cycle L2 misses.
* Avoid `__syncthreads` in tight inner loops — each sync is a barrier that
  forces all 48 warps in a block to stall.

---

## cuOpt MIP Tuning for A1000

### Python-level (no rebuild)

```python
from cuopt.linear_programming import SolverSettings

settings = SolverSettings()
settings.set_parameter("mip_heuristics_only", True)  # skip CPU B&B overhead
settings.set_parameter("mip_cut_passes", 3)           # default 10 — cuts are CPU-bound
settings.set_parameter("num_cpu_threads", 4)          # match your laptop's P-cores
settings.set_parameter("time_limit", 120.0)
```

### Source-level (requires rebuild)

File: `cpp/src/mip_heuristics/feasibility_jump/feasibility_jump.cuh`

| Parameter | Default | A1000 value | Rationale |
|---|---|---|---|
| `TPB_heavyvars` | `WarpSize*16` (512) | `WarpSize*8` (256) | Better occupancy, lower register pressure |
| `TPB_setval` | `WarpSize*16` (512) | `WarpSize*8` (256) | Same reason; setval is bandwidth-bound |
| `max_sampled_moves` | `WarpSize*16` (512) | `WarpSize*4` (128) | 4× less DRAM traffic per FJ iteration |
| `load_balancing_codepath_min_varcount` | 3200 | 800 | Engage load balancing earlier on 16 SMs |

---

## Profiling Checklist

1. **Roofline**: run `ncu --target-processes all --set full -o profile.ncu-rep <binary>` and open in Nsight Compute. Check whether kernels are bandwidth-bound or compute-bound.
2. **Achieved occupancy**: `ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active`.
3. **L2 hit rate**: `ncu --metrics l2_global_hit_rate` — should be > 50% on medium problems with the tuned settings.
4. **Warp stall sampling**: `ncu --metrics smsp__warp_issue_stalled_long_scoreboard_pct` for memory latency stalls.

---

## Keywords

`a1000`, `RTX A1000`, `laptop GPU`, `mobile GPU`, `Ampere`, `GA107`, `cc 8.6`,
`compute capability 8.6`, `16 SM`, `128 GB/s`, `6 GB GDDR6`, `MIP`, `FJ`,
`feasibility jump`, `low-end GPU`, `bandwidth bound`, `occupancy`
