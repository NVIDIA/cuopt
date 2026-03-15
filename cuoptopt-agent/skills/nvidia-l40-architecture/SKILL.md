---
name: nvidia-l40-architecture
version: "26.04.00"
description: NVIDIA L40/L40S GPU architecture — Ada Lovelace, compute capability 8.9, GDDR6, Tensor Cores, cache hierarchy, optimization notes.
---

# NVIDIA L40 / L40S Architecture

Ada Lovelace generation (2022). The L40 and L40S are PCIe data-center GPUs
optimized for graphics, AI inference, and compute workloads.

---

## Key Specifications

| Property | L40 | L40S |
|----------|-----|------|
| Architecture | Ada Lovelace | Ada Lovelace |
| Compute Capability | 8.9 | 8.9 |
| CUDA Cores | 18,176 | 18,176 |
| Tensor Cores (gen) | 4th | 4th |
| VRAM | 48 GB GDDR6 | 48 GB GDDR6 |
| Memory Bandwidth | 864 GB/s | 864 GB/s |
| FP32 TFLOPS | 91.6 | 91.6 |
| TF32 TFLOPS | 183 | 183 |
| FP16/BF16 TFLOPS | 362 (sparsity: 724) | 362 (sparsity: 724) |
| L2 Cache | 96 MB | 96 MB |
| SMs | 142 | 142 |
| L1 / Shared Mem per SM | 128 KB | 128 KB |
| TDP | 300 W | 350 W |
| NVLink | None | None |
| PCIe | Gen 4 ×16 | Gen 4 ×16 |

---

## Cache Hierarchy

```
Thread registers
     ↓
L1 / Shared memory: 128 KB per SM (configurable split)
     ↓
L2 Cache: 96 MB (shared across all SMs)
     ↓
GDDR6 VRAM: 48 GB @ 864 GB/s
     ↓
PCIe Gen4 ×16: ~64 GB/s (host ↔ device)
```

- 96 MB L2 is unusually large; cuOpt sparse matrix data and graph adjacency structures
  can often fit entirely in L2, eliminating repeated DRAM fetches.
- L1 carve-out: use `cudaFuncAttributePreferredSharedMemoryCarveout` to bias toward
  shared memory (better for reduction kernels) or L1 (better for streaming kernels).

---

## 4th-Generation Tensor Cores

Supported precisions for MMA operations:
- **FP16, BF16** — standard mixed-precision
- **TF32** — 10-bit mantissa FP32-like; 2× FP32 throughput without code changes
- **FP8 (E4M3 / E5M2)** — first consumer Ada feature; 4× FP16 throughput with sparsity
- **INT8, INT4** — structured sparsity doubles effective throughput again

For cuOpt: LP/MILP coefficient matrices are typically FP64; TF32/FP16 Tensor Cores
are most useful for inner-loop GEMM in iterative LP solvers where FP32 accumulation
is acceptable.

---

## Optimization Notes for L40

### No NVLink
- L40 is a standalone card — multi-GPU data sharing requires PCIe or NVSwitch fabric.
- For cuOpt single-GPU workloads this is not a concern.
- For multi-GPU decomposition, use CUDA-aware MPI or NCCL over PCIe.

### GDDR6 vs HBM
- GDDR6 bandwidth (864 GB/s) is 3–4× lower than H100 HBM3 (3.35 TB/s).
- Memory-bound kernels will be more constrained than on A100/H100.
- **Action:** Prioritize memory access coalescing and L2 reuse over raw throughput.

### Large L2 Cache (96 MB)
- Much larger than A100 (40 MB) or H100 (50 MB).
- Effective strategy: structure cuOpt data (constraint matrices, node-arc data) to
  fit repeated-access working sets inside L2.
- Use `cudaDeviceSetLimit(cudaLimitL2FetchGranularity, ...)` to tune prefetch granularity.

### Ada Warp Engine
- Warp scheduler improved vs Ampere; better instruction-level parallelism for
  mixed compute + memory workloads.
- Double-speed FP32 units (2× FP32 ops per clock vs Ampere) benefit float-heavy kernels.

---

## Profiling on L40

```bash
# System-level timeline
nsys profile --trace=cuda,nvtx --output l40_profile python -m cuopt solve --mps problem.mps

# Kernel-level metrics
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_hit_rate.pct,lts__t_hit_rate.pct,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed \
python -m cuopt solve --mps problem.mps
```

Key metrics to watch:
- `lts__t_hit_rate.pct` (L2 hit rate) — want > 80% for working-set-fits-L2 kernels
- `gpu__dram_throughput` — if saturated, increase L2 reuse
- `sm__throughput` — if < 60%, likely memory-bound or low occupancy
