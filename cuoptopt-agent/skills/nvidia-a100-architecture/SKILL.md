---
name: nvidia-a100-architecture
version: "26.04.00"
description: NVIDIA A100 GPU architecture — Ampere, compute capability 8.0, HBM2e, NVLink 3.0, MIG, 3rd-gen Tensor Cores, TF32/BF16, optimization notes.
---

# NVIDIA A100 Architecture

Ampere generation (2020). The A100 is the primary data-center GPU for HPC and AI
workloads; the most common cuOpt deployment target at the time of writing.

---

## Key Specifications

| Property | A100 SXM4 80GB | A100 PCIe 80GB |
|----------|---------------|----------------|
| Architecture | Ampere | Ampere |
| Compute Capability | 8.0 | 8.0 |
| CUDA Cores | 6,912 | 6,912 |
| Tensor Cores (gen) | 3rd | 3rd |
| VRAM | 80 GB HBM2e | 80 GB HBM2e |
| Memory Bandwidth | 2,039 GB/s | 1,935 GB/s |
| FP64 TFLOPS | 9.7 | 9.7 |
| FP32 TFLOPS | 19.5 | 19.5 |
| TF32 TFLOPS | 156 (sparsity: 312) | 156 |
| FP16/BF16 TFLOPS | 312 (sparsity: 624) | 312 |
| L2 Cache | 40 MB | 40 MB |
| SMs | 108 | 108 |
| L1 / Shared Mem per SM | 192 KB (max 164 KB shared) | 192 KB |
| NVLink | 3.0, 600 GB/s (6 links) | None |
| TDP | 400 W | 300 W |

---

## Cache Hierarchy

```
Thread registers
     ↓
L1 / Shared memory: up to 164 KB per SM
     ↓
L2 Cache: 40 MB
     ↓
HBM2e: 80 GB @ 2,039 GB/s
     ↓
NVLink 3.0 (SXM): 600 GB/s bidirectional
     ↓
PCIe Gen4 ×16 (PCIe variant): ~64 GB/s
```

---

## 3rd-Generation Tensor Cores

New in Ampere:
- **TF32**: Full FP32 dynamic range, 10-bit mantissa; 2× FP32 throughput; transparent via cuBLAS.
- **BF16**: Same exponent as FP32; better training convergence than FP16.
- **Sparsity acceleration**: 2:4 structured sparsity doubles FP16/TF32 throughput.
- Tile sizes for WMMA: `16×16×16` (FP16), `16×16×8` (TF32).

For cuOpt LP:
- TF32 Tensor Cores accelerate dense GEMM in PDLP iterations with minimal accuracy loss.
- BF16 is preferred over FP16 for numerical stability in iterative solvers.

---

## Multi-Instance GPU (MIG)

A100 supports MIG: partition one GPU into up to 7 independent instances (GI).
- Each GI has isolated memory partitions, SM groups, and L2 slices.
- Useful for running multiple independent cuOpt solves concurrently on shared hardware.
- MIG profiles: `1g.10gb`, `2g.20gb`, `3g.40gb`, `4g.40gb`, `7g.80gb`.
- MIG cannot be mixed with multi-GPU NVLink topologies.

---

## NVLink 3.0 (SXM4 Only)

- 600 GB/s total bidirectional bandwidth between GPUs.
- Enables `cudaMemcpyPeerAsync` at near-memory-bandwidth speeds.
- cuOpt multi-GPU: use `cudaEnablePeerAccess` + unified virtual addressing for
  seamless cross-GPU pointer sharing in distributed LP decomposition.

---

## Async Memory Copy (`cp.async`)

Introduced in Ampere — key for hiding HBM latency:
```cuda
#include <cuda/pipeline>
__pipeline_memcpy_async(smem_ptr, gmem_ptr, 16);
__pipeline_commit();
// compute with previously staged data
__pipeline_wait_prior(0);
```
- Requires SM ≥ 8.0; falls back to synchronous copy on older hardware.
- Combine with double-buffering in shared memory for maximum overlap.

---

## Shared Memory Configuration

A100 has up to 164 KB shared memory per SM (largest of any pre-Hopper GPU).
```cuda
cudaFuncSetAttribute(
    my_kernel,
    cudaFuncAttributeMaxDynamicSharedMemorySize,
    163840  // 160 KB
);
```
This enables larger tile sizes for matrix kernels.

---

## Optimization Notes for A100

### HBM2e Memory-Bound Workloads
- 2 TB/s bandwidth means most well-written kernels are compute-bound.
- If memory-bound, check for uncoalesced access (stride > 1 cache line).

### Register File
- 256 KB register file per SM; 255 registers per thread maximum.
- Use `--maxrregcount=80` for most cuOpt kernels to maintain occupancy.

### Warp Scheduling
- Ampere has 4 warp schedulers per SM, each issuing 2 instructions/cycle.
- Achieving full throughput requires ≥ 4 independent instruction streams per SM.

---

## Profiling on A100

```bash
# High-level timeline
nsys profile --trace=cuda,nvtx -o a100_report python -m cuopt solve --mps problem.mps

# SM efficiency and memory
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_hit_rate.pct,lts__t_hit_rate.pct \
--target-processes all python -m cuopt solve --mps problem.mps
```
