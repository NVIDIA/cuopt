---
name: nvidia-h100-architecture
version: "26.04.00"
description: NVIDIA H100/H200 GPU architecture — Hopper, compute capability 9.0, HBM3, NVLink 4.0, Transformer Engine, FP8, TMA, warp specialization.
---

# NVIDIA H100 / H200 Architecture

Hopper generation (2022/2023). H100 is NVIDIA's flagship HPC/AI GPU; H200 adds
HBM3e memory for higher bandwidth.

---

## Key Specifications

| Property | H100 SXM5 | H100 PCIe | H200 SXM |
|----------|-----------|-----------|----------|
| Architecture | Hopper | Hopper | Hopper |
| Compute Capability | 9.0 | 9.0 | 9.0 |
| CUDA Cores | 16,896 | 14,592 | 16,896 |
| Tensor Cores (gen) | 4th | 4th | 4th |
| VRAM | 80 GB HBM3 | 80 GB HBM2e | 141 GB HBM3e |
| Memory Bandwidth | 3,350 GB/s | 2,000 GB/s | 4,800 GB/s |
| FP64 TFLOPS | 33.5 | 24.0 | 33.5 |
| FP32 TFLOPS | 67.0 | 48.0 | 67.0 |
| FP16/BF16 TFLOPS | 989 (sparsity: 1,978) | 756 | 989 |
| FP8 TFLOPS | 1,979 (sparsity: 3,958) | 1,513 | 1,979 |
| L2 Cache | 50 MB | 50 MB | 50 MB |
| SMs | 132 | 114 | 132 |
| L1 / Shared Mem per SM | 256 KB | 256 KB | 256 KB |
| NVLink | 4.0, 900 GB/s | None | 4.0, 900 GB/s |
| TDP | 700 W | 350 W | 700 W |

---

## Cache Hierarchy

```
Thread registers
     ↓
L1 / Shared memory: up to 228 KB per SM
     ↓
L2 Cache: 50 MB
     ↓
HBM3: 80 GB @ 3,350 GB/s  (H100 SXM)
     ↓
NVLink 4.0: 900 GB/s bidirectional
```

---

## 4th-Generation Tensor Cores (Hopper)

New in Hopper over Ampere:
- **FP8 (E4M3/E5M2)**: Native 8-bit float inference; 2× FP16 throughput.
- **Warp-group MMA (`wgmma`)**: New asynchronous MMA instruction that reads directly
  from shared memory without explicit register loads; allows pipelining compute and data movement.
- Tile sizes: up to 64×256×16 (FP16 wgmma) vs 16×16×16 (wmma).
- Transformer Engine: hardware FP8↔FP16 cast + scaling; transparent via cuBLAS/cuDNN.

For cuOpt:
- `wgmma` instructions are the preferred API for dense GEMM kernels on H100.
- FP8 accumulation with FP32 output is suitable for LP residual computations.

---

## Tensor Memory Accelerator (TMA)

New hardware unit in Hopper that offloads address generation for bulk data copies.

```cuda
// TMA descriptor (created once on host, passed to kernel)
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(&tensor_map, type, rank, global_addr, shape, stride, tile_shape, ...);

// In kernel: async bulk copy without per-thread address math
__cluster_barrier_arrive();
cp.async.bulk.tensor.3d.shared::cluster.global [...] [%tensor_map_desc, %coords];
cp.async.bulk.commit_group;
cp.async.bulk.wait_group 0;
```

- TMA eliminates warp-level addressing overhead for structured tensor access.
- Enables **thread block clusters**: groups of up to 8 thread blocks that share a
  distributed shared memory space via TMA.
- Key for cuOpt: large constraint matrix blocks can be staged to shared memory via TMA,
  freeing all threads for computation.

---

## Thread Block Clusters (Hopper)

```cuda
__cluster_dims__(2, 1, 1)  // 2-block cluster
__global__ void kernel() {
    namespace cg = cooperative_groups;
    auto cluster = cg::this_cluster();
    // Access shared memory of any block in cluster via dst_smem ptr
}
```
- Cluster-level synchronization: `cluster.sync()`.
- Distributed shared memory: read another block's smem directly via `__cluster_barrier_arrive`.
- Benefit for cuOpt: multi-block reduction kernels can exchange partial sums via cluster
  smem instead of global memory atomics.

---

## Warp Specialization Pattern

Separate warps into producer (data load) and consumer (compute) roles:
```cuda
// Producer warp: async loads data into shared memory pipeline
if (warp_role == PRODUCER) {
    while (...) {
        cp.async.bulk(...);
        arrive_on_barrier(compute_barrier);
    }
}
// Consumer warp: waits for data, then computes
if (warp_role == CONSUMER) {
    wait_on_barrier(compute_barrier);
    wgmma.mma_async(...);
}
```
This is the pattern used by cuBLAS GEMM kernels on H100 for near-peak throughput.

---

## Optimization Notes for H100

### Memory Bandwidth
- 3.35 TB/s (SXM5) means compute-bound is the norm for well-written kernels.
- FP64 throughput (33.5 TFLOPS) is 3.5× A100 — significant for cuOpt's double-precision LP.

### NVLink 4.0
- 900 GB/s bidirectional (50% increase over A100 NVLink 3.0).
- NVSwitch in DGX H100 provides full any-to-any topology at 3.6 TB/s aggregate.
- Enables cuOpt multi-GPU decomposition with low communication overhead.

### GPC / SM Architecture
- 132 SMs, each with 256 KB L1/shared — nearly double A100.
- Use `cudaFuncAttributeMaxDynamicSharedMemorySize` to allocate up to 228 KB.

---

## Profiling on H100

```bash
nsys profile --trace=cuda,nvtx -o h100_report python -m cuopt solve --mps problem.mps

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_inst_executed_op_memory_ld.sum,\
smsp__sass_inst_executed_op_wgmma_64_8.sum \
python -m cuopt solve --mps problem.mps
```
