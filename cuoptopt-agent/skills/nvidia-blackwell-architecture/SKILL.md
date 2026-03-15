---
name: nvidia-blackwell-architecture
version: "26.04.00"
description: NVIDIA Blackwell GPU architecture — B100/B200/GB200, dual-die NV-HBI, 5th-gen Tensor Cores, tcgen05 single-thread MMA, Tensor Memory (TMEM), FP4/FP6, hardware Decompression Engine, CTA-pair execution, NVLink 5.0.
---

# NVIDIA Blackwell Architecture (B100 / B200 / GB200)

Blackwell generation (2024). Successor to Hopper; the first dual-die GPU design
with coherent unified memory across both dies. Fundamentally changes how Tensor Core
MMA instructions are issued.

---

## Key Specifications

| Property | B200 SXM | B100 SXM | GB200 (per GPU) |
|----------|----------|----------|-----------------|
| Architecture | Blackwell | Blackwell | Blackwell |
| Compute Capability | 10.0 | 10.0 | 10.0 |
| Dies | 2 (dual-die) | 2 (dual-die) | 2 (dual-die) |
| CUDA Cores | ~20,480 | ~20,480 | ~20,480 |
| Tensor Cores (gen) | 5th | 5th | 5th |
| SMs | 148 | 132 | 148 |
| VRAM | 192 GB HBM3e | 192 GB HBM3e | 192 GB HBM3e |
| Memory Bandwidth | ~8,000 GB/s | ~8,000 GB/s | ~8,000 GB/s |
| FP4 TFLOPS | 18,000 (sparsity: 36,000) | 14,000 | 18,000 |
| FP16/BF16 TFLOPS | ~4,500 | ~3,500 | ~4,500 |
| FP64 TFLOPS | ~120 | ~100 | ~120 |
| L2 Cache | ~4 × partitions | ~4 × partitions | large |
| L1 / Shared per SM | TBD (≥ 256 KB) | TBD | TBD |
| NVLink | 5.0 | 5.0 | 5.0 |
| TDP | ~1,000 W | ~700 W | ~1,000 W |

*Certain micro-architectural values are not yet publicly disclosed.*

---

## Dual-Die Design (NV-HBI)

- Two GPU dies connected by **NVIDIA High-Bandwidth Interface (NV-HBI)** at
  10 TB/s chip-to-chip bandwidth.
- Appears as a **single device** to software: unified virtual address space,
  single `cudaSetDevice()` call, single CUDA context.
- 4 L2 cache partitions (double Hopper's 2); enables lower average L2 latency
  for the larger SM count.
- 8 HBM3e memory stacks; 192 GB unified.

---

## 5th-Generation Tensor Cores and `tcgen05`

### Breaking Change from All Prior Generations

**Hopper and earlier:** MMA operations use `wgmma` (warp-group synchronous) or `mma.sync`
(warp synchronous). All 32 threads in a warp (or all warps in a warp-group) must synchronize
before executing the MMA.

**Blackwell (`tcgen05.mma`):** MMA is a **single-thread instruction**. Each thread
independently issues matrix multiply-accumulate operations. There is no warp-level barrier
requirement for the MMA itself.

```ptx
// Blackwell-only: single-thread MMA dispatch
tcgen05.mma.cta_group::1.kind::f32 [tmem_d_ptr], [smem_a_ptr], [tmem_b_ptr], [tmem_c_ptr];
```

Implications:
- Eliminates idle cycles caused by warp synchronization barriers before MMA.
- Compiler has full freedom to schedule MMA around unrelated instructions.
- New instruction latency and saturation characteristics — not documented by NVIDIA;
  measured empirically at ~10–12 cycles MMA latency (B200 microbenchmarks, Dec 2024).

---

## Tensor Memory (TMEM)

A new dedicated on-chip memory tier exclusively for Tensor Core operands.

### What TMEM Is
- Per-SM scratchpad that sits between shared memory and Tensor Core register files.
- Holds matrix A, B, and accumulator (C/D) tiles for tcgen05 operations.
- Allocated and managed explicitly in software; the compiler does NOT automatically
  place data in TMEM.

### Critical: Entirely New Instruction Set
The traditional data-movement instructions **cannot interface with TMEM**:

| Old instruction | Can use TMEM? |
|----------------|--------------|
| `wmma.load` | NO |
| `wgmma.mma_async` | NO |
| `ldmatrix` | NO |
| `cp.async` | NO |
| `ld.shared` | NO |

New TMEM instructions (all prefixed `tcgen05`):
```ptx
// Allocate TMEM
tcgen05.alloc.cta_group::1 [tmem_ptr], num_cols;

// Copy smem → tmem
tcgen05.cp.cta_group::1 [tmem_dst], [smem_src];

// Load into register from tmem
tcgen05.ld.cta_group::1 {r0, r1, r2, r3}, [tmem_ptr];

// Store register to tmem
tcgen05.st.cta_group::1 [tmem_ptr], {r0, r1, r2, r3};

// Deallocate
tcgen05.dealloc.cta_group::1 [tmem_ptr], num_cols;
```

For cuOpt: existing CUDA kernels using `wgmma` or `wmma` must be rewritten with
`tcgen05` instructions to exploit Blackwell Tensor Core throughput.

---

## FP4 and FP6 Native Precision

Blackwell is the first GPU with native FP4 (E2M1) and FP6 (E3M2/E2M3) Tensor Core support.

| Precision | TFLOPS (B200) | Use case for cuOpt |
|-----------|--------------|-------------------|
| FP4 | 18,000 | Ultra-low-precision inner loops (research) |
| FP6 | ~12,000 | Quantized LP coefficient operations |
| FP8 | ~9,000 | Residual computation |
| FP16/BF16 | ~4,500 | Standard LP/MIP iterations |
| TF32 | ~2,250 | Full-precision-like GEMM |
| FP64 | ~120 | Exact LP arithmetic |

FP4/FP6 require `tcgen05` instructions; no higher-level API equivalent exists yet (as of 2025).

---

## CTA-Pair Execution

Two CTAs (Cooperative Thread Arrays) with adjacent ranks form a **CTA pair**:
- Mapped to a single TPC (Texture Processor Cluster).
- Share a dedicated intra-TPC communication network.
- Can share Tensor Core operands, reducing redundant loads.

```cuda
// Launch with CTA pairs (requires cooperative launch with cluster dims)
__cluster_dims__(2, 1, 1)
__global__ void blackwell_kernel() {
    // CTA pair shares operand B tile via cluster SMEM
    auto cluster = cooperative_groups::this_cluster();
    // ...
}
```

---

## Hardware Decompression Engine (DE)

Dedicated hardware subsystem that decompresses model weights from HBM3e
transparently during memory access:
- Supported algorithms: (publicly: sparse formats, quantized weights)
- Decompresses at memory-bandwidth rate — no SM cycles consumed.
- Use case for cuOpt: compressed sparse constraint matrix storage → decompress on load,
  reducing effective memory footprint and increasing L2 hit rate.

---

## Memory Bandwidth and Latency

- 8 TB/s HBM3e bandwidth (vs. H200's 4.8 TB/s).
- **58% reduction in cache-miss access latency** vs H200 (measured, Jarmusch et al. 2024).
- Revised L2 cache architecture with 4 partitions enables better scaling across 148 SMs.

---

## NVLink 5.0

- Full NVSwitch fabric in GB200 NVL72: 36 Grace CPUs + 72 Blackwell GPUs.
- 1.8 TB/s NVLink bandwidth per GPU (2× H100's 900 GB/s).
- Aggregate fabric bandwidth: 130 TB/s.
- SHARP all-reduce: 7.2 TB/s (vs. H100's 1.8 TB/s).

---

## Porting Guide: Hopper → Blackwell for cuOpt

| Hopper API | Blackwell Replacement |
|-----------|----------------------|
| `wgmma.mma_async` | `tcgen05.mma` |
| `cp.async.bulk.tensor` (TMA) | `tcgen05.cp` for TMEM, TMA still valid for SMEM |
| `__pipeline_memcpy_async` | Unchanged for SMEM transfers |
| `wmma::mma_sync` | `tcgen05.mma` (preferred) or `wmma` (legacy) |
| Shared memory accumulators | TMEM accumulators |

---

## Profiling on Blackwell

```bash
nsys profile --trace=cuda,nvtx -o b200_report python -m cuopt solve --mps problem.mps

ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__inst_executed_pipe_tensor_op_hmma.sum \
python -m cuopt solve --mps problem.mps
```

Note: Blackwell-specific `tcgen05` metrics may require NCU 2025.1+.
