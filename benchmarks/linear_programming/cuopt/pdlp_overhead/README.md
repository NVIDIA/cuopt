<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PDLP overhead benchmark

This optional C API driver reads an original MPS file, constructs its continuous LP
relaxation, and invokes cuOpt's native PSLP presolve, solve, and postsolve path. It
checks the returned solution in the original coordinates. Integer markers are
relaxed; objective sense and offset are preserved.

From a configured cuOpt development build, enable the LP benchmarks and build:

```bash
cmake -S cpp -B build -DBUILD_LP_BENCHMARKS=ON
cmake --build build --target cuopt_pdlp_overhead cuopt_pdlp_overhead_checks
build/pdlp_overhead/cuopt_pdlp_overhead_checks
```

The driver links the current build's cuOpt, client, RMM, and CUDA targets. Its
exported RMM resource symbols let the executable install the resource used by
cuOpt. The compressed mode keeps the scoped, shared resource and its initial
1 GiB pool alive until all cuOpt objects have been destroyed, then restores the
previous resource and verifies cleanup. It requires CUDA virtual memory and
generic allocation compression support.

Run one GPU at a time with the same model and settings in both configurations.
These commands use the first visible GPU; change `CUDA_VISIBLE_DEVICES` to select
the device for the experiment.

```bash
# Baseline controls.
env CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER OMP_NUM_THREADS=8 \
  CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE=1 \
  LPFEAS_REORDER=off LPFEAS_REORDER_HUB_THRESHOLD=1024 \
  LPFEAS_RMM_MEMORY_MODE=default \
  build/pdlp_overhead/cuopt_pdlp_overhead model.mps \
  --method 0 --threads 8 --time-limit 300 \
  --relative-tolerance 1e-6 --absolute-tolerance 1e-4

# Retained overhead changes, RCM, and compressed allocation.
env CUDA_VISIBLE_DEVICES=0 CUDA_MODULE_LOADING=EAGER OMP_NUM_THREADS=8 \
  CUOPT_EXPERIMENTAL_GRAPH_BATCH_SIZE=24 \
  LPFEAS_REORDER=rcm LPFEAS_REORDER_HUB_THRESHOLD=1024 \
  LPFEAS_RMM_MEMORY_MODE=compressed \
  build/pdlp_overhead/cuopt_pdlp_overhead model.mps \
  --method 0 --threads 8 --time-limit 300 \
  --relative-tolerance 1e-6 --absolute-tolerance 1e-4
```

Here `--method 0` means the **C API default/concurrent** method. Use `--method 1`
in both commands for **PDLP only**. The relative tolerance applies to primal,
dual, and gap termination; all three absolute tolerances are also set explicitly.
Native PSLP is enabled, crossover is disabled, and cuOpt owns the 300-second solve
budget. Native PSLP uses its configured default folding behavior. RCM
automatically preserves the spectral probe's coordinates; diagnostic probe flags
are unnecessary. Graph batching can fall back to size one on unsupported paths;
inspect the logged effective size and batch counts.

For comparisons of library changes, build and run each revision with its own
matching dependencies. Switching the environment controls above in one binary
does not disable the compiled vector and scalar-readback changes. Keep diagnostic
overrides, solution imports, and probe dumps disabled during timings.

`RESULT_JSON` includes native status, timings, objectives, and original-space
residual checks. `RCM_JSON` reports ordering and inverse times, and `ALLOCATOR_JSON`
reports allocation coverage and cleanup. Treat the timers separately:

- `cuopt_reported_seconds` includes native presolve and solver setup. A PDLP
  concurrent winner can finish before the other workers have joined.
- `solve_api_wall_seconds` covers the complete solve API call, including worker
  joins and native postsolve.
- `total_wall_seconds` covers driver work through validation; external process
  timing additionally includes final destruction and allocator teardown.

Configured `Optimal` and the independent original-space diagnostic flags answer
different questions. Preserve both when comparing speed. A returned native time
limit status identifies a solver timeout; API or process time alone does not.
