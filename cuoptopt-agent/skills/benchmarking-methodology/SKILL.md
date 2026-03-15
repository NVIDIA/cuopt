---
name: benchmarking-methodology
version: "26.04.00"
description: Correct GPU benchmarking methodology — warmup runs, CUDA event timing, statistical significance, thermal throttling, before/after comparison for cuOpt.
---

# Benchmarking Methodology

Correct benchmarking is essential for detecting genuine performance improvements.
Common mistakes lead to false positives (reporting speedup that does not exist)
or false negatives (missing real improvements).

---

## Core Principles

1. **Measure the same thing**: identical problem instances, identical seeds, identical hardware state.
2. **Warm up the GPU**: JIT compilation, kernel caching, and thermal state affect first runs.
3. **Use multiple timed runs**: take the median, not the mean (outlier-resistant).
4. **Control variance**: thermal throttling, background processes, CPU frequency scaling.
5. **Use geometric mean for aggregates**: ratios should use geometric mean, not arithmetic mean.

---

## GPU Warm-Up

### Why Warm-Up Is Critical
- CUDA JIT compilation (PTX → SASS) happens on first kernel launch: can take 100ms–10s.
- GPU clock states (P-states) may start in power-saving mode.
- L2 cache cold-start effects.

### Warm-Up Protocol
```python
# Python: two full warm-up solves before timing
for _ in range(WARMUP_RUNS):
    result = cuopt_solve(problem)  # result discarded

# Then time TIMED_RUNS solves
times = []
for _ in range(TIMED_RUNS):
    t0 = time.perf_counter()
    result = cuopt_solve(problem)
    cuda.synchronize()  # ensure GPU is done before recording time
    times.append(time.perf_counter() - t0)

median_time = sorted(times)[len(times) // 2]
```

### CUDA Event Timing (More Precise)
```cpp
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
my_kernel<<<grid, block>>>(args);
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float ms = 0;
cudaEventElapsedTime(&ms, start, stop);
// ms is in milliseconds with ~0.5μs resolution
```
Use CUDA events for kernel-level timing; `time.perf_counter()` for end-to-end solve time.

---

## Statistical Correctness

### Sample Size
- Minimum 5 timed runs per instance; 10 for high-variance kernels.
- Use **median** (50th percentile), not mean — GPU timing has right-skewed outliers.
- For noisy benchmarks: also report IQR (interquartile range).

### Aggregate Across Instances
Use **geometric mean** of ratios when comparing two conditions:
```python
import math

ratios = [t_candidate[i] / t_baseline[i] for i in range(n_instances)]
geomean_ratio = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
speedup_pct = 100 * (1.0 - geomean_ratio)  # positive = faster
```

Do NOT use arithmetic mean of absolute times — it over-weights large instances.

### Significance Testing
For small sample sizes (< 30 runs), use the Wilcoxon signed-rank test:
```python
from scipy.stats import wilcoxon
stat, p_value = wilcoxon(times_baseline, times_candidate)
# Reject null (no difference) if p_value < 0.05
```
A 5% improvement is only reliable if p < 0.05 with ≥ 10 timed runs.

---

## Thermal Throttling Control

Modern GPUs reduce clock frequency when temperature exceeds a threshold (typically 83°C).
This reduces performance and introduces variance.

### Detection
```bash
# Monitor GPU temperature and clock frequency
nvidia-smi dmon -s pct -d 1  # 1-second polling
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw --format=csv -l 1
```

### Mitigation
```bash
# Lock GPU clocks (requires root; not persistent across reboot)
sudo nvidia-smi -pm 1               # enable persistence mode
sudo nvidia-smi --lock-gpu-clocks=<target_MHz>  # lock SM clock
sudo nvidia-smi --lock-memory-clocks=<target_MHz>

# Restore
sudo nvidia-smi --reset-gpu-clocks
```

For benchmarking without root: add a 10s sleep between each solve to allow cooling,
or run in a thermally conditioned data center environment.

---

## CPU-GPU Time Isolation

Distinguish CPU time from GPU time:
```python
import time

# End-to-end wall time (includes CPU overhead)
t0 = time.perf_counter()
result = cuopt_solve(problem)
total_wall = time.perf_counter() - t0

# GPU-only time (requires cudaDeviceSynchronize inside solve)
# Or use CUDA events around the GPU work only
```

For solver benchmarks, **wall time** (including CPU scheduling and data setup) is the
correct metric for end-users. **GPU kernel time** (from CUDA events) is useful for
identifying the GPU-specific contribution.

---

## Before/After Comparison Protocol

When evaluating a code change:

### Step 1: Record Baseline
```bash
git stash  # or git checkout main
python benchmark.py --runs 5 --warmup 2 > baseline.json
```

### Step 2: Apply Change and Record Candidate
```bash
git stash pop  # or checkout your branch
python benchmark.py --runs 5 --warmup 2 > candidate.json
```

### Step 3: Compare
```python
import json, math

baseline = json.load(open("baseline.json"))
candidate = json.load(open("candidate.json"))

for inst in baseline:
    b = baseline[inst]["median_s"]
    c = candidate[inst]["median_s"]
    delta = 100 * (c - b) / b
    print(f"{inst}: {b:.4f}s → {c:.4f}s  ({delta:+.2f}%)")

# Aggregate
ratios = [candidate[i]["median_s"] / baseline[i]["median_s"] for i in baseline]
gm = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
print(f"\nGeometric mean speedup: {100*(1-gm):+.2f}%")
```

### Step 4: Solution Quality Check
Compare best objective values:
```python
for inst in baseline:
    b_obj = baseline[inst]["objective"]
    c_obj = candidate[inst]["objective"]
    if b_obj is not None and c_obj is not None and b_obj != 0:
        delta = 100 * (c_obj - b_obj) / abs(b_obj)
        if abs(delta) > 0.01:  # 0.01% threshold
            print(f"QUALITY CHANGE {inst}: {b_obj:.6f} → {c_obj:.6f} ({delta:+.4f}%)")
```

---

## cuOpt Benchmark Dataset Strategy

Available test sets (in `datasets/`):
- `linear_programming/` — LP instances (afiro, benchmark MPS files)
- `mixed_integer_programming/` — MIP instances
- `mip/` — Additional MIP (sudoku, presolve, MIPLIB)
- `quadratic_programming/` — QP instances

Recommended benchmark subsets by change type:
| Change | Primary dataset | Secondary |
|--------|----------------|-----------|
| Presolver | `mip/presolve/` | `mixed_integer_programming/` |
| LP solver | `linear_programming/` | `mixed_integer_programming/` |
| Routing | `distance_engine/` | `cuopt_service_data/` |
| QP solver | `quadratic_programming/` | — |
| CUDA kernel | All | — |
