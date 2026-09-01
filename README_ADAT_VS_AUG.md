# ADAT vs augmented cache-reuse bench

Compares **ADAT** (`CUOPT_AUGMENTED=0`) vs **forced augmented** (`CUOPT_AUGMENTED=1`) on the factor-lifted portfolio QP, with cache **off** (baseline) vs cache **on** (`sequence_solve` + `update_linear_objective`).

Script: `benchmark_review/native_api_test/scripts/resolve_adat_vs_aug_total.py`

This tree is local/untracked. You need the files under `benchmark_review/native_api_test/` (script + `portfolio.npz` + `resolve_seq.npz`), plus a build of `cuopt_cache_reuse_FSI` as in `README_CACHE_REUSE.md`.

## What it runs

For each of **4** combinations (path × config):

| path | `CUOPT_AUGMENTED` | config |
|------|-------------------|--------|
| ADAT | 0 | default |
| ADAT | 0 | tuned |
| augmented | 1 | default |
| augmented | 1 | tuned |

Each combo:

1. **Baseline** — `K` independent solves, cache off (`sequence_solve=False`).
2. **Cache** — t=0 cold solve with `sequence_solve=True`, then t=1..K−1 `update_linear_objective(c)` + `Solve` on the same `DataModel`.

`K` comes from `resolve_seq.npz` (20 periods). Reports **solver total** warm means only (not wall time). Asserts:

- linear system log matches the path name
- objectives match baseline vs cache (`atol=1e-5`, `rtol=1e-8`)
- every warm cache solve logs  
  `Barrier: reusing cache (skip convert/presolve/scaling)`

IR and bound-free-var presolve are forced off. Tuned adds `CUOPT_BARRIER_DUAL_INITIAL_POINT=1` and `CUOPT_BARRIER_STEP_SCALE=0.99`.

## Prerequisites

1. Build and install this branch (`./build.sh libcuopt cuopt --install`) per `README_CACHE_REUSE.md`.
2. Activate the same conda env.
3. Have `portfolio.npz` and `resolve_seq.npz` next to the script (already in this repo’s `benchmark_review/native_api_test/scripts/`).
4. A free GPU.

## Run

```bash
mamba activate ./.cuopt_env          # or: conda activate cuopt129
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0        # pick a free GPU
export BENCH_RMM_POOL=0              # raw cudaMalloc (no RMM pool)

cd benchmark_review/native_api_test/scripts
python -u resolve_adat_vs_aug_total.py
```

A full 4-combo run is on the order of several minutes (warmup + 20 baseline + 20 cache, four times).

## Optional filters

```bash
# only forced augmented, tuned
export CUOPT_BENCH_PATHS=augmented
export CUOPT_BENCH_CONFIGS=tuned

# only ADAT
export CUOPT_BENCH_PATHS=ADAT
```

Allocator (default without `BENCH_RMM_POOL=0` is a 1 GiB RMM pool):

```bash
export BENCH_RMM_POOL=0              # off (cudaMalloc)
export BENCH_RMM_KIND=pool           # PoolMemoryResource
export BENCH_RMM_KIND=async          # CudaAsyncMemoryResource
export CUOPT_GIGABYTES_PER_PROC=1    # pool size when kind=pool
```

## What success looks like

Stdout ends with a table like:

```text
=== ADAT vs forced augmented (solver total, warm mean, ms) ===
path       config     baseline      cache       save   save%
----------------------------------------------------------------
ADAT       default       ...        ...        ...     ...
ADAT       tuned         ...        ...        ...     ...
augmented  default       ...        ...        ...     ...
augmented  tuned         ...        ...        ...     ...
```

Per-combo lines include `cache_reuse=19/19` and `obj_match=YES`. Warm reuse is 19/19 because t=0 is the cold store.

If cache reuse is missing, the script **asserts** and exits. Check that `settings.sequence_solve` and `DataModel.update_linear_objective` exist (you are on this branch’s Python install, not a released wheel).

## Outputs

| Path | Contents |
|------|----------|
| `benchmark_review/native_api_test/results/resolve_adat_vs_aug_total_nopool.json` | Summary JSON (`_nopool` when `BENCH_RMM_POOL=0`) |
| `benchmark_review/native_api_test/results/resolve_adat_vs_aug_logs/` | Per-solve logs (`ADAT_default_baseline/t00.log`, …) |
| `benchmark_review/native_api_test/results/{path}_{config}_all_solver.log` | Combined solver stdout per combo |

Override the per-solve log root with `BENCH_LOG_ROOT`.
