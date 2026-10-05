# cuOpt vs ClarabelGPU — SOCP re-solve reproduction package

Everything needed to reproduce the SOCP part of our comparison on your own machine.

```
data/portfolio.npz                     F (5000x50), d, mu, n, k, gamma, cap      — raw data, seed 0 (synthetic)
data/portfolio_socp.npz                sigma^2 = 1.000764358e-04 (risk of the equal-weight portfolio; cone is active)
data/resolve_seq.npz                   mu_seq (20x5000): mu_t = mu*(1 + 2% iid U(-1,1)), seed 12345
data/portfolio_socp_n5000_k50.mps      the SOCP as MPS (QCMATRIX), exported by cuOpt's DataModel.writeMPS()
data/portfolio_socp_n5000_k50_mu_t00.mps   same with objective = -mu_0 (first re-solve step)
data/portfolio_qp_n5000_k50.mps        QP variant (QUADOBJ), for reference
data/mu_seq_20x5000.csv                the 20 objective vectors as CSV
cuopt_socp_resolve.py                  cuOpt side   (needs the cuopt Python package; --mps to load the MPS file)
clarabel_socp_resolve.py               ClarabelGPU side (needs our clarabel_gpu extension, see below)
```

## The problem

```
min  -mu'x                                   x in R^5000 (weights), y in R^50 (factor exposures)
s.t. sum(x) = 1
     F'x - y = 0                             (50 rows)
     0 <= x <= 0.05,   y free
     sum_i d_i x_i^2 + sum_j y_j^2 <= sigma^2      <- risk budget, a second-order cone of dim 5051
```

| | ClarabelGPU | cuOpt |
|---|---|---|
| cone input | native `SecondOrderConeT(5051)`: rows `[0; -sqrt(d).*x; -y]`, `b = [sigma; 0; 0]` | quadratic constraint `z'Qz <= sigma^2`, `Q = diag(d, 1_k)` → converted internally to one SOC of dim 5052 (+5050 aux vars, +5052 linking rows) |
| standard-form size | 5050 vars, 15102 rows (Zero 51 + Nonneg 10000 + SOC 5051), nnz 270k | user: 5050 vars / 51 rows / 1 QC; barrier: 10102 vars / 5103 rows / 265k nnz |

## Methodology (identical on both sides)

1. One warm-up solve (discarded) so CUDA/cuDSS initialisation is out of the timing.
2. One cold solve of the base problem.
3. **20 re-solves** where only the linear objective changes (`q = [-mu_t; 0]`); the constraint matrix, bounds and cone are untouched.
4. Report per-step time / iterations / status, then mean / min / max over the 20 steps.
5. `--check` recomputes the true objective and feasibility (budget, box, cone excess) from the raw data, so the
   comparison does not rely on either solver's self-reported residuals.

What "re-solve" means differs, and that is the point of the comparison:

| | ClarabelGPU | cuOpt |
|---|---|---|
| re-solve path | `update_q()` → KKT values untouched, cuDSS symbolic + structure reused; with `warm_start_enable` the previous iterate is reused (IPM warm start; iters 23.8 → 18.1) | full **cold** solve each step: `sequence_solve` / `update_linear_objective` is disabled when a quadratic constraint is present (`cpp/src/dual_simplex/solve.cpp:423-428`); `--try-reuse` shows the exception |
| timer | solver-internal `solve_time` (host clock around the IPM loop, results synced) | wall time of `Solve()` (within a few % of cuOpt's own barrier timer) |

## Settings we used

ClarabelGPU: `tol_feas = tol_gap_abs = tol_gap_rel = 1e-8`, `warm_start_enable=True`, `ir_adaptive=True`, `equilibrate_enable=True`
(the best of the 8 on/off combinations; `equilibrate=False` was marginally better on A100).

cuOpt: Barrier, `CUOPT_AUGMENTED=1`, `CUOPT_BARRIER_DUAL_INITIAL_POINT=1`, `CUOPT_BARRIER_STEP_SCALE=0.99`,
`CUOPT_BARRIER_PRESOLVE_BOUND_FREE_VARIABLES=0`, `set_optimality_tolerance(1e-8)`, RMM pool 1 GiB,
**`CUOPT_BARRIER_ITERATIVE_REFINEMENT=1`** (see note 1).

```bash
python cuopt_socp_resolve.py --check --try-reuse                # build from npz
python cuopt_socp_resolve.py --mps data/portfolio_socp_n5000_k50.mps   # same numbers from the MPS file
python cuopt_socp_resolve.py --ir 0                             # reproduce the IR=0 instability
python clarabel_socp_resolve.py --check                         # needs clarabel_gpu on PYTHONPATH
python clarabel_socp_resolve.py --ws 0 --ir-adaptive 0 --equil 0    # plain factorization-reuse baseline
```

## Reference results (per-step re-solve, ms; all with the settings above)

| GPU | ClarabelGPU re-solve | ClarabelGPU cold | cuOpt cold re-solve | ratio |
|---|---|---|---|---|
| L20 | 38.2 | 47.8 | 307 | 8.0x |
| B200 | 28.3 | 35.3 | 226 | 8.0x |
| A100 80GB | 40.5 | 42.8 | 330 | 8.2x |
| RTX PRO 6000 Blackwell | 29.8 | 36.7 | 205 | 6.9x |

The two scripts in this package were validated on L20 (2026-09-23): ClarabelGPU 38.1 ms re-solve / 47.3 ms cold,
cuOpt 343 ms cold with IR=1 (IR=1 costs ~10% over the IR=0 numbers in the table above). Absolute numbers depend on
host CPU, driver and GPU, so compare the two solvers from the same session; the ratio is what is stable.

Precision at tol 1e-8 (true relative objective error vs CPU Clarabel @1e-12, f* = -0.098364846996):
ClarabelGPU 2.6e-09; cuOpt IR=1 2.5e-09; cuOpt IR=0 2.8e-07 (and see note 1). Objectives with IR=1: cuOpt
-0.098364846753, ClarabelGPU -0.098364846740.

## A100 session, 2026-10-02: Clarabel warm re-solve vs cuOpt cold default

Same 20 `mu` steps (`resolve_seq.npz`). ClarabelGPU ran in container `cufolio-dev` (`cufolio:py3.12`) via `/tmp/socp_bench/clarabel_socp_resolve.py` with the script defaults: `tol_feas = tol_gap_abs = tol_gap_rel = 1e-8`, `warm_start_enable=True`, `ir_adaptive=True`, `equilibrate_enable=True`. Those re-solves are `update_q` with the previous iterate reused, not cold solves. Clarabel's base cold solve, before the sequence, was 23 iterations, 52.65 ms, objective −0.098364845819.

cuOpt cold default is `cuopt_socp_resolve_cache.py --configs default --cold-only` (fresh `DataModel` per `mu`, `sequence_solve` off):

- method Barrier, augmented system, iterative refinement on, crossover off
- `barrier_dual_initial_point` automatic (−1), which this cone model resolves to SeDuMi-μ
- `barrier_step_scale` 0.9 (the default; the tuned config's 0.99 is not used here)
- `barrier_presolve_bound_free_variables` 0
- `barrier_csr_ir_matvec` 0, `cudss_deterministic` off
- `set_optimality_tolerance(1e-8)` sets the six primal/dual/gap tolerances. On a cone the barrier still stops the objective gap at `barrier_relative_objective_gap_tol` = 1e-6 (relative). The 1e-8 value is copied onto the relaxed gap tolerance, used only for a suboptimal return
- RMM pool 1 GiB

`barr_ms` is cuOpt's barrier timer. Clarabel's column is its internal `solve_time`. Obj delta is Clarabel minus cuOpt. Every step was solved / optimal. The largest objective difference, 4.5e-7, is inside that 1e-6 relative gap allowance.

| t | Clara it | cuOpt it | Clara ms | cuOpt ms | Clara obj | cuOpt obj | obj delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22 | 23 | 52.24 | 302.0 | −0.098563711172 | −0.098563656886 | −5.43e-08 |
| 1 | 22 | 22 | 52.01 | 250.0 | −0.098508019036 | −0.098507869614 | −1.49e-07 |
| 2 | 20 | 22 | 48.05 | 227.0 | −0.098668251839 | −0.098668133384 | −1.18e-07 |
| 3 | 20 | 22 | 48.06 | 230.0 | −0.098823647183 | −0.098823386234 | −2.61e-07 |
| 4 | 19 | 22 | 46.06 | 227.0 | −0.098492897498 | −0.098492443729 | −4.54e-07 |
| 5 | 20 | 23 | 50.46 | 230.0 | −0.098754449733 | −0.098754341540 | −1.08e-07 |
| 6 | 17 | 22 | 45.82 | 225.0 | −0.098474176334 | −0.098473843010 | −3.33e-07 |
| 7 | 18 | 23 | 46.51 | 232.0 | −0.098553327151 | −0.098553256025 | −7.11e-08 |
| 8 | 19 | 22 | 46.06 | 227.0 | −0.098707702192 | −0.098707407884 | −2.94e-07 |
| 9 | 21 | 23 | 50.09 | 235.0 | −0.098388850483 | −0.098388793689 | −5.68e-08 |
| 10 | 17 | 23 | 42.11 | 233.0 | −0.098566863561 | −0.098566803932 | −5.96e-08 |
| 11 | 17 | 22 | 42.02 | 225.0 | −0.098488444745 | −0.098488234983 | −2.10e-07 |
| 12 | 19 | 22 | 45.99 | 225.0 | −0.098505708962 | −0.098505321629 | −3.87e-07 |
| 13 | 17 | 23 | 44.49 | 230.0 | −0.098419540402 | −0.098419428631 | −1.12e-07 |
| 14 | 15 | 23 | 40.59 | 229.0 | −0.098597566926 | −0.098597499608 | −6.73e-08 |
| 15 | 16 | 22 | 42.46 | 227.0 | −0.098633143898 | −0.098632897210 | −2.47e-07 |
| 16 | 19 | 22 | 46.07 | 231.0 | −0.098596171107 | −0.098595760038 | −4.11e-07 |
| 17 | 15 | 22 | 40.54 | 226.0 | −0.098524051315 | −0.098523691569 | −3.60e-07 |
| 18 | 15 | 23 | 40.46 | 229.0 | −0.098379007806 | −0.098378878335 | −1.29e-07 |
| 19 | 14 | 22 | 36.05 | 229.0 | −0.098662547718 | −0.098662137910 | −4.10e-07 |

Means over the 20 steps: Clarabel 18.1 iterations and 45.3 ms; cuOpt cold default 22.4 iterations and 233 ms.

## Two notes for the cuOpt team

1. **`barrier_iterative_refinement=0` is unstable on this instance.** 10 cold solves at tol 1e-8:
   A100 4–6/10 `NumericalError`, RTX PRO 6000 3/10. Residuals reach ~1e-8 at iteration 12, the solver keeps
   going, dual infeasibility grows four orders of magnitude, NaN at ~18, and the returned primal is an all-zero
   vector of the *internal* length (10102), not the user length. IR=1: 0/10 failures. `CUDSS_DETERMINISTIC=1`
   alone only gets to 1/10. (Our earlier QP scripts had IR=0; on QP it makes no difference.)
2. **Reuse is off for any problem with a quadratic constraint**, so the 20-step sequence costs 20 cold solves
   (~200–350 ms each) versus ~16–23 ms per re-solve for the QP with cache reuse. `update_linear_objective`
   then also fails the length check because the cached column count is the internal 10102.

## Running the ClarabelGPU side

`clarabel_gpu` is our internal GPU interior-point solver (DiffSolver repo, branch `jershi/bench-hooks` @ 18d22af,
built with CUDA 13.x for sm_80/86/89/90/100/120). It is not pip-installable; if you want to run that side, we can
give you the `.so` plus its conda env spec, or run it for you on a GPU of your choice. The script is included mainly
so the methodology is explicit and auditable.
