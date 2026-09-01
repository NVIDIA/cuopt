# Barrier sequence-update APIs

QP barrier cache reuse on `DataModel`, opt-in via `settings.sequence_solve = True`.
Public solve stays `Solve(dm, settings)`. Updates write original-problem data onto the
same `DataModel`, crush into the cached barrier workspace, then the next `Solve`
skips convert / presolve / scaling.

QP-only (quadratic objective, no quadratic constraints / SOC). First solve must
be Optimal. Mehrotra-start each reuse (does **not** reuse the last iterate).
Variable bounds, row senses, and problem size stay fixed.

Naming: **`update_linear_objective` updates linear `c`, not `Q`.** `update_P` is the quadratic
matrix (not implemented yet).

---

## Shared first solve (all APIs)

Required once before any update:

```text
settings.sequence_solve = True
Solve(dm, settings)   # Optimal
```

On Optimal, the cache stores:

| Saved | Role |
|-------|------|
| `iteration_data_t` | GPU IPM workspace: `c`/`d_c_`, `b`/`d_b_`, `A`, `Q`, `chol` (symbolic done), ADAT/augmented CSRs |
| `barrier_transform_t` | Frozen user↔barrier maps from that first convert / presolve / scaling |
| `barrier_lp` | Scaled equality LP the barrier actually solved |

**Maps in `barrier_transform_t` (frozen at t=0):**

| Map | Used for |
|-----|----------|
| `row_sense` | Convert `G` rows (`≥`) to `L` (`≤`) by negating RHS (and originally A) |
| `presolve_info.negated_variables` | Flip `c` (and later `Q` columns) for `-∞ < x ≤ u` |
| `presolve_info.remaining_variables` | Drop empty / fixed columns |
| `presolve_info.remaining_constraints` | Keep non-empty rows (gather for crush) |
| `presolve_info.removed_constraints` | Empty rows dropped at t=0; crush checks new `b` for infeasibility |
| `presolve_info.free_variable_pairs` | LP free-var split `x = v − w` (QP usually uses `direct_free_variables` instead) |
| `presolve_info.direct_free_variables` | QP free vars kept in the KKT (not split) |
| `column_scales` | `c_bar = c_crushed / col_scale` (Ruiz / column scaling) |
| `row_scales` | `b_bar = b_crushed / row_scale` |
| `linear_obj_shift` | `c_bar += shift` for terms convert/presolve added (e.g. `Q·ℓ` after bound shift) |
| `rhs_shift` | `b_bar += shift` for terms convert/presolve added (e.g. fixed-var `b -= A_{:j} ℓ_j`) |
| `rhs_update_supported` | False if first solve had range rows or folding |

Reuse **does not** re-decide which rows/cols presolve would drop. If a later
`P`/`A`/`b` would have changed that, the cheap path is wrong.

Next `Solve` when `cache.dirty()`:

1. Slim `user_problem_from_transform` (sizes + `c`; dummy `Q_values={1}`; **rhs left 0** — crush already wrote barrier `b`).
2. Log `Barrier: reusing cache (skip convert/presolve/scaling)`. If `update_b` marked an empty row infeasible, return `INFEASIBLE` here and skip IPM.
3. `barrier_advanced_solve` → `prepare_for_reuse` (reset `D`, `form_adat/aug(false)`, invalidate numeric factor, keep symbolic).
4. `run_ipm` Mehrotra-starts with the updated vectors/matrices already in `iteration_data_t`.
5. Unscale / uncrush with the **same** maps. Optimal → `mark_clean()`. Failure → `cache.clear()`.

---

## Status

| API | User data | Status |
|-----|-----------|--------|
| `update_linear_objective(c)` | Linear objective `c` | **Done** (shipped on cache-reuse branch) |
| `update_b(b)` | Constraint RHS `b` | **Coded, not verified** (build/install interrupted; no bench) |
| `update_P(Q)` | Quadratic objective values, same nnz pattern | **Not started** |
| `update_A(A)` | Constraint matrix values, same nnz pattern | **Not started** |
| Pattern-changing `A` or `P` | New CSR structure | **Not started** (would need new symbolic / maybe hash) |

Also remaining for `update_b`: rebuild Python (`./build.sh cuopt --install`), a small QP sequence test, and decide whether the dummy `user_problem.rhs = 0` on reuse is fine (it is, because uncrush does not read `b` and IPM reads `iteration_data.b`).

---

## `update_linear_objective` — linear `c` (done)

### Maps used

`negated_variables`, `remaining_variables`, `free_variable_pairs`, `column_scales`, `linear_obj_shift`.

Does **not** use row maps.

### On `update_linear_objective(c)`

1. Crush user `c` through the maps above → barrier-length vector.
2. Add `linear_obj_shift`.
3. Write `iteration_data.c` and `d_c_`.
4. Set `c_dirty`.
5. Store user `c` on the DataModel (after crush succeeds).

`A`, `Q`, `b`, bounds unchanged.

### On next `Solve`

Skip convert/presolve/scaling. `prepare_for_reuse` rebuilds KKT **values** from existing `A`/`Q` and reset `D` (symbolic stays). IPM uses the new `c`. Dual residual / objective see the new linear term.

---

## `update_b` — RHS (coded)

### Maps used

`row_sense`, `remaining_constraints`, `removed_constraints`, `row_scales`, `rhs_shift`.

Does **not** use column maps. `G` rows: crush stores \(b_i \leftarrow -b_i\) (same as convert).

One-sided / equality rows (`E` / `L` / `G` via `set_row_types` + `set_constraint_bounds`) are the intended case.

### On `update_b(b)`

1. If `!rhs_update_supported`, throw (range or folding at t=0).
2. Crush user `b` (`G` negate → check removed empty rows → keep remaining rows → divide `row_scales`). Empty-row infeasible → set `rhs_infeasible`, do not write barrier `b`.
3. Else add `rhs_shift` (fixed/lower-bound affine terms from t=0; independent of new `b` while `A` and bounds stay fixed).
4. Write `iteration_data.b` / `d_b_` and `barrier_lp.rhs`.
5. Set `b_dirty`.
6. Store user `b` on the DataModel.

`A`, `Q`, `c`, bounds unchanged. KKT sparsity unchanged.

### On next `Solve`

Same skip path as `update_linear_objective`, unless `rhs_infeasible`: then `INFEASIBLE` immediately (no IPM). Cache stays so a later feasible `update_b` can reuse again. Otherwise IPM primal residual is \(Ax - b\) with the new `b`. `norm_b` is recomputed from `data.b` at the start of `run_ipm`.

### Limitations

Reuse **does not re-run convert/presolve**. Whatever slacks, dropped rows, and reduced space t=0 chose stay.

#### Range rows — `rhs_update_supported = false`

A range row is a **two-sided finite** constraint \(h \le a^\top x \le u\), not “variable bound 0 vs \(\infty\)”.

Detection at t=0 (DataModel path): both constraint lower and upper finite and unequal → `range_rows` / `range_value = u - \ell`. Convert then fills `new_slacks`. `update_b` does **not** inspect infinities; it only sees `new_slacks.empty()` from that first convert.

Convert does **not** keep that interval as a single `b` entry. It rewrites

\[
a^\top x - s = 0, \qquad h \le s \le u
\]

with \(h,u\) computed from **this solve’s** `b`, range `r`, and row sense. Barrier **RHS for that row is 0**. The user’s `b` lives on **slack bounds**.

`update_b` only patches barrier `b`. It never updates `s` lower/upper. A new user `b` would need new \(h',u'\) on `s`. We refuse rather than write a nonzero onto a row defined as \(a^\top x - s = 0\).

Updating `s` is possible later if we store `r` and the slack column index; that is bound updates, not `update_b`.

Ordinary `'E'` / `'L'` / `'G'` (one finite side, or equality) are fine.

#### Empty rows removed at t=0 — reuse, detect infeasibility

Presolve drops rows with \(A_{i,:} = 0\). Those original indices are `removed_constraints`. Crush only gathers `remaining_constraints`, so a later `b` on a dropped row would never reach the barrier.

- \(0 = b_i\) (`E`): infeasible iff \(b_i \neq 0\)
- \(0 \le b_i\) (`L`, or `G` after negate): infeasible iff converted RHS \(< 0\)

If every dropped row stays feasible, reuse is valid (same as t=0 dropping them again). If any is infeasible, `update_b` sets a flag and the next `Solve` returns **infeasible**. A full solve would also be infeasible; it does not recover a feasible model.

#### Folding — `rhs_update_supported = false`

Folding is an LP **symmetry** presolve (`folding != 0`, **`Q.n == 0`**, no cones). QP `update_b` usually never folds. We still refuse if `folding_info.is_folded` so a folded LP cache cannot be patched.

It is **not** “delete some rows.” It rebuilds a smaller LP whose rows/columns are **aggregates**:

\[
A' = C_s^\top A D, \quad b' = C_s^\top b, \quad c' = D^\top c
\]

`C_s` and `D` live on `folding_info`. Barrier row \(k\) is **not** original row \(i\). Crush’s gather (`remaining_constraints` then `row_scales`) would write the wrong \(b'\).

Folding also requires original rows in the same color to share one RHS. A new `b` can break that coloring; then the saved fold is invalid.

To support it later: \(b' = C_s^\top b_{\text{new}}\) and re-check equal-RHS-per-color; otherwise full-solve. That is a different map than `update_linear_objective` / one-sided `update_b`.

| | Empty-row drop | Folding |
|--|----------------|---------|
| Barrier row \(k\) | some original row \(i\) | mix of original rows |
| New `b` | gather + check dropped rows | must use \(C_s\); coloring may die |
| v1 | detect infeasible empty rows | refuse |

Also assumed even when supported: same \(m\), same row senses, same `A` and variable bounds, QP-only, first solve Optimal with `sequence_solve`, bound-free-var presolve off. A new `b` that *would* have changed other presolve decisions (not empty-row drop) is still ignored.

---

## `update_P` — quadratic values, same pattern (todo)

### Maps to use

Same **column** maps as `c`: `negated_variables`, `remaining_variables`, `free_variable_pairs`, `column_scales`.

Scale: `Q_bar_ij = Q_user_ij / (col_scale_i * col_scale_j)` (and obj-scale if any), with sign flips on negated columns.

Also recompute or refresh **`linear_obj_shift`** if convert used a bound shift (`½xᵀQx` produces a linear term in `c`).

### On `update_P`

1. Require same CSR pattern as t=0 (`indptr`/`indices`).
2. Crush `Q.data` into `iteration_data.Q` / `Qdiag` / `d_Q_diag_` / `cusparse_Q_view_`.
3. Mark dirty.
4. Drop the dummy `Q_values = {1}` gate; reuse must know this is still a QP.

**No sparsity hash** if pattern is unchanged.

- ADAT: `Q` is diagonal → only `Qdiag` in `D`; `AAᵀ` pattern unchanged.
- Augmented: off-diagonal `Q` lives in the KKT; same `Q` nnz → same symbolic.

Diagonal ↔ non-diagonal switches ADAT ↔ augmented: **full solve / new analyze**, not a hash.

### On next `Solve`

Skip convert/presolve/scaling. `prepare_for_reuse` already calls `form_*(false)` using `Qdiag` / `device_Q`. Symbolic stays. New Mehrotra start.

---

## `update_A` — constraint values, same pattern (todo)

### Maps to use

**Both** row and column maps: `row_sense` (negate `G` rows of `A`), `remaining_variables` / `remaining_constraints`, `column_scales`, `row_scales`.

`A_bar_ij = A_user_ij / (row_scale_i * col_scale_j)` after convert/presolve gather.

If t=0 had fixed-var RHS shifts, **`rhs_shift` depends on `A`**. Same-sparsity `A` value changes that hit those columns need a new shift, or those models are unsupported (same class as range/removed rows).

### On `update_A` (same CSR)

1. Crush `A.data` into `barrier_lp.A` **and** every device copy (`original_a_values`, `a_mat`, cusparse views, ADAT inputs).
2. Mark dirty.
3. Keep `chol` symbolic (`AAᵀ` / augmented pattern unchanged).

### On next `Solve`

Skip convert/presolve/scaling. `form_adat/aug(false)` rebuilds numeric KKT values from the new `A`. Symbolic stays.

### Later: pattern change

New `A` (or `Q`) nnz → new KKT sparsity → new `analyze`. Frozen t=0 presolve maps are likely invalid. Options: reject (full solve), or bring back sparsity hash only as “reuse analyze if KKT pattern matches a previous one.”

---

## Suggested order

1. Finish `update_b`: install, QP sequence test (`update_b` + `cache_reuse` log + obj check).
2. `update_P` same pattern.
3. `update_A` same pattern (watch all A buffers + `rhs_shift`).
4. Pattern-changing `A`/`P` only if needed.
