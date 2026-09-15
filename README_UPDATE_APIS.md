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
3. `barrier_advanced_solve` → `reset_iterate_state` (reset `D`, `form_adat/aug(false)`, invalidate numeric factor, keep symbolic).
4. `run_ipm` Mehrotra-starts with the updated vectors/matrices already in `iteration_data_t`.
5. Unscale / uncrush with the **same** maps. Optimal → `mark_clean()`. Failure → `cache.clear()`.

---

## Status

| API | User data | Status |
|-----|-----------|--------|
| `update_linear_objective(c)` | Linear objective `c` | **Shipped, two known silent bugs** — see "Suggested order" item 2 |
| `update_rhs(b)` | Constraint RHS `b` | **Done** (verified against fresh full solves) |
| `update_P(Q)` | Quadratic objective values, same nnz pattern | **Not started** |
| `update_A(A)` | Constraint matrix values, same nnz pattern | **Not started** |
| Pattern-changing `A` or `P` | New CSR structure | **Not started** (would need new symbolic / maybe hash) |

Named `update_rhs` (not `update_b`) to match the C++ cache method.

Verified on a QP with a `G` row (`min xᵀx` s.t. `x0 + x1 >= b`): two successive
`update_rhs` re-solves hit the reuse path, skipped presolve / reordering / symbolic
factorization, and matched a fresh full solve exactly. The dummy `user_problem.rhs = 0`
on reuse is fine, because uncrush does not read `b` and IPM reads `iteration_data.b`.
Committed coverage lives in
`python/cuopt/cuopt/tests/linear_programming/test_barrier_sequence_solve.py`. It compares
every cached re-solve against a fresh full solve and asserts the reuse log line, so a test
cannot pass while the gate rejects the model and falls back. `update_linear_objective` is
only exercised there as a way to dirty the cache for the free-variable gate test; it still
has no test of its own correctness, which is how both bugs above went unnoticed.

### Free-variable bounding and the reuse gate

Presolve can give free variables implied bounds derived from the rows, recording them in
`presolve_info.bounded_free_variables`. That state is not something the reuse path can
replay, so a cache built with it is not reusable.

`sequence_solve = True` resolves the `-1` (automatic) default of
`barrier_presolve_bound_free_variables` to `0` in `run_barrier`, so callers no longer have to
set the parameter themselves — reuse fires out of the box. Passing an explicit `1` is
honored, and those solves simply do not get reuse.

The gate checks the cache as well as the setting. Keying on the setting alone was a bug: it
read the value of the solve asking for reuse, not the one the cache was built with, so a
first solve at `-1` that bounded a free variable could be picked up by a later solve passing
`0`, and the IPM then failed to converge on the mismatched workspace.

Cost: models with genuine free variables give up that presolve step on their first solve in
exchange for reuse on every later one.

### Known gap — plain setters do not invalidate the cache (future TODO)

The reuse gate validates only shape (`num_cols` / `num_rows`), never content. The
non-cache-aware setters — `set_constraint_bounds`, `set_csr_constraint_matrix`,
`set_quadratic_objective_matrix`, and the variable/constraint bound setters — write the
DataModel without touching the cache or its dirty flags. So a sequence like

```text
dm.update_rhs(b_new)          # sets b_dirty
dm.set_csr_constraint_matrix(...)  # cache not told
Solve(dm, settings)           # reuses cached A, silently ignores the new one
```

returns a confidently wrong answer rather than an error. Documented in each `update_*`
docstring but not enforced. Options when we pick this up: clear the cache from those
setters, raise from them while a cache is live, or extend the gate to compare `A`/`Q`/bounds
content. This gets more important with every update API added.

A content fingerprint is the cheapest version of the third option, and the pieces exist:
`cpp/src/utilities/hashing.hpp` has FNV-1a `compute_hash`, and MIP already fingerprints a
whole problem with it (`mip_heuristics/problem/problem.cu:2315`). Hash the user data at the
cold solve, re-check on reuse, refuse on mismatch. Widen the hash or confirm with an exact
compare before trusting it as a correctness gate — a collision here is a wrong answer, not a
slow one.

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

**Where the difficulty actually is.** Not the GPU plumbing. `c` and `b` could freeze their
shift vectors because neither changes them, so `linear_obj_shift` is simply *measured* once as
`barrier_objective - crush(c)`. Q breaks that: the bound translation `x = x' + ℓ` puts `Qℓ`
into the linear objective and `½ℓᵀQℓ` into `obj_constant`, so both are functions of Q and have
to be **derived**, matching presolve's arithmetic (`presolve.cpp:1323-1342`) closely enough
that a fresh solve agrees. Get it wrong and the failure is the quiet one: correct solution
vector, wrong reported objective.

Both terms are *linear in Q's entries*, which is what makes a delta against the cached Q
workable: `ΔQ̄ = crush_user_Q(new) − barrier_lp->Q.x` is positional once the pattern is
pinned, and `ℓ` in crushed coordinates is `column_scales[j] * removed_lower_bounds[j]`.

### Stage 1 — pin the pattern

- [ ] Store the user Q CSR `offsets` / `indices` on `barrier_transform_t` at the cold solve.
      Nothing records them today, so there is nothing to validate an update against.
- [ ] Reject a caller whose pattern differs; same pattern also keeps `Q.is_diagonal()` stable,
      so `use_augmented` cannot flip under us.

### Stage 2 — `crush_user_Q`

Column maps only, but applied to **both** indices of each entry.

- [ ] Sign flip `Q_ij` iff exactly one of `i,j` is in `negated_variables`
      (`presolve.cpp:1271-1285`).
- [ ] Two-sided gather through `remaining_variables`.
- [ ] Scale. Two paths exist and they are not the same formula: Ruiz does
      `Q.x *= c[row]*c[col]` (`scaling.cpp:215`), standard does
      `Q.x / (column_scaling[row]*column_scaling[col])` (`scaling.cpp:293`). Either record
      which ran or store one effective factor. A diagonal entry scales by `col_scale_j²`.
- [ ] Feed the CSR→CSC + slack-padding conversion `create_Q` does (`barrier.cu:2442`).
- [ ] QP keeps free variables in the KKT via `direct_free_variables`, so no Q expansion.

### Stage 3 — move the objective shift and constant (the hard part)

- [ ] Shift the barrier linear objective by `ΔQ̄ ℓ` and `obj_constant` by `½ ℓᵀ ΔQ̄ ℓ`. No ½ on
      the first and a ½ on the second, both full-matrix sums, because Q is stored **full
      symmetric** under a `½xᵀQx` objective: presolve's two half-updates
      (`presolve.cpp:1333-1336`) sum by symmetry to `(Qℓ)_k`, its constant term
      (`presolve.cpp:1339`) to `½ℓᵀQℓ`, and the reported objective uses `0.5 * xTQx`
      (`barrier.cu:4162`).
- [ ] Order must not matter: `update_P` and `update_linear_objective` both write the barrier
      objective, so whatever each reads as its baseline has to survive the other.
- [ ] This is also where `update_linear_objective`'s own `obj_constant` staleness has to be
      fixed first — the same `ℓ` drives both, and Q's delta lands on top of it.

### Stage 4 — push to buffers (cheap; free for diagonal Q)

- [ ] Write host `barrier_Q` and `barrier_lp->Q`.
- [ ] Recompute `Qdiag` and `d_Q_diag_`. These are built **once** in the `iteration_data_t`
      constructor (`barrier.cu:599-638`) and never refreshed; `reset_iterate_state` reads the
      cached copy, so a new Q is invisible without this.
- [ ] Re-run the PSD check. Diagonal Q is just `Qdiag[j] >= 0`; the general case has a TODO
      where the check should be (`barrier.cu:631`). Decide the rejection path for an
      indefinite update.
- [ ] Diagonal Q stops here: `reset_iterate_state` already refolds `Qdiag` into `diag`.
- [ ] Non-diagonal Q also needs `device_Q_csc_`, the `cusparse_Q_view_` device copy, and the
      **off-diagonal Q entries inside `device_augmented.x`** — `form_augmented(false)` only
      rewrites diagonals (`barrier.cu:1085-1103`), so those keep first-build values.

### Gates

- [ ] Refuse when `presolve_info.removed_variables` is non-empty: the empty-column rule fixes
      a variable by minimizing `c_j x_j + ½ q_jj x_j²` (`presolve.cpp:105-132`), a decision
      frozen against the old Q *values*.
- [ ] Drop the dummy `Q_values = {1}` on the reuse path so the gate still sees a QP.

### Tests

Mirror `test_barrier_sequence_solve.py`: oracle against a fresh full solve, assert the reuse
log line, and cover diagonal Q, non-diagonal Q, non-unit column scaling, and — the one that
catches a botched stage 3 — **nonzero variable lower bounds**. Mutation-check each.

### Naming

The code symbol is `iteration_data_t::reset_iterate_state`, not `prepare_for_reuse`.

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

### Rejected: rebuild A and reuse only the symbolic factorization

Tempting, because redoing convert / presolve / scaling recomputes `rhs_shift`, the
equilibration, and every presolve decision, which deletes the whole staleness bug class and
the device-buffer surface at once — a pattern hash over the rebuilt `row_start` / `j` would
then say whether `analyze` can be kept.

Not worth it: symbolic is only ~10% of a solve on the ADAT path and ~20-30% on augmented, so
this keeps roughly half the benefit of surgical reuse in exchange for a large restructuring.
Revisit only if those fractions change.

### Booby trap

`form_adat(false)` restores `device_AD.x` from the `d_original_A_values` snapshot on **every**
call (`barrier.cu:1156`). An `update_A` that misses that one buffer gets silently reverted on
the next reuse and returns answers from the old `A` with no error anywhere.

---

## Suggested order

1. ~~`update_rhs`~~ — done, with sequence_solve tests.
2. Fix `update_linear_objective`, which has two known bugs of its own, both silent. The
   translation `x = x' + ℓ` folds `Σ c_j ℓ_j` into `obj_constant`, so an objective update on a
   model with nonzero lower bounds reports an objective short by `Σ (c_new − c_old)_j ℓ_j`
   while returning the right solution vector. Separately, the max → min rewrite negates `c`
   before the cache ever sees it, so crushing the user's own coefficients onto a maximize
   model minimizes `+cᵀx` and terminates Optimal on the wrong answer.
3. Close the setter-invalidation gap (content fingerprint). Smallest job left, and it removes
   the sharpest edge: today a plain setter plus `Solve` returns a confidently wrong answer.
4. `update_P` same pattern. Stage 3 (moving `Qℓ` and `½ℓᵀQℓ`) is the real work and builds on
   item 2; the GPU side is nearly free for diagonal Q.
5. `update_A` same pattern (watch all A buffers + `rhs_shift` + the `d_original_A_values`
   trap).
6. Pattern-changing `A`/`P` only if needed.
