# `update_rhs`: re-solving a QP after changing only the RHS

`DataModel.update_rhs(b)` lets you re-solve a barrier QP with a new constraint
right-hand side while skipping convert, presolve, scaling and the symbolic
factorization. `A`, `Q`, the bounds and the row senses must stay unchanged.

This is the RHS counterpart to the existing `update_linear_objective`. For the
full design notes, see `README_UPDATE_APIS.md`; this file is the short version.

## Using it

```python
from cuopt.linear_programming import data_model, solver, solver_settings

settings = solver_settings.SolverSettings()
settings.set_parameter("sequence_solve", True)

model = data_model.DataModel()
# ... set A, Q, bounds, row types, and the first RHS ...

first = solver.Solve(model, settings)      # full solve, fills the cache
assert first.get_termination_reason() == "Optimal"

model.update_rhs(new_b)                    # crush new_b into the cached workspace
second = solver.Solve(model, settings)     # reuses it
```

`update_rhs` always writes the RHS onto the `DataModel`, so the call is safe
even when no cache is in play; you just get a normal full solve next time.

To confirm you actually got the fast path, look for this in the solver log:

```
Barrier: reusing cache (skip convert/presolve/scaling)
```

Checking the log matters. A cache miss silently falls back to a full solve and
still returns the correct answer, so numbers alone will not tell you whether
reuse happened.

## What happens to your `b`

The cached solve keeps a `barrier_transform_t` holding the presolve and scaling
maps, which is enough to replay the original RHS pipeline on new numbers
without rerunning the algorithms that produced it.

```
user b (length = user_num_rows)
  |
  | 1. 'G' rows are negated          convert rewrites G rows as L by negating
  |                                  the row and its RHS
  |
  | 2. rows presolve dropped as      an empty row has no variables, so the new
  |    empty are feasibility-        b can only be checked, never applied:
  |    checked, not applied            'E' needs |b_i| <= primal_tol
  |                                     others need  b_i >= -primal_tol
  |                                  a violation is exact, not approximate:
  |                                  the next Solve reports PrimalInfeasible
  |                                  without running IPM, and keeps the cache
  |
  | 3. surviving rows are selected   remaining_constraints
  |
  | 4. divide by row_scales          undo equilibration
  |
  | 5. add rhs_shift                 constant recorded on the first solve for
  |                                  the x = x' + l translation and fixed
  |                                  variables: barrier rhs - crush(user b)
  v
barrier RHS -> barrier_lp->rhs and the device copy in iteration_data (b, d_b_)
```

Both halves have to be updated: the next solve rebuilds its solver from
`barrier_lp` and takes its Mehrotra starting point from `barrier_lp->rhs`,
while the IPM iterations read the device copy.

`obj_constant` is deliberately left alone. It depends on `c` and the translated
lower bounds, not on `b`.

## When reuse is refused

`update_rhs` raises when the cached maps cannot express a new RHS:

- **Range rows or folded rows.** `convert_range_rows` zeroes `rhs[i]` and moves
  the bounds onto the slack, and folding aggregates rows, so neither leaves the
  user RHS recoverable. Error: `cached convert used range rows or folding; run
  a full Solve.` Plain inequality and equality rows are fine, slacks and
  artificials included.
- **Wrong length.** Validated against the cached user row count.
- **No cache yet.** You need an `Optimal` solve with `sequence_solve` on first.

Reuse is also skipped, falling back to a full solve, when the model is not a QP
(a quadratic objective is required, quadratic constraints and second-order
cones are not supported), when the dimensions no longer match the cache, or
when the first solve's presolve bounded a free variable. That last case leaves
state in `presolve_info` the reuse path cannot replay, so the cache is refused
even if the later solve asks for `barrier_presolve_bound_free_variables = 0`.

Note that `barrier_presolve_bound_free_variables` defaults to `-1`
(automatic), which resolves to `0` when `sequence_solve` is on, so you normally
do not need to set it yourself.

## Where the code is

| Piece | File |
|---|---|
| `crush_user_rhs`, `rhs_shift`, `rhs_update_supported` | `cpp/src/barrier/barrier_transform.hpp` |
| `barrier_cache_t::update_rhs` | `cpp/src/barrier/barrier_cache.cu` |
| `apply_barrier_rhs` (host + device copies) | `cpp/src/barrier/barrier.cu` |
| Reuse gate, records `rhs_shift` / `primal_tol` | `cpp/src/dual_simplex/solve.cpp` |
| Matching gate one level up | `cpp/src/pdlp/solve.cu` |
| Python entry point | `python/cuopt/.../data_model/data_model{.py,_wrapper.pyx}` |
| Tests | `python/cuopt/cuopt/tests/linear_programming/test_barrier_sequence_solve.py` |

The two gates must stay in lockstep. The `pdlp` one also swaps in a slim
problem built from the transform, so if it allows reuse while the lower gate
refuses, presolve runs on a fabricated problem.

## Test coverage

Every test compares the reused solve against an oracle, a freshly built model
with the new RHS solved with default settings and no cache, and asserts the
reuse log line so a silent fallback cannot pass. Covered: mixed E/L/G senses,
row norms spanning seven orders of magnitude, a nonzero `rhs_shift`, dropped
empty rows both feasible and infeasible, cache survival across an infeasible
update, `obj_constant` staying put, and the free-variable gate with paired
positive and negative controls.

Not covered yet: the range-row and folding refusal, `maximize` combined with
`update_rhs`, composing `update_rhs` with `update_linear_objective`, and
anything beyond the Python API or larger than a few variables.
