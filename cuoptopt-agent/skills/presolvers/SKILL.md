---
name: presolvers
version: "26.04.00"
description: LP and MILP presolver techniques — bound tightening, probing, redundancy removal, coefficient strengthening, aggregation, clique detection. GPU parallelization strategies.
---

# Presolvers for LP and MILP

A presolver reduces problem size and tightens formulations before the main solver
runs. Effective presolving can reduce solve time by 10–1000× on structured problems.

---

## Why Presolve Matters

- Removes redundant constraints → smaller LP relaxation → faster simplex/IPM
- Tightens variable bounds → smaller branch-and-bound tree
- Detects infeasibility early → avoids wasted solve time
- Exploits problem structure that the main solver cannot see

---

## Core Presolver Techniques

### 1. Bound Tightening (Feasibility-Based)

For constraint `aᵢxⱼ + Σₖ≠ⱼ aᵢₖxₖ ≤ bᵢ`:

**Upper bound tightening** on xⱼ (aᵢⱼ > 0):
```
xⱼ ≤ (bᵢ - Σₖ≠ⱼ min_contribution(aᵢₖ)) / aᵢⱼ
```

**Lower bound tightening** (aᵢⱼ < 0):
```
xⱼ ≥ (bᵢ - Σₖ≠ⱼ min_contribution(aᵢₖ)) / aᵢⱼ
```

Repeat until fixed point (typically 3–10 passes for dense problems).

**GPU parallelization:**
- Each thread handles one constraint; parallel bound update with atomic min/max.
- For dense systems: launch one thread per (constraint, variable) pair.
- Use warp-level `__reduce_min_sync` / `__reduce_max_sync` for per-constraint sums.

### 2. Probing (MILP Only)

Fix each binary variable xⱼ ∈ {0, 1} and propagate:
- Run bound tightening with xⱼ = 0 → record bounds L₀, U₀
- Run bound tightening with xⱼ = 1 → record bounds L₁, U₁
- Merge: final lⱼ = max(L₀, L₁), final uⱼ = min(U₀, U₁)
- If lⱼ > uⱼ: infeasible fixing detected

**GPU:** Launch one thread block per binary variable; each block runs full propagation.

### 3. Redundant Constraint Removal

Constraint `i` is redundant if, given current variable bounds, it is always satisfied:
```
Σⱼ max(aᵢⱼ lⱼ, aᵢⱼ uⱼ) ≤ bᵢ   →   constraint i always satisfied → remove
```

**Dense systems:**  
For relatively dense matrices (density > 10%), the activity computation is a dense
matrix-vector product: `activity = A × bounds_max`. Use cuBLAS `cublasDgemv` to
compute all constraint activities in one call, then compare to rhs vector.

### 4. Singleton Rows and Columns

- **Singleton row**: constraint with exactly one variable → directly fixes variable or tightens bound.
- **Singleton column** (free column): variable appears in only one constraint → substitute out.
- Detect via row/column nnz count vector; parallelize with `thrust::count_if`.

### 5. Coefficient Strengthening (MILP)

For constraint `Σ aᵢⱼ xⱼ ≤ bᵢ` with xⱼ ∈ {0,1}, if `aᵢⱼ > bᵢ`:
- Strengthen: replace `aᵢⱼ` with `bᵢ` (tighter constraint, same feasible set).
- Reduces LP relaxation gap.

### 6. Aggregation / Substitution

When two constraints uniquely determine one variable:
```
a₁x + b₁y = c₁
a₂x + b₂y = c₂
```
Solve for x, substitute into all other constraints → eliminate x.
Applicable to equality-constrained LP problems.

### 7. Clique Detection (MILP)

For binary variables with constraint `Σⱼ∈S xⱼ ≤ 1`:
- Variables in S form a clique in the conflict graph.
- Replace individual bounds with clique inequalities (tighter LP relaxation).
- Detect via graph coloring or maximal clique algorithms.

---

## Dense System Strategies (the cuOpt-Specific Case)

For "relatively dense" constraint matrices (e.g., LP relaxations of routing problems):

1. **Reformulate as GEMM**: Activity = A × x_bounds is dense matrix-vector multiply.
   Use `cublasDgemv` for baseline; or `cublasHgemm` with TF32/FP16 for speed.

2. **Parallel row processing**: Each SM owns a block of rows; processes bound
   tightening passes in shared memory before writing back to global bounds vector.

3. **Convergence detection**: Use a flag in global memory (`bool changed`).
   Atomic `OR` within each pass; stop when `changed == false`.

4. **Column-major vs row-major**: Bound tightening per constraint → row access →
   row-major layout preferred. Activity sum per row is a dot product.

---

## Presolver Pass Ordering

Standard industry ordering (Achterberg et al., 2007):
1. Remove fixed variables
2. Remove redundant rows
3. Singleton rows/columns
4. Bound tightening (3–5 passes)
5. Probing (MILP only)
6. Coefficient strengthening (MILP only)
7. Clique detection (MILP only)
8. Repeat until no further reduction

---

## Parallelism Summary

| Technique | Parallelism | GPU Approach |
|-----------|------------|--------------|
| Bound tightening | Per constraint | One thread block per row |
| Redundancy removal | Per constraint | cublasDgemv + compare |
| Probing | Per binary variable | One block per variable |
| Singleton detection | Global scan | thrust::count_if |
| Clique detection | Graph algorithm | Graph coloring |

---

## Key References

- Achterberg, T. et al. (2007). "Presolving Mixed Integer Linear Programs." ZIB Report.
- Gamrath, G. et al. (2015). "Progress in Presolving for Mixed Integer Programming."
- Lodi, A. (2010). "Mixed Integer Programming Computation." Springer.
