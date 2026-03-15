---
name: mip-algorithms
version: "26.04.00"
description: MILP solving algorithms — branch-and-bound, cutting planes (Gomory, cover, clique), LP relaxations, primal heuristics, diving, column generation. GPU acceleration strategies.
---

# MILP Algorithms

Reference for understanding and improving cuOpt's Mixed-Integer Linear Programming solver.

---

## Branch-and-Bound (B&B)

The standard exact MILP algorithm.

### Core Loop
```
1. Solve LP relaxation at current node
2. If LP is infeasible → prune (infeasibility)
3. If all integer variables are integer-valued → update incumbent (feasibility)
4. If LP bound ≥ incumbent → prune (bound)
5. Select branching variable (fractional integer variable)
6. Branch: create two children with xⱼ ≤ ⌊x*ⱼ⌋ and xⱼ ≥ ⌈x*ⱼ⌉
7. Add children to the tree; recurse
```

### Branching Variable Selection
- **Most fractional**: branch on variable closest to 0.5 → balanced tree
- **Strong branching**: evaluate both children's LP bounds before committing → expensive but better
- **Pseudo-costs**: cheap estimate of strong branching by history → good default
- **Reliability branching**: blend strong + pseudo-cost based on observation count

### Node Selection
- **Best-first**: minimize LP bound → fastest convergence, high memory
- **Depth-first**: reduce memory, find feasible solutions quickly
- **Best-estimate**: mix bound + depth heuristic

### GPU Parallelism for B&B
- **Tree parallelism**: solve multiple nodes simultaneously on different SMs.
  Requires a shared work queue (GPU work-stealing or atomic queue).
- **Node parallelism**: solve the LP relaxation at each node using GPU-accelerated simplex/IPM.
- cuOpt approach: parallelize LP solves; B&B tree management on CPU.

---

## Cutting Planes

Cuts tighten the LP relaxation without removing integer-feasible solutions.

### Gomory Cuts (General MILP)
Derived from the optimal LP tableau:
```
For row i: Σⱼ f̄ᵢⱼ xⱼ ≥ f̄ᵢ   (where f̄ = fractional part)
```
- Pure Gomory cuts close the integrality gap systematically.
- Fractional Gomory cuts are stronger; Mixed-Integer Gomory (MIG) cuts generalize.

### Cover Cuts (Binary Variables)
For knapsack constraint `Σⱼ aⱼxⱼ ≤ b` with binary xⱼ:
- A **cover** is a set C where `Σⱼ∈C aⱼ > b` (cannot all be 1).
- Cover inequality: `Σⱼ∈C xⱼ ≤ |C| - 1`
- **Lifted cover inequality**: stronger form using lifting coefficients.

### Clique Cuts
For conflict graph cliques: `Σⱼ∈K xⱼ ≤ 1`
- Stronger than individual binary bounds.
- GPU: build conflict graph via parallel edge detection; find cliques via greedy coloring.

### Flow Cover Cuts
For flow/capacity constraints with binary activation variables.
Highly effective for routing-type MILPs (cuOpt vehicle routing).

### Cut Selection Strategy
1. Generate all applicable cut types
2. Score cuts: violation × normalization factor
3. Select diverse, non-dominated cuts (limit to avoid LP density explosion)
4. Add cuts to LP; resolve

---

## LP Relaxation Solvers

### Simplex Method
- **Revised simplex**: maintain B⁻¹ (basis inverse); update via eta-factors.
- **Dual simplex** (standard for MIP): warm-starts efficiently after adding cuts/bounds.
- GPU: basis factorization (LU) is challenging to parallelize; row operations are parallelizable.

### Interior Point Method (IPM/PDLP)
- Primal-Dual Linear Programming (PDLP): first-order IPM suitable for GPU.
- cuOpt uses PDLP for LP relaxations; highly parallel: each iteration is sparse
  matrix-vector products.
- Warm-starting IPM after node branching is non-trivial; crossover to simplex needed
  for exact vertex solutions.

### GPU LP Speedup Targets
- Sparse matrix-vector (SpMV): cuSPARSE `cusparseSpMV`
- Dense GEMM for dense sub-blocks: cuBLAS
- Vector operations (dot products, norms): cuBLAS Level-1

---

## Primal Heuristics

Find good feasible solutions early to provide strong upper bounds.

### Rounding Heuristics
- Round fractional LP solution toward nearest integer.
- **Feasibility pump**: alternate between rounding and LP solving to find feasibility.
- **RINS** (Relaxation Induced Neighborhood Search): fix variables with same value
  in LP and incumbent; solve restricted MIP.

### Diving
Fix variables iteratively (deepest fractional → integer) and resolve LP at each step.
- Coefficient diving, fractional diving, guided diving (using pseudo-costs).
- Backtrack on infeasibility.

### Large Neighborhood Search (LNS)
- Fix a large fraction of variables to incumbent values; re-optimize the rest.
- **RENS**: fix variables near-integer in LP relaxation; solve remaining.
- GPU: parallel neighborhood evaluation.

---

## Column Generation

For LPs with many variables (pricing problem):
```
1. Start with subset of columns (variables)
2. Solve restricted LP (master problem)
3. Solve pricing subproblem: find most negative reduced cost column
4. If found: add column; repeat. Else: LP is optimal.
```
For cuOpt routing: columns = routes; pricing = shortest path (GPU-parallelizable).

---

## GPU Acceleration Summary

| Component | GPU technique |
|-----------|--------------|
| LP solve (PDLP) | SpMV (cuSPARSE) + BLAS-1 (cuBLAS) |
| Gomory cut separation | Parallel tableau row processing |
| Cover cut separation | Parallel knapsack evaluation |
| Clique detection | Parallel graph coloring |
| Primal rounding | Parallel floor/ceil + feasibility check |
| Node queue | GPU priority queue (atomic operations) |
| Branching variable | Parallel LP sensitivity analysis |
