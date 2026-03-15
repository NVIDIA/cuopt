---
name: research-literature
version: "26.04.00"
description: How the cuoptopt-agent research module searches academic literature, constructs queries, parses abstracts, and evaluates papers for applicability to cuOpt GPU optimization.
---

# Research Literature Search

The research agent automatically searches Google Scholar and arxiv for papers
relevant to the optimization query. This skill documents how that search works
and how to interpret and apply findings.

---

## Automated Search (cuoptopt-agent)

The `research.py` module runs two searches in parallel:

### Google Scholar (`scholarly`)
```python
from scholarly import scholarly
results = scholarly.search_pubs("GPU presolver linear programming dense constraints")
```
- Returns title, authors, year, abstract, URL.
- Rate-limited by Google; the agent uses exponential backoff with jitter.
- May be blocked by reCAPTCHA in cloud environments; use arxiv as fallback.

### arxiv (`arxiv` package)
```python
import arxiv
search = arxiv.Search(
    query="GPU presolver linear programming dense constraints",
    max_results=5,
    sort_by=arxiv.SortCriterion.Relevance
)
```
- More reliable in automated environments; no rate limiting for small requests.
- Covers CS (cs.MS, cs.DC, cs.DS) and optimization (math.OC) categories.
- Returns full abstract and PDF URL.

---

## Query Construction

### Effective Query Patterns

For a query like "Improve presolver for dense LP on L40":

1. **Component + technique**: `"LP presolver GPU parallel bound tightening"`
2. **Algorithm + hardware**: `"interior point method CUDA GPU acceleration"`
3. **General survey**: `"GPU linear programming solver survey 2020 2021 2022 2023 2024"`
4. **Routing specific**: `"parallel vehicle routing problem GPU branch-and-bound"`

### Query Templates by Problem Type

| cuOpt Component | Search Query Template |
|----------------|----------------------|
| LP presolver | `"linear programming presolver GPU parallel {technique}"` |
| MILP B&B | `"mixed integer programming branch bound GPU CUDA parallel"` |
| LP solver (PDLP) | `"first-order LP solver GPU ADMM primal-dual acceleration"` |
| Routing | `"vehicle routing problem GPU parallel exact heuristic"` |
| QP | `"quadratic programming GPU CUDA solver"` |
| CUDA kernels | `"sparse matrix-vector GPU optimization {architecture}"` |

---

## Evaluating Paper Relevance

When reading paper abstracts, assess:

### Applicability Criteria
1. **Algorithm transferability**: Does the paper describe an algorithm implementable in C++/CUDA?
2. **Problem match**: Is the problem class (LP, MILP, graph, sparse linear algebra) a match?
3. **Hardware target**: Is the GPU architecture compatible with cuOpt's deployment GPUs?
4. **Practical speedup**: Does the paper report wall-clock speedup on realistic problem sizes?
5. **Reproducibility**: Is code available? Are benchmarks on standard datasets (MIPLIB, etc.)?

### Red Flags
- Papers with only theoretical analysis, no implementation results.
- Results only on tiny toy problems (< 1000 constraints).
- Speedups measured only in GPU kernel time (not end-to-end).
- Architecture-specific tricks that do not generalize (e.g., old Kepler-only).

---

## Key Journals and Venues

| Venue | Focus |
|-------|-------|
| Mathematical Programming | LP/MILP algorithms |
| INFORMS Journal on Computing | Computational OR |
| SC / ICS / PPoPP | HPC, parallel algorithms |
| NeurIPS / ICML | Learning-based optimization |
| IPDPS / Euro-Par | Parallel and distributed computing |
| Operations Research | Routing, scheduling |
| arxiv:math.OC | Optimization (fast preprints) |
| arxiv:cs.MS | Mathematical software |

---

## Notable Papers Relevant to cuOpt

### Presolvers
- Achterberg et al. (2007). "Presolving Mixed Integer Linear Programs." ZIB-Report 07-17.
- Gamrath et al. (2015). "Progress in Presolving for Mixed Integer Programming." INFORMS JoC.

### GPU LP Solvers
- Applegate et al. (2023). "Faster First-Order Primal-Dual Methods for LP Using Restarts and Sharpness." arxiv:2201.12519.
- Lu & Yang (2023). "cuPDLP-C: A Strengthened CUDA Implementation of PDLP for Linear Programming." arxiv:2312.14832.

### MILP on GPU
- Huchette et al. (2023). "Deep Learning for Integer and Mixed-Integer Programming." arxiv:2307.13101.
- Nair et al. (2020). "Solving Mixed Integer Programs Using Neural Networks." arxiv:2012.13349.

### Routing on GPU
- Cire & Erway (2013). "Parallel Search for Combinatorial Problems on GPU." CP 2013.
- Rochat et al. (2017). "GPU Acceleration for Heuristic Routing Algorithms."

### Blackwell / Hopper Optimization
- Jarmusch & Chandrasekaran (2024). "Microbenchmarking NVIDIA's Blackwell Architecture." arxiv:2512.02189.

---

## Extracting Actionable Insights

From each paper, extract:
1. **Algorithm name** and pseudocode (if available)
2. **Key data structure** changes needed
3. **Parallelism strategy** (how to map to CUDA threads/warps/blocks)
4. **Reported speedup** and on what benchmark
5. **Limitations** (problem size, density, GPU model)

Format for injection into the implementation prompt:
```
Paper: "Title" (Author, Year)
URL: <url>
Key technique: <1-2 sentences>
Applicability: <how it maps to cuOpt>
Expected benefit: <speedup estimate>
```
