---
name: cuoptopt-agent
version: "26.04.00"
description: Agentic optimization loop for cuOpt — improves solver speed and solution quality via literature research, LLM-driven code changes, and automated regression testing. Use when asked to optimize, speed up, or improve cuOpt.
---

# cuoptopt-agent

An autonomous optimization agent for the NVIDIA cuOpt codebase.

**Use this skill** when the user asks you to:
- Optimize, speed up, or improve cuOpt performance
- Reduce solve time for a specific problem class or GPU
- Improve solution quality for a solver component
- Automate investigation and implementation of algorithmic improvements

---

## Two Modes of Operation

### Mode A — Run the Python CLI (preferred when installed)
```bash
cd cuoptopt-agent
python -m cuoptopt_agent "your optimization query" --model claude
```
Or from Cursor: `Ctrl+Shift+P` → **Tasks: Run Task** → **cuoptopt-agent: Run (interactive)**

### Mode B — Cursor AI Acts as Orchestrator (no installation needed)
If the Python agent is not installed, YOU (the Cursor AI) execute the workflow steps
below manually, using the available tools.

---

## Workflow (execute in order)

### Step 1: Understand the Query
Parse the query to identify:
- **Target component**: presolver, LP solver, MILP, routing kernel, QP, CUDA kernel
- **Target GPU**: L40, A100, H100, Blackwell — load the matching architecture skill
- **Metric**: speed (solve time), quality (objective value), or both

### Step 2: Load Relevant Skills
Read skills from `cuoptopt-agent/skills/` that match the query. Priority order:
1. GPU architecture skill (e.g., `nvidia-l40-architecture`, `nvidia-blackwell-architecture`)
2. Algorithm skill (e.g., `presolvers`, `mip-algorithms`)
3. `cuda-optimization` for any kernel-level work
4. `gpu-profiling` if you need to identify bottlenecks first
5. `benchmarking-methodology` for proper before/after comparison

Skill directory: `cuoptopt-agent/skills/`

### Step 3: Search Literature
Formulate 2–3 search queries combining the component and technique, e.g.:
- `"GPU presolver linear programming dense constraints"`
- `"CUDA parallel branch-and-bound MILP"`

Sources: Google Scholar, arxiv.org/search, Semantic Scholar.
Extract: algorithm names, key insights, applicability to cuOpt.

### Step 4: Explore the Codebase
Before proposing changes, read the relevant source files:
- `cpp/src/` — C++/CUDA solver internals
- `cpp/include/` — headers
- `python/cuopt/` — Python API layer
- `regression/` — existing tests to understand what is benchmarked

### Step 5: Implement Changes
Generate unified diffs targeting the identified bottleneck.
- Keep changes minimal and focused
- Preserve existing API contracts
- Add comments explaining non-obvious choices

### Step 6: Run Regression Tests
```bash
# Baseline first (before changes, if not already recorded)
pytest regression/ -x -q

# After changes
pytest regression/ -x -q
```
Also run targeted benchmarks:
```bash
python -m cuopt solve --mps datasets/linear_programming/<file>.mps
```

### Step 7: Evaluate Results
- **Speed regression > 5%**: Revert and re-approach. Explain why the change was rejected.
- **Quality regression > 1%**: Report to the user with exact numbers and ask for approval.
- **No regression**: Ask for final user approval.

### Step 8: Create PR (on user approval)
```bash
git checkout -b YYYY-MM-DD-<type>   # e.g., 2026-03-15-presolve
git add <changed files>
git commit -m "perf: <short description>"
git push -u origin HEAD
gh pr create --title "perf: <description>" --body "<benchmark table + literature citations>"
```
Branch type suffixes: `presolve`, `mip-solver`, `lp-solver`, `routing`, `cuda-kernel`, `memory`, `qp-solver`.

---

## Regression Thresholds
See `cuoptopt-agent/config/thresholds.yaml`:
- Auto-reject if solve time increases > 5%
- Ask human if objective degrades > 1%
- Max 5 implementation iterations before stopping

## Available Skills Index
| Skill | When to use |
|-------|------------|
| `cuda-optimization` | Any CUDA kernel change |
| `nvidia-l40-architecture` | Targeting L40/L40S |
| `nvidia-a100-architecture` | Targeting A100 |
| `nvidia-h100-architecture` | Targeting H100/H200 |
| `nvidia-blackwell-architecture` | Targeting B100/B200/GB200 |
| `presolvers` | LP/MILP presolver changes |
| `mip-algorithms` | Branch-and-bound, cuts, heuristics |
| `gpu-profiling` | Finding bottlenecks with Nsight |
| `benchmarking-methodology` | Correct timing methodology |
| `research-literature` | Searching papers for techniques |
