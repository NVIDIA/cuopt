---
name: cuopt-lp-milp-api-cli
description: LP and MILP with cuOpt — CLI only (MPS files, cuopt_cli). Use when the user is solving from MPS via command line.
---

# cuOpt LP/MILP — CLI

Confirm problem type and formulation (variables, objective, constraints, variable types) before coding.

This skill is **CLI only** (MPS input).

## Basic usage

```bash
# Solve LP or MILP from MPS file
cuopt_cli problem.mps

# With options
cuopt_cli problem.mps --time-limit 120 --mip-relative-tolerance 0.01
```

## Common options

```bash
cuopt_cli --help

# Time limit (seconds)
cuopt_cli problem.mps --time-limit 120

# MIP gap tolerance (stop when within X% of optimal)
cuopt_cli problem.mps --mip-relative-tolerance 0.001

# MIP absolute tolerance
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001

# Presolve, iteration limit, method
cuopt_cli problem.mps --presolve --iteration-limit 10000 --method 1
```

## MPS format (required sections, in order)

1. **NAME** — problem name  
2. **ROWS** — N (objective), L/G/E (constraints)  
3. **COLUMNS** — variable names, row names, coefficients  
4. **RHS** — right-hand side values  
5. **BOUNDS** (optional) — LO, UP, FX, BV, LI, UI  
6. **ENDATA**

Integer variables: use `'MARKER' 'INTORG'` before and `'MARKER' 'INTEND'` after the integer columns.

## Examples

- [examples.md](resources/examples.md) — LP and MILP MPS examples, format reference, troubleshooting
- **Sample MPS files:** This skill's `assets/` — [lp_simple](assets/lp_simple/), [lp_production](assets/lp_production/), [milp_facility](assets/milp_facility/). See [assets/README.md](assets/README.md).

## Getting the CLI

CLI is included with the Python package (`cuopt`). Install via pip or conda; then run `cuopt_cli --help` to verify.
