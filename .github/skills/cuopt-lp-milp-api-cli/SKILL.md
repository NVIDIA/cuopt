---
name: cuopt-lp-milp-api-cli
description: LP and MILP with cuOpt — CLI only (MPS files, cuopt_cli). Use with cuopt-lp-milp-formulation for concepts. Use when the user is solving from MPS via command line.
---

# cuOpt LP/MILP — CLI

**Concepts:** Read `cuopt-lp-milp-formulation/SKILL.md` for problem type and formulation.

This skill is **CLI only**. For Python or C, use `cuopt-lp-milp-api-python` or `cuopt-lp-milp-api-c`.

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

## Getting the CLI

CLI is included with the Python package (`cuopt`). Install via pip or conda (see `cuopt-installation-common` + `cuopt-installation-api-python`); then run `cuopt_cli --help` to verify.
