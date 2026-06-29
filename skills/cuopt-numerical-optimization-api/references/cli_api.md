# cuOpt Numerical Optimization — CLI

Solve LP, MILP, and QP problems from MPS files via `cuopt_cli`. The CLI is included with the `cuopt` Python package — install via pip or conda, then verify with `cuopt_cli --help`.

## Basic Usage

```bash
# Solve LP or MILP
cuopt_cli problem.mps

# With options
cuopt_cli problem.mps --time-limit 120 --mip-relative-tolerance 0.01
```

## Common Options

```bash
cuopt_cli problem.mps --time-limit 120
cuopt_cli problem.mps --mip-relative-tolerance 0.001
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001
cuopt_cli problem.mps --presolve --iteration-limit 10000 --method 1
```

## MPS Format (required sections, in order)

1. **NAME** — problem name
2. **ROWS** — `N` (objective), `L`/`G`/`E` (constraints)
3. **COLUMNS** — variable names, row names, coefficients
4. **RHS** — right-hand side values
5. **BOUNDS** (optional) — `LO`, `UP`, `FX`, `BV`, `LI`, `UI`
6. **ENDATA**

Integer variables: wrap columns with `'MARKER' 'INTORG'` before and `'MARKER' 'INTEND'` after.

## QP via CLI (beta)

Same `cuopt_cli` command and options. Quadratic objectives use the standard MPS quadratic-objective extension. See `docs/cuopt/source/cuopt-cli/` for the format.

**QP rules:** MINIMIZE only — negate coefficients in the MPS file to maximize. Continuous variables only — do not mix integer markers with quadratic objectives.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Failed to parse MPS | Check ENDATA, section order, integer markers |
| Infeasible | Check constraint directions (L/G/E) and RHS values |
| Wrong flag name | Use `--mip-relative-tolerance` (not `--mip-relative-gap`) |

## Reference Models

| Model | Type | Location |
|-------|------|----------|
| Minimal LP | LP | [assets/cli/lp_simple/](../assets/cli/lp_simple/) |
| Production planning | LP | [assets/cli/lp_production/](../assets/cli/lp_production/) |
| Facility location | MILP | [assets/cli/milp_facility/](../assets/cli/milp_facility/) |
