---
name: cuopt-qp-api-cli
description: QP with cuOpt — CLI (e.g. cuopt_cli with QP-capable input). Use with cuopt-qp-formulation for concepts. Use when the user is solving QP from the command line.
---

# cuOpt QP — CLI

**Concepts:** Read `cuopt-qp-formulation/SKILL.md` for when QP applies and the **minimize-only** rule.

This skill is **CLI only** for QP. For Python or C, use `cuopt-qp-api-python` or `cuopt-qp-api-c`.

## QP via CLI

cuOpt CLI supports QP (quadratic objectives). Use the same `cuopt_cli` tool; input format and options may extend the LP/MILP MPS workflow to allow quadratic terms (see repo docs or `cuopt_cli --help` for QP-specific options).

**Important:** QP objectives must be **minimization**. For maximization, negate the objective.

## Basic usage

```bash
# Solve QP (syntax may match or extend LP/MILP CLI; check --help)
cuopt_cli problem.mps

# With time limit
cuopt_cli problem.mps --time-limit 60
```

Check `cuopt_cli --help` and the repository documentation (e.g. `docs/cuopt/source/cuopt-cli/`) for QP file format and any QP-specific flags.

**Reference:** This skill's [assets/README.md](assets/README.md) — pointers to LP/MILP CLI samples and options.

## Getting the CLI

CLI is included with the Python package (`cuopt`). Install via pip or conda (see `cuopt-installation-api-python`); then run `cuopt_cli --help` to verify.

## Escalate

See `cuopt-qp-formulation` for when to use cuopt-lp-milp-formulation or cuopt-developer.
