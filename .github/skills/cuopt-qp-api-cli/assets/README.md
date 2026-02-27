# Assets — QP CLI reference

QP can be solved via `cuopt_cli` when the input format supports quadratic objectives (see repo docs and `cuopt_cli --help` for QP-specific options and file format).

**Important:** QP objectives must be **minimization**. For maximization, negate the objective.

**Reference:**

| Resource | Description |
|----------|-------------|
| `cuopt-lp-milp-api-cli/assets/` | Sample MPS files and CLI usage for LP/MILP |
| `cuopt-lp-milp-api-cli/resources/examples.md` | CLI options (time limit, tolerances) |
| Repo docs | `docs/cuopt/source/cuopt-cli/` for QP file format and flags |

No sample QP input files are included here; use LP/MILP assets as the CLI pattern and check documentation for quadratic term format.
