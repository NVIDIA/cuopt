# Assets — QP C API reference

QP uses the same cuOpt C library as LP/MILP; the API extends to quadratic objectives.

**Build and run:** Use the same include/lib paths and link steps as in `cuopt-lp-milp-api-c` (see that skill's `assets/` and `resources/examples.md`). Then use QP-specific creation and solve calls from the cuOpt C headers.

**Reference locations (in repo):**

| Resource | Description |
|----------|-------------|
| `cuopt-lp-milp-api-c/assets/` | LP/MILP C examples and build pattern |
| `cuopt-lp-milp-api-c/resources/examples.md` | Parameter constants, CSR format |
| Repo docs | `docs/cuopt/source/cuopt-c/lp-qp-milp/` for QP C API and examples |

No standalone QP C source files are included in this skill; copy the build pattern from LP/MILP and adapt for quadratic objective APIs from the headers.
