---
name: cuopt
version: "26.12.00"
description: Entry point for cuOpt — routing, LP, MILP and QP, from Python, C, Java, the CLI or the REST and gRPC server. Where to find settings and API details, plus gotchas. Use when writing or debugging code that calls cuOpt.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
---

# cuOpt

GPU-accelerated routing, LP, MILP and QP. Source: https://github.com/NVIDIA/cuopt

Look details up locally rather than from memory — the installed package always matches the
version in use, and needs no network:

- `help(obj)` and `inspect.signature(f)` — signatures and expected arguments
- `solver_parameters` — every parameter name
- `SolverSettings().get_parameter(name)` — its default
- `SolverSettings().dump_parameters_to_file(path)` — description, type and range

For anything the package does not carry — install, system requirements, examples, the other
interfaces, the FAQ — see the user guide: https://docs.nvidia.com/cuopt/user-guide/latest/

## Gotchas

Solver statuses and results that surprise you are answered under **Common Pitfalls** in the
FAQ: `docs/cuopt/source/faq.rst` in this repo, published at
https://docs.nvidia.com/cuopt/user-guide/latest/faq.html

Additional gotchas are listed below.

**`addConstraint` requires a comparison** (`<=`, `>=`, `==`). Accumulate an expression in a
loop, then compare it in the call; passing a bare expression raises `ValueError:
addConstraint requires a Constraint object`. Constraints cannot be built as strings.

**`LinearExpression` takes `constant` as a required positional**, not an optional keyword:
`LinearExpression([x, y], [1.0, 2.0], 0.0)`. For expressions with thousands of terms, build
the lists once and pass them in a single call rather than accumulating with `sum(...)`; see
the LP / QP examples page.

**The solver prints its own log to stdout.** That is expected, not an error.
