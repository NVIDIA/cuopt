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
version in use, and needs no network.

For what the package does not carry — install, examples, parameter ranges, the other
interfaces — the user guide:

| | |
|---|---|
| Install, channels, Docker | https://docs.nvidia.com/cuopt/user-guide/latest/install.html |
| GPU / CUDA / driver support | https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html |
| LP / QP settings | https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html |
| MILP settings | https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html |
| Python API reference | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/ |
| LP / QP examples | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-examples.html |
| MILP examples | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/mip/mip-examples.html |
| Routing (Python) | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/routing/ |
| C, CLI, Java, gRPC, REST server | https://docs.nvidia.com/cuopt/user-guide/latest/ |
| FAQ | https://docs.nvidia.com/cuopt/user-guide/latest/faq.html |

These pages track the newest release. For any other version, read `docs/cuopt/source/` on
its branch in the repo: `release/<major>.<minor>`, or `main` if that branch does not exist.

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
