---
name: cuopt
version: "26.12.00"
description: Entry point for working with cuOpt: links into the user guide for install, settings and the API reference, plus API gotchas. Use when writing or debugging code that calls cuOpt.
license: Apache-2.0
metadata:
  author: NVIDIA cuOpt Team
  tags:
    - cuopt
---

# cuOpt

## Where things are

| | |
|---|---|
| Source | https://github.com/NVIDIA/cuopt |
| Docs | https://docs.nvidia.com/cuopt/user-guide/latest/ |
| Install, channels, Docker | https://docs.nvidia.com/cuopt/user-guide/latest/install.html |
| GPU / CUDA / driver support | https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html |
| LP / QP settings | https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html |
| MILP settings | https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html |
| Python API reference | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/ |
| LP / QP examples | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-examples.html |
| MILP examples | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/mip/mip-examples.html |
| Routing (Python only) | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/routing/ |
| REST server | https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/ |
| FAQ | https://docs.nvidia.com/cuopt/user-guide/latest/faq.html |

If a documentation URL carries a version number, replace it with `latest`.

With no network access, read from the installed package — `solver_parameters` has every
parameter name and `help()` gives signatures — or, inside the repo, from
`docs/cuopt/source/*.rst`, which carries the prose pages but not the Python API reference.

Settings names, defaults and ranges change between releases — look them up, never recall
them from memory.

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
