---
name: cuopt-qp-api-c
description: Quadratic Programming (QP) with cuOpt — C API. Use with cuopt-qp-common for concepts. Use when the user is embedding QP in C/C++.
---

# cuOpt QP — C API

**Concepts:** Read `cuopt-qp-common/SKILL.md` for when QP applies, minimize-only, when to escalate.

This skill is **C only**. For Python (beta), use `cuopt-qp-api-python`.

QP uses the same cuOpt C library as LP/MILP; the API extends to quadratic objectives. See LP/MILP C API docs and build instructions in `cuopt-lp-milp-api-c` for the base setup; then use the QP-specific creation/solve calls from the cuOpt C headers.

## Escalate

See `cuopt-qp-common` for when to use cuopt-lp-milp-common or cuopt-developer.
