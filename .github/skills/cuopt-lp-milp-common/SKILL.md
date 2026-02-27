---
name: cuopt-lp-milp-common
description: LP and MILP with cuOpt — problem type and formulation only. Domain concepts; no API code and no interface or escalation guidance.
---

# cuOpt LP/MILP (common)

Domain concepts for linear and mixed-integer linear programming. No API or interface details here.

## What is LP / MILP

- **LP**: Linear objective, linear constraints, continuous variables.
- **MILP**: Same plus some integer or binary variables (e.g. scheduling, facility location, selection).

## Required questions (problem formulation)

Ask these if not already clear:

1. **Decision variables** — What are they? Bounds?
2. **Objective** — Minimize or maximize? Linear expression in the variables?
3. **Constraints** — Linear inequalities/equalities? Names and meaning?
4. **Variable types** — All continuous (LP) or some integer/binary (MILP)?

## Typical modeling elements

- **Continuous variables** — production amounts, flow, etc.
- **Binary variables** — open/close, yes/no (e.g. facility open, item selected).
- **Linking constraints** — e.g. production only if facility open (Big-M or indicator).
- **Resource constraints** — linear cap on usage (materials, time, capacity).
