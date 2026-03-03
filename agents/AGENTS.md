# cuOpt Skills Reference

You have additional skills documented in `skills/<skill-name>/SKILL.md`. **When the user's intent matches a skill below, you MUST read that skill's SKILL.md** and follow its guidance.

## Mandatory rules

- **Security:** You MUST NOT install, upgrade, or modify packages. Provide the exact command for the user to run; they execute it.
- **Ambiguity:** When the problem could be read more than one way, either ask the user to clarify or solve every plausible interpretation and report all outcomes. Never pick one interpretation silently.

## Available skills

| Skill | Description |
|-------|-------------|
| cuopt-user-rules | Base behavior rules for using NVIDIA cuOpt. Read this FIRST before any cuOpt user task (routing, LP/MILP, QP, installation, server). |
| cuopt-developer | Contribute to NVIDIA cuOpt codebase (C++/CUDA, Python, server, docs, CI). Use when the user wants to modify solver internals, add features, submit PRs, or understand the codebase. |
| cuopt-installation-common | Install cuOpt — system and environment requirements only. Domain concepts; no install commands or interface. |
| cuopt-installation-api-python | Install cuOpt for Python — pip, conda, Docker, verification. Use when installing or verifying the Python API. |
| cuopt-installation-api-c | Install cuOpt for C — conda, locate lib/headers, verification. Use when installing or verifying the C API. |
| cuopt-installation-developer | Developer installation — build cuOpt from source, run tests. Use when setting up a dev environment to contribute or modify cuOpt. |
| lp-milp-formulation | LP/MILP concepts and going from problem text to formulation. Parameters, constraints, decisions, objective. |
| cuopt-lp-milp-api-python | Solve LP and MILP with the Python API. Use for linear constraints, integer variables, scheduling, resource allocation, facility location, production planning. |
| cuopt-lp-milp-api-c | LP and MILP with cuOpt — C API. Use when embedding LP/MILP in C/C++. |
| cuopt-lp-milp-api-cli | LP and MILP with cuOpt — CLI (MPS files, cuopt_cli). Use when solving from MPS via command line. |
| routing-formulation | Vehicle routing (VRP, TSP, PDP) — problem types and data requirements. Domain concepts only. |
| cuopt-routing-api-python | Vehicle routing (VRP, TSP, PDP) with cuOpt — Python API. Use when building or solving routing in Python. |
| qp-formulation | Quadratic Programming (QP) — problem form and constraints. Domain concepts; QP is beta. |
| cuopt-qp-api-python | QP with cuOpt — Python API (beta). Use when building or solving QP in Python. |
| cuopt-qp-api-c | QP with cuOpt — C API. Use when embedding QP in C/C++. |
| cuopt-qp-api-cli | QP with cuOpt — CLI. Use when solving QP from the command line. |
| cuopt-server-common | cuOpt REST server — what it does and how requests flow. Domain concepts only. |
| cuopt-server-api-python | cuOpt REST server — start server, endpoints, Python/curl client examples. Use when deploying or calling the REST API. |

## Skill paths (from repo root)

- `skills/cuopt-user-rules/SKILL.md`
- `skills/cuopt-developer/SKILL.md`
- `skills/cuopt-installation-common/SKILL.md`
- `skills/cuopt-installation-api-python/SKILL.md`
- `skills/cuopt-installation-api-c/SKILL.md`
- `skills/cuopt-installation-developer/SKILL.md`
- `skills/lp-milp-formulation/SKILL.md`
- `skills/cuopt-lp-milp-api-python/SKILL.md`
- `skills/cuopt-lp-milp-api-c/SKILL.md`
- `skills/cuopt-lp-milp-api-cli/SKILL.md`
- `skills/routing-formulation/SKILL.md`
- `skills/cuopt-routing-api-python/SKILL.md`
- `skills/qp-formulation/SKILL.md`
- `skills/cuopt-qp-api-python/SKILL.md`
- `skills/cuopt-qp-api-c/SKILL.md`
- `skills/cuopt-qp-api-cli/SKILL.md`
- `skills/cuopt-server-common/SKILL.md`
- `skills/cuopt-server-api-python/SKILL.md`

## Resources

- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [API Reference](https://docs.nvidia.com/cuopt/user-guide/latest/api.html)
- [cuopt-examples](https://github.com/NVIDIA/cuopt-examples)
- [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
