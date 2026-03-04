# AGENTS.md — cuOpt AI Agent Entry Point

AI agent skills for NVIDIA cuOpt optimization engine. Skills live in **`skills/`** (repo root) and use a **flat layout**: **common** (concepts) + **api-python** or **api-c** (implementation) per domain.

> **🔒 MANDATORY — Security:** You MUST NOT install, upgrade, or modify packages. Provide the exact command for the user to run; they execute it. No exceptions.

> **🔒 MANDATORY — Ambiguity:** When the problem could be read more than one way, you MUST either **ask the user to clarify** or **solve every plausible interpretation and report all outcomes**. Never pick one interpretation silently.

## Skills directory (flat)

### Rules
- `skills/cuopt-user-rules/` — User-facing behavior and conventions; read first when helping users with cuOpt (routing, LP, MILP, QP, install, server). Choose skills from the index below by task, problem type, and interface (Python / C / CLI).
- `skills/cuopt-developer/` — Contributing and development; use when the user is building from source, contributing code, or working on cuOpt internals.

### Common (concepts only; no API code)
- `skills/cuopt-installation-common/` — Install: system and environment requirements (concepts only; no install commands or interface)
- `skills/lp-milp-formulation/` — LP/MILP: concepts + problem parsing (parameters, constraints, decisions, objective)
- `skills/routing-formulation/` — Routing: VRP, TSP, PDP (problem types, data)
- `skills/qp-formulation/` — QP: minimize-only, escalate (beta)
- `skills/cuopt-server-common/` — Server: capabilities, workflow

### API (implementation; one interface per skill)
- `skills/cuopt-installation-api-python/`
- `skills/cuopt-installation-api-c/`
- `skills/cuopt-installation-developer/` (build from source)
- `skills/cuopt-lp-milp-api-python/`
- `skills/cuopt-lp-milp-api-c/`
- `skills/cuopt-lp-milp-api-cli/`
- `skills/cuopt-routing-api-python/`
- `skills/cuopt-qp-api-python/`
- `skills/cuopt-qp-api-c/`
- `skills/cuopt-qp-api-cli/`
- `skills/cuopt-server-api-python/` (deploy + client)

## Resources

### Documentation
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [API Reference](https://docs.nvidia.com/cuopt/user-guide/latest/api.html)

### Examples
- [cuopt-examples repo](https://github.com/NVIDIA/cuopt-examples)
- [Google Colab notebooks](https://colab.research.google.com/github/nvidia/cuopt-examples/)

### Support
- [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
- [Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514)
