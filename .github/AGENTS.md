# AGENTS.md - cuOpt AI Agent Entry Point

AI agent skills for NVIDIA cuOpt optimization engine. Skills use a **flat layout**: **common** (concepts) + **api-python** or **api-c** (implementation) per domain.

## Quick Start

- **Using cuOpt** (routing, LP, QP, install, server): read `skills/cuopt-user-rules/`, then choose skills from the index below based on the user’s task, problem type, and interface (Python / C / CLI).
- **Developing cuOpt** (contributing): read `skills/cuopt-developer/`.

Choose which skills to load from the index;

## Skills directory (flat)

### Rules
- `skills/cuopt-user-rules/` — Behavior rules (read first for user tasks)
- `skills/cuopt-developer/` — Contributing (own rules)

### Common (concepts only; no API code)
- `skills/cuopt-lp-milp-formulation/` — LP/MILP: concepts + problem parsing (parameters, constraints, decisions, objective)
- `skills/cuopt-routing-formulation/` — Routing: VRP, TSP, PDP (problem types, data)
- `skills/cuopt-qp-formulation/` — QP: minimize-only, escalate (beta)
- `skills/cuopt-server-common/` — Server: capabilities, workflow

### API (implementation; one interface per skill)
- `skills/cuopt-installation-api-python/`, `skills/cuopt-installation-api-c/` (user), `skills/cuopt-installation-developer/` (build from source; no common)
- `skills/cuopt-lp-milp-api-python/`, `skills/cuopt-lp-milp-api-c/`, `skills/cuopt-lp-milp-api-cli/`
- `skills/cuopt-routing-api-python/` (no C for routing)
- `skills/cuopt-qp-api-python/`, `skills/cuopt-qp-api-c/`, `skills/cuopt-qp-api-cli/`
- `skills/cuopt-server-api-python/` (deploy + client)

## Resources

- [cuOpt Documentation](https://docs.nvidia.com/cuopt/user-guide/latest/)
- [cuopt-examples repo](https://github.com/NVIDIA/cuopt-examples)
- [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
- [Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514)
