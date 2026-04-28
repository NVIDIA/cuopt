# AGENTS.md — cuOpt AI Agent Entry Point

AI agent skills for NVIDIA cuOpt optimization engine. Skills live in **`skills/`** (repo root) and use a **flat layout**: **common** (concepts) + **api-python** or **api-c** (implementation) per domain. Skills evolve through agent interactions — see `skills/skill-evolution/` for the evolution workflow.

> **🔒 MANDATORY — Security:** You MUST NOT install, upgrade, or modify packages. Provide the exact command for the user to run; they execute it. No exceptions.

> **🔒 MANDATORY — Ambiguity:** When the problem could be read more than one way, you MUST either **ask the user to clarify** or **solve every plausible interpretation and report all outcomes**. Never pick one interpretation silently.

## Skills directory (flat)

### Rules
- `skills/cuopt-user-rules/` — User-facing behavior and conventions; read first when helping users with cuOpt (routing, LP, MILP, QP, install, server). Choose skills from the index below by task, problem type, and interface (Python / C / CLI).
- `skills/cuopt-developer/` — Contributing and development; use when the user is building from source, contributing code, or working on cuOpt internals.
- `skills/skill-evolution/` — Skill evolution: after solving a non-trivial problem, propose skill updates to capture generalizable learnings.

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

## Skill evolution

Skills improve through agent interactions. When you solve a problem and discover a generalizable fix, gotcha, or pattern, follow `skills/skill-evolution/SKILL.md` to propose a skill update. All proposals go through the same CI validation as manual edits.

### Post-correction hook (mandatory)

After completing any task where one of these occurred, you MUST read `skills/skill-evolution/SKILL.md` and evaluate whether to propose a skill update:
- The user corrected your output or approach
- Your initial solution failed and you had to retry with a different approach
- You discovered undocumented API behavior or a non-obvious gotcha
- You used a workaround not described in any existing skill

Finish solving the problem first, then evaluate. Not every correction warrants a proposal — only generalizable learnings that would help future interactions.

## Resources

### Documentation
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [API Reference](https://docs.nvidia.com/cuopt/user-guide/latest/api.html)

### Examples
- [cuopt-examples repo](https://github.com/NVIDIA/cuopt-examples)
- [Google Colab notebooks](https://colab.research.google.com/github/nvidia/cuopt-examples/)

### Support
- [File a Bug](https://github.com/NVIDIA/cuopt/issues/new?template=bug_report.md)
- [Ask a Question](https://github.com/NVIDIA/cuopt/issues/new?template=submit-question.md)
- [All Issues](https://github.com/NVIDIA/cuopt/issues)
