# AGENTS.md — cuOpt AI Agent Entry Point

AI agent skills for NVIDIA cuOpt optimization engine. Skills live in **`skills/`** (repo root) and use a **flat layout**: per domain, a concept skill (formulation / problem types) plus implementation skills — typically one per interface (Python, C, CLI, server), or consolidated when the content is shared across interfaces (e.g. installation). Skills evolve through agent interactions — see `skills/skill-evolution/` for the evolution workflow.

> **🔒 MANDATORY — Ambiguity:** When the problem could be read more than one way, you MUST either **ask the user to clarify** or **solve every plausible interpretation and report all outcomes**. Never pick one interpretation silently.

## Skills directory (flat)

### Rules
- `skills/cuopt-user-rules/` — Base rules for end users calling cuOpt (routing, LP, MILP, QP, install, server). Not for cuOpt internals — see `skills/cuopt-developer/`. Read first for user-facing tasks; choose skills from the index below by task and interface.
- `skills/cuopt-developer/` — Modify, build, test, debug, and contribute to cuOpt internals (C++/CUDA, Python, server, CI). Use for solver internals, PRs, DCO, and code conventions. For **VRP dimension** work (combine invariants, fwd/bwd propagation, new constraints/objectives in the routing engine), read **`skills/cuopt-developer/references/vrp_skills.md`** in addition to `SKILL.md`.
- `skills/skill-evolution/` — Skill evolution: after solving a non-trivial problem, propose skill updates to capture generalizable learnings.

### Common (concepts only; no API code)
- `skills/cuopt-numerical-optimization-formulation/` — LP / MILP / QP: concepts + problem parsing + common formulation patterns
- `skills/cuopt-routing-formulation/` — Routing: VRP, TSP, PDP (problem types, data)
- `skills/cuopt-server-common/` — Server: capabilities, workflow

### Installation
- `skills/cuopt-install/` — User install for Python, C, and server (pip, conda, Docker, verification). For building cuOpt from source, see `skills/cuopt-developer/`.

### API (implementation; one interface per skill)
- `skills/cuopt-numerical-optimization-api-python/` (LP, MILP, QP)
- `skills/cuopt-numerical-optimization-api-c/` (LP, MILP, QP)
- `skills/cuopt-numerical-optimization-api-cli/` (LP, MILP, QP)
- `skills/cuopt-routing-api-python/`
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
