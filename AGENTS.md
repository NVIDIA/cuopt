# AGENTS.md — cuOpt AI Agent Entry Point

NVIDIA cuOpt solves LP, MILP, QP (beta) and vehicle routing on the GPU.

**Start with the documentation.** It is the current, authoritative reference for installation,
system requirements, solver settings, and every interface:
<https://docs.nvidia.com/cuopt/user-guide/latest/>

Skills live in **`skills/`** (repo root), flat layout. There are deliberately few of them: a
skill should exist only when it carries something the documentation does not, because every
skill costs agent context on tasks that may not need it.

> **🔒 MANDATORY — Ambiguity:** When the problem could be read more than one way, you MUST either **ask the user to clarify** or **solve every plausible interpretation and report all outcomes**. Never pick one interpretation silently.

## Skills directory (flat)

- `skills/cuopt-developer/` — Modify, build, test, debug, and contribute to cuOpt internals (C++/CUDA, Python, server, CI). Use for solver internals, PRs, DCO, and code conventions. For **VRP dimension** work (combine invariants, fwd/bwd propagation, new constraints/objectives in the routing engine), read **`skills/cuopt-developer/references/vrp_skills.md`** in addition to `SKILL.md`.
- `skills/cuopt-multi-objective-exploration/` — Multi-objective: trace and interpret the Pareto frontier across competing objectives (ε-constraint / weighted-sum over repeated cuOpt solves).

## Using cuOpt

For installing cuOpt, writing models against the Python/C/CLI/server interfaces, choosing
solver settings, or understanding termination statuses, read the user guide:

- [Install](https://docs.nvidia.com/cuopt/user-guide/latest/install.html)
- [System requirements](https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html)
- [Python API](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/)
- [REST server](https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-server/)
- [LP / QP settings](https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html)
- [MILP settings](https://docs.nvidia.com/cuopt/user-guide/latest/mip-settings.html)
- [FAQ](https://docs.nvidia.com/cuopt/user-guide/latest/faq.html)

If the documentation is missing something or is wrong, the fix belongs in the docs — please
open an issue rather than adding it here.

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
