---
name: "Numerical issues"
about: "Report numerical issues (e.g., NaNs/Infs, ill-conditioning, tolerance or scaling issues)"
title: "[NUMERICAL_ISSUES]"
labels: "? - Needs Triage, numerical issues"
assignees: ''

---

**Describe the numerical issue**
A clear description of the numerical problem (e.g., NaN/Inf values, unexpected divergence, extreme sensitivity to small input changes).

**Steps / minimal reproduction**
Follow the [RAPIDS issue guidelines](https://docs.rapids.ai/contributing/issues/) (search existing issues first, then describe the problem so it can be understood and reproduced). Use the cuOpt issue template that best fits; the RAPIDS page describes general practices. Include inputs, API calls, and solver settings (tolerances, scaling, precision) that reproduce the behavior.

**Expected vs actual behavior**
What you expected numerically, and what you observed (including any error messages or logs).

**Environment details (please complete the following information):**
 - Environment location: [Bare-metal, Docker, Cloud (specify cloud provider)]
 - Method of cuOpt install: [conda, Docker, or from source]
   - If method of install is [Docker], provide `docker pull` & `docker run` commands used

**Additional context**
Add any other context (hardware, CUDA/driver versions, data scaling, problem size) that may help diagnose the issue.
