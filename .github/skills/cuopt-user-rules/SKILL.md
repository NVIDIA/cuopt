---
name: cuopt-user-rules
description: Base behavior rules for using NVIDIA cuOpt. Read this FIRST before any cuOpt user task (routing, LP/MILP, QP, debugging, installation, server). Covers handling incomplete questions, clarifying data requirements, verifying understanding, and running commands safely.
---

# cuOpt User Rules

**Read this before using any cuOpt skill.** These rules ensure you help users effectively and safely.

---

## 1. Ask Before Assuming

**Always clarify ambiguous requirements before implementing:**

- What interface? (Python API / REST Server / C API / CLI)
- What problem type? (Routing / LP / MILP / QP)
- What constraints matter? (time windows, capacities, etc.)
- What output format? (solution values, routes, visualization)

**Skip asking only if:**
- User explicitly stated the requirement
- Context makes it unambiguous (e.g., user shows Python code)

---

## 2. Handle Incomplete Questions

**If a question seems partial or incomplete, ask follow-up questions:**

- "Could you tell me more about [missing detail]?"
- "What specifically would you like to achieve with this?"
- "Are there any constraints or requirements I should know about?"

**Common missing information to probe for:**
- Problem size (number of vehicles, locations, variables, constraints)
- Specific constraints (time windows, capacities, precedence)
- Performance requirements (time limits, solution quality)
- Integration context (existing codebase, deployment environment)

**Don't guess — ask.** A brief clarifying question saves time vs. solving the wrong problem.

---

## 3. Clarify Data Requirements

**Before generating examples, ask about data:**

1. **Check if user has data:**
   - "Do you have specific data you'd like to use, or should I create a sample dataset?"
   - "Can you share the format of your input data?"

2. **If using synthesized data:**
   - State clearly: "I'll create a sample dataset for demonstration"
   - Keep it small and understandable (e.g., 5-10 locations, 2-3 vehicles)
   - Make values realistic and meaningful

3. **Always document what you used:**
   ```
   "For this example I'm using:
   - [X] locations/variables/constraints
   - [Key assumptions: e.g., all vehicles start at depot, 8-hour shifts]
   - [Data source: synthesized / user-provided / from docs]"
   ```

4. **State assumptions explicitly:**
   - "I'm assuming [X] — let me know if this differs from your scenario"
   - List any default values or simplifications made

---

## 4. MUST Verify Understanding

**Before writing substantial code, you MUST confirm your understanding:**

```
"Let me confirm I understand:
- Problem: [restate in your words]
- Constraints: [list them]
- Objective: [minimize/maximize what]
- Interface: [Python/REST/C/CLI]
Is this correct?"
```

---

## 5. Follow Requirements Exactly

- Use the **exact** variable names, formats, and structures the user specifies
- Don't add features the user didn't ask for
- Don't change the problem formulation unless asked
- If user provides partial code, extend it—don't rewrite from scratch

---

## 6. Read Examples First

Before generating code, **read the canonical example** for that problem type:

| Problem | Example Location |
|---------|------------------|
| Routing | `docs/cuopt/source/cuopt-python/routing/examples/` |
| LP/MILP | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/` |
| QP | `docs/cuopt/source/cuopt-python/lp-qp-milp/examples/simple_qp_example.py` |
| Server | `docs/cuopt/source/cuopt_spec.yaml` (OpenAPI) |
| C API | `docs/cuopt/source/cuopt-c/lp-qp-milp/examples/` |

**Don't invent API patterns.** Copy from examples.

---

## 7. Check Results

After providing a solution, guide the user to verify:

- **Status check**: Is it `Optimal` / `FeasibleFound` / `SUCCESS`?
- **Constraint satisfaction**: Are all constraints met?
- **Objective value**: Is it reasonable for the problem?

Provide diagnostic code snippets when helpful.

---

## 8. Check Environment First

**Before writing code or suggesting installation, verify the user's setup:**

1. **Ask how they access cuOpt:**
   - "Do you have cuOpt installed? If so, which interface?"
   - "What environment are you using? (local GPU, cloud, Docker, server, etc.)"

2. **Different packages for different interfaces:**

   | Interface | Package | Check |
   |-----------|---------|-------|
   | Python API | `cuopt` (pip/conda) | `import cuopt` |
   | C API | `libcuopt` (conda/system) | `find libcuopt.so` or header check |
   | REST Server | `cuopt-server` or Docker | `curl /cuopt/health` |
   | CLI | `cuopt` package includes CLI | `cuopt_cli --help` |

   **Note:** `libcuopt` (C library) installed via conda is NOT available through Python import — they are separate packages.

3. **If not installed, ask how they want to access:**
   - "Would you like help installing cuOpt, or do you have access another way?"
   - Options: pip, conda, Docker, cloud instance, existing remote server

4. **Never assume installation is needed** — the user may:
   - Already have it installed
   - Be connecting to a remote server
   - Prefer a specific installation method
   - Only need the C library (not Python)

5. **Ask before running any verification commands:**
   ```python
   # Python API check - ask first
   import cuopt
   print(cuopt.__version__)
   ```
   ```bash
   # C API check - ask first
   find ${CONDA_PREFIX} -name "libcuopt.so"
   ```
   ```bash
   # Server check - ask first
   curl http://localhost:8000/cuopt/health
   ```

---

## 9. Ask Before Running

**Do not execute commands or code without explicit permission:**

| Action | Rule |
|--------|------|
| Shell commands | Show command, explain what it does, ask "Should I run this?" |
| Package installs | **Never** run `pip`, `conda`, `apt` without asking first |
| Examples/scripts | Show the code first, ask "Would you like me to run this?" |
| File writes | Explain what will change, ask before writing |

**Exceptions (okay without asking):**
- Read-only commands the user explicitly requested
- Commands the user just provided and asked you to run

---

## 10. No Privileged Operations

**Never do these without explicit user request AND confirmation:**

- Use `sudo` or run as root
- Modify system files or configurations
- Add package repositories or keys
- Change firewall, network, or driver settings
- Write files outside the workspace

---

## Resources

### Documentation
- [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
- [API Reference](https://docs.nvidia.com/cuopt/user-guide/latest/api.html)

### Examples
- [cuopt-examples repo](https://github.com/NVIDIA/cuopt-examples)
- [Google Colab notebooks](https://colab.research.google.com/github/nvidia/cuopt-examples/)

### Support
- [NVIDIA Developer Forums](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514)
- [GitHub Issues](https://github.com/NVIDIA/cuopt/issues)
