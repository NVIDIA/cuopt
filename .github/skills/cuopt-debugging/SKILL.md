---
name: cuopt-debugging
description: Troubleshoot cuOpt problems including errors, wrong results, infeasible solutions, performance issues, and status codes. Use when the user says something isn't working, gets unexpected results, or needs help diagnosing issues.
---

# cuOpt Debugging Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Diagnose and fix issues with cuOpt solutions, errors, and performance.

## Before You Start: Required Questions

**Ask these to understand the problem:**

1. **What's the symptom?**
   - Error message?
   - Wrong/unexpected results?
   - Empty solution?
   - Performance too slow?

2. **What's the status?**
   - For LP/MILP: `problem.Status.name`
   - For Routing: `solution.get_status()`
   - For Server: HTTP response code

3. **Can you share?**
   - The error message (exact text)
   - The code that produces it
   - Problem size (variables, constraints, locations)
   - Share important log messages or status of on-going run.

## Quick Diagnosis by Symptom

### "Solution is empty/None but status looks OK"

**Most common cause: Wrong status string case**

```python
# ❌ WRONG - "OPTIMAL" never matches, silently fails
if problem.Status.name == "OPTIMAL":
    print(problem.ObjValue)  # Never runs!

# ✅ CORRECT - use PascalCase
if problem.Status.name in ["Optimal", "FeasibleFound"]:
    print(problem.ObjValue)
```

**Diagnostic code:**
```python
print(f"Actual status: '{problem.Status.name}'")
print(f"Matches 'Optimal': {problem.Status.name == 'Optimal'}")
print(f"Matches 'OPTIMAL': {problem.Status.name == 'OPTIMAL'}")
```

### "Objective value is wrong/zero"

**Check if variables are actually used:**
```python
for var in [x, y, z]:
    print(f"{var.name}: {var.getValue()}")
print(f"Objective: {problem.ObjValue}")
```

**Common causes:**
- Constraints too restrictive (all zeros is feasible)
- Objective coefficients have wrong sign
- Wrong variable in objective

### "Infeasible" status

**For LP/MILP:**
```python
if problem.Status.name == "Infeasible":
    print("Problem has no feasible solution")
    # Check constraints manually
    for name in constraint_names:
        c = problem.getConstraint(name)
        print(f"{name}: {c}")
```

**Common causes:**
- Conflicting constraints (x <= 5 AND x >= 10)
- Bounds too tight
- Missing a "slack" variable for soft constraints

**For Routing:**
```python
if solution.get_status() != 0:
    print(f"Error: {solution.get_error_message()}")
    infeasible = solution.get_infeasible_orders()
    print(f"Infeasible orders: {infeasible.to_list()}")
```

**Common routing infeasibility causes:**
- Time windows too tight (earliest > vehicle latest)
- Total demand > total capacity
- Order location unreachable in time

### "Integer variable has fractional value"

```python
# Check how variable was defined
int_var = problem.addVariable(
    lb=0, ub=10,
    vtype=INTEGER,  # Must be INTEGER, not CONTINUOUS
    name="count"
)

# Also check if status is actually optimal
if problem.Status.name == "FeasibleFound":
    print("Warning: not fully optimal, may have fractional intermediate values")
```

### Server returns 422 Validation Error

**Check payload against OpenAPI spec:**

Common field name mistakes:
```
❌ transit_time_matrix_data → ✅ travel_time_matrix_data
❌ vehicle_capacities       → ✅ capacities
❌ locations                → ✅ task_locations
```

**Capacity format:**
```json
// ❌ WRONG
"capacities": [[50], [50]]

// ✅ CORRECT
"capacities": [[50, 50]]
```

### OutOfMemoryError

**Check problem size:**
```python
print(f"Variables: {problem.num_variables}")
print(f"Constraints: {problem.num_constraints}")

# For routing
print(f"Locations: {n_locations}")
print(f"Orders: {n_orders}")
print(f"Fleet: {n_fleet}")
```

**Mitigations:**
- Reduce problem size
- Use sparse constraint matrix
- For routing: reduce time limit, simplify constraints

### cudf Type Errors

**Always use explicit dtypes:**
```python
cost_matrix = cost_matrix.astype("float32")
demand = cudf.Series([...], dtype="int32")
order_locations = cudf.Series([...], dtype="int32")
time_windows = cudf.Series([...], dtype="int32")
```

### MPS Parsing Fails

**Check MPS format:**
```bash
head -30 problem.mps
```

**Required sections in order:**
1. NAME
2. ROWS
3. COLUMNS
4. RHS
5. (optional) BOUNDS
6. ENDATA

**Common issues:**
- Missing ENDATA
- Integer markers malformed: `'MARKER'`, `'INTORG'`, `'INTEND'`
- Invalid characters or encoding

## Status Code Reference

### LP Status Values
| Status | Meaning |
|--------|---------|
| `Optimal` | Found optimal solution |
| `PrimalFeasible` | Found feasible but may not be optimal |
| `PrimalInfeasible` | No feasible solution exists |
| `DualInfeasible` | Problem is unbounded |
| `TimeLimit` | Stopped due to time limit |
| `IterationLimit` | Stopped due to iteration limit |
| `NumericalError` | Numerical issues encountered |
| `NoTermination` | Solver didn't converge |

### MILP Status Values
| Status | Meaning |
|--------|---------|
| `Optimal` | Found optimal solution |
| `FeasibleFound` | Found feasible, within gap tolerance |
| `Infeasible` | No feasible solution exists |
| `Unbounded` | Problem is unbounded |
| `TimeLimit` | Stopped due to time limit |
| `NoTermination` | No solution found yet |

### Routing Status Values
| Code | Meaning |
|------|---------|
| 0 | SUCCESS |
| 1 | FAIL |
| 2 | TIMEOUT |
| 3 | EMPTY |

## Performance Debugging

### Slow LP/MILP Solve

```python
settings = SolverSettings()
settings.set_parameter("log_to_console", 1)  # See progress
settings.set_parameter("time_limit", 60)      # Don't wait forever

# For MILP, accept good-enough solution
settings.set_parameter("mip_relative_gap", 0.05)  # 5% gap
```

### Slow Routing Solve

```python
ss = routing.SolverSettings()
ss.set_time_limit(60)           # Increase time for better solutions
ss.set_verbose_mode(True)       # See progress during solve
```

## Diagnostic Checklist

```
□ Status checked with correct case (PascalCase)?
□ All variables have correct vtype (INTEGER vs CONTINUOUS)?
□ Constraint directions correct (<= vs >= vs ==)?
□ Objective sense correct (MINIMIZE vs MAXIMIZE)?
□ For QP: using MINIMIZE (not MAXIMIZE)?
□ Data types explicit (float32, int32)?
□ Matrix dimensions match n_locations?
□ Time windows have transit_time_matrix?
```

## Diagnostic Code Snippets

See [resources/diagnostic_snippets.md](resources/diagnostic_snippets.md) for copy-paste diagnostic code:
- Status checking
- Variable inspection
- Constraint analysis
- Routing infeasibility diagnosis
- Server response debugging
- Memory and performance checks

## When to Escalate

Switch to **cuopt-developer** if:
- Bug appears to be in cuOpt itself
- Need to examine solver internals

File a GitHub issue if:
- Reproducible bug with minimal example
- Include: cuOpt version, CUDA version, error message, minimal repro code
