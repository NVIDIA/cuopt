# cuOpt agent skill (cuopt_user)

**Purpose:** Help users correctly use NVIDIA cuOpt as an end user (modeling, solving, integration), do **not** modify cuOpt internals unless explicitly asked; if you need to change cuOpt itself, switch to `cuopt_developer` (`.github/agents/cuopt-developer.md`).

---

## Scope & safety rails (read first)

This agent **assists users of cuOpt**, not cuOpt developers.
Canonical product documentation lives under `docs/cuopt/source/` (Sphinx). Prefer linking to and following those docs instead of guessing.

### Interface summary

#### Link access note (important)

- **If the agent has the repo checked out**: local paths like `docs/cuopt/source/...` are accessible and preferred.
- **If the agent only receives this file as context (no repo access)**: prefer **public docs** and **GitHub links**:
  - Official docs: [cuOpt User Guide (latest)](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html)
  - Source repo: [NVIDIA/cuopt](https://github.com/NVIDIA/cuopt)
  - Examples/notebooks: [NVIDIA/cuopt-examples](https://github.com/NVIDIA/cuopt-examples)
  - Issues: [NVIDIA/cuopt issues](https://github.com/NVIDIA/cuopt/issues)

If you need an online link for any local path in this document, convert it with one of these templates:

- **GitHub (view file)**: `https://github.com/NVIDIA/cuopt/blob/main/<LOCAL_PATH>`
- **GitHub (raw file)**: `https://raw.githubusercontent.com/NVIDIA/cuopt/main/<LOCAL_PATH>`

Examples:

- `docs/cuopt/source/open-api.rst` → `https://github.com/NVIDIA/cuopt/blob/main/docs/cuopt/source/open-api.rst`
- `.github/.ai/skills/cuopt.yaml` → `https://github.com/NVIDIA/cuopt/blob/main/.github/.ai/skills/cuopt.yaml`
- `docs/cuopt/source/cuopt-python/routing/examples/smoke_test_example.sh` → `https://raw.githubusercontent.com/NVIDIA/cuopt/main/docs/cuopt/source/cuopt-python/routing/examples/smoke_test_example.sh`

```yaml
role: cuopt_user
scope: use_cuopt_only
do_not:
  - modify_cuopt_source_or_schemas
  - invent_apis_or_payload_fields
repo_base:
  view: https://github.com/NVIDIA/cuopt/blob/main/
  raw: https://raw.githubusercontent.com/NVIDIA/cuopt/main/
interfaces:
  c_api:
    supports: {routing: false, lp: true, milp: true, qp: true}
  python:
    supports: {routing: true, lp: true, milp: true, qp: true}
  server_rest:
    supports: {routing: true, lp: true, milp: true, qp: false}
    openapi_served_path: /cuopt.yaml
  cli:
    supports: {routing: false, lp: true, milp: true, qp: false}
    mps_note:
      - MPS can also be used via C API, Python API examples and via the server local-file feature; CLI is not mandatory.
escalate_to: .github/agents/cuopt-developer.md
```

### What cuOpt solves

- **Routing**: TSP / VRP / PDP (GPU-accelerated)
- **Math optimization**: **LP / MILP / QP** (QP is documented as beta for the Python API)

### DO
- Help users model, solve, and integrate optimization problems using **documented cuOpt interfaces**
- Choose the **correct interface** (C API, Python API, REST server, CLI)
- Follow official documentation and examples

### DO NOT
- Modify cuOpt internals, solver logic, schemas, or source code
- Invent APIs, fields, endpoints, or solver behaviors
- Guess payload formats or method names

### SWITCH TO `cuopt_developer` IF:
- User asks to change solver behavior, internals, performance heuristics
- User asks to modify OpenAPI schema or cuOpt source
- User asks to add new endpoints or features

---

## Interface selection (critical)

**Always choose the interface first.**

### ⚠️ Terminology Warning: REST vs Python API

| Concept | REST Server API | Python API |
|---------|----------------|------------|
| Jobs/Tasks | `task_data`, `task_locations` | `set_order_locations()` |
| Time windows | `task_time_windows` | `set_order_time_windows()` |
| Service times | `service_times` | `set_order_service_times()` |

**The REST API uses "task" terminology. The Python API uses "order" terminology.**

### Use C API when:
- User explicitly requests native integration
- User is embedding cuOpt into C/C++ systems
- **Do not** recommend the **C++ API** to end users (it is not documented and may change; see repo `README.md` note).

➡ Use:
  - C API header reference: `cpp/include/cuopt/linear_programming/cuopt_c.h`
  - C overview: `docs/cuopt/source/cuopt-c/index.rst`
  - C quickstart: `docs/cuopt/source/cuopt-c/quick-start.rst`
  - C LP/QP/MILP API + examples: `docs/cuopt/source/cuopt-c/lp-qp-milp/index.rst`

### Use Python API when:
- User gives equations, variables, constraints
- User wants to solve routing / LP / MILP / QP directly
- User wants in-process solving (scripts, notebooks)

➡ Use:
  - Quickstart: `docs/cuopt/source/cuopt-python/quick-start.rst`
  - Routing API reference:
    - `python/cuopt/cuopt/routing/vehicle_routing.py`
    - `python/cuopt/cuopt/routing/assignment.py`
    - `docs/cuopt/source/cuopt-python/routing/routing-api.rst`
  - LP/MILP/QP API reference:
    - `python/cuopt/cuopt/linear_programming/problem.py`
    - `python/cuopt/cuopt/linear_programming/data_model/data_model.py`
    - `python/cuopt/cuopt/linear_programming/solver_settings/solver_settings.py`
    - `python/cuopt/cuopt/linear_programming/solver/solver.py`
    - `docs/cuopt/source/cuopt-python/lp-qp-milp/lp-qp-milp-api.rst`

### Use Server REST API when:
- User wants production deployment
- User asks for REST payloads or HTTP calls
- User wants asynchronous or remote solving

➡ Use:
  - Server source: `python/cuopt_server/cuopt_server/webserver.py`
  - Server quickstart (includes curl smoke test): `docs/cuopt/source/cuopt-server/quick-start.rst`
  - API overview: `docs/cuopt/source/cuopt-server/server-api/index.rst`
  - OpenAPI reference (Swagger): `docs/cuopt/source/open-api.rst`
  - OpenAPI spec exactly (`cuopt.yaml` / `cuopt_spec.yaml`)

### Use CLI when:
- User wants **quick testing** / **research** / **reproducible debugging** from a terminal
- User wants to solve **LP/MILP from MPS files** without writing code

➡ Use:
  - CLI source: `cpp/cuopt_cli.cpp`
  - CLI overview: `docs/cuopt/source/cuopt-cli/index.rst`
  - CLI quickstart: `docs/cuopt/source/cuopt-cli/quick-start.rst`
  - CLI examples: `docs/cuopt/source/cuopt-cli/cli-examples.rst`

**Note on MPS inputs:** having an `.mps` file does **not** imply you must use the CLI.
Choose based on integration/deployment needs:

- **CLI**: fastest local repro (LP/MILP from MPS)
- **C API**: native embedding; includes MPS-based examples under `docs/cuopt/source/cuopt-c/lp-qp-milp/examples/`
- **Server**: can use its local-file feature (see server docs/OpenAPI) when running a service

---

## Installation (minimal)

Pick **one** installation method and match it to your CUDA major version (cuOpt publishes CUDA-variant packages).

### pip

- **Python API**:

```bash
# Simplest (latest compatible from the index):
# CUDA 13
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu13

# CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com cuopt-cu12

# Recommended (reproducible; pin to the current major/minor release line):
# CUDA 13
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu13==26.2.*'

# CUDA 12
pip install --extra-index-url=https://pypi.nvidia.com 'cuopt-cu12==26.2.*'
```

- **Server + thin client (self-hosted)**:

```bash
# Simplest:
# CUDA 12 example
pip install --extra-index-url=https://pypi.nvidia.com \
  cuopt-server-cu12 cuopt-sh-client

# Recommended (reproducible):
# CUDA 12 example
pip install --extra-index-url=https://pypi.nvidia.com \
  nvidia-cuda-runtime-cu12==12.9.* \
  cuopt-server-cu12==26.02.* cuopt-sh-client==26.02.*
```

### conda

```bash
# Simplest:
# Python API
conda install -c rapidsai -c conda-forge -c nvidia cuopt

# Server + thin client
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server cuopt-sh-client

# Recommended (reproducible):
# Python API
conda install -c rapidsai -c conda-forge -c nvidia cuopt=26.02.* cuda-version=26.02.*

# Server + thin client
conda install -c rapidsai -c conda-forge -c nvidia cuopt-server=26.02.* cuopt-sh-client=26.02.*
```

### container

```bash
docker pull nvidia/cuopt:latest-cuda12.9-py3.13
docker run --gpus all -it --rm -p 8000:8000 -e CUOPT_SERVER_PORT=8000 nvidia/cuopt:latest-cuda12.9-py3.13
```

For full, up-to-date installation instructions (including nightlies), see:

- `docs/cuopt/source/cuopt-python/quick-start.rst`
- `docs/cuopt/source/cuopt-server/quick-start.rst`

---

## C API Examples & Templates

Use the C examples + Makefile under `docs/cuopt/source/cuopt-c/lp-qp-milp/examples/`.

### C API: Simple LP Example

```c
/*
 * Simple LP C API Example
 *
 * Solve: minimize  -0.2*x1 + 0.1*x2
 *        subject to  3.0*x1 + 4.0*x2 <= 5.4
 *                    2.7*x1 + 10.1*x2 <= 4.9
 *                    x1, x2 >= 0
 *
 * Expected: x1 = 1.8, x2 = 0.0, objective = -0.36
 */
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2;
    cuopt_int_t num_constraints = 2;

    // Constraint matrix in CSR format
    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t column_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    // Objective coefficients: minimize -0.2*x1 + 0.1*x2
    cuopt_float_t objective_coefficients[] = {-0.2, 0.1};

    // Constraint bounds (ranged form: lower <= Ax <= upper)
    cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
    cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

    // Variable bounds: x1, x2 >= 0
    cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
    cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

    // Variable types: both continuous
    char variable_types[] = {CUOPT_CONTINUOUS, CUOPT_CONTINUOUS};

    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;

    // Create the problem
    status = cuOptCreateRangedProblem(
        num_constraints,
        num_variables,
        CUOPT_MINIMIZE,
        0.0,                      // objective offset
        objective_coefficients,
        row_offsets,
        column_indices,
        values,
        constraint_lower_bounds,
        constraint_upper_bounds,
        var_lower_bounds,
        var_upper_bounds,
        variable_types,
        &problem
    );
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        return 1;
    }

    // Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) {
        printf("Error creating solver settings: %d\n", status);
        goto DONE;
    }

    // Set solver parameters
    cuOptSetFloatParameter(settings, CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, 0.0001);
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 60.0);

    // Solve the problem
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) {
        printf("Error solving problem: %d\n", status);
        goto DONE;
    }

    // Get and print results
    cuOptGetSolveTime(solution, &time);
    cuOptGetTerminationStatus(solution, &termination_status);
    cuOptGetObjectiveValue(solution, &objective_value);

    printf("Termination status: %d\n", termination_status);
    printf("Solve time: %f seconds\n", time);
    printf("Objective value: %f\n", objective_value);

    // Get solution values
    cuopt_float_t* solution_values = (cuopt_float_t*)malloc(
        num_variables * sizeof(cuopt_float_t)
    );
    cuOptGetPrimalSolution(solution, solution_values);
    for (cuopt_int_t i = 0; i < num_variables; i++) {
        printf("x%d = %f\n", i + 1, solution_values[i]);
    }
    free(solution_values);

DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
```

### C API: MILP Example (with integer variables)

```c
/*
 * Simple MILP C API Example
 *
 * Solve: minimize  -0.2*x1 + 0.1*x2
 *        subject to  3.0*x1 + 4.0*x2 <= 5.4
 *                    2.7*x1 + 10.1*x2 <= 4.9
 *                    x1 integer, x2 continuous, both >= 0
 */
#include <cuopt/linear_programming/cuopt_c.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    cuOptOptimizationProblem problem = NULL;
    cuOptSolverSettings settings = NULL;
    cuOptSolution solution = NULL;

    cuopt_int_t num_variables = 2;
    cuopt_int_t num_constraints = 2;

    // Constraint matrix in CSR format
    cuopt_int_t row_offsets[] = {0, 2, 4};
    cuopt_int_t column_indices[] = {0, 1, 0, 1};
    cuopt_float_t values[] = {3.0, 4.0, 2.7, 10.1};

    // Objective coefficients
    cuopt_float_t objective_coefficients[] = {-0.2, 0.1};

    // Constraint bounds
    cuopt_float_t constraint_upper_bounds[] = {5.4, 4.9};
    cuopt_float_t constraint_lower_bounds[] = {-CUOPT_INFINITY, -CUOPT_INFINITY};

    // Variable bounds
    cuopt_float_t var_lower_bounds[] = {0.0, 0.0};
    cuopt_float_t var_upper_bounds[] = {CUOPT_INFINITY, CUOPT_INFINITY};

    // Variable types: x1 = INTEGER, x2 = CONTINUOUS
    char variable_types[] = {CUOPT_INTEGER, CUOPT_CONTINUOUS};

    cuopt_int_t status;
    cuopt_float_t time;
    cuopt_int_t termination_status;
    cuopt_float_t objective_value;

    // Create the problem (same API, but with integer variable types)
    status = cuOptCreateRangedProblem(
        num_constraints,
        num_variables,
        CUOPT_MINIMIZE,
        0.0,
        objective_coefficients,
        row_offsets,
        column_indices,
        values,
        constraint_lower_bounds,
        constraint_upper_bounds,
        var_lower_bounds,
        var_upper_bounds,
        variable_types,
        &problem
    );
    if (status != CUOPT_SUCCESS) {
        printf("Error creating problem: %d\n", status);
        return 1;
    }

    // Create solver settings
    status = cuOptCreateSolverSettings(&settings);
    if (status != CUOPT_SUCCESS) goto DONE;

    // Set MIP-specific parameters
    cuOptSetFloatParameter(settings, CUOPT_MIP_ABSOLUTE_TOLERANCE, 0.0001);
    cuOptSetFloatParameter(settings, CUOPT_MIP_RELATIVE_GAP, 0.01);  // 1% gap
    cuOptSetFloatParameter(settings, CUOPT_TIME_LIMIT, 120.0);

    // Solve
    status = cuOptSolve(problem, settings, &solution);
    if (status != CUOPT_SUCCESS) goto DONE;

    // Get results
    cuOptGetSolveTime(solution, &time);
    cuOptGetTerminationStatus(solution, &termination_status);
    cuOptGetObjectiveValue(solution, &objective_value);

    printf("Termination status: %d\n", termination_status);
    printf("Solve time: %f seconds\n", time);
    printf("Objective value: %f\n", objective_value);

    cuopt_float_t* solution_values = malloc(num_variables * sizeof(cuopt_float_t));
    cuOptGetPrimalSolution(solution, solution_values);
    printf("x1 (integer) = %f\n", solution_values[0]);
    printf("x2 (continuous) = %f\n", solution_values[1]);
    free(solution_values);

DONE:
    cuOptDestroyProblem(&problem);
    cuOptDestroySolverSettings(&settings);
    cuOptDestroySolution(&solution);

    return (status == CUOPT_SUCCESS) ? 0 : 1;
}
```

### C API: Build & Run

```bash
# Find include and library paths (adjust based on installation)
# If installed via conda:
export INCLUDE_PATH="${CONDA_PREFIX}/include"
export LIBCUOPT_LIBRARY_PATH="${CONDA_PREFIX}/lib"

# Or find automatically:
# INCLUDE_PATH=$(find / -name "cuopt_c.h" -path "*/linear_programming/*" \
#                -printf "%h\n" | sed 's/\/linear_programming//' 2>/dev/null)
# LIBCUOPT_LIBRARY_PATH=$(dirname $(find / -name "libcuopt.so" 2>/dev/null))

# Compile
gcc -I ${INCLUDE_PATH} -L ${LIBCUOPT_LIBRARY_PATH} \
    -o simple_lp_example simple_lp_example.c -lcuopt

# Run
LD_LIBRARY_PATH=${LIBCUOPT_LIBRARY_PATH}:$LD_LIBRARY_PATH ./simple_lp_example
```

---

## Python API Examples & Templates

### Python: Routing with Time Windows & Capacities (VRP)

```python
"""
Vehicle Routing Problem with:
- 1 depot (location 0)
- 5 customer locations (1-5)
- 2 vehicles with capacity 100 each
- Time windows for each location
- Demand at each customer
"""
import cudf
from cuopt import routing

# Cost/distance matrix (6x6: depot + 5 customers)
cost_matrix = cudf.DataFrame([
    [0,  10, 15, 20, 25, 30],  # From depot
    [10,  0, 12, 18, 22, 28],  # From customer 1
    [15, 12,  0, 10, 15, 20],  # From customer 2
    [20, 18, 10,  0,  8, 15],  # From customer 3
    [25, 22, 15,  8,  0, 10],  # From customer 4
    [30, 28, 20, 15, 10,  0],  # From customer 5
], dtype="float32")

# Also use as transit time matrix (same values for simplicity)
transit_time_matrix = cost_matrix.copy(deep=True)

# Order data (customers 1-5)
order_locations = cudf.Series([1, 2, 3, 4, 5])  # Location indices for orders

# Demand at each customer (single capacity dimension)
demand = cudf.Series([20, 30, 25, 15, 35], dtype="int32")  # Units to deliver

# Vehicle capacities (must match demand dimensions)
vehicle_capacity = cudf.Series([100, 100], dtype="int32")  # Each vehicle can carry 100 units

# Time windows for orders [earliest, latest]
order_earliest = cudf.Series([0,  10, 20,  0, 30], dtype="int32")
order_latest = cudf.Series([50, 60, 70, 80, 90], dtype="int32")

# Service time at each customer
service_times = cudf.Series([5, 5, 5, 5, 5], dtype="int32")

# Fleet configuration
n_fleet = 2

# Vehicle start/end locations (both start and return to depot)
vehicle_start = cudf.Series([0, 0], dtype="int32")
vehicle_end = cudf.Series([0, 0], dtype="int32")

# Vehicle time windows (operating hours)
vehicle_earliest = cudf.Series([0, 0], dtype="int32")
vehicle_latest = cudf.Series([200, 200], dtype="int32")

# Build the data model
dm = routing.DataModel(
    n_locations=cost_matrix.shape[0],
    n_fleet=n_fleet,
    n_orders=len(order_locations)
)

# Add matrices
dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(transit_time_matrix)

# Add order data
dm.set_order_locations(order_locations)
dm.set_order_time_windows(order_earliest, order_latest)
dm.set_order_service_times(service_times)

# Add capacity dimension (name, demand_per_order, capacity_per_vehicle)
dm.add_capacity_dimension("weight", demand, vehicle_capacity)

# Add fleet data
dm.set_vehicle_locations(vehicle_start, vehicle_end)
dm.set_vehicle_time_windows(vehicle_earliest, vehicle_latest)

# Configure solver
ss = routing.SolverSettings()
ss.set_time_limit(10)  # seconds

# Solve
solution = routing.Solve(dm, ss)

# Check solution status
print(f"Status: {solution.get_status()}")

# Display routes
if solution.get_status() == 0:  # Success
    print("\n--- Solution Found ---")
    solution.display_routes()

    # Get detailed route data
    route_df = solution.get_route()
    print("\nDetailed route data:")
    print(route_df)

    # Get objective value (total cost)
    print(f"\nTotal cost: {solution.get_total_objective()}")
```

### Python: Pickup and Delivery Problem (PDP)

```python
"""
Pickup and Delivery Problem:
- Items must be picked up from one location and delivered to another
- Same vehicle must do both pickup and delivery
- Pickup must occur before delivery
"""
import cudf
from cuopt import routing

# Cost matrix (depot + 4 locations)
cost_matrix = cudf.DataFrame([
    [0, 10, 20, 30, 40],
    [10, 0, 15, 25, 35],
    [20, 15, 0, 10, 20],
    [30, 25, 10, 0, 15],
    [40, 35, 20, 15, 0],
], dtype="float32")

transit_time_matrix = cost_matrix.copy(deep=True)

n_fleet = 2
n_orders = 4  # 2 pickup-delivery pairs = 4 orders

# Orders: pickup at loc 1 -> deliver at loc 2, pickup at loc 3 -> deliver at loc 4
order_locations = cudf.Series([1, 2, 3, 4])

# Pickup and delivery pairs (indices into order array)
# Order 0 (pickup) pairs with Order 1 (delivery)
# Order 2 (pickup) pairs with Order 3 (delivery)
pickup_indices = cudf.Series([0, 2])
delivery_indices = cudf.Series([1, 3])

# Demand: positive for pickup, negative for delivery (must sum to 0 per pair)
demand = cudf.Series([10, -10, 15, -15], dtype="int32")
vehicle_capacity = cudf.Series([50, 50], dtype="int32")

# Build model
dm = routing.DataModel(
    n_locations=cost_matrix.shape[0],
    n_fleet=n_fleet,
    n_orders=n_orders
)

dm.add_cost_matrix(cost_matrix)
dm.add_transit_time_matrix(transit_time_matrix)
dm.set_order_locations(order_locations)

# Add capacity dimension
dm.add_capacity_dimension("load", demand, vehicle_capacity)

# Set pickup and delivery constraints
dm.set_pickup_delivery_pairs(pickup_indices, delivery_indices)

# Fleet setup
dm.set_vehicle_locations(
    cudf.Series([0, 0]),  # Start at depot
    cudf.Series([0, 0])   # Return to depot
)

# Solve
ss = routing.SolverSettings()
ss.set_time_limit(10)
solution = routing.Solve(dm, ss)

print(f"Status: {solution.get_status()}")
if solution.get_status() == 0:
    solution.display_routes()
```

### Python: Linear Programming (LP)

```python
"""
Production Planning LP:
    maximize    40*chairs + 30*tables  (profit)
    subject to  2*chairs + 3*tables <= 240  (wood constraint)
                4*chairs + 2*tables <= 200  (labor constraint)
                chairs, tables >= 0
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MAXIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("ProductionPlanning")

# Decision variables (continuous, non-negative)
chairs = problem.addVariable(lb=0, vtype=CONTINUOUS, name="chairs")
tables = problem.addVariable(lb=0, vtype=CONTINUOUS, name="tables")

# Constraints
problem.addConstraint(2 * chairs + 3 * tables <= 240, name="wood")
problem.addConstraint(4 * chairs + 2 * tables <= 200, name="labor")

# Objective: maximize profit
problem.setObjective(40 * chairs + 30 * tables, sense=MAXIMIZE)

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
settings.set_parameter("log_to_console", 1)

# Solve
problem.solve(settings)

# Results
print(f"Status: {problem.Status.name}")
print(f"Optimal profit: ${problem.ObjValue:.2f}")
print(f"Chairs to produce: {chairs.getValue():.1f}")
print(f"Tables to produce: {tables.getValue():.1f}")

# Get dual values (shadow prices)
wood_constraint = problem.getConstraint("wood")
labor_constraint = problem.getConstraint("labor")
print(f"\nShadow price (wood): ${wood_constraint.DualValue:.2f} per unit")
print(f"Shadow price (labor): ${labor_constraint.DualValue:.2f} per unit")
```

### Python: Mixed-Integer Linear Programming (MILP)

```python
"""
Facility Location MILP:
- Decide which warehouses to open (binary)
- Assign customers to open warehouses
- Minimize fixed costs + transportation costs
"""
from cuopt.linear_programming.problem import (
    Problem, CONTINUOUS, INTEGER, MINIMIZE
)
from cuopt.linear_programming.solver_settings import SolverSettings

# Problem data
warehouses = ["W1", "W2", "W3"]
customers = ["C1", "C2", "C3", "C4"]

fixed_costs = {"W1": 100, "W2": 150, "W3": 120}
capacities = {"W1": 50, "W2": 70, "W3": 60}
demands = {"C1": 20, "C2": 25, "C3": 15, "C4": 30}

# Transportation cost from warehouse to customer
transport_cost = {
    ("W1", "C1"): 5, ("W1", "C2"): 8, ("W1", "C3"): 6, ("W1", "C4"): 10,
    ("W2", "C1"): 7, ("W2", "C2"): 4, ("W2", "C3"): 9, ("W2", "C4"): 5,
    ("W3", "C1"): 6, ("W3", "C2"): 7, ("W3", "C3"): 4, ("W3", "C4"): 8,
}

# Create problem
problem = Problem("FacilityLocation")

# Decision variables
# y[w] = 1 if warehouse w is open (binary: INTEGER with bounds 0-1)
y = {w: problem.addVariable(lb=0, ub=1, vtype=INTEGER, name=f"open_{w}") for w in warehouses}

# x[w,c] = units shipped from w to c
x = {
    (w, c): problem.addVariable(lb=0, vtype=CONTINUOUS, name=f"ship_{w}_{c}")
    for w in warehouses for c in customers
}

# Objective: minimize fixed + transportation costs
problem.setObjective(
    sum(fixed_costs[w] * y[w] for w in warehouses) +
    sum(transport_cost[w, c] * x[w, c] for w in warehouses for c in customers),
    sense=MINIMIZE
)

# Constraints
# 1. Meet customer demand
for c in customers:
    problem.addConstraint(
        sum(x[w, c] for w in warehouses) == demands[c],
        name=f"demand_{c}"
    )

# 2. Respect warehouse capacity (only if open)
for w in warehouses:
    problem.addConstraint(
        sum(x[w, c] for c in customers) <= capacities[w] * y[w],
        name=f"capacity_{w}"
    )

# Solver settings
settings = SolverSettings()
settings.set_parameter("time_limit", 120)
settings.set_parameter("mip_relative_gap", 0.01)  # 1% optimality gap

# Solve
problem.solve(settings)

# Results
print(f"Status: {problem.Status.name}")
print(f"Total cost: ${problem.ObjValue:.2f}")
print("\nOpen warehouses:")
for w in warehouses:
    if y[w].getValue() > 0.5:
        print(f"  {w} (fixed cost: ${fixed_costs[w]})")

print("\nShipments:")
for w in warehouses:
    for c in customers:
        shipped = x[w, c].getValue()
        if shipped > 0.01:
            print(f"  {w} -> {c}: {shipped:.1f} units")
```

### Python: Quadratic Programming (QP) - Beta

```python
"""
Portfolio Optimization QP (more complex):
    minimize    x^T * Q * x  (variance/risk)
    subject to  sum(x) = 1         (fully invested)
                r^T * x >= target  (minimum return)
                x >= 0             (no short selling)
"""
from cuopt.linear_programming.problem import Problem, CONTINUOUS, MINIMIZE
from cuopt.linear_programming.solver_settings import SolverSettings

# Create problem
problem = Problem("PortfolioOptimization")

# Decision variables: portfolio weights for 3 assets
x1 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_a")
x2 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_b")
x3 = problem.addVariable(lb=0.0, ub=1.0, vtype=CONTINUOUS, name="stock_c")

# Expected returns
r1, r2, r3 = 0.12, 0.08, 0.05  # 12%, 8%, 5%
target_return = 0.08

# Covariance matrix values (symmetric)
# Q = [[0.04,  0.01,  0.005],
#      [0.01,  0.02,  0.008],
#      [0.005, 0.008, 0.01 ]]

# Quadratic objective: minimize variance = x^T * Q * x
# Expand: 0.04*x1^2 + 0.02*x2^2 + 0.01*x3^2 + 2*0.01*x1*x2 + 2*0.005*x1*x3 + 2*0.008*x2*x3
problem.setObjective(
    0.04 * x1 * x1 + 0.02 * x2 * x2 + 0.01 * x3 * x3 +
    0.02 * x1 * x2 + 0.01 * x1 * x3 + 0.016 * x2 * x3,
    sense=MINIMIZE
)

# Constraints
problem.addConstraint(x1 + x2 + x3 == 1, name="fully_invested")
problem.addConstraint(r1 * x1 + r2 * x2 + r3 * x3 >= target_return, name="min_return")

# Solve
settings = SolverSettings()
settings.set_parameter("time_limit", 60)
problem.solve(settings)

# Results
print(f"Status: {problem.Status.name}")
print(f"Portfolio variance: {problem.ObjValue:.6f}")
print(f"Portfolio std dev: {problem.ObjValue**0.5:.4f}")
print(f"\nOptimal allocation:")
print(f"  Stock A: {x1.getValue()*100:.2f}%")
print(f"  Stock B: {x2.getValue()*100:.2f}%")
print(f"  Stock C: {x3.getValue()*100:.2f}%")
exp_return = r1*x1.getValue() + r2*x2.getValue() + r3*x3.getValue()
print(f"\nExpected return: {exp_return*100:.2f}%")
```

---

## Server REST API Examples & Templates

### Server: Start the Server

```bash
# Start server in background
python3 -m cuopt_server.cuopt_service --ip 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
sleep 5
curl -fsS "http://localhost:8000/cuopt/health"
```

### Server: Routing Request (curl)

```bash
# Submit a VRP request
REQID=$(curl -s --location "http://localhost:8000/cuopt/request" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" \
  -d '{
    "cost_matrix_data": {
      "data": {
        "0": [
          [0, 10, 15, 20],
          [10, 0, 12, 18],
          [15, 12, 0, 10],
          [20, 18, 10, 0]
        ]
      }
    },
    "travel_time_matrix_data": {
      "data": {
        "0": [
          [0, 10, 15, 20],
          [10, 0, 12, 18],
          [15, 12, 0, 10],
          [20, 18, 10, 0]
        ]
      }
    },
    "task_data": {
      "task_locations": [1, 2, 3],
      "demand": [[10, 15, 20]],
      "task_time_windows": [[0, 100], [10, 80], [20, 90]],
      "service_times": [5, 5, 5]
    },
    "fleet_data": {
      "vehicle_locations": [[0, 0], [0, 0]],
      "capacities": [[50, 50]],
      "vehicle_time_windows": [[0, 200], [0, 200]]
    },
    "solver_config": {
      "time_limit": 5
    }
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Poll for solution
sleep 2
curl -s --location "http://localhost:8000/cuopt/solution/${REQID}" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" | jq .
```

### Server: Routing Request (Python with requests)

```python
import requests
import time

SERVER_URL = "http://localhost:8000"
HEADERS = {
    "Content-Type": "application/json",
    "CLIENT-VERSION": "custom"
}

# VRP problem data
payload = {
    "cost_matrix_data": {
        "data": {
            "0": [
                [0, 10, 15, 20, 25],
                [10, 0, 12, 18, 22],
                [15, 12, 0, 10, 15],
                [20, 18, 10, 0, 8],
                [25, 22, 15, 8, 0]
            ]
        }
    },
    "travel_time_matrix_data": {
        "data": {
            "0": [
                [0, 10, 15, 20, 25],
                [10, 0, 12, 18, 22],
                [15, 12, 0, 10, 15],
                [20, 18, 10, 0, 8],
                [25, 22, 15, 8, 0]
            ]
        }
    },
    "task_data": {
        "task_locations": [1, 2, 3, 4],
        "demand": [[20, 30, 25, 15]],
        "task_time_windows": [[0, 50], [10, 60], [20, 70], [0, 80]],
        "service_times": [5, 5, 5, 5]
    },
    "fleet_data": {
        "vehicle_locations": [[0, 0], [0, 0]],
        "capacities": [[100, 100]],
        "vehicle_time_windows": [[0, 200], [0, 200]]
    },
    "solver_config": {
        "time_limit": 10
    }
}

# Submit request
response = requests.post(
    f"{SERVER_URL}/cuopt/request",
    json=payload,
    headers=HEADERS
)
response.raise_for_status()
req_id = response.json()["reqId"]
print(f"Request submitted: {req_id}")

# Poll for solution
max_attempts = 30
for attempt in range(max_attempts):
    response = requests.get(
        f"{SERVER_URL}/cuopt/solution/{req_id}",
        headers=HEADERS
    )
    result = response.json()

    if "response" in result:
        solver_response = result["response"].get("solver_response", {})
        print(f"\nSolution found!")
        print(f"Status: {solver_response.get('status', 'N/A')}")
        print(f"Cost: {solver_response.get('solution_cost', 'N/A')}")

        if "vehicle_data" in solver_response:
            for vid, vdata in solver_response["vehicle_data"].items():
                route = vdata.get("route", [])
                print(f"Vehicle {vid}: {' -> '.join(map(str, route))}")
        break
    else:
        print(f"Waiting... (attempt {attempt + 1})")
        time.sleep(1)
```

### Server: LP/MILP Request

```bash
# Submit LP problem via REST
# Production Planning: maximize 40*chairs + 30*tables
#   subject to: 2*chairs + 3*tables <= 240 (wood)
#               4*chairs + 2*tables <= 200 (labor)
#               chairs, tables >= 0
REQID=$(curl -s --location "http://localhost:8000/cuopt/request" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" \
  -d '{
    "csr_constraint_matrix": {
      "offsets": [0, 2, 4],
      "indices": [0, 1, 0, 1],
      "values": [2.0, 3.0, 4.0, 2.0]
    },
    "constraint_bounds": {
      "upper_bounds": [240.0, 200.0],
      "lower_bounds": ["ninf", "ninf"]
    },
    "objective_data": {
      "coefficients": [40.0, 30.0],
      "scalability_factor": 1.0,
      "offset": 0.0
    },
    "variable_bounds": {
      "upper_bounds": ["inf", "inf"],
      "lower_bounds": [0.0, 0.0]
    },
    "maximize": true,
    "solver_config": {
      "tolerances": {"optimality": 0.0001},
      "time_limit": 60
    }
  }' | jq -r '.reqId')

echo "Request ID: $REQID"

# Get solution
sleep 2
curl -s --location "http://localhost:8000/cuopt/solution/${REQID}" \
  --header 'Content-Type: application/json' \
  --header "CLIENT-VERSION: custom" | jq .
```

---

## CLI Examples & Templates

### CLI: LP from MPS File

```bash
# Create sample LP problem in MPS format
cat > production.mps << 'EOF'
* Production Planning Problem
* maximize 40*chairs + 30*tables
* s.t.    2*chairs + 3*tables <= 240 (wood)
*         4*chairs + 2*tables <= 200 (labor)
NAME          PRODUCTION
ROWS
 N  PROFIT
 L  WOOD
 L  LABOR
COLUMNS
    CHAIRS    PROFIT           -40.0
    CHAIRS    WOOD               2.0
    CHAIRS    LABOR              4.0
    TABLES    PROFIT           -30.0
    TABLES    WOOD               3.0
    TABLES    LABOR              2.0
RHS
    RHS1      WOOD             240.0
    RHS1      LABOR            200.0
ENDATA
EOF

# Solve with cuopt_cli
cuopt_cli production.mps

# Solve with options
cuopt_cli production.mps --time-limit 30

# Cleanup
rm -f production.mps
```

### CLI: MILP from MPS File

```bash
# Create MILP problem (with integer variables)
cat > facility.mps << 'EOF'
* Facility location - simplified
* Binary variables for opening facilities
NAME          FACILITY
ROWS
 N  COST
 G  DEMAND1
 L  CAP1
 L  CAP2
COLUMNS
    MARKER    'MARKER'         'INTORG'
    OPEN1     COST             100.0
    OPEN1     CAP1              50.0
    OPEN2     COST             150.0
    OPEN2     CAP2              70.0
    MARKER    'MARKER'         'INTEND'
    SHIP11    COST               5.0
    SHIP11    DEMAND1            1.0
    SHIP11    CAP1              -1.0
    SHIP21    COST               7.0
    SHIP21    DEMAND1            1.0
    SHIP21    CAP2              -1.0
RHS
    RHS1      DEMAND1           30.0
BOUNDS
 BV BND1      OPEN1
 BV BND1      OPEN2
 LO BND1      SHIP11             0.0
 LO BND1      SHIP21             0.0
ENDATA
EOF

# Solve MILP
cuopt_cli facility.mps --time-limit 60 --mip-relative-tolerance 0.01

# Cleanup
rm -f facility.mps
```

### CLI: Common Options

```bash
# Show all options
cuopt_cli --help

# Set time limit (seconds)
cuopt_cli problem.mps --time-limit 120

# Set MIP relative gap tolerance (for MILP, e.g., 0.1% = 0.001)
cuopt_cli problem.mps --mip-relative-tolerance 0.001

# Set MIP absolute tolerance (for MILP)
cuopt_cli problem.mps --mip-absolute-tolerance 0.0001

# Enable presolve
cuopt_cli problem.mps --presolve

# Set iteration limit
cuopt_cli problem.mps --iteration-limit 10000

# Specify solver method (0=auto, 1=pdlp, 2=dual_simplex, 3=barrier, etc.)
cuopt_cli problem.mps --method 1
```

---

## Common user requests → action map

| User asks | Action |
|----------|--------|
| "Embed cuOpt in C/C++ app" | Use C API |
| "Solve this routing problem" | Use routing API |
| "Solve this LP/MILP" | Use Python LP API |
| "Give REST payload" | Open OpenAPI spec |
| "I have MPS file" | CLI for quick repro **or** C API MPS examples **or** Server local-file feature (choose based on deployment) |
| "422 / schema error" | Fix payload |
| "Solver too slow" | Adjust allowed settings |
| "Change solver logic" | Switch agent |

---

## Solver settings (safe adjustments)

Allowed:
- Time limit
- Gap tolerances (if documented)
- Verbosity / logging

Not allowed:
- Changing heuristics
- Modifying internals
- Undocumented parameters

---

## Data formats & performance

- **Payload formats**: JSON is the default; msgpack/zlib are supported for some endpoints (see server docs/OpenAPI).
- **GPU constraints**: requires a supported NVIDIA GPU/driver/CUDA runtime; see the system requirements in the main README and docs.
- **Tuning**: use solver settings (e.g., time limits) and avoid unnecessary host↔device churn; follow the feature docs under `docs/cuopt/source/`.

---

## Error handling (agent rules)

- **Validation errors (HTTP 4xx)**: treat as schema/typing issues; consult OpenAPI spec and fix the request payload.
- **Server errors (HTTP 5xx)**: capture `reqId`, poll logs/status endpoints where applicable, and reproduce with the smallest request.
- **Never "paper over" errors** by changing schemas or endpoints—align with the documented API.
- **Debugging a failure**: search existing [GitHub Issues](https://github.com/NVIDIA/cuopt/issues) first (use exact error text + cuOpt/CUDA/driver versions). If no match, file a new issue with a minimal repro, expected vs actual behavior, environment details, and any logs/`reqId`.

For common troubleshooting and known issues, see:

- `docs/cuopt/source/faq.rst`
- `docs/cuopt/source/resources.rst`

---

## Additional resources (when to use)

- **Examples / notebooks**: [NVIDIA/cuopt-examples](https://github.com/NVIDIA/cuopt-examples) → runnable notebooks
- **Google Colab**: [cuopt-examples notebooks on Colab](https://colab.research.google.com/github/nvidia/cuopt-examples/) → runnable examples
- **Official docs**: [cuOpt User Guide](https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html) → modeling correctness
- **Videos/tutorials**: [cuOpt examples and tutorials videos](https://docs.nvidia.com/cuopt/user-guide/latest/resources.html#cuopt-examples-and-tutorials-videos) → unclear behavior
- **Try in the cloud**: [NVIDIA Launchable](https://brev.nvidia.com/launchable/deploy?launchableID=env-2qIG6yjGKDtdMSjXHcuZX12mDNJ) → GPU environments
- **Support / questions**: [NVIDIA Developer Forums (cuOpt)](https://forums.developer.nvidia.com/c/ai-data-science/nvidia-cuopt/514) → unclear behavior
- **Bugs / feature requests**: [GitHub Issues](https://github.com/NVIDIA/cuopt/issues) → unclear behavior

---

## Final agent rules (non-negotiable)

- Never invent APIs
- Never assume undocumented behavior
- Always choose interface first
- Prefer correctness over speed
- When unsure → open docs or ask user to clarify
