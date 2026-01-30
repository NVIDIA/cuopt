---
name: cuopt-routing
description: Solve vehicle routing problems (VRP, TSP, PDP) with NVIDIA cuOpt. Use when the user asks about delivery optimization, fleet routing, time windows, capacities, pickup-delivery pairs, or traveling salesman problems.
---

# cuOpt Routing Skill

> **Prerequisites**: Read `cuopt-user-rules/SKILL.md` first for behavior rules.

Model and solve vehicle routing problems using NVIDIA cuOpt's GPU-accelerated solver.

## Before You Start: Required Questions

**Ask these if not already clear:**

1. **Problem type?**
   - TSP (single vehicle, visit all locations)
   - VRP (multiple vehicles, capacity constraints)
   - PDP (pickup and delivery pairs)

2. **What constraints?**
   - Time windows (earliest/latest arrival)?
   - Vehicle capacities?
   - Service times at locations?
   - Multiple depots?
   - Vehicle-specific start/end locations?

3. **What data do you have?**
   - Cost/distance matrix or coordinates?
   - Demand per location?
   - Fleet size fixed or to optimize?

4. **Interface preference?**
   - Python API (in-process)
   - REST Server (production/async)

## Interface Support

| Interface | Routing Support |
|-----------|:---------------:|
| Python    | ✓               |
| REST      | ✓               |
| C API     | ✗               |
| CLI       | ✗               |

## Quick Reference: Python API

### Minimal VRP Example

```python
import cudf
from cuopt import routing

# Cost matrix (n_locations x n_locations)
cost_matrix = cudf.DataFrame([
    [0, 10, 15, 20],
    [10, 0, 12, 18],
    [15, 12, 0, 10],
    [20, 18, 10, 0],
], dtype="float32")

# Build data model
dm = routing.DataModel(
    n_locations=4,      # Total locations including depot
    n_fleet=2,          # Number of vehicles
    n_orders=3          # Orders to fulfill (locations 1,2,3)
)

# Required: cost matrix
dm.add_cost_matrix(cost_matrix)

# Required: order locations (which location each order is at)
dm.set_order_locations(cudf.Series([1, 2, 3]))

# Solve
solution = routing.Solve(dm, routing.SolverSettings())

# Check result
if solution.get_status() == 0:  # SUCCESS
    solution.display_routes()
```

### Adding Constraints

```python
# Time windows (need transit time matrix)
dm.add_transit_time_matrix(transit_time_matrix)
dm.set_order_time_windows(
    cudf.Series([0, 10, 20]),    # earliest
    cudf.Series([50, 60, 70])    # latest
)

# Capacities
dm.add_capacity_dimension(
    "weight",
    cudf.Series([20, 30, 25]),       # demand per order
    cudf.Series([100, 100])          # capacity per vehicle
)

# Service times
dm.set_order_service_times(cudf.Series([5, 5, 5]))

# Vehicle locations (start/end)
dm.set_vehicle_locations(
    cudf.Series([0, 0]),  # start at depot
    cudf.Series([0, 0])   # return to depot
)

# Vehicle time windows
dm.set_vehicle_time_windows(
    cudf.Series([0, 0]),      # earliest start
    cudf.Series([200, 200])   # latest return
)
```

### Pickup and Delivery (PDP)

```python
# Demand: positive=pickup, negative=delivery (must sum to 0 per pair)
demand = cudf.Series([10, -10, 15, -15])

# Pair indices: order 0 pairs with 1, order 2 pairs with 3
dm.set_pickup_delivery_pairs(
    cudf.Series([0, 2]),   # pickup order indices
    cudf.Series([1, 3])    # delivery order indices
)
```

## Quick Reference: REST Server

### Terminology Difference

| Concept | Python API | REST Server |
|---------|------------|-------------|
| Jobs | `order_locations` | `task_locations` |
| Time windows | `set_order_time_windows()` | `task_time_windows` |
| Service times | `set_order_service_times()` | `service_times` |

### Minimal REST Payload

```json
{
  "cost_matrix_data": {
    "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
  },
  "travel_time_matrix_data": {
    "data": {"0": [[0,10,15],[10,0,12],[15,12,0]]}
  },
  "task_data": {
    "task_locations": [1, 2]
  },
  "fleet_data": {
    "vehicle_locations": [[0, 0]],
    "capacities": [[100]]
  },
  "solver_config": {
    "time_limit": 10
  }
}
```

## Solution Checking

```python
status = solution.get_status()
# 0 = SUCCESS
# 1 = FAIL  
# 2 = TIMEOUT
# 3 = EMPTY

if status == 0:
    solution.display_routes()
    route_df = solution.get_route()
    total_cost = solution.get_total_objective()
else:
    print(f"Error: {solution.get_error_message()}")
    infeasible = solution.get_infeasible_orders()
    if len(infeasible) > 0:
        print(f"Infeasible orders: {infeasible.to_list()}")
```

## Common Issues

| Problem | Likely Cause | Fix |
|---------|--------------|-----|
| Empty solution | Time windows too tight | Widen windows or check travel times |
| Infeasible orders | Demand > capacity | Increase fleet or capacity |
| Status != 0 | Missing transit time matrix | Add `add_transit_time_matrix()` when using time windows |
| Wrong route cost | Matrix not symmetric | Check cost_matrix values |

## Data Type Requirements

```python
# Always use explicit dtypes
cost_matrix = cost_matrix.astype("float32")
order_locations = cudf.Series([...], dtype="int32")
demand = cudf.Series([...], dtype="int32")
vehicle_capacity = cudf.Series([...], dtype="int32")
time_windows = cudf.Series([...], dtype="int32")
```

## Solver Settings

```python
ss = routing.SolverSettings()
ss.set_time_limit(30)           # seconds
ss.set_number_of_climbers(64)   # parallel search threads
ss.set_solution_scope(0)        # 0=optimize, 1=feasibility only
```

## Examples

See `resources/` for complete examples:
- [Python API](resources/python_examples.md) — VRP, PDP, multi-depot
- [REST Server](resources/server_examples.md) — curl and Python requests

## When to Escalate

Switch to **cuopt-debugging** if:
- Solution is infeasible and you can't determine why
- Performance is unexpectedly slow

Switch to **cuopt-developer** if:
- User wants to modify solver behavior
- User wants to add new constraint types
