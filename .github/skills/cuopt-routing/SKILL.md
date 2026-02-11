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

### Precedence Constraints

Use `add_order_precedence()` to require certain orders to be visited before others.

**Important:** This is a per-node API — call it once for each order that has predecessors.

```python
import numpy as np

# Order 2 must come after orders 0 and 1
dm.add_order_precedence(
    node_id=2,                           # this order
    preceding_nodes=np.array([0, 1])     # must come after these
)

# Order 3 must come after order 2
dm.add_order_precedence(
    node_id=3,
    preceding_nodes=np.array([2])
)
```

**Rules:**
- Call once per order that has predecessors
- `preceding_nodes` is a numpy array of order indices
- Circular dependencies are NOT allowed (A before B before A)
- Orders without precedence constraints don't need a call

**Example: Assembly sequence**
```python
# Task B requires Task A to be done first
# Task C requires Tasks A and B to be done first
dm.add_order_precedence(1, np.array([0]))     # B after A
dm.add_order_precedence(2, np.array([0, 1]))  # C after A and B
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

## Solution DataFrame Schema

`solution.get_route()` returns a `cudf.DataFrame` with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `route` | int | Order/task index in the route sequence |
| `truck_id` | int | Vehicle ID assigned to this stop |
| `location` | int | Location index (0 = depot typically) |
| `arrival_stamp` | float | Arrival time at this location |

**Example output:**
```
   route  arrival_stamp  truck_id  location
0      0            0.0         1         0    # Vehicle 1 starts at depot
1      3            2.0         1         3    # Vehicle 1 visits location 3
2      2            4.0         1         2    # Vehicle 1 visits location 2
3      0            5.0         1         0    # Vehicle 1 returns to depot
4      0            0.0         0         0    # Vehicle 0 starts at depot
5      1            1.0         0         1    # Vehicle 0 visits location 1
6      0            3.0         0         0    # Vehicle 0 returns to depot
```

**Working with results:**
```python
route_df = solution.get_route()

# Routes per vehicle
for vid in route_df["truck_id"].unique().to_arrow().tolist():
    vehicle_route = route_df[route_df["truck_id"] == vid]
    locations = vehicle_route["location"].to_arrow().tolist()
    print(f"Vehicle {vid}: {locations}")

# Total travel time
max_arrival = route_df["arrival_stamp"].max()
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
ss.set_verbose_mode(True)       # enable progress output
ss.set_error_logging_mode(True) # log constraint errors if infeasible
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
