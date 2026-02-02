# Debugging: Diagnostic Snippets

## LP/MILP Diagnostics

### Check Status Properly

```python
# Print actual status value
print(f"Status: '{problem.Status.name}'")
print(f"Status type: {type(problem.Status.name)}")

# Common mistake: wrong case
print(f"== 'Optimal': {problem.Status.name == 'Optimal'}")      # ✅
print(f"== 'OPTIMAL': {problem.Status.name == 'OPTIMAL'}")      # ❌ Always False
```

### Inspect Variables

```python
# Check all variable values
for var in [x, y, z]:
    print(f"{var.name}: lb={var.lb}, ub={var.ub}, value={var.getValue()}")

# Check if integer variables are actually integer
for var in integer_vars:
    val = var.getValue()
    is_int = abs(val - round(val)) < 1e-6
    print(f"{var.name}: {val} (is_integer: {is_int})")
```

### Inspect Constraints

```python
# Check constraint values
for name in ["constraint1", "constraint2"]:
    c = problem.getConstraint(name)
    print(f"{name}: dual={c.DualValue}")
```

### Check Problem Size

```python
print(f"Variables: {problem.num_variables}")
print(f"Constraints: {problem.num_constraints}")
```

## Routing Diagnostics

### Check Solution Status

```python
status = solution.get_status()
print(f"Status code: {status}")
# 0 = SUCCESS
# 1 = FAIL
# 2 = TIMEOUT
# 3 = EMPTY

if status != 0:
    print(f"Message: {solution.get_message()}")
    print(f"Error: {solution.get_error_message()}")
```

### Find Infeasible Orders

```python
infeasible = solution.get_infeasible_orders()
if len(infeasible) > 0:
    print(f"Infeasible orders: {infeasible.to_list()}")

    # Check why each is infeasible
    for order_idx in infeasible.to_list():
        print(f"\nOrder {order_idx}:")
        print(f"  Location: {order_locations[order_idx]}")
        print(f"  Time window: [{order_earliest[order_idx]}, {order_latest[order_idx]}]")
        print(f"  Demand: {demand[order_idx]}")
```

### Verify Data Dimensions

```python
print(f"Cost matrix shape: {cost_matrix.shape}")
print(f"n_locations declared: {dm.n_locations}")
print(f"n_orders: {len(order_locations)}")
print(f"n_fleet: {dm.n_fleet}")

# Check consistency
assert cost_matrix.shape[0] == cost_matrix.shape[1], "Matrix not square"
assert cost_matrix.shape[0] == dm.n_locations, "Matrix size != n_locations"
```

### Check Data Types

```python
# For numpy arrays, use .dtype directly
# For pandas/cudf DataFrames, use .values.dtype or .to_numpy().dtype
print(f"cost_matrix dtype: {cost_matrix.values.dtype}")  # float32
print(f"order_locations dtype: {order_locations.values.dtype}")  # int32
print(f"demand dtype: {demand.values.dtype}")  # int32
```

### Verify Time Windows Feasibility

```python
# Check for impossible time windows
for i in range(len(order_earliest)):
    if order_earliest[i] > order_latest[i]:
        print(f"Order {i}: earliest ({order_earliest[i]}) > latest ({order_latest[i]})")

# Check if orders are reachable from depot in time
depot = 0
for i in range(len(order_locations)):
    loc = order_locations[i]
    travel_time = transit_time_matrix.iloc[depot, loc]
    if travel_time > order_latest[i]:
        print(f"Order {i}: unreachable (travel={travel_time}, latest={order_latest[i]})")
```

### Check Capacity Feasibility

```python
total_demand = demand.sum()
total_capacity = vehicle_capacity.sum()
print(f"Total demand: {total_demand}")
print(f"Total capacity: {total_capacity}")
if total_demand > total_capacity:
    print("WARNING: Total demand exceeds total capacity!")
```

## Server Diagnostics

### Check Response Structure

```python
import json

response = requests.get(f"{SERVER}/cuopt/solution/{req_id}", headers=HEADERS)
print(f"Status code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
```

### Validate Payload Against Schema

```bash
# Get OpenAPI spec
curl -s http://localhost:8000/cuopt.yaml > cuopt_spec.yaml

# Check payload structure manually
```

### Common 422 Error Fixes

```python
# ❌ Wrong field name
payload = {"transit_time_matrix_data": {...}}

# ✅ Correct field name
payload = {"travel_time_matrix_data": {...}}

# ❌ Wrong capacity format
"capacities": [[50], [50]]

# ✅ Correct capacity format
"capacities": [[50, 50]]
```

## Memory Diagnostics

### Check GPU Memory

```python
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

### Estimate Problem Memory

```python
# Rough estimate for routing
n_locations = 1000
n_fleet = 50
n_orders = 500

# Cost matrix: n_locations² * 4 bytes (float32)
matrix_size = n_locations * n_locations * 4 / 1e9  # GB
print(f"Cost matrix: ~{matrix_size:.2f} GB")
```

## Performance Diagnostics

### Time the Solve

```python
import time

start = time.time()
problem.solve(settings)
elapsed = time.time() - start
print(f"Solve time: {elapsed:.2f}s")
```

### Enable Solver Logging

```python
settings = SolverSettings()
settings.set_parameter("log_to_console", 1)
```

---

## Additional References

| Topic | Resource |
|-------|----------|
| Python API docstrings | `python/cuopt/cuopt/routing/vehicle_routing.py` |
| LP/MILP Problem class | `python/cuopt/cuopt/linear_programming/problem.py` |
| Server API spec | `docs/cuopt/source/cuopt_spec.yaml` |
| Troubleshooting guide | [NVIDIA cuOpt Docs](https://docs.nvidia.com/cuopt/user-guide/latest/troubleshooting.html) |
