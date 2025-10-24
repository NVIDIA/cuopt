# Vehicle Distance Tiers - Python Guide

This guide explains how to use the distance-based tiered pricing feature in cuOpt Python API.

## Overview

Distance tiers allow you to define different cost structures for vehicles based on the total distance traveled. This is useful for:

- **Progressive pricing**: Higher rates for longer distances
- **Fixed fees**: Flat rates for short trips (e.g., urban deliveries)
- **Heterogeneous fleets**: Different pricing models for different vehicle types
- **Real-world scenarios**: Modeling actual transportation costs with fuel, tolls, and driver compensation

## Files Created

### 1. Test Suite: `python/cuopt/cuopt/tests/routing/test_vehicle_distance_tiers.py`

Comprehensive test suite with two main tests:

- **`test_vehicle_distance_tiers_uniform()`**: Tests homogeneous fleet where all vehicles have the same tier configuration
- **`test_vehicle_distance_tiers_heterogeneous()`**: Tests heterogeneous fleet with different tier configurations per vehicle

**Run the tests:**
```bash
cd python/cuopt
pytest cuopt/tests/routing/test_vehicle_distance_tiers.py -v
```

Or run a specific test:
```bash
pytest cuopt/tests/routing/test_vehicle_distance_tiers.py::test_vehicle_distance_tiers_uniform -v
```

### 2. Examples: `examples/vehicle_distance_tiers_example.py`

Three practical examples showing different use cases:

**Example 1: Uniform Tiers**
- All vehicles have the same pricing structure
- Good for validating the feature works correctly

**Example 2: Heterogeneous Tiers**
- Different vehicles with different pricing (Economy, Standard, Premium)
- Demonstrates how the solver chooses cost-effective vehicles

**Example 3: Realistic Delivery Scenario**
- Small vans, medium trucks, and large trucks
- Each vehicle type has realistic pricing based on distance
- Shows optimal fleet allocation

**Run the examples:**
```bash
python examples/vehicle_distance_tiers_example.py
```

## API Usage

### Basic Structure

```python
from cuopt import routing
import cudf
import numpy as np

# 1. Create data model
data_model = routing.DataModel(n_locations, n_vehicles)
data_model.add_cost_matrix(cost_matrix)

# 2. Configure distance tiers
vehicle_ids = cudf.Series([0, 0, 0, 1, 1], dtype=np.int32)
thresholds = cudf.Series([50.0, 100.0, 1e9, 80.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([100.0, 0.0, 0.0, 120.0, 0.0], dtype=np.float32)
costs_per_unit = cudf.Series([0.0, 2.0, 5.0, 0.0, 1.5], dtype=np.float32)

data_model.set_vehicle_distance_tiers(
    vehicle_ids,
    thresholds,
    fixed_costs,
    costs_per_unit
)

# 3. Solve
solver_settings = routing.SolverSettings()
solution = routing.Solve(data_model, solver_settings)
```

### Parameters Explained

**`vehicle_ids`** (cudf.Series[int32])
- Vehicle ID for each tier entry
- Tiers for the same vehicle should be consecutive
- Example: `[0, 0, 0, 1, 1]` = 3 tiers for vehicle 0, 2 tiers for vehicle 1

**`thresholds`** (cudf.Series[float32])
- Distance thresholds for each tier (in same units as cost matrix)
- Must be sorted in ascending order for each vehicle
- Use `1e9` or `float('inf')` for the last tier

**`fixed_costs`** (cudf.Series[float32])
- Fixed cost for the tier (applied regardless of distance)
- Set to `0.0` if using `cost_per_unit` instead
- If `fixed_cost > 0`, it overrides `cost_per_unit`

**`costs_per_unit`** (cudf.Series[float32])
- Cost per distance unit for the tier
- Set to `0.0` if using `fixed_cost` instead
- Applied as: `cost = distance * cost_per_unit`

## Configuration Examples

### Example 1: Simple Fixed Fee + Variable Cost

**Scenario**: $50 fixed fee for short trips, $2/km for longer trips

```python
# For one vehicle
vehicle_ids = cudf.Series([0, 0], dtype=np.int32)
thresholds = cudf.Series([30.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([50.0, 0.0], dtype=np.float32)
costs_per_unit = cudf.Series([0.0, 2.0], dtype=np.float32)
```

**Cost calculation**:
- Distance < 30 km: Pay $50 (fixed)
- Distance ≥ 30 km: Pay distance × $2/km

### Example 2: Progressive Pricing

**Scenario**: Economy tier → Standard tier → Premium tier

```python
vehicle_ids = cudf.Series([0, 0, 0], dtype=np.int32)
thresholds = cudf.Series([50.0, 100.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([0.0, 0.0, 0.0], dtype=np.float32)
costs_per_unit = cudf.Series([1.0, 2.0, 4.0], dtype=np.float32)
```

**Cost calculation**:
- Distance < 50 km: Pay distance × $1/km
- Distance 50-100 km: Pay distance × $2/km
- Distance > 100 km: Pay distance × $4/km

### Example 3: Multiple Vehicles with Different Configs

**Scenario**: Small van vs Large truck

```python
# Small van (vehicle 0): Good for short trips
# Large truck (vehicle 1): Better for long hauls

vehicle_ids = cudf.Series([0, 0, 1, 1], dtype=np.int32)
thresholds = cudf.Series([25.0, 1e9, 60.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([40.0, 0.0, 100.0, 0.0], dtype=np.float32)
costs_per_unit = cudf.Series([0.0, 3.5, 0.0, 1.2], dtype=np.float32)
```

**Cost calculation**:
- **Small van (0)**:
  - < 25 km: $40 fixed
  - ≥ 25 km: $3.5/km
- **Large truck (1)**:
  - < 60 km: $100 fixed
  - ≥ 60 km: $1.2/km

## How It Works

The solver will:

1. **Evaluate each vehicle's potential cost** based on distance tiers
2. **Choose vehicles optimally** to minimize total cost
3. **Apply the appropriate tier** based on actual route distance

### Cost Calculation Logic

For each vehicle's route:
```python
total_distance = sum of all edges in the route

for each tier in vehicle_tiers:
    if total_distance < tier.threshold:
        if tier.fixed_cost > 0:
            cost = tier.fixed_cost
        else:
            cost = total_distance * tier.cost_per_unit
        break
```

## Best Practices

### 1. **Always define a final "catch-all" tier**
```python
# Last tier should have threshold = 1e9 (infinity)
thresholds = [..., 1e9]
```

### 2. **Sort tiers by threshold in ascending order**
```python
# ✅ Correct
thresholds = [30.0, 60.0, 100.0, 1e9]

# ❌ Wrong
thresholds = [100.0, 30.0, 60.0, 1e9]
```

### 3. **Group tiers by vehicle consecutively**
```python
# ✅ Correct - vehicle 0 tiers, then vehicle 1 tiers
vehicle_ids = [0, 0, 0, 1, 1]

# ❌ Wrong - interleaved
vehicle_ids = [0, 1, 0, 1, 0]
```

### 4. **For each tier, use EITHER fixed_cost OR cost_per_unit**
```python
# ✅ Correct - tier 1 uses fixed, tier 2 uses per-unit
fixed_costs = [50.0, 0.0]
costs_per_unit = [0.0, 2.0]

# ⚠️  Avoid - both non-zero (fixed_cost takes precedence)
fixed_costs = [50.0, 30.0]
costs_per_unit = [1.0, 2.0]
```

### 5. **Test with uniform configuration first**
Start with all vehicles having the same tiers to validate your setup, then introduce heterogeneity.

## Troubleshooting

### Issue: "vehicle_ids contains X but fleet size is Y"
**Solution**: Make sure all vehicle IDs in `vehicle_ids` are less than `n_vehicles`

```python
# If n_vehicles = 3
vehicle_ids = [0, 1, 2]  # ✅ Valid
vehicle_ids = [0, 1, 3]  # ❌ Invalid - vehicle 3 doesn't exist
```

### Issue: Costs don't match expectations
**Solution**:
1. Check that tiers are sorted by threshold
2. Verify fixed_cost vs cost_per_unit logic
3. Print actual route distances to validate tier application

### Issue: All vehicles use the same route despite different tiers
**Solution**: The cost differences might not be significant enough. Try:
1. Increase the difference between tier costs
2. Add more constraints (capacity, time windows) to force differentiation
3. Increase problem size to give more optimization opportunities

## Comparison: C++ vs Python API

| Aspect | C++ API | Python API |
|--------|---------|------------|
| **Tier Data** | Flat arrays with offsets | cuDF Series with vehicle IDs |
| **Configuration** | `set_vehicle_distance_tiers(thresholds*, fixed_costs*, ...)` | `set_vehicle_distance_tiers(vehicle_ids, thresholds, ...)` |
| **Indexing** | Manual offset management | Automatic grouping by vehicle_id |
| **Data Location** | Device pointers | cuDF Series (handles device memory) |

### C++ Example (for reference)
```cpp
// Flat arrays for all vehicles
std::vector<float> all_thresholds = {40, 80, 1e9,  40, 80, 1e9};  // 2 vehicles
std::vector<int> tier_offsets = {0, 3, 6};  // Vehicle 0: [0-3), Vehicle 1: [3-6)

dm.set_vehicle_distance_tiers(
    d_thresholds.data(),
    d_fixed_costs.data(),
    d_costs_per_unit.data(),
    d_tier_offsets.data(),
    total_tiers
);
```

### Python Equivalent
```python
# Series with explicit vehicle IDs
vehicle_ids = cudf.Series([0, 0, 0, 1, 1, 1], dtype=np.int32)
thresholds = cudf.Series([40, 80, 1e9, 40, 80, 1e9], dtype=np.float32)

data_model.set_vehicle_distance_tiers(
    vehicle_ids, thresholds, fixed_costs, costs_per_unit
)
```

The Python API is more intuitive as it explicitly associates each tier with a vehicle ID.

## Related Documentation

- C++ Test: `cpp/tests/routing/unit_tests/test_distance_tier_dummy.cu`
- Python Tests: `python/cuopt/cuopt/tests/routing/test_vehicle_distance_tiers.py`
- Python Examples: `examples/vehicle_distance_tiers_example.py`
- API Documentation: See `python/cuopt/cuopt/routing/vehicle_routing.py:1249`

## Additional Examples

### Ride-sharing with surge pricing
```python
# Normal hours: $2/km
# Rush hour multiplier: $4/km
vehicle_ids = cudf.Series([0, 0], dtype=np.int32)
thresholds = cudf.Series([20.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([0.0, 0.0], dtype=np.float32)
costs_per_unit = cudf.Series([2.0, 4.0], dtype=np.float32)
```

### Freight with weight-based tiers
```python
# Light load (< 50km): $1.5/km
# Heavy load (≥ 50km): $2.5/km + fuel surcharge
vehicle_ids = cudf.Series([0, 0], dtype=np.int32)
thresholds = cudf.Series([50.0, 1e9], dtype=np.float32)
fixed_costs = cudf.Series([0.0, 30.0], dtype=np.float32)  # $30 surcharge
costs_per_unit = cudf.Series([1.5, 2.5], dtype=np.float32)
```

## Summary

Vehicle distance tiers provide powerful flexibility for modeling real-world transportation costs. Use the tests to validate your setup and the examples as templates for your specific use case.

**Quick Start**: Run `python examples/vehicle_distance_tiers_example.py` to see it in action!
