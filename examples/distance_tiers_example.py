"""
Example: Using Distance-Based Tiered Pricing in cuOpt

This example demonstrates how to use the distance_tiers feature to apply
different cost structures based on the total route distance.

Scenario:
- 2 vehicles with different pricing tiers
- Vehicle 0: < 100 km = 50 fixed, 100-200 km = 0.1/km, > 200 km = 0.5/km
- Vehicle 1: < 150 km = 75 fixed, > 150 km = 0.3/km
"""

import cudf
import numpy as np
from cuopt import routing

# Create data model
n_locations = 6  # 1 depot + 5 customers
n_vehicles = 2

data_model = routing.DataModel(n_locations, n_vehicles)

# Define cost matrix (distances in km)
cost_matrix = np.array(
    [
        # Depot, C1,  C2,  C3,  C4,  C5
        [0, 30, 40, 50, 80, 100],  # Depot
        [30, 0, 20, 35, 60, 85],  # Customer 1
        [40, 20, 0, 25, 55, 75],  # Customer 2
        [50, 35, 25, 0, 40, 60],  # Customer 3
        [80, 60, 55, 40, 0, 30],  # Customer 4
        [100, 85, 75, 60, 30, 0],  # Customer 5
    ],
    dtype=np.float32,
)

data_model.add_cost_matrix(cudf.DataFrame(cost_matrix))

# Set vehicle locations (both start at depot - location 0)
vehicle_starts = cudf.Series([0, 0], dtype=np.int32)
vehicle_returns = cudf.Series([0, 0], dtype=np.int32)
data_model.set_vehicle_locations(vehicle_starts, vehicle_returns)

# Define order locations (customers to visit)
order_locations = cudf.Series([1, 2, 3, 4, 5], dtype=np.int32)
data_model.set_order_locations(order_locations)

# ============================================================================
# SET DISTANCE TIERS - This is the new feature!
# ============================================================================

# Vehicle 0 tiers: < 100km = 50 fixed, 100-200km = 0.1/km, > 200km = 0.5/km
# Vehicle 1 tiers: < 150km = 75 fixed, > 150km = 0.3/km

vehicle_ids = cudf.Series(
    [
        0,
        0,
        0,  # Vehicle 0 has 3 tiers
        1,
        1,  # Vehicle 1 has 2 tiers
    ],
    dtype=np.int32,
)

thresholds = cudf.Series(
    [
        100.0,
        200.0,
        1e9,  # Vehicle 0 thresholds
        150.0,
        1e9,  # Vehicle 1 thresholds
    ],
    dtype=np.float32,
)

fixed_costs = cudf.Series(
    [
        50.0,
        0.0,
        0.0,  # Vehicle 0: only first tier has fixed cost
        75.0,
        0.0,  # Vehicle 1: only first tier has fixed cost
    ],
    dtype=np.float32,
)

costs_per_unit = cudf.Series(
    [
        0.0,
        0.1,
        0.5,  # Vehicle 0: 0, 0.1/km, 0.5/km
        0.0,
        0.3,  # Vehicle 1: 0, 0.3/km
    ],
    dtype=np.float32,
)

data_model.set_vehicle_distance_tiers(
    vehicle_ids, thresholds, fixed_costs, costs_per_unit
)

# ============================================================================
# SOLVE
# ============================================================================

solver_settings = routing.SolverSettings()
solver_settings.set_time_limit(5)  # 5 seconds

routing_solution = routing.Solver(data_model, solver_settings).solve()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if routing_solution.get_status() == 0:
    print("✓ Solution found!")
    print("\nRoute Details:")
    print("-" * 80)

    vehicle_routes = routing_solution.get_route()

    for vehicle_id in range(n_vehicles):
        route = vehicle_routes[vehicle_routes["truck_id"] == vehicle_id]

        if len(route) > 0:
            # Get route distance
            route_distance = 0.0
            locations = route["route"].to_arrow().to_pylist()

            for i in range(len(locations) - 1):
                from_loc = locations[i]
                to_loc = locations[i + 1]
                route_distance += cost_matrix[from_loc][to_loc]

            print(f"\nVehicle {vehicle_id}:")
            print(f"  Route: {' -> '.join(map(str, locations))}")
            print(f"  Total Distance: {route_distance:.2f} km")

            # Calculate cost based on tiers
            if vehicle_id == 0:
                if route_distance < 100:
                    cost = 50.0
                    tier_info = "< 100 km: Fixed cost 50"
                elif route_distance < 200:
                    cost = route_distance * 0.1
                    tier_info = "100-200 km: 0.1/km"
                else:
                    cost = route_distance * 0.5
                    tier_info = "> 200 km: 0.5/km"
            else:  # vehicle_id == 1
                if route_distance < 150:
                    cost = 75.0
                    tier_info = "< 150 km: Fixed cost 75"
                else:
                    cost = route_distance * 0.3
                    tier_info = "> 150 km: 0.3/km"

            print(f"  Applied Tier: {tier_info}")
            print(f"  Route Cost: {cost:.2f}")

    print("\n" + "-" * 80)
    print(f"Total Objective Cost: {routing_solution.final_cost}")

else:
    print(f"✗ No solution found. Status: {routing_solution.get_status()}")


# ============================================================================
# HELPER FUNCTION: Simplified tier creation
# ============================================================================


def create_distance_tiers_simple(tiers_by_vehicle):
    """
    Helper to create distance tiers from a more readable dictionary format.

    Parameters
    ----------
    tiers_by_vehicle : list of list of dict
        Each element is a list of tier dictionaries for that vehicle.
        Each tier dict should have 'threshold' and either 'fixed_cost' or 'cost_per_unit'.

    Returns
    -------
    tuple of cudf.Series
        (vehicle_ids, thresholds, fixed_costs, costs_per_unit)

    Example
    -------
    >>> tiers = [
    ...     # Vehicle 0
    ...     [
    ...         {"threshold": 100, "fixed_cost": 50},
    ...         {"threshold": 200, "cost_per_unit": 0.1},
    ...         {"threshold": 1e9, "cost_per_unit": 0.5}
    ...     ],
    ...     # Vehicle 1
    ...     [
    ...         {"threshold": 150, "fixed_cost": 75},
    ...         {"threshold": 1e9, "cost_per_unit": 0.3}
    ...     ]
    ... ]
    >>> vehicle_ids, thresholds, fixed_costs, costs_per_unit = create_distance_tiers_simple(tiers)
    >>> data_model.set_vehicle_distance_tiers(vehicle_ids, thresholds, fixed_costs, costs_per_unit)
    """
    vehicle_ids_list = []
    thresholds_list = []
    fixed_costs_list = []
    costs_per_unit_list = []

    for vehicle_id, tiers in enumerate(tiers_by_vehicle):
        for tier in tiers:
            vehicle_ids_list.append(vehicle_id)
            thresholds_list.append(tier["threshold"])
            fixed_costs_list.append(tier.get("fixed_cost", 0.0))
            costs_per_unit_list.append(tier.get("cost_per_unit", 0.0))

    return (
        cudf.Series(vehicle_ids_list, dtype=np.int32),
        cudf.Series(thresholds_list, dtype=np.float32),
        cudf.Series(fixed_costs_list, dtype=np.float32),
        cudf.Series(costs_per_unit_list, dtype=np.float32),
    )


# Example using the helper function:
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Using helper function:")
    print("=" * 80 + "\n")

    tiers_definition = [
        # Vehicle 0 tiers
        [
            {"threshold": 100.0, "fixed_cost": 50.0},
            {"threshold": 200.0, "cost_per_unit": 0.1},
            {"threshold": 1e9, "cost_per_unit": 0.5},
        ],
        # Vehicle 1 tiers
        [
            {"threshold": 150.0, "fixed_cost": 75.0},
            {"threshold": 1e9, "cost_per_unit": 0.3},
        ],
    ]

    vids, thresh, fixed, per_unit = create_distance_tiers_simple(
        tiers_definition
    )

    print("Generated tier data:")
    print(f"  Vehicle IDs: {vids.to_arrow().to_pylist()}")
    print(f"  Thresholds: {thresh.to_arrow().to_pylist()}")
    print(f"  Fixed Costs: {fixed.to_arrow().to_pylist()}")
    print(f"  Costs per Unit: {per_unit.to_arrow().to_pylist()}")
