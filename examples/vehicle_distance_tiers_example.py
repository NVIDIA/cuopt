#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Vehicle Distance Tiers Example

This example demonstrates how to use distance-based tiered pricing for vehicles
in cuOpt. Distance tiers allow you to define different cost structures based on
the total distance traveled by each vehicle.

Use cases:
- Progressive pricing: higher rates for longer distances
- Fixed fees for short trips
- Different pricing models for different vehicle types
"""

import numpy as np
import cudf
from cuopt import routing


def create_simple_problem():
    """
    Create a simple routing problem with 10 locations and 3 vehicles
    """
    n_locations = 10
    n_vehicles = 3

    # Create a simple distance matrix (symmetric)
    np.random.seed(42)
    distances = np.random.uniform(10, 50, (n_locations, n_locations))
    # Make symmetric and zero diagonal
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)

    cost_df = cudf.DataFrame(distances.astype(np.float32))

    # Simple demands and capacities
    n_orders = n_locations - 1  # Exclude depot
    demand = cudf.Series([10] * n_orders, dtype=np.int32)
    capacities = cudf.Series([40] * n_vehicles, dtype=np.int32)

    return cost_df, demand, capacities


def example_uniform_tiers():
    """
    Example 1: Uniform tiers - All vehicles have the same pricing structure

    Pricing structure:
    - Distance < 50 km: Fixed fee of $100
    - Distance 50-100 km: $2 per km
    - Distance > 100 km: $5 per km
    """
    print("=" * 70)
    print("EXAMPLE 1: UNIFORM DISTANCE TIERS")
    print("=" * 70)
    print("\nAll vehicles have the same pricing structure:")
    print("  • Distance < 50 km: Fixed fee of $100")
    print("  • Distance 50-100 km: $2 per km")
    print("  • Distance > 100 km: $5 per km\n")

    # Create problem
    cost_df, demand, capacities = create_simple_problem()
    n_locations = len(cost_df)
    n_vehicles = len(capacities)

    # Create data model
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(cost_df)

    # Set order locations (all locations except depot at 0)
    order_locations = cudf.Series(range(1, n_locations), dtype=np.int32)
    data_model.set_order_locations(order_locations)

    # Add capacity constraint
    data_model.add_capacity_dimension("demand", demand, capacities)

    # Configure distance tiers - same for all vehicles
    vehicle_ids = []
    thresholds = []
    fixed_costs = []
    costs_per_unit = []

    for v in range(n_vehicles):
        # Tier 1: < 50 km = fixed cost 100
        vehicle_ids.append(v)
        thresholds.append(50.0)
        fixed_costs.append(100.0)
        costs_per_unit.append(0.0)

        # Tier 2: 50-100 km = 2.0 per km
        vehicle_ids.append(v)
        thresholds.append(100.0)
        fixed_costs.append(0.0)
        costs_per_unit.append(2.0)

        # Tier 3: > 100 km = 5.0 per km
        vehicle_ids.append(v)
        thresholds.append(1e9)  # infinity
        fixed_costs.append(0.0)
        costs_per_unit.append(5.0)

    # Set tiers
    data_model.set_vehicle_distance_tiers(
        cudf.Series(vehicle_ids, dtype=np.int32),
        cudf.Series(thresholds, dtype=np.float32),
        cudf.Series(fixed_costs, dtype=np.float32),
        cudf.Series(costs_per_unit, dtype=np.float32),
    )

    # Solve
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(10.0)

    print("Solving...")
    solution = routing.Solve(data_model, solver_settings)

    # Display results
    print(f"\nStatus: {solution.get_status()}")
    print(f"Total objective: {solution.get_total_objective():.2f}")

    objectives = solution.get_objective_values()
    if routing.Objective.COST in objectives:
        print(
            f"Total cost (with tiers): ${objectives[routing.Objective.COST]:.2f}"
        )

    print("\n✅ Example 1 completed\n")


def example_heterogeneous_tiers():
    """
    Example 2: Heterogeneous tiers - Different vehicles have different pricing

    Vehicle types:
    - Vehicle 0 (Economy): Cheaper for short trips, expensive for long
    - Vehicle 1 (Standard): Balanced pricing
    - Vehicle 2 (Premium): More expensive upfront, cheaper for long distances
    """
    print("=" * 70)
    print("EXAMPLE 2: HETEROGENEOUS DISTANCE TIERS")
    print("=" * 70)
    print("\nDifferent vehicles have different pricing structures:\n")
    print("🔵 Vehicle 0 (Economy):")
    print("  • < 30 km: $50 fixed")
    print("  • 30-80 km: $3 per km")
    print("  • > 80 km: $6 per km")
    print("\n🟡 Vehicle 1 (Standard):")
    print("  • < 60 km: $80 fixed")
    print("  • 60-100 km: $2 per km")
    print("  • > 100 km: $4 per km")
    print("\n🟢 Vehicle 2 (Premium):")
    print("  • < 100 km: $120 fixed")
    print("  • > 100 km: $1.5 per km\n")

    # Create problem
    cost_df, demand, capacities = create_simple_problem()
    n_locations = len(cost_df)
    n_vehicles = len(capacities)

    # Create data model
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(cost_df)

    # Set order locations
    order_locations = cudf.Series(range(1, n_locations), dtype=np.int32)
    data_model.set_order_locations(order_locations)

    # Add capacity constraint
    data_model.add_capacity_dimension("demand", demand, capacities)

    # Configure distance tiers - different for each vehicle
    vehicle_ids = []
    thresholds = []
    fixed_costs = []
    costs_per_unit = []

    # Vehicle 0: Economy
    vehicle_ids.extend([0, 0, 0])
    thresholds.extend([30.0, 80.0, 1e9])
    fixed_costs.extend([50.0, 0.0, 0.0])
    costs_per_unit.extend([0.0, 3.0, 6.0])

    # Vehicle 1: Standard
    vehicle_ids.extend([1, 1, 1])
    thresholds.extend([60.0, 100.0, 1e9])
    fixed_costs.extend([80.0, 0.0, 0.0])
    costs_per_unit.extend([0.0, 2.0, 4.0])

    # Vehicle 2: Premium
    vehicle_ids.extend([2, 2])
    thresholds.extend([100.0, 1e9])
    fixed_costs.extend([120.0, 0.0])
    costs_per_unit.extend([0.0, 1.5])

    # Set tiers
    data_model.set_vehicle_distance_tiers(
        cudf.Series(vehicle_ids, dtype=np.int32),
        cudf.Series(thresholds, dtype=np.float32),
        cudf.Series(fixed_costs, dtype=np.float32),
        cudf.Series(costs_per_unit, dtype=np.float32),
    )

    # Solve
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(10.0)

    print("Solving...")
    solution = routing.Solve(data_model, solver_settings)

    # Display results
    print(f"\nStatus: {solution.get_status()}")
    print(f"Total objective: {solution.get_total_objective():.2f}")

    objectives = solution.get_objective_values()
    if routing.Objective.COST in objectives:
        print(
            f"Total cost (with tiers): ${objectives[routing.Objective.COST]:.2f}"
        )

    # Show which vehicles were used
    truck_ids = solution.get_truck_id().to_numpy()
    routes = solution.get_route().to_numpy()

    print("\nVehicle usage:")
    for v in range(n_vehicles):
        orders = np.sum((truck_ids == v) & (routes != 0))
        vehicle_type = ["Economy", "Standard", "Premium"][v]
        if orders > 0:
            print(f"  Vehicle {v} ({vehicle_type}): {orders} orders")
        else:
            print(f"  Vehicle {v} ({vehicle_type}): Not used")

    print("\n✅ Example 2 completed\n")


def example_realistic_scenario():
    """
    Example 3: Realistic delivery scenario

    A delivery company has:
    - Small vans: Best for short urban deliveries
    - Medium trucks: Good for medium distances
    - Large trucks: Efficient for long hauls despite higher base cost
    """
    print("=" * 70)
    print("EXAMPLE 3: REALISTIC DELIVERY SCENARIO")
    print("=" * 70)
    print("\nA delivery company optimizing their fleet:\n")
    print("🚐 Small Vans (2 available):")
    print("  • < 20 km: $30 fixed (urban deliveries)")
    print("  • > 20 km: $4 per km (expensive for long trips)")
    print("\n🚚 Medium Trucks (2 available):")
    print("  • < 50 km: $60 fixed")
    print("  • 50-100 km: $1.5 per km")
    print("  • > 100 km: $3 per km")
    print("\n🚛 Large Trucks (1 available):")
    print("  • < 80 km: $100 fixed")
    print("  • > 80 km: $1 per km (efficient for long hauls)\n")

    # Create a larger problem
    n_locations = 15
    n_vehicles = 5  # 2 small + 2 medium + 1 large

    # Create distance matrix with some structure
    np.random.seed(123)
    distances = np.random.uniform(5, 80, (n_locations, n_locations))
    distances = (distances + distances.T) / 2
    np.fill_diagonal(distances, 0)
    cost_df = cudf.DataFrame(distances.astype(np.float32))

    # Different capacities for different vehicle types
    n_orders = n_locations - 1
    demand = cudf.Series([8] * n_orders, dtype=np.int32)
    capacities = cudf.Series([30, 30, 50, 50, 80], dtype=np.int32)

    # Create data model
    data_model = routing.DataModel(n_locations, n_vehicles)
    data_model.add_cost_matrix(cost_df)

    order_locations = cudf.Series(range(1, n_locations), dtype=np.int32)
    data_model.set_order_locations(order_locations)
    data_model.add_capacity_dimension("demand", demand, capacities)

    # Configure distance tiers
    vehicle_ids = []
    thresholds = []
    fixed_costs = []
    costs_per_unit = []

    # Small vans (vehicles 0, 1)
    for v in [0, 1]:
        vehicle_ids.extend([v, v])
        thresholds.extend([20.0, 1e9])
        fixed_costs.extend([30.0, 0.0])
        costs_per_unit.extend([0.0, 4.0])

    # Medium trucks (vehicles 2, 3)
    for v in [2, 3]:
        vehicle_ids.extend([v, v, v])
        thresholds.extend([50.0, 100.0, 1e9])
        fixed_costs.extend([60.0, 0.0, 0.0])
        costs_per_unit.extend([0.0, 1.5, 3.0])

    # Large truck (vehicle 4)
    vehicle_ids.extend([4, 4])
    thresholds.extend([80.0, 1e9])
    fixed_costs.extend([100.0, 0.0])
    costs_per_unit.extend([0.0, 1.0])

    # Set tiers
    data_model.set_vehicle_distance_tiers(
        cudf.Series(vehicle_ids, dtype=np.int32),
        cudf.Series(thresholds, dtype=np.float32),
        cudf.Series(fixed_costs, dtype=np.float32),
        cudf.Series(costs_per_unit, dtype=np.float32),
    )

    # Solve
    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(15.0)

    print("Solving...")
    solution = routing.Solve(data_model, solver_settings)

    # Display results
    print(f"\nStatus: {solution.get_status()}")
    print(f"Total objective: {solution.get_total_objective():.2f}")

    objectives = solution.get_objective_values()
    if routing.Objective.COST in objectives:
        print(
            f"Total cost (with tiers): ${objectives[routing.Objective.COST]:.2f}"
        )

    # Detailed vehicle usage
    truck_ids = solution.get_truck_id().to_numpy()
    routes = solution.get_route().to_numpy()

    print("\nOptimal fleet allocation:")
    vehicle_types = [
        "Small Van",
        "Small Van",
        "Medium Truck",
        "Medium Truck",
        "Large Truck",
    ]

    for v in range(n_vehicles):
        orders = np.sum((truck_ids == v) & (routes != 0))
        capacity_used = orders * 8  # Each order is 8 units
        capacity_total = capacities[v]

        if orders > 0:
            print(f"  Vehicle {v} ({vehicle_types[v]}):")
            print(f"    • Orders: {orders}")
            print(
                f"    • Capacity used: {capacity_used}/{capacity_total} units"
            )
        else:
            print(f"  Vehicle {v} ({vehicle_types[v]}): Not used")

    print("\n💡 The solver automatically selected the most cost-effective")
    print("   vehicles based on distance tiers and capacity constraints!")

    print("\n✅ Example 3 completed\n")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("VEHICLE DISTANCE TIERS - EXAMPLES")
    print("=" * 70)
    print("\nThese examples demonstrate how to use distance-based tiered")
    print("pricing in cuOpt to optimize vehicle routing costs.\n")

    try:
        example_uniform_tiers()
        print("-" * 70 + "\n")

        example_heterogeneous_tiers()
        print("-" * 70 + "\n")

        example_realistic_scenario()

        print("=" * 70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY ✅")
        print("=" * 70)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
