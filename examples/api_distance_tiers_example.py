"""
Example: Using Distance Tiers through cuOpt REST API

This example shows how to call the cuOpt API (self-hosted) with distance tiers
for tiered pricing based on route distance.
"""

import requests
import json

# API endpoint (change to your server address)
API_URL = "http://localhost:5000/cuopt/request"

# Create the request payload
payload = {
    "cost_waypoint_graph_data": None,
    "travel_time_waypoint_graph_data": None,
    "cost_matrix_data": {
        "data": {
            "1": [
                [0, 30, 40, 50, 80, 100],
                [30, 0, 20, 35, 60, 85],
                [40, 20, 0, 25, 55, 75],
                [50, 35, 25, 0, 40, 60],
                [80, 60, 55, 40, 0, 30],
                [100, 85, 75, 60, 30, 0],
            ]
        }
    },
    "travel_time_matrix_data": None,
    "fleet_data": {
        "vehicle_locations": [[0, 0], [0, 0]],
        "vehicle_ids": ["vehicle-0", "vehicle-1"],
        "capacities": None,
        "vehicle_time_windows": None,
        "vehicle_break_time_windows": None,
        "vehicle_break_durations": None,
        "vehicle_break_locations": None,
        "vehicle_types": None,
        "vehicle_order_match": None,
        "skip_first_trips": None,
        "drop_return_trips": None,
        "min_vehicles": None,
        "vehicle_max_costs": None,
        "vehicle_max_times": None,
        "vehicle_fixed_costs": None,
        # NEW FIELD: Distance tiers for tiered pricing
        "vehicle_distance_tiers": [
            # Vehicle 0 tiers
            [
                {
                    "threshold": 100.0,
                    "fixed_cost": 50.0,
                    "cost_per_unit": 0.0,
                },  # < 100 km = 50 fixed
                {
                    "threshold": 200.0,
                    "fixed_cost": 0.0,
                    "cost_per_unit": 0.1,
                },  # 100-200 km = 0.1/km
                {
                    "threshold": 1e9,
                    "fixed_cost": 0.0,
                    "cost_per_unit": 0.5,
                },  # > 200 km = 0.5/km
            ],
            # Vehicle 1 tiers
            [
                {
                    "threshold": 150.0,
                    "fixed_cost": 75.0,
                    "cost_per_unit": 0.0,
                },  # < 150 km = 75 fixed
                {
                    "threshold": 1e9,
                    "fixed_cost": 0.0,
                    "cost_per_unit": 0.3,
                },  # > 150 km = 0.3/km
            ],
        ],
    },
    "task_data": {
        "task_locations": [1, 2, 3, 4, 5],
        "task_ids": [
            "customer-1",
            "customer-2",
            "customer-3",
            "customer-4",
            "customer-5",
        ],
        "demand": None,
        "pickup_and_delivery_pairs": None,
        "task_time_windows": None,
        "service_times": None,
        "prizes": None,
        "order_vehicle_match": None,
        "soft_time_windows": None,
        "task_order_precedence": None,
    },
    "solver_config": {"time_limit": 5},
}


def call_cuopt_api():
    """
    Call the cuOpt API with the distance tiers payload
    """
    print("=" * 80)
    print("Calling cuOpt API with Distance Tiers")
    print("=" * 80)

    print("\nPayload (fleet_data.vehicle_distance_tiers):")
    print(
        json.dumps(payload["fleet_data"]["vehicle_distance_tiers"], indent=2)
    )

    try:
        # Send POST request
        response = requests.post(
            API_URL, json=payload, headers={"Content-Type": "application/json"}
        )

        # Check if request was successful
        if response.status_code == 200:
            result = response.json()

            # Check if we got a request ID (async mode)
            if "reqId" in result:
                req_id = result["reqId"]
                print("\n✓ Request submitted successfully!")
                print(f"  Request ID: {req_id}")

                # Poll for result
                print("\nPolling for result...")
                status_url = f"{API_URL}/{req_id}"

                import time

                max_attempts = 60
                for attempt in range(max_attempts):
                    status_response = requests.get(status_url)
                    status_data = status_response.json()

                    if status_data.get("status") == "Finished":
                        print("\n✓ Solution found!")
                        display_results(
                            status_data["response"]["solver_response"]
                        )
                        break
                    elif status_data.get("status") == "Failed":
                        print(
                            f"\n✗ Solving failed: {status_data.get('error')}"
                        )
                        break

                    time.sleep(1)
                else:
                    print("\n✗ Timeout waiting for solution")

            # Direct response (sync mode)
            elif "response" in result:
                print("\n✓ Solution found!")
                display_results(result["response"]["solver_response"])

            else:
                print(f"\n✗ Unexpected response format: {result}")

        else:
            print(f"\n✗ API call failed with status {response.status_code}")
            print(f"  Error: {response.text}")

    except requests.exceptions.ConnectionError:
        print("\n✗ Could not connect to cuOpt server")
        print(f"  Make sure the server is running at {API_URL}")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def display_results(solution_data):
    """
    Display the routing solution with distance tier information
    """
    print("\n" + "-" * 80)
    print("ROUTING SOLUTION")
    print("-" * 80)

    if "vehicle_data" in solution_data:
        vehicle_data = solution_data["vehicle_data"]

        for i, (route, route_type) in enumerate(
            zip(vehicle_data.get("routes", []), vehicle_data.get("type", []))
        ):
            if route_type == 0:  # Valid route
                print(f"\nVehicle {i}:")
                print(f"  Route: {' -> '.join(map(str, route))}")

                # Calculate route distance (simplified - using cost as proxy)
                # In real scenario, you'd calculate actual distance from cost matrix
                # For this example, we'll use the cost value from solution

        # Display cost information
        if "cost" in solution_data:
            print(f"\n{'=' * 80}")
            print(
                f"Total Cost (with tiered pricing): {solution_data['cost']:.2f}"
            )
            print(f"{'=' * 80}")

    else:
        print("No solution data available")


def show_tier_interpretation():
    """
    Show how the tiers are interpreted
    """
    print("\n" + "=" * 80)
    print("DISTANCE TIER CONFIGURATION")
    print("=" * 80)

    tiers = payload["fleet_data"]["vehicle_distance_tiers"]

    for vehicle_id, vehicle_tiers in enumerate(tiers):
        print(f"\nVehicle {vehicle_id}:")
        for i, tier in enumerate(vehicle_tiers):
            threshold = tier["threshold"]
            fixed_cost = tier["fixed_cost"]
            cost_per_unit = tier["cost_per_unit"]

            if threshold >= 1e9:
                distance_range = (
                    f"Distance ≥ {vehicle_tiers[i - 1]['threshold']} km"
                )
            elif i == 0:
                distance_range = f"Distance < {threshold} km"
            else:
                prev_threshold = vehicle_tiers[i - 1]["threshold"]
                distance_range = (
                    f"{prev_threshold} km ≤ Distance < {threshold} km"
                )

            if fixed_cost > 0:
                cost_desc = f"Fixed cost: {fixed_cost}"
            else:
                cost_desc = f"{cost_per_unit}/km"

            print(f"  Tier {i + 1}: {distance_range} → {cost_desc}")


if __name__ == "__main__":
    # Show the tier configuration
    show_tier_interpretation()

    # Call the API
    print("\n")
    call_cuopt_api()

    print("\n" + "=" * 80)
    print("EXAMPLE CURL COMMAND")
    print("=" * 80)
    print(f"""
curl -X POST {API_URL} \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload, indent=2)}'
    """)
