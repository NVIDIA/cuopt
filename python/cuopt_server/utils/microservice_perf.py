# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import time

import numpy as np
import requests


def write_json_results(results, full_path):
    # Serializing json
    json_object = json.dumps(results, indent=4)

    # Writing to sample.json
    with open(full_path, "w") as outfile:
        outfile.write(json_object)


def generate_random_cost_matrix(n_locs, scale_factor=100):
    rand_cost_mat = np.random.rand(n_locs, n_locs)
    np.fill_diagonal(rand_cost_mat, 0.0)
    return rand_cost_mat * scale_factor


def convert_cost_matrix_to_csr(np_cost_matrix):
    num_nodes = len(np_cost_matrix)

    offsets = []
    edges = []
    weights = []

    cur_offset = 0
    for ind_loc in range(num_nodes):
        offsets.append(cur_offset)

        new_edges = list(range(num_nodes))
        new_edges.pop(ind_loc)
        new_weights = list(np_cost_matrix[ind_loc])
        new_weights.pop(ind_loc)

        edges += new_edges
        weights += new_weights

        cur_offset = len(edges)
    offsets.append(cur_offset)

    return offsets, edges, weights


def generate_random_task_data(
    n_tasks,
    demand_range=5,
    serve_time_range=5,
):
    task_locations = np.random.choice(n_tasks, n_tasks, replace=False).tolist()

    demand = [np.random.randint(1, demand_range, n_tasks).tolist()]

    pd_pairs = np.arange(n_tasks)
    np.random.shuffle(pd_pairs)
    if n_tasks % 2 != 0:
        pd_pairs = np.append(pd_pairs, pd_pairs[0])
    pd_pairs = pd_pairs.reshape((len(pd_pairs) // 2), 2).tolist()

    task_time_windows = [[1, 2]] * n_tasks

    service_times = np.random.randint(1, serve_time_range, n_tasks).tolist()

    task_data = {
        "task_locations": task_locations,
        "demand": demand,
        "pickup_and_delivery_pairs": pd_pairs,
        "task_time_windows": task_time_windows,
        "service_times": service_times,
    }

    return task_data


def generate_random_fleet_data(n_vehicles, capacity_range=5):
    vehicle_locations = (
        np.random.randint(1, n_vehicles, n_vehicles * 2)
        .reshape((n_vehicles, 2))
        .tolist()
    )

    capacities = [np.random.randint(1, capacity_range, n_vehicles).tolist()]

    vehicle_time_windows = [[1, 2]] * n_vehicles

    vehicle_break_time_windows = [[[1, 2]] * n_vehicles]

    vehicle_break_durations = [[1] * n_vehicles]

    vehicle_break_locations = list(range(len(vehicle_locations)))

    skip_first_trips = np.random.choice(
        [True, False], n_vehicles, replace=True
    ).tolist()
    drop_return_trips = skip_first_trips[::-1]

    min_vehicles = n_vehicles - 1

    vehicle_max_costs = [1000] * n_vehicles

    max_slack = np.random.randint(1, 100, 1)[0].item()

    max_lateness_per_vehicle = np.random.randint(1, 30, 1)[0].item()

    fleet_data = {
        "vehicle_locations": vehicle_locations,
        "capacities": capacities,
        "vehicle_time_windows": vehicle_time_windows,
        "vehicle_break_time_windows": vehicle_break_time_windows,
        "vehicle_break_durations": vehicle_break_durations,
        "vehicle_break_locations": vehicle_break_locations,
        "skip_first_trips": skip_first_trips,
        "drop_return_trips": drop_return_trips,
        "min_vehicles": min_vehicles,
        "vehicle_max_costs": vehicle_max_costs,
        "max_slack": max_slack,
        "max_lateness_per_vehicle": max_lateness_per_vehicle,
    }

    return fleet_data


def generate_random_config_data():
    time_limit = np.round(np.random.random_sample(), 3)

    solver_config = {
        "time_limit": time_limit,
    }

    return solver_config


def verify_cuopt_up(cuopt_url):
    cuopt_response = requests.get(f"{cuopt_url}/health")
    return cuopt_response.status_code


def time_health_endpoint(cuopt_url, n_runs):
    health_timing_results = np.array([])
    for _ in range(n_runs):
        start_health_perf = time.perf_counter()
        _ = requests.get(f"{cuopt_url}/health")
        end_health_perf = time.perf_counter()

        health_timing_results = np.append(
            health_timing_results, end_health_perf - start_health_perf
        )

    average_health_time = np.mean(health_timing_results)

    return average_health_time


def time_get_optimization_data_state_endpoint(cuopt_url, n_runs):
    opti_data_state_timing_results = np.array([])
    for _ in range(n_runs):
        start_get_opti_data_perf = time.perf_counter()
        _ = requests.get(f"{cuopt_url}/get_optimization_data_state")
        end_get_opti_data_perf = time.perf_counter()

        opti_data_state_timing_results = np.append(
            opti_data_state_timing_results,
            end_get_opti_data_perf - start_get_opti_data_perf,
        )

    average_opti_data_state = np.mean(opti_data_state_timing_results)

    return average_opti_data_state


def time_clear_optimization_data_endpoint(cuopt_url, n_runs):
    clear_opti_data_timing_results = np.array([])
    for _ in range(n_runs):
        start_clear_opti_data_perf = time.perf_counter()
        _ = requests.delete(f"{cuopt_url}/clear_optimization_data")
        end_clear_opti_data_perf = time.perf_counter()

        clear_opti_data_timing_results = np.append(
            clear_opti_data_timing_results,
            end_clear_opti_data_perf - start_clear_opti_data_perf,
        )

    average_clear_opti_data = np.mean(clear_opti_data_timing_results)

    return average_clear_opti_data


def time_set_update_cost_matrix_endpoints(
    cuopt_url,
    n_runs,
    test_set=True,
    test_update=True,
    payload=None,
    n_locs=None,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    if payload is None:
        generated_cost_matrix = generate_random_cost_matrix(n_locs).tolist()
        endpoint_json = {"cost_matrix": {0: generated_cost_matrix}}
    else:
        endpoint_json = payload

    set_matrix_timing_results = np.array([])
    update_matrix_timing_results = np.array([])
    for _ in range(n_runs):
        if test_set:
            start_set_matrix_perf = time.perf_counter()
            _ = requests.post(
                f"{cuopt_url}/set_cost_matrix",
                json=endpoint_json,
                params=params,
            )
            end_set_matrix_perf = time.perf_counter()

            set_matrix_timing_results = np.append(
                set_matrix_timing_results,
                end_set_matrix_perf - start_set_matrix_perf,
            )

        if test_update:
            start_update_matrix_perf = time.perf_counter()
            _ = requests.put(
                f"{cuopt_url}/update_cost_matrix",
                json=endpoint_json,
                params=params,
            )
            end_update_matrix_perf = time.perf_counter()

            update_matrix_timing_results = np.append(
                update_matrix_timing_results,
                end_update_matrix_perf - start_update_matrix_perf,
            )

    if len(set_matrix_timing_results) != 0:
        set_response = np.mean(set_matrix_timing_results)
    else:
        set_response = None

    if len(update_matrix_timing_results) != 0:
        update_response = np.mean(update_matrix_timing_results)
    else:
        update_response = None

    return (set_response, update_response)


def time_set_update_waypoint_graph_endpoints(
    cuopt_url,
    n_runs,
    test_set=True,
    test_update=True,
    payload=None,
    n_locs=None,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    if payload is None:
        generated_cost_matrix = generate_random_cost_matrix(n_locs)

        (
            generated_offsets,
            generated_edges,
            generated_weights,
        ) = convert_cost_matrix_to_csr(generated_cost_matrix)

        graph_data = {
            "edges": generated_edges,
            "offsets": generated_offsets,
            "weights": generated_weights,
        }

        endpoint_json = {"waypoint_graph": {0: graph_data}}
    else:
        endpoint_json = payload

    set_waypoint_graph_timing_results = np.array([])
    update_waypoint_graph_timing_results = np.array([])
    for _ in range(n_runs):
        if test_set:
            start_set_waypoint_graph_perf = time.perf_counter()
            _ = requests.post(
                f"{cuopt_url}/set_cost_waypoint_graph",
                json=endpoint_json,
                params=params,
            )
            end_set_waypoint_graph_perf = time.perf_counter()

            set_waypoint_graph_timing_results = np.append(
                set_waypoint_graph_timing_results,
                end_set_waypoint_graph_perf - start_set_waypoint_graph_perf,
            )

        if test_update:
            endpoint_update_json = {
                "weights": {0: endpoint_json["waypoint_graph"][0]["weights"]}
            }
            start_update_waypoint_graph_perf = time.perf_counter()
            _ = requests.put(
                f"{cuopt_url}/update_cost_waypoint_graph_weights",
                json=endpoint_update_json,
                params=params,
            )
            end_update_waypoint_graph_perf = time.perf_counter()

            update_waypoint_graph_timing_results = np.append(
                update_waypoint_graph_timing_results,
                end_update_waypoint_graph_perf
                - start_update_waypoint_graph_perf,
            )

    if len(set_waypoint_graph_timing_results) != 0:
        set_response = np.mean(set_waypoint_graph_timing_results)
    else:
        set_response = None

    if len(update_waypoint_graph_timing_results) != 0:
        update_response = np.mean(update_waypoint_graph_timing_results)
    else:
        update_response = None

    return (set_response, update_response)


def time_set_update_task_endpoints(
    cuopt_url,
    n_runs,
    test_set=True,
    test_update=True,
    payload=None,
    n_tasks=None,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    if payload is None:
        endpoint_json = generate_random_task_data(n_tasks)
    else:
        endpoint_json = payload

    set_tasks_timing_results = np.array([])
    update_tasks_timing_results = np.array([])
    for _ in range(n_runs):
        if test_set:
            start_set_tasks_perf = time.perf_counter()
            _ = requests.post(
                f"{cuopt_url}/set_task_data",
                json=endpoint_json,
                params=params,
            )
            end_set_tasks_perf = time.perf_counter()

            set_tasks_timing_results = np.append(
                set_tasks_timing_results,
                end_set_tasks_perf - start_set_tasks_perf,
            )

        if test_update:
            start_update_tasks_perf = time.perf_counter()
            _ = requests.put(
                f"{cuopt_url}/update_task_data",
                json=endpoint_json,
                params=params,
            )
            end_update_tasks_perf = time.perf_counter()

            update_tasks_timing_results = np.append(
                update_tasks_timing_results,
                end_update_tasks_perf - start_update_tasks_perf,
            )

    if len(set_tasks_timing_results) != 0:
        set_response = np.mean(set_tasks_timing_results)
    else:
        set_response = None

    if len(update_tasks_timing_results) != 0:
        update_response = np.mean(update_tasks_timing_results)
    else:
        update_response = None

    return (set_response, update_response)


def time_set_update_fleet_endpoints(
    cuopt_url,
    n_runs,
    test_set=True,
    test_update=True,
    payload=None,
    n_vehicles=None,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    if payload is None:
        endpoint_json = generate_random_fleet_data(n_vehicles)
    else:
        endpoint_json = payload

    set_fleet_timing_results = np.array([])
    update_fleet_timing_results = np.array([])
    for _ in range(n_runs):
        if test_set:
            start_set_fleet_perf = time.perf_counter()
            _ = requests.post(
                f"{cuopt_url}/set_fleet_data",
                json=endpoint_json,
                params=params,
            )
            end_set_fleet_perf = time.perf_counter()

            set_fleet_timing_results = np.append(
                set_fleet_timing_results,
                end_set_fleet_perf - start_set_fleet_perf,
            )

        if test_update:
            start_update_fleet_perf = time.perf_counter()
            _ = requests.put(
                f"{cuopt_url}/update_fleet_data",
                json=endpoint_json,
                params=params,
            )
            end_update_fleet_perf = time.perf_counter()

            update_fleet_timing_results = np.append(
                update_fleet_timing_results,
                end_update_fleet_perf - start_update_fleet_perf,
            )

    if len(set_fleet_timing_results) != 0:
        set_response = np.mean(set_fleet_timing_results)
    else:
        set_response = None

    if len(update_fleet_timing_results) != 0:
        update_response = np.mean(update_fleet_timing_results)
    else:
        update_response = None

    return (set_response, update_response)


def time_set_update_config_endpoints(
    cuopt_url,
    n_runs,
    test_set=True,
    test_update=True,
    payload=None,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    if payload is None:
        endpoint_json = generate_random_config_data()
    else:
        endpoint_json = payload

    set_config_timing_results = np.array([])
    update_config_timing_results = np.array([])
    for _ in range(n_runs):
        if test_set:
            start_set_config_perf = time.perf_counter()
            _ = requests.post(
                f"{cuopt_url}/set_solver_config",
                json=endpoint_json,
                params=params,
            )
            end_set_config_perf = time.perf_counter()

            set_config_timing_results = np.append(
                set_config_timing_results,
                end_set_config_perf - start_set_config_perf,
            )

        if test_update:
            start_update_config_perf = time.perf_counter()
            _ = requests.put(
                f"{cuopt_url}/update_solver_config",
                json=endpoint_json,
                params=params,
            )
            end_update_config_perf = time.perf_counter()

            update_config_timing_results = np.append(
                update_config_timing_results,
                end_update_config_perf - start_update_config_perf,
            )

    if len(set_config_timing_results) != 0:
        set_response = np.mean(set_config_timing_results)
    else:
        set_response = None

    if len(update_config_timing_results) != 0:
        update_response = np.mean(update_config_timing_results)
    else:
        update_response = None

    return (set_response, update_response)


def time_solve_full_problem_async(
    cuopt_url,
    n_runs,
    env_type,
    env_data,
    task_data,
    fleet_data,
    config_data,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    solve_time_limit = config_data["time_limit"]

    full_solve_async_timing_results = np.array([])
    full_solve_async_solve_delta_results = np.array([])
    for _ in range(n_runs):
        # Clear any existing data
        _ = requests.delete(f"{cuopt_url}/clear_optimization_data")

        start_full_solve_perf = time.perf_counter()
        # Set environment data
        if env_type == "matrix":
            _ = requests.post(
                f"{cuopt_url}/set_cost_matrix",
                json=env_data,
                params=params,
            )
        elif env_type == "waypoint_graph":
            _ = requests.post(
                f"{cuopt_url}/set_cost_waypoint_graph",
                json=env_data,
                params=params,
            )

        # Set task data
        _ = requests.post(
            f"{cuopt_url}/set_task_data",
            json=task_data,
            params=params,
        )

        # Set fleet data
        _ = requests.post(
            f"{cuopt_url}/set_fleet_data",
            json=fleet_data,
            params=params,
        )

        # Set config data
        _ = requests.post(
            f"{cuopt_url}/set_solver_config",
            json=config_data,
            params=params,
        )

        # Solve the problem
        start_solve_timer = time.perf_counter()
        _ = requests.get(
            f"{cuopt_url}/routes",
        )
        end_solve_timer = time.perf_counter()
        end_full_solve_perf = time.perf_counter()

        # Record timings
        full_solve_time = (
            end_full_solve_perf - start_full_solve_perf
        ) - solve_time_limit

        solve_delta = (end_solve_timer - start_solve_timer) - solve_time_limit

        full_solve_async_timing_results = np.append(
            full_solve_async_timing_results, full_solve_time
        )
        full_solve_async_solve_delta_results = np.append(
            full_solve_async_solve_delta_results, solve_delta
        )
    full_solve_results = np.mean(full_solve_async_timing_results)
    solve_delta_results = np.mean(full_solve_async_solve_delta_results)

    return (full_solve_results, solve_delta_results)


def time_solve_full_problem_sync(
    cuopt_url,
    n_runs,
    env_type,
    env_data,
    task_data,
    fleet_data,
    config_data,
    return_data_state=False,
):
    params = {
        "return_data_state": return_data_state,
    }

    solve_time_limit = config_data["time_limit"]

    if env_type == "matrix":
        json_data = {
            "cost_matrix_data": env_data,
            "cost_waypoint_graph_data": [],
            "travel_time_matrix_data": [],
            "travel_time_waypoint_graph_data": [],
            "fleet_data": fleet_data,
            "task_data": task_data,
            "solver_config": config_data,
        }

    elif env_type == "waypoint_graph":
        json_data = {
            "cost_matrix_data": [],
            "cost_waypoint_graph_data": env_data,
            "travel_time_matrix_data": [],
            "travel_time_waypoint_graph_data": [],
            "fleet_data": fleet_data,
            "task_data": task_data,
            "solver_config": config_data,
        }

    full_solve_sync_timing_results = np.array([])
    for _ in range(n_runs):
        # Clear any existing data
        _ = requests.delete(f"{cuopt_url}/clear_optimization_data")

        start_full_solve_timer = time.perf_counter()
        _ = requests.post(
            f"{cuopt_url}/get_optimized_routes_sync",
            json=json_data,
            params=params,
        )
        end_full_solve_timer = time.perf_counter()

        full_solve_sync_timing_results = np.append(
            full_solve_sync_timing_results,
            (end_full_solve_timer - start_full_solve_timer) - solve_time_limit,
        )

    full_solve_results = np.mean(full_solve_sync_timing_results)

    return full_solve_results


def run_endpoint_tests(
    cuopt_url,
    n_runs,
    n_locs_array=[10, 100, 200, 400, 600, 800, 1000],
    n_task_array=[10, 100, 200, 400, 600, 800, 1000],
    n_vehicles_array=[2, 10, 50, 100, 200, 400, 800],
):
    end_point_results = {}

    # Health
    print("Testing Health Endpoint")
    end_point_results["health"] = time_health_endpoint(cuopt_url, n_runs)

    # Get Data
    print("Testing Get Optimization Data Endpoint : Initial")
    end_point_results["get_optimization_data_state_empty"] = (
        time_get_optimization_data_state_endpoint(cuopt_url, n_runs)
    )

    # Clear Data
    print("Testing Clear Optimization Data Endpoint")
    end_point_results["clear_optimization_data"] = (
        time_clear_optimization_data_endpoint(cuopt_url, n_runs)
    )

    # Environment Data
    for number_of_locations in n_locs_array:
        # Cost Matrix
        print(
            f"Testing Set/Update Matrix Endpoints: \
                {number_of_locations} Locations"
        )

        (
            set_matrix_results,
            update_matrix_results,
        ) = time_set_update_cost_matrix_endpoints(
            cuopt_url,
            n_runs,
            test_set=True,
            test_update=True,
            payload=None,
            n_locs=number_of_locations,
        )

        end_point_results[f"set_matrix_{number_of_locations}"] = (
            set_matrix_results
        )
        end_point_results[f"update_matrix_{number_of_locations}"] = (
            update_matrix_results
        )

        # Waypoint Graph
        print(
            f"Testing Set/Update Waypoint Graph Endpoints: \
                {number_of_locations} Nodes, {number_of_locations * (number_of_locations - 1)} Edges"  # noqa
        )

        (
            set_waypoint_graph_results,
            update_waypoint_graph_results,
        ) = time_set_update_waypoint_graph_endpoints(
            cuopt_url,
            n_runs,
            test_set=True,
            test_update=True,
            payload=None,
            n_locs=number_of_locations,
        )

        end_point_results[f"set_waypoint_graph_{number_of_locations}"] = (
            set_waypoint_graph_results
        )
        end_point_results[f"update_waypoint_graph_{number_of_locations}"] = (
            update_waypoint_graph_results
        )

    # Order Data
    for number_of_tasks in n_task_array:
        print(
            f"Testing Set/Update Tasks Endpoints: \
                {number_of_tasks} Tasks"
        )

        (
            set_tasks_results,
            update_tasks_results,
        ) = time_set_update_task_endpoints(
            cuopt_url,
            n_runs,
            test_set=True,
            test_update=True,
            payload=None,
            n_tasks=number_of_tasks,
            return_data_state=False,
        )

        end_point_results[f"set_tasks_{number_of_tasks}"] = set_tasks_results
        end_point_results[f"update_tasks_{number_of_tasks}"] = (
            update_tasks_results
        )

    # Vehicles Data
    for number_of_vehicles in n_vehicles_array:
        print(
            f"Testing Set/Update Fleet Endpoints: \
                {number_of_vehicles} Vehicles"
        )

        (
            set_fleet_results,
            update_fleet_results,
        ) = time_set_update_fleet_endpoints(
            cuopt_url,
            n_runs,
            test_set=True,
            test_update=True,
            payload=None,
            n_vehicles=number_of_vehicles,
            return_data_state=False,
        )

        end_point_results[f"set_fleet_{number_of_vehicles}"] = (
            set_fleet_results
        )
        end_point_results[f"update_fleet_{number_of_vehicles}"] = (
            update_fleet_results
        )

    # Solver Config
    print("Testing Set/Update Solver Config Endpoint")
    (
        set_config_results,
        update_config_results,
    ) = time_set_update_config_endpoints(
        cuopt_url,
        n_runs,
        test_set=True,
        test_update=True,
        payload=None,
        return_data_state=False,
    )

    end_point_results["set_config"] = set_config_results
    end_point_results["update_config"] = update_config_results
    print("\nENDPOINT TESTING COMPLETE\n")
    return end_point_results


def run_methods_tests(
    cuopt_url,
    n_runs,
    n_locs_array=[10, 100, 200, 400, 600, 800, 1000],
):
    methods_results = {}
    for n_locations in n_locs_array:
        # generate random cost matrix
        np_cost_matrix = generate_random_cost_matrix(n_locations, 100)
        cost_matrix = {"cost_matrix": {0: np_cost_matrix.tolist()}}

        # generate waypoint graph from cost matrix
        (
            generated_offsets,
            generated_edges,
            generated_weights,
        ) = convert_cost_matrix_to_csr(np_cost_matrix)

        waypoint_graph = {
            "waypoint_graph": {
                0: {
                    "edges": generated_edges,
                    "offsets": generated_offsets,
                    "weights": generated_weights,
                }
            }
        }

        # generate minimal task data
        task_locations = np.random.choice(
            n_locations, n_locations // 5, replace=False
        ).tolist()
        demand = [[1] * len(task_locations)]
        task_data = {
            "task_locations": task_locations,
            "demand": demand,
        }
        assert len(task_locations) == len(demand[0])

        # generate minimal fleet data
        vehicle_locations = [[0, 0]] * (n_locations // 10)
        capacities = [
            [1 + (len(demand[0]) // (n_locations // 10))]
            * len(vehicle_locations)
        ]
        fleet_data = {
            "vehicle_locations": vehicle_locations,
            "capacities": capacities,
        }
        assert len(vehicle_locations) == len(capacities[0])
        assert np.sum(demand[0]) <= np.sum(capacities[0])

        # generate reasonable config
        config_data = {
            "time_limit": 1.0,
        }

        # run matrix data off
        print(f"Running Cost Matrix, Data Off, {n_locations} Locations")
        (
            data_off_matrix_full_solve_results,
            data_off_matrix_solve_delta_results,
        ) = time_solve_full_problem_async(
            cuopt_url=cuopt_url,
            n_runs=n_runs,
            env_type="matrix",
            env_data=cost_matrix,
            task_data=task_data,
            fleet_data=fleet_data,
            config_data=config_data,
            return_data_state=False,
        )

        methods_results[f"full_solve_matrix_data_off_{n_locations}"] = (
            data_off_matrix_full_solve_results
        )

        methods_results[f"solve_delta_matrix_{n_locations}"] = (
            data_off_matrix_solve_delta_results
        )

        # run matrix data on
        print(f"Running Cost Matrix, Data On, {n_locations} Locations")
        (
            data_on_matrix_full_solve_results,
            data_on_matrix_solve_delta_results,
        ) = time_solve_full_problem_async(
            cuopt_url=cuopt_url,
            n_runs=n_runs,
            env_type="matrix",
            env_data=cost_matrix,
            task_data=task_data,
            fleet_data=fleet_data,
            config_data=config_data,
            return_data_state=True,
        )

        methods_results[f"full_solve_matrix_data_on_{n_locations}"] = (
            data_on_matrix_full_solve_results
        )

        # run waypoint_graph data off
        print(f"Running Waypoint Graph, Data Off, {n_locations} Locations")
        (
            data_off_wpg_full_solve_results,
            data_off_wpg_solve_delta_results,
        ) = time_solve_full_problem_async(
            cuopt_url=cuopt_url,
            n_runs=n_runs,
            env_type="waypoint_graph",
            env_data=waypoint_graph,
            task_data=task_data,
            fleet_data=fleet_data,
            config_data=config_data,
            return_data_state=False,
        )

        methods_results[f"full_solve_wpg_data_off_{n_locations}"] = (
            data_off_wpg_full_solve_results
        )

        methods_results[f"solve_delta_wpg_{n_locations}"] = (
            data_off_wpg_solve_delta_results
        )

        # Test sync waypoint graph
        print(
            f"Running Sync Waypoint Graph, Data Off, {n_locations} Locations"
        )
        data_off_sync_wpg_full_solve_results = time_solve_full_problem_sync(
            cuopt_url=cuopt_url,
            n_runs=n_runs,
            env_type="waypoint_graph",
            env_data=waypoint_graph,
            task_data=task_data,
            fleet_data=fleet_data,
            config_data=config_data,
            return_data_state=False,
        )

        methods_results[f"full_solve_sync_wpg_data_off_{n_locations}"] = (
            data_off_sync_wpg_full_solve_results
        )

        # Test sync waypoint graph
        print(f"Running Sync Matrix, Data Off, {n_locations} Locations")
        data_off_sync_matrix_full_solve_results = time_solve_full_problem_sync(
            cuopt_url=cuopt_url,
            n_runs=n_runs,
            env_type="matrix",
            env_data=cost_matrix,
            task_data=task_data,
            fleet_data=fleet_data,
            config_data=config_data,
            return_data_state=False,
        )

        methods_results[f"full_solve_sync_matrix_data_off_{n_locations}"] = (
            data_off_sync_matrix_full_solve_results
        )
    print("\nMETHODS TESTING COMPLETE\n")
    return methods_results


def main(cuopt_ip, cuopt_port, test_type, n_runs, out_dir):
    cuopt_url = f"http://{cuopt_ip}:{cuopt_port}/cuopt"

    # Ensure that cuOpt is up at the location provided
    assert verify_cuopt_up(cuopt_url) == 200

    # If the server is up clear the current data
    time_clear_optimization_data_endpoint(cuopt_url, n_runs=1)

    endpoint_data = None
    methods_data = None

    if test_type == "all":
        endpoint_data = run_endpoint_tests(cuopt_url, n_runs)
        methods_data = run_methods_tests(cuopt_url, n_runs)

    elif test_type == "endpoints":
        endpoint_data = run_endpoint_tests(cuopt_url, n_runs)

    elif test_type == "methods":
        methods_data = run_methods_tests(cuopt_url, n_runs)

    if endpoint_data is not None:
        write_json_results(endpoint_data, f"{out_dir}/endpoint_results.json")

    if methods_data is not None:
        write_json_results(methods_data, f"{out_dir}/methods_results.json")


if __name__ == "__main__":
    out_dir = os.getcwd()

    parser = argparse.ArgumentParser(
        description="A benchmark script for cuOpt microservice",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-t",
        "--test_type",
        help="Test type options. Select from [all, endpoints, methods]",
        required=False,
        default="all",
        type=str,
    )

    parser.add_argument(
        "-ip",
        "--ip",
        help="IP of the machine running the cuOpt microservice",
        required=False,
        default="0.0.0.0",
        type=str,
    )

    parser.add_argument(
        "-p",
        "--port",
        help="Port at the given IP running the cuOpt microservice",
        required=False,
        default="5000",
        type=str,
    )

    parser.add_argument(
        "-n",
        "--n_runs",
        help="Dictates the number of runs to average over",
        required=False,
        default=10,
        type=int,
    )

    args = parser.parse_args()
    test_type = args.test_type
    cuopt_ip = args.ip
    cuopt_port = args.port
    n_runs = args.n_runs

    main(cuopt_ip, cuopt_port, test_type, n_runs, out_dir)
