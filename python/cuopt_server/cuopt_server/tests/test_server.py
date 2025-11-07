# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import (
    RAPIDS_DATASET_ROOT_DIR,
    RequestClient,
    cuopt_service_sync,
    get_routes,
)

client = RequestClient()


def validate_solver_sol(
    res,
    expected_status=0,
    expected_cost=-1,
    expected_vehicle_count=-1,
    error=None,
    expected_objective_values={},
):
    assert res["status"] == expected_status

    if expected_status == 0:
        if expected_cost != -1:
            assert res["solution_cost"] == expected_cost
        if expected_vehicle_count != -1:
            assert res["num_vehicles"] == expected_vehicle_count
    else:
        assert res["error"] == error

    if expected_objective_values:
        objective_values = res["objective_values"]
        for exp_type, exp_val in expected_objective_values.items():
            assert exp_val == objective_values[exp_type]


def test_sync_endpoint(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}
    time_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    vehicle_max_times = [10, 10]
    objectives = {"cost": 1}

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        travel_time_matrix=time_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        vehicle_max_times=vehicle_max_times,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
        expected_objective_values={"cost": 3.0},
    )

    # Set time_limit to None, will estimate
    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=None,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
    )


def test_request_filter(cuoptproc):  # noqa
    cost_matrix = {
        0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        1: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    }

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_types = [0, 1]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
    )


def test_service_endpoint(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )

    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        # expected_cost=3.0,
        expected_vehicle_count=1,
    )


def test_service_endpoint_with_headers(cuoptproc):  # noqa
    asset_dir = RAPIDS_DATASET_ROOT_DIR + "/cuopt_service_data"
    headers = {
        "NVCF-ASSET-DIR": asset_dir,
        # we only expect 1, but we should test the code just the same
        "NVCF-FUNCTION-ASSET-IDS": "cuopt_problem_data.msgpack,"
        "someotherfile.json",
    }

    # Prove that the new image works with data set to None as well
    optimization_data = {"action": "cuOpt_OptimizedRouting", "data": None}
    res = client.post("/cuopt/cuopt", json=optimization_data, headers=headers)
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
    )

    # Ensure that for backward compat, the current container handles
    # unpickled data as well
    headers["NVCF-FUNCTION-ASSET-IDS"] = "cuopt_problem_data.json"
    res = client.post("/cuopt/cuopt", json=optimization_data, headers=headers)
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=3.0,
        expected_vehicle_count=1,
    )


def test_pickup_and_delivery(cuoptproc):  # noqa
    offsets = [0, 3, 5, 7, 8, 11, 13, 15, 17]
    edges = [1, 2, 3, 0, 2, 0, 3, 4, 5, 6, 7, 2, 4, 1, 5, 0, 6]
    weights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    cost_waypoint_graph_data = {
        0: {"offsets": offsets, "edges": edges, "weights": weights}
    }
    # fleet data
    v_locations = [[0, 0], [1, 4], [0, 4], [1, 0]]

    # task data
    # 2->4, 3->6, 6->4, 4->5, 0->2
    t_locations = [2, 4, 3, 6, 6, 4, 4, 5, 0, 2]
    pick_up_delivery_pair = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

    solver_time_limit = 4

    res = get_routes(
        client,
        cost_waypoint_graph=cost_waypoint_graph_data,
        vehicle_locations=v_locations,
        task_locations=t_locations,
        pickup_and_delivery_pairs=pick_up_delivery_pair,
        time_limit=solver_time_limit,
    )

    assert res.status_code == 200

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        # expected_cost=6.0, FIXME:: Turn off cost for now. We have to
        # re-enable this after we add appropriate operators for
        # heterogenous fleet in the new solver
        expected_vehicle_count=1,
    )


def test_vehicle_order_match(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    }

    # fleet data
    v_locations = [[0, 0], [0, 0], [0, 0]]
    # Order index in task locations
    v_order_match = [
        {"vehicle_id": 0, "order_ids": [0, 1, 2]},
        {"vehicle_id": 1, "order_ids": [0, 1, 2]},
        {"vehicle_id": 2, "order_ids": [3]},
    ]

    # task data
    t_locations = [1, 2, 3, 4]

    solver_time_limit = 4

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        vehicle_order_match=v_order_match,
        task_locations=t_locations,
        time_limit=solver_time_limit,
    )

    assert res.status_code == 200

    veh_data = res.json()["response"]["solver_response"]["vehicle_data"]

    # Vehicle 2 should be part of the solution and 2 should be only serving
    # order 3 located at location 4
    assert (
        "2" in veh_data.keys()
        and sum(veh_data["2"]["route"]) == t_locations[3]
    )


def test_order_vehicle_match(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    }

    # fleet data
    v_locations = [[0, 0], [0, 0], [0, 0]]

    # task data
    t_locations = [1, 2, 3, 4]

    # Only vehicle 2 can serve order 4
    t_vehicle_match = [
        {"order_id": 0, "vehicle_ids": [0, 1]},
        {"order_id": 1, "vehicle_ids": [0, 1]},
        {"order_id": 2, "vehicle_ids": [0, 1]},
        {"order_id": 3, "vehicle_ids": [2]},
    ]

    solver_time_limit = 4

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        order_vehicle_match=t_vehicle_match,
        task_locations=t_locations,
        time_limit=solver_time_limit,
    )

    assert res.status_code == 200

    veh_data = res.json()["response"]["solver_response"]["vehicle_data"]

    # Vehicle 2 should be part of the solution and 2 should be only serving
    # order 3 located at location 4
    assert (
        "2" in veh_data.keys()
        and sum(veh_data["2"]["route"]) == t_locations[3]
    )


def test_prize_collection(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    }

    # fleet data
    v_locations = [[0, 0]]
    v_capacities = [[1]]

    # task data
    t_locations = [1, 2, 3, 4]
    t_demand = [[1, 1, 1, 1]]
    t_prizes = [3.0, 50.0, 1.0, 12.0]
    objectives = {"prize": 2.0}

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        task_locations=t_locations,
        demand=t_demand,
        prizes=t_prizes,
        objectives=objectives,
    )

    assert res.status_code == 200

    vehicle_data = res.json()["response"]["solver_response"]["vehicle_data"]
    assert vehicle_data["0"]["task_id"] == ["Depot", "1", "Depot"]

    dropped_tasks = sorted(
        res.json()["response"]["solver_response"]["dropped_tasks"][
            "task_index"
        ]
    )

    assert [0, 2, 3] == dropped_tasks
    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_objective_values={"cost": 2.0, "prize": -50.0},
    )


def test_result_display(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[4, 4], [8, 8]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_break_time_windows = [
        [[0, 2], [0, 2]],
        [[5, 6], [5, 6]],
    ]
    v_break_durations = [[1, 1], [1, 1]]

    # task data
    t_locations = [0, 1, 2, 1, 2]
    t_demand = [[0, 2, 2, 2, 2], [0, 4, 4, 4, 4]]
    t_time_window = [[0, 10], [0, 2], [7, 8], [0, 2], [7, 8]]
    t_service_time = [0, 1, 1, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        vehicle_break_time_windows=v_break_time_windows,
        vehicle_break_durations=v_break_durations,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )

    assert res.status_code == 200
    vehicle_data = res.json()["response"]["solver_response"]["vehicle_data"][
        "0"
    ]

    assert vehicle_data["type"].count("Break") == 2
    assert vehicle_data["type"][0] == "Depot"
    assert vehicle_data["type"][-1] == "Depot"

    v_skip_first_trips = [True, True]
    v_drop_return_trips = [True, True]

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        vehicle_break_time_windows=v_break_time_windows,
        vehicle_break_durations=v_break_durations,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )

    assert res.status_code == 200
    vehicle_data = res.json()["response"]["solver_response"]["vehicle_data"][
        "0"
    ]
    assert vehicle_data["type"].count("Break") == 2
    assert vehicle_data["type"][0] != "Depot"
    assert vehicle_data["type"][-1] != "Depot"


def test_runtime_ceil(cuoptproc):  # noqa
    cost_matrix = {
        0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    }

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
    )
    assert res.status_code == 200

    solver_time_limit = 4
    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
    )
    assert res.status_code == 200

    solver_time_limit = 120
    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
    )
    assert res.status_code == 200

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
    )
    assert res.status_code == 200


def test_action_value(cuoptproc):  # noqa
    cost_matrix = {
        0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        1: [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
    }

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_types = [0, 1]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    action = "some wrong action"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 422
    assert (
        "Input should be 'cuOpt_OptimizedRouting', 'cuOpt_RoutingValidator', 'cuOpt_LP', 'cuOpt_LPValidator', 'cuOpt_Solver' or 'cuOpt_Validator'"  # noqa
        in res.json()["error"]
    )


def test_mismatch_capacity_demand_dims(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1], [0, 4, 5]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
    )
    assert res.status_code == 400

    assert (
        res.json()["error"]
        == "Mismatch in Capacity and Demand dimension, (capacity_dim) 2 != (demand_dim) 3"  # noqa
    )


def test_deprecation_warnings(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]

    # fields = []

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
    )
    assert res.status_code == 200
    # assert "warnings" in res.json()
    # present = {}
    # for f in fields:
    #    present[f] = False

    # for w in res.json()["warnings"]:
    #    for f in fields:
    #        if f in w:
    #            present[f] = True
    #            break
    # assert False not in present.values()

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
    )
    assert res.status_code == 200
    # assert "warnings" not in res.json()


def test_validator(cuoptproc):  # noqa
    cost_matrix = {0: [[0, 1, 1], [1, 0, 1], [1, 1, 0]]}

    # fleet data
    v_locations = [[0, 0], [0, 0]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]

    # task data
    t_locations = [0, 1, 2]
    t_demand = [[0, 1, 1], [0, 3, 1]]
    t_time_window = [[0, 10], [0, 4], [2, 4]]
    t_service_time = [0, 1, 1]

    solver_time_limit = 4

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        validation_only=True,
    )

    assert res.status_code == 200
    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=-1,
        expected_vehicle_count=-1,
    )
    assert res.json()["response"]["solver_response"]["msg"] == "Input is Valid"

    res = cuopt_service_sync(
        client,
        action="cuOpt_RoutingValidator",
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_window,
        service_times=t_service_time,
        time_limit=solver_time_limit,
    )

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=-1,
        expected_vehicle_count=-1,
    )
    assert res.json()["response"]["solver_response"]["msg"] == "Input is Valid"


def test_vehicle_fixed_costs(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    }

    # fleet data
    v_locations = [[0, 0], [0, 0], [0, 0], [0, 0]]
    v_capacities = [[30, 30, 20, 20]]
    v_fixed_costs = [50, 50, 10, 10]
    objectives = {"cost": 0, "vehicle_fixed_cost": 1.0}

    # task data
    t_locations = [1, 2, 3, 4]
    t_demand = [[10, 10, 10, 10]]

    res = get_routes(
        client,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        capacities=v_capacities,
        vehicle_fixed_costs=v_fixed_costs,
        task_locations=t_locations,
        demand=t_demand,
        objectives=objectives,
        time_limit=10,
    )

    assert res.status_code == 200

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_vehicle_count=2,
        expected_objective_values={"vehicle_fixed_cost": 20.0},
    )


def test_cost_matrix_solution(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        1: [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
    }

    # fleet data
    v_locations = [[6, 6], [7, 7]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_break_time_windows = [[[5, 6], [2, 3]]]
    v_break_durations = [[1, 1]]
    v_break_locations = [2]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_types = [0, 1]

    # task data
    t_locations = [1, 3]
    t_demand = [[1, 1], [3, 1]]
    t_time_windows = [[0, 4], [8, 10]]
    t_service_time = [1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        vehicle_break_time_windows=v_break_time_windows,
        vehicle_break_durations=v_break_durations,
        vehicle_break_locations=v_break_locations,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_windows,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=4.0,
        expected_vehicle_count=1,
    )

    veh_data = res.json()["response"]["solver_response"]["vehicle_data"]

    for vehicle_id, route_data in veh_data.items():
        route_df = pd.DataFrame(route_data)
        sol_start_loc = route_df["route"].iloc[0]
        sol_end_loc = route_df["route"].iloc[-1]
        sol_break_location = route_df["route"][route_df["type"] == "Break"]
        sol_veh_start_time = route_df["arrival_stamp"].iloc[0]
        sol_veh_end_time = route_df["arrival_stamp"].iloc[-1]
        sol_task_loc = route_df["route"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Pickup")
        ]
        sol_arrival_stamp = route_df["arrival_stamp"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Picup")
        ]
        sol_task_id = route_df["task_id"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Picup")
        ]

        vehicle_id = int(vehicle_id)

        assert v_locations[vehicle_id][0] == sol_start_loc
        assert v_locations[vehicle_id][1] == sol_end_loc
        assert v_time_windows[vehicle_id][0] == sol_veh_start_time
        assert v_time_windows[vehicle_id][1] == sol_veh_end_time
        assert sol_break_location.isin(v_break_locations).all()
        assert (
            sol_veh_start_time >= v_time_windows[vehicle_id][0]
            and sol_veh_start_time <= v_time_windows[vehicle_id][1]
        )
        assert sol_task_loc.isin(t_locations).all()
        for i in range(len(sol_arrival_stamp)):
            task_id = int(sol_task_id.iloc[i])
            arrival_time = int(sol_arrival_stamp.iloc[i])

            start_time = t_time_windows[task_id][0]
            end_time = t_time_windows[task_id][1]

            assert arrival_time >= start_time and arrival_time <= end_time


def test_waypoint_graph_solution(cuoptproc):  # noqa
    offsets = [0, 8, 16, 24, 32, 40, 48, 56, 64]
    edges = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
    ]
    weights = [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    cost_waypoint_graph_data = {
        0: {"offsets": offsets, "edges": edges, "weights": weights},
        1: {"offsets": offsets, "edges": edges, "weights": weights},
    }

    # fleet data
    v_locations = [[6, 6], [7, 7]]
    v_capacities = [[2, 2], [4, 1]]
    v_time_windows = [[0, 10], [0, 10]]
    v_break_time_windows = [[[5, 6], [2, 3]]]
    v_break_durations = [[1, 1]]
    v_break_locations = [2]
    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_types = [0, 1]

    # task data
    t_locations = [1, 3]
    t_demand = [[1, 1], [3, 1]]
    t_time_windows = [[0, 4], [8, 10]]
    t_service_time = [1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [20, 20]
    objectives = {"cost": 1}

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_waypoint_graph=cost_waypoint_graph_data,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        vehicle_break_time_windows=v_break_time_windows,
        vehicle_break_durations=v_break_durations,
        vehicle_break_locations=v_break_locations,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_windows,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=4.0,
        expected_vehicle_count=1,
    )

    veh_data = res.json()["response"]["solver_response"]["vehicle_data"]

    for vehicle_id, route_data in veh_data.items():
        route_df = pd.DataFrame(route_data)
        sol_start_loc = route_df["route"].iloc[0]
        sol_end_loc = route_df["route"].iloc[-1]
        sol_break_location = route_df["route"][route_df["type"] == "Break"]
        sol_veh_start_time = route_df["arrival_stamp"].iloc[0]
        sol_veh_end_time = route_df["arrival_stamp"].iloc[-1]
        sol_task_loc = route_df["route"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Pickup")
        ]
        sol_arrival_stamp = route_df["arrival_stamp"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Picup")
        ]
        sol_task_id = route_df["task_id"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Picup")
        ]

        vehicle_id = int(vehicle_id)

        assert v_locations[vehicle_id][0] == sol_start_loc
        assert v_locations[vehicle_id][1] == sol_end_loc
        assert v_time_windows[vehicle_id][0] == sol_veh_start_time
        assert v_time_windows[vehicle_id][1] == sol_veh_end_time
        assert sol_break_location.isin(v_break_locations).all()
        assert (
            sol_veh_start_time >= v_time_windows[vehicle_id][0]
            and sol_veh_start_time <= v_time_windows[vehicle_id][1]
        )
        assert sol_task_loc.isin(t_locations).all()
        for i in range(len(sol_arrival_stamp)):
            task_id = int(sol_task_id.iloc[i])
            arrival_time = int(sol_arrival_stamp.iloc[i])

            start_time = t_time_windows[task_id][0]
            end_time = t_time_windows[task_id][1]

            assert arrival_time >= start_time and arrival_time <= end_time


def test_heterogeneous_breaks(cuoptproc):  # noqa
    cost_matrix = {
        0: [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
        1: [
            [0, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 0],
        ],
    }

    v_locations = [[6, 6], [7, 7]]
    v_capacities = [[2, 2], [4, 4]]
    v_break_locations = [[[2]], [[4], [2, 5]]]

    v_breaks = [
        {
            "vehicle_id": 0,
            "earliest": 5,
            "latest": 6,
            "duration": 1,
            "locations": v_break_locations[0][0],
        },
        {
            "vehicle_id": 1,
            "earliest": 2,
            "latest": 3,
            "duration": 1,
            "locations": v_break_locations[1][0],
        },
        {
            "vehicle_id": 1,
            "earliest": 5,
            "latest": 6,
            "duration": 1,
            "locations": v_break_locations[1][1],
        },
    ]

    v_time_windows = [[0, 10], [0, 10]]

    v_skip_first_trips = [False, False]
    v_drop_return_trips = [False, False]
    v_types = [0, 1]

    # task data
    t_locations = [1, 5, 3, 6]
    t_demand = [[1, 1, 1, 1], [2, 1, 2, 1]]
    t_time_windows = [[0, 4], [0, 4], [8, 10], [8, 10]]
    t_service_time = [1, 1, 1, 1]

    solver_time_limit = 4
    vehicle_max_costs = [10, 10]
    objectives = {"cost": 1}

    action = "cuOpt_OptimizedRouting"

    res = cuopt_service_sync(
        client,
        action=action,
        cost_matrix=cost_matrix,
        vehicle_locations=v_locations,
        vehicle_types=v_types,
        capacities=v_capacities,
        vehicle_time_windows=v_time_windows,
        vehicle_breaks=v_breaks,
        skip_first_trips=v_skip_first_trips,
        drop_return_trips=v_drop_return_trips,
        task_locations=t_locations,
        demand=t_demand,
        task_time_windows=t_time_windows,
        service_times=t_service_time,
        time_limit=solver_time_limit,
        vehicle_max_costs=vehicle_max_costs,
        objectives=objectives,
    )
    assert res.status_code == 200
    assert "reqId" not in res.json()

    validate_solver_sol(
        res.json()["response"]["solver_response"],
        expected_status=0,
        expected_cost=7.0,
        expected_vehicle_count=2,
    )

    veh_data = res.json()["response"]["solver_response"]["vehicle_data"]

    for vehicle_id, route_data in veh_data.items():
        route_df = pd.DataFrame(route_data)
        sol_start_loc = route_df["route"].iloc[0]
        sol_end_loc = route_df["route"].iloc[-1]
        sol_break_location = route_df["route"][route_df["type"] == "Break"]
        sol_veh_start_time = route_df["arrival_stamp"].iloc[0]
        sol_veh_end_time = route_df["arrival_stamp"].iloc[-1]
        sol_task_loc = route_df["route"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Pickup")
        ]
        sol_arrival_stamp = route_df["arrival_stamp"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Pickup")
        ]
        sol_task_id = route_df["task_id"][
            (route_df["type"] == "Delivery") | (route_df["type"] == "Pickup")
        ]

        vehicle_id = int(vehicle_id)

        assert v_locations[vehicle_id][0] == sol_start_loc
        assert v_locations[vehicle_id][1] == sol_end_loc
        assert v_time_windows[vehicle_id][0] == sol_veh_start_time
        assert v_time_windows[vehicle_id][1] >= sol_veh_end_time

        for i, break_location in enumerate(sol_break_location):
            assert break_location in v_break_locations[vehicle_id][i]
        assert (
            sol_veh_start_time >= v_time_windows[vehicle_id][0]
            and sol_veh_start_time <= v_time_windows[vehicle_id][1]
        )
        assert sol_task_loc.isin(t_locations).all()
        for i in range(len(sol_arrival_stamp)):
            task_id = int(sol_task_id.iloc[i])
            arrival_time = int(sol_arrival_stamp.iloc[i])

            start_time = t_time_windows[task_id][0]
            end_time = t_time_windows[task_id][1]

            assert arrival_time >= start_time and arrival_time <= end_time
