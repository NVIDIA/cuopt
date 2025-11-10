# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy

from cuopt_server.tests.utils.utils import cuoptproc  # noqa
from cuopt_server.tests.utils.utils import RequestClient

client = RequestClient()

valid_data = {
    "cost_matrix_data": {
        "data": {
            0: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        }
    },
    "travel_time_matrix_data": {
        "data": {
            0: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ]
        }
    },
    "fleet_data": {
        "vehicle_locations": [[1, 1], [2, 2], [3, 3], [4, 4]],
        "capacities": [[1, 2, 3, 4], [2, 3, 4, 5]],
        "vehicle_time_windows": [[0, 10], [0, 10], [0, 10], [0, 10]],
        "vehicle_break_time_windows": [
            [[1, 2], [1, 2], [2, 3], [2, 3]],
            [[3, 4], [3, 4], [5, 6], [5, 6]],
        ],  # noqa
        "vehicle_break_durations": [[1, 1, 1, 1], [1, 1, 1, 1]],
        "vehicle_break_locations": [1, 2],
        "vehicle_order_match": [{"vehicle_id": 0, "order_ids": [1]}],
        "skip_first_trips": [False, False, True, True],
        "drop_return_trips": [True, False, True, False],
        "min_vehicles": 1,
        "vehicle_max_costs": [150, 150, 150, 150],
        "vehicle_max_times": [100, 30, 50, 70],
        "vehicle_fixed_costs": [50, 50, 50, 50],
    },
    "task_data": {
        "task_locations": [1],
        "demand": [[1], [1]],
        "task_time_windows": [[0, 10]],
        "service_times": [1],
    },
    "solver_config": {"time_limit": 10},
}


# FLEET DATA TESTING


# Test validation error when multiple cost matrices set without vehicle types
def test_invalid_vehicle_types(cuoptproc):  # noqa
    matrix_data = {
        "data": {
            0: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ],
            1: [
                [0, 1, 1, 1, 1],
                [1, 0, 1, 1, 1],
                [1, 1, 0, 1, 1],
                [1, 1, 1, 0, 1],
                [1, 1, 1, 1, 0],
            ],
        }
    }

    test_data = copy.deepcopy(valid_data)

    test_data["cost_matrix_data"] = matrix_data

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Set vehicle types when using multiple matrices",
        "error_result": True,
    }


# Testing valid with all fleet parameters
def test_valid_full_set_fleet_data(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)
    assert response_set.status_code == 200


# Testing valid with minimal required parameters
def test_valid_minimal_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"] = {
        "vehicle_locations": [[1, 1], [2, 2], [3, 3], [4, 4]]
    }

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 200


# Testing invalid, all fleet parameters need to in range
def test_invalid_values_set_fleet_data(cuoptproc):  # noqa
    invalid_fleet_data_values = {
        "vehicle_locations": [[-1, 1], [2, 2], [3, 3], [4, 4]],
        "capacities": [[0, -2, 3, 4], [2, 3, 4, 5]],
        "vehicle_time_windows": [[-1, 2], [1, 2], [1, 2], [1, 2]],
        "vehicle_break_time_windows": [
            [[1, 2], [1, 2], [-2, 3], [2, 3]],
            [[2, 3], [2, 3], [1, 2], [1, 2]],
        ],  # noqa
        "vehicle_break_durations": [[1, 1, 1, -1], [1, 1, 1, 1]],
        "vehicle_break_locations": [1, -2],
        "vehicle_order_match": [{"vehicle_id": 6, "order_ids": [1, 2]}],
        "skip_first_trips": [False, False, True, True],
        "drop_return_trips": [True, False, True, False],
        "min_vehicles": 0,
        "vehicle_max_costs": [0, 0, 0, 0],
        "vehicle_max_times": [0, 0, 0, 0],
        "vehicle_fixed_costs": [-1, 50, 50, 50],
    }

    # all task locations must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_locations"] = invalid_fleet_data_values[
        "vehicle_locations"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Fleet locations represent index locations and must be greater than or equal to 0",  # noqa
        "error_result": True,
    }

    # task locations should be list of pairs of start and end locations
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_locations"] = [
        [1, 1],
        [2, 2, 2],
        [3, 3],
        [4, 4],
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Vehicle locations should be list of pairs of start and end location for each vehicle",  # noqa
        "error_result": True,
    }

    # capacity values if provided must be greater than 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["capacities"] = invalid_fleet_data_values[
        "capacities"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All capacity dimensions values must be 0 or greater",
        "error_result": True,
    }

    # vehicle time windows if provided must be greater than or equal
    # to 0 for each vehicle
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_time_windows"] = (
        invalid_fleet_data_values["vehicle_time_windows"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_time_windows: All vehicle time window values must be greater than or equal to 0",  # noqa
        "error_result": True,
    }

    # vehicle break time windows if provided must be greater than or equal
    # to 0 for each vehicle
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_break_time_windows"] = (
        invalid_fleet_data_values["vehicle_break_time_windows"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_break_time_windows: All vehicle time window values must be greater than or equal to 0",  # noqa
        "error_result": True,
    }

    # vehicle break durations if provided must be greater than or equal
    # to 0 for each vehicle
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_break_durations"] = (
        invalid_fleet_data_values["vehicle_break_durations"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Vehicle break duration must be greater than or equal to 0",
        "error_result": True,
    }

    # vehicle break locations if provided must be greater than or equal
    # to 0 for each vehicle
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_break_locations"] = (
        invalid_fleet_data_values["vehicle_break_locations"]
    )

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Vehicle break location must be greater than or equal to 0",
        "error_result": True,
    }

    # vehicle order match, vehicle id should be with [0, num_vehicles),
    # and order ids should be non-negative values.
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_order_match"] = invalid_fleet_data_values[
        "vehicle_order_match"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "One or more Vehicle IDs provided are not in the expected range, should be within [0,  number of vehicle )",  # noqa
        "error_result": True,
    }

    # min_vehicles must be greater than 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["min_vehicles"] = invalid_fleet_data_values[
        "min_vehicles"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Minimum vehicles must be greater than 0",
        "error_result": True,
    }

    # vehicle_max_costs must be greater than 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_max_costs"] = invalid_fleet_data_values[
        "vehicle_max_costs"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Maximum distance any vehicle can travel must be greater than 0",  # noqa
        "error_result": True,
    }

    # vehicle_max_times must be greater than 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_max_times"] = invalid_fleet_data_values[
        "vehicle_max_times"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Maximum time any vehicle can travel must be greater than 0",  # noqa
        "error_result": True,
    }

    # vehicle_fixed_costs must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_fixed_costs"] = invalid_fleet_data_values[
        "vehicle_fixed_costs"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Fixed cost of vehicle must be greater than or equal to 0",
        "error_result": True,
    }

    # vehicle_fixed_costs improper length
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_fixed_costs"] = [50, 50, 50]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining vehicle properties must be of consistent length",  # noqa
        "error_result": True,
    }


def test_invalid_length_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_time_windows"] = [[1, 2], [1, 2], [1, 2]]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining vehicle properties must be of consistent length",  # noqa
        "error_result": True,
    }


def test_invalid_capacities_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["capacities"] = [[1, 2, 3, 4], [2, 3, 4]]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All capacity dimensions must have length equal to the number of vehicles",  # noqa
        "error_result": True,
    }


def test_invalid_time_windows_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_time_windows"] = [
        [1, 2],
        [4, 2],
        [1, 2],
        [1, 2],
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_time_windows: All vehicles time windows must have vehicle_x_time_window[0] < vehicle_x_time_window[1]",  # noqa
        "error_result": True,
    }

    test_data["fleet_data"]["vehicle_time_windows"] = [
        [1, 2],
        [0, 2],
        [1, 2],
        [1, 2, 9],
    ]
    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_time_windows: Time windows for each vehicle must be of length 2. 0: earliest, 1: latest",  # noqa
        "error_result": True,
    }


def test_invalid_break_time_windows_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_break_time_windows"] = [
        [[1, 2], [1, 2], [2, 3], [2, 3]],
        [[2, 3], [2, 3], [1, 2], [2, 1]],
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_break_time_windows: All vehicles time windows must have vehicle_x_time_window[0] < vehicle_x_time_window[1]",  # noqa
        "error_result": True,
    }

    test_data["fleet_data"]["vehicle_break_time_windows"] = [
        [[1], [1, 2], [2, 3], [2, 3]],
        [[2, 3], [2, 3], [1, 2], [1, 2]],
    ]  # noqa
    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle_break_time_windows: Time windows for each vehicle must be of length 2. 0: earliest, 1: latest",  # noqa
        "error_result": True,
    }


def test_invalid_skip_first_trips_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["skip_first_trips"] = [False, False, True]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining vehicle properties must be of consistent length",  # noqa
        "error_result": True,
    }


def test_invalid_drop_return_trips_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["drop_return_trips"] = [False, False, True]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining vehicle properties must be of consistent length",  # noqa
        "error_result": True,
    }


# Invalid order ids in vehicle order match
def test_vehicle_order_match(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["vehicle_order_match"] = [
        {"vehicle_id": 0, "order_ids": [0, -1]}
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Order Id should be greater than or equal to zero",
        "error_result": True,
    }


def test_invalid_extra_arg_set_fleet_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["fleet_data"]["extra_arg"] = 1

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 422
