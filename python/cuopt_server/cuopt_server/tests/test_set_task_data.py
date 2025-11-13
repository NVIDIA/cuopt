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
        "capacities": [[50, 50, 50, 50], [50, 50, 50, 50]],
        "vehicle_time_windows": [[0, 100], [0, 100], [0, 100], [0, 100]],
    },
    "task_data": {
        "task_locations": [1, 2, 3, 4],
        "demand": [[1, -1, 1, -1], [1, -1, 1, -1]],
        "pickup_and_delivery_pairs": [[0, 1], [2, 3]],
        "task_time_windows": [[0, 100], [0, 100], [0, 100], [0, 100]],
        "service_times": [1, 1, 1, 1],
        "prizes": [10.0, 10.0, 20.0, 20.0],
        "order_vehicle_match": [{"order_id": 0, "vehicle_ids": [1, 2]}],
    },
    "solver_config": {"time_limit": 1},
}


# TASK DATA TESTING


def test_valid_full_set_task_data(cuoptproc):  # noqa
    response_set = client.post("/cuopt/request", json=valid_data)
    assert response_set.status_code == 200


def test_valid_minimal_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)

    test_data["fleet_data"] = {
        "vehicle_locations": [[1, 1], [2, 2], [3, 3], [4, 4]]
    }
    test_data["task_data"] = {"task_locations": [1, 2, 3, 4]}

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 200


# Testing invalid, all task parameters need to in range
def test_invalid_values_set_task_data(cuoptproc):  # noqa
    invalid_task_data = {
        "task_locations": [1, -2, 3, 4],
        "demand": [[1, 2, 3, 4], [2, 3, 4, -5]],
        "pickup_and_delivery_pairs": [[1, -2], [3, 4]],
        "task_time_windows": [[1, 2], [-1, 2], [1, 2], [1, 2]],
        "service_times": [1, 2, 1, -2],
        "prizes": [1.0, 3.0, 10.0, 10.0, 5.0],
        "order_vehicle_match": [{"order_id": 6, "vehicle_ids": [1, 2]}],
    }

    # all task locations must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["task_locations"] = invalid_task_data[
        "task_locations"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "task_locations represent index locations and must be greater than or equal to 0",  # noqa
        "error_result": True,
    }

    # demand can be negative in the pickup and delivery case as of 2022-04-16.
    # Not tested for range.

    # all pickup and delivery locations must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["pickup_and_delivery_pairs"] = invalid_task_data[
        "pickup_and_delivery_pairs"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "pickup_and_delivery_pairs represent order index and must be greater than or equal to 0",  # noqa
        "error_result": True,
    }

    # all task time windows must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["task_time_windows"] = invalid_task_data[
        "task_time_windows"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "task_time_windows must be greater than or equal to 0",
        "error_result": True,
    }

    # all task service_times must be greater than or equal to 0
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["service_times"] = invalid_task_data[
        "service_times"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "service_times must be greater than or equal to 0",
        "error_result": True,
    }

    # Length of prizes should be equal to number of task locations
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["prizes"] = invalid_task_data["prizes"]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "Size of the task prizes should be equal to number of tasks",  # noqa
        "error_result": True,
    }

    # Order Id should be non negative and should be less than number of
    # orders/task locations
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["order_vehicle_match"] = invalid_task_data[
        "order_vehicle_match"
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400

    print("response = ", response_set.json())
    assert response_set.json() == {
        "error": "One or more Order IDs provided are not in the expected range in task vehicle match, should be within [0,  len(Task Locations) )",  # noqa
        "error_result": True,
    }


# Test invalid tasks arrays of consistent length
def test_invalid_length_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["task_time_windows"] = [[1, 2], [1, 2], [1, 2]]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining task properties must be of consistent length",  # noqa
        "error_result": True,
    }


# Test invalid demand dimension
def test_invalid_demand_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["demand"] = [[1, 2, 3, 4], [2, 3, 4]]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All demand dimensions must have length equal to the number of tasks",  # noqa
        "error_result": True,
    }


# Test invlaid pickup and delivery, all task locations
# used & all task locations from task locations array
def test_invalid_pickup_and_delivery_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["pickup_and_delivery_pairs"] = [[0, 2], [2, 3]]

    response_set_1 = client.post("/cuopt/request", json=test_data)
    assert response_set_1.status_code == 400
    assert response_set_1.json() == {
        "error": "pickup_and_delivery_pairs assignments must be in the set of task/order indices and all task location indices must be used",  # noqa
        "error_result": True,
    }

    test_data["task_data"]["pickup_and_delivery_pairs"] = [[1, 1], [2, 3]]
    response_set_2 = client.post("/cuopt/request", json=test_data)
    assert response_set_2.status_code == 400
    assert response_set_2.json() == {
        "error": "pickup_and_delivery_pairs assignments must be in the set of task/order indices and all task location indices must be used",  # noqa
        "error_result": True,
    }


# Test invlaid time windows
def test_invalid_time_windows_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["task_time_windows"] = [
        [1, 2],
        [1, 2, 2],
        [1, 2],
        [1, 2],
    ]

    response_set_1 = client.post("/cuopt/request", json=test_data)
    assert response_set_1.status_code == 400
    assert response_set_1.json() == {
        "error": "All task_time_windows must be of length 2. 0:earliest, 1:latest",  # noqa
        "error_result": True,
    }

    test_data["task_data"]["task_time_windows"] = [
        [1, 2],
        [3, 2],
        [1, 2],
        [1, 2],
    ]
    response_set_2 = client.post("/cuopt/request", json=test_data)
    assert response_set_2.status_code == 400
    assert response_set_2.json() == {
        "error": "All task time windows must have task_x_time_window[0] < task_x_time_window[1]",  # noqa
        "error_result": True,
    }


# Test invalid Service Time
def test_invalid_service_time_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["service_times"] = [1, 2, 1, 2, 5]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "All arrays defining task properties must be of consistent length",  # noqa
        "error_result": True,
    }


# Test invalid vehicle ids in order vehicle match
def test_invalid_order_vehicle_match(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["order_vehicle_match"] = [
        {"order_id": 0, "vehicle_ids": [1, -2]}
    ]

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 400
    assert response_set.json() == {
        "error": "vehicle Id should be greater than or equal to zero",  # noqa
        "error_result": True,
    }


# Test set/get vehicle specific service times
def test_vehicle_specific_service_times(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"]["service_times"] = {
        0: [1.0, 1.0, 1.0, 1.0],
        1: [1.0, 1.0, 1.0, 1.0],
        2: [1.0, 1.0, 1.0, 1.0],
        3: [1.0, 1.0, 1.0, 1.0],
    }

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 200


def test_invalid_extra_arg_set_task_data(cuoptproc):  # noqa
    test_data = copy.deepcopy(valid_data)
    test_data["task_data"] = {"task_locations": [1, 2, 3, 4], "extra_arg": 1}

    response_set = client.post("/cuopt/request", json=test_data)
    assert response_set.status_code == 422
