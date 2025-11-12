# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


def test_uniform_breaks():
    vehicle_num = 25
    run_nodes = 100
    nodes = run_nodes + 1
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    break_times = [[40, 50], [170, 180]]

    num_breaks = len(break_times)
    vehicle_breaks_earliest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_latest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_duration = np.zeros([vehicle_num, num_breaks])
    for b in range(num_breaks):
        break_begin = break_times[b][0]
        break_end = break_times[b][1]
        break_duration = break_end - break_begin
        vehicle_breaks_earliest[:, b] = [break_begin] * vehicle_num
        vehicle_breaks_latest[:, b] = [break_begin] * vehicle_num
        vehicle_breaks_duration[:, b] = [break_duration] * vehicle_num

    # Add all nodes as the vehicle break location
    break_locations = cudf.Series([i for i in range(nodes)])

    d.set_break_locations(break_locations)
    for b in range(num_breaks):
        d.add_break_dimension(
            cudf.Series(vehicle_breaks_earliest[:, b]).astype(np.int32),
            cudf.Series(vehicle_breaks_latest[:, b]).astype(np.int32),
            cudf.Series(vehicle_breaks_duration[:, b]).astype(np.int32),
        )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    routing_solution = routing.Solve(d, s)
    ret_break_locations = d.get_break_locations()
    ret_break_dimensions = d.get_break_dimensions()

    assert (ret_break_locations == break_locations).all()

    for b, break_dimension in enumerate(ret_break_dimensions.keys()):
        vehicle_break = ret_break_dimensions[break_dimension]
        assert (
            vehicle_break["earliest"]
            == cudf.Series(vehicle_breaks_earliest[:, b])
        ).all()
        assert (
            vehicle_break["latest"] == cudf.Series(vehicle_breaks_latest[:, b])
        ).all()
        assert (
            vehicle_break["duration"]
            == cudf.Series(vehicle_breaks_duration[:, b])
        ).all()

    # TO DO: Check if breaks are adhered to
    assert routing_solution.get_status() == 0


def test_non_uniform_breaks():
    vehicle_num = 30
    run_nodes = 100
    nodes = run_nodes + 1
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    num_v_type_1 = int(vehicle_num / 2)
    break_times_1 = [[40, 50], [100, 120], [170, 180]]
    break_durations_1 = [5, 20, 10]

    num_v_type_2 = vehicle_num - num_v_type_1
    break_times_2 = [[60, 90], [110, 120], [200, 210]]
    break_durations_2 = [20, 10, 5]

    num_breaks = 3
    vehicle_breaks_earliest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_latest = np.zeros([vehicle_num, num_breaks])
    vehicle_breaks_duration = np.zeros([vehicle_num, num_breaks])
    for b in range(num_breaks):
        vehicle_breaks_earliest[:, b] = [
            break_times_1[b][0]
        ] * num_v_type_1 + [break_times_2[b][0]] * num_v_type_2
        vehicle_breaks_latest[:, b] = [break_times_1[b][1]] * num_v_type_1 + [
            break_times_2[b][1]
        ] * num_v_type_2
        vehicle_breaks_duration[:, b] = [
            break_durations_1[b]
        ] * num_v_type_1 + [break_durations_2[b]] * num_v_type_2

    # Depot should not be a break node
    break_locations = cudf.Series([i + 1 for i in range(nodes - 1)])

    d.set_break_locations(break_locations)
    for b in range(num_breaks):
        d.add_break_dimension(
            cudf.Series(vehicle_breaks_earliest[:, b]).astype(np.int32),
            cudf.Series(vehicle_breaks_latest[:, b]).astype(np.int32),
            cudf.Series(vehicle_breaks_duration[:, b]).astype(np.int32),
        )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    routing_solution = routing.Solve(d, s)
    ret_break_locations = d.get_break_locations()
    ret_break_dimensions = d.get_break_dimensions()

    assert (ret_break_locations == break_locations).all()

    for b, break_dimension in enumerate(ret_break_dimensions.keys()):
        vehicle_break = ret_break_dimensions[break_dimension]
        assert (
            vehicle_break["earliest"]
            == cudf.Series(vehicle_breaks_earliest[:, b])
        ).all()
        assert (
            vehicle_break["latest"] == cudf.Series(vehicle_breaks_latest[:, b])
        ).all()
        assert (
            vehicle_break["duration"]
            == cudf.Series(vehicle_breaks_duration[:, b])
        ).all()

    # TO DO: Check if breaks are adhered to
    assert routing_solution.get_status() == 0


def test_heterogenous_breaks():
    vehicle_num = 20
    run_nodes = 100
    d = utils.create_data_model(
        filename, run_nodes=run_nodes, num_vehicles=vehicle_num
    )

    """
    Half of vehicles have three breaks and the remaining half have two breaks.
    Break locations are also different. First set of vehicles have specified
    subset of locations while the second set of vehicles have default, i.e. any
    location can be a break
    """
    num_breaks_1 = 2
    num_v_type_1 = int(vehicle_num / 2)
    break_times_1 = [[90, 100], [150, 170]]
    break_durations_1 = [15, 15]
    break_locations_1 = cudf.Series([5 * i for i in range(1, 18)])

    num_breaks_2 = 3
    num_v_type_2 = vehicle_num - num_v_type_1
    break_times_2 = [[40, 50], [110, 120], [160, 170]]
    break_durations_2 = [10, 10, 10]

    for i in range(num_v_type_1):
        for b in range(num_breaks_1):
            d.add_vehicle_break(
                i,
                break_times_1[b][0],
                break_times_1[b][1],
                break_durations_1[b],
                break_locations_1,
            )

    for i in range(num_v_type_2):
        for b in range(num_breaks_2):
            d.add_vehicle_break(
                i + num_v_type_1,
                break_times_2[b][0],
                break_times_2[b][1],
                break_durations_2[b],
            )

    s = routing.SolverSettings()
    s.set_time_limit(30)
    routing_solution = routing.Solve(d, s)

    # TO DO: Check if breaks are adhered to
    assert routing_solution.get_status() == 0
    print("num vehicles = ", routing_solution.get_vehicle_count())
    counters = {}
    routes = routing_solution.get_route().to_pandas()
    break_locations_1 = break_locations_1.to_arrow().to_pylist()
    # make sure the break locations are the right ones and
    # the arrival stamps satisfy the break time constraints
    for i in range(routes.shape[0]):
        truck_id = routes["truck_id"][i]
        if truck_id not in counters:
            counters[truck_id] = 0
        if routes["type"][i] == "Break":
            break_dim = routes["route"][i]
            location = routes["location"][i]
            arrival_time = routes["arrival_stamp"][i]
            if truck_id < num_v_type_1:
                assert location in break_locations_1
                assert arrival_time >= break_times_1[break_dim][0]
                assert arrival_time <= break_times_1[break_dim][1]
            else:
                assert arrival_time >= break_times_2[break_dim][0]
                assert arrival_time <= break_times_2[break_dim][1]
            counters[truck_id] = counters[truck_id] + 1

    # Make sure the achieved number of breaks is same as the specified
    for truck_id, num_breaks in counters.items():
        if truck_id < num_v_type_1:
            assert num_breaks == num_breaks_1
        else:
            assert num_breaks == num_breaks_2
