# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

import cudf

from cuopt import distance_engine, routing
from cuopt.utilities import InputValidationError


def start_compute_waypoint_sequence(
    locations, n_vehicles, min_vehicles, set_order_locations
):
    data = {
        "start": [0, 1, 2, 3, 4, 4],
        "offsets": [0, 3, 5, 7, 8, 9],
        "edges": [1, 2, 3, 0, 2, 0, 3, 4, 0],
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    # expected_cost_matrix = [
    #     0.0,
    #     1.0,
    #     2.0,
    #     11.0,
    #     4.0,
    #     0.0,
    #     5.0,
    #     15.0,
    #     6.0,
    #     7.0,
    #     0.0,
    #     15.0,
    #     9.0,
    #     10.0,
    #     11.0,
    #     0.0,
    # ]
    # expected_full_path = [0, 2, 2, 3, 4, 4, 0, 0, 0, 1, 1, 0]
    # expected_sequence_offsets = [0, 2, 5, 7, 8, 10, 12]

    offsets = np.array(data["offsets"])
    edges = np.array(data["edges"])
    weights = np.array(data["weights"])

    w_matrix = distance_engine.WaypointMatrix(offsets, edges, weights)

    locations = np.array(locations)

    cost_matrix = w_matrix.compute_cost_matrix(np.array(locations))

    n_locations = len(locations)
    n_orders = n_locations

    if set_order_locations:
        n_orders = n_locations - 1

    dm = routing.DataModel(n_locations, n_vehicles, n_orders)
    dm.add_cost_matrix(cost_matrix)
    dm.set_min_vehicles(min_vehicles)

    # If order locations are being used, the depot has to be dropped
    # because it is handled inside the solver
    if set_order_locations:
        dm.set_order_locations(cudf.Series([1, 2, 3]))

    solver_settings = routing.SolverSettings()
    solver_settings.set_time_limit(5)

    sol = routing.Solve(dm, solver_settings)

    assert sol.get_status() == 0

    sol_df = sol.get_route()

    val = w_matrix.compute_waypoint_sequence(locations, sol_df)
    val = val["waypoint_sequence"]

    # FIXME: Determinism PR
    # assert np.array_equal(
    #     cost_matrix.to_pandas().to_numpy(),
    #     np.array(expected_cost_matrix).reshape(n_locations, n_locations),
    # )
    # assert np.array_equal(
    #     sol_df["sequence_offset"].to_numpy(),
    #     np.array(expected_sequence_offsets),
    # )
    # assert np.array_equal(val.to_numpy(), np.array(expected_full_path))


def start_compute_waypoint_sequence_no_matrix_call(locations):
    data = {
        "start": [0, 1, 2, 3, 4, 4],
        "offsets": [0, 3, 5, 7, 8, 9],
        "edges": [1, 2, 3, 0, 2, 0, 3, 4, 0],
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    offsets = np.array(data["offsets"])
    edges = np.array(data["edges"])
    weights = np.array(data["weights"])

    w_matrix = distance_engine.WaypointMatrix(offsets, edges, weights)

    locations = np.array(locations)

    with pytest.raises(InputValidationError):
        w_matrix.compute_waypoint_sequence(
            locations, cudf.DataFrame({"location": [0]})
        )


def start_waypoint_matrix_validity():
    data = {
        "offsets": [0, 3, 5, 7, 8, 9],
        "edges": [1, 2, 3, 0, 2, 0, 3, 4, 0],
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    }

    offsets = np.array(data["offsets"])
    edges = np.array(data["edges"])
    weights = np.array(data["weights"])

    # -- Offsets checks --

    # Negative val
    offsets[3] = -1

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)

    # Greater or equal to number of edges
    offsets[3] = 9

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)

    # Not sorted increasingly
    offsets[3] = 3

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)

    # Set back to previous
    offsets[3] = data["offsets"][3]

    # -- Indices checks --

    # Negative val
    edges[3] = -1

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)

    # Greater or equal to number of vertices
    edges[3] = 5

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)

    # Set back to previous
    edges[3] = data["edges"][3]

    # -- Weights checks --

    # Negative val
    weights[3] = -1

    with pytest.raises(InputValidationError):
        distance_engine.WaypointMatrix(offsets, edges, weights)


def start_target_locations_validity():
    data = {
        "offsets": [0, 3, 5, 7, 8, 9],
        "edges": [1, 2, 3, 0, 2, 0, 3, 4, 0],
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_locations": [0, 1, 2, 4],
    }

    offsets = np.array(data["offsets"])
    edges = np.array(data["edges"])
    weights = np.array(data["weights"])
    target_locations = np.array(data["target_locations"])

    w_matrix = distance_engine.WaypointMatrix(offsets, edges, weights)

    # Working call for next compute waypoint sequence call
    w_matrix.compute_cost_matrix(target_locations)

    # Negative value
    target_locations[3] = -1

    with pytest.raises(InputValidationError):
        w_matrix.compute_cost_matrix(target_locations)

    with pytest.raises(InputValidationError):
        w_matrix.compute_waypoint_sequence(
            target_locations, cudf.DataFrame({"location": [0]})
        )

    with pytest.raises(InputValidationError):
        w_matrix.compute_shortest_path_costs(target_locations, weights)

    # Greater or equal to number of vertices
    target_locations[3] = 5

    with pytest.raises(InputValidationError):
        w_matrix.compute_cost_matrix(target_locations)

    with pytest.raises(InputValidationError):
        w_matrix.compute_waypoint_sequence(
            target_locations, cudf.DataFrame({"location": [0]})
        )

    with pytest.raises(InputValidationError):
        w_matrix.compute_shortest_path_costs(target_locations, weights)


def start_locations_validity():
    data = {
        "offsets": [0, 3, 5, 7, 8, 9],
        "edges": [1, 2, 3, 0, 2, 0, 3, 4, 0],
        "weights": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "target_locations": [0, 2, 4],
        # location value higher than number of target locations
        "locations": [0, 1, 3, 2],
    }

    offsets = np.array(data["offsets"])
    edges = np.array(data["edges"])
    weights = np.array(data["weights"])
    target_locations = np.array(data["target_locations"])
    locations = np.array(data["locations"])

    w_matrix = distance_engine.WaypointMatrix(offsets, edges, weights)

    w_matrix.compute_cost_matrix(target_locations)

    with pytest.raises(InputValidationError):
        w_matrix.compute_waypoint_sequence(
            target_locations,
            cudf.DataFrame(
                {"location": cudf.Series(locations, dtype=np.int32)}
            ),
        )


def test_compute_waypoint_sequence_set_order_locations():
    start_compute_waypoint_sequence([0, 1, 2, 4], 3, 2, True)


def test_compute_waypoint_sequence_no_set_order_locations():
    start_compute_waypoint_sequence([0, 1, 2, 4], 3, 2, False)


def test_compute_waypoint_sequence_no_matrix_call():
    start_compute_waypoint_sequence_no_matrix_call([0, 1, 2, 4])


def test_waypoint_matrix_validity():
    start_waypoint_matrix_validity()


def test_target_locations_validity():
    start_target_locations_validity()


def test_locations_validity():
    start_locations_validity()
