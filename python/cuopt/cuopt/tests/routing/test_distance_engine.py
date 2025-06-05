# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.  # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import pytest

import cudf

from cuopt import distance_engine, routing
from cuopt.routing import utils
from cuopt.utilities import InputValidationError

depot = "HP 127"

# Helper functions


def get_graph_data(file, type_t):
    graph_info = []
    graph_info_file = open(file, "r")
    for i, line in enumerate(graph_info_file):
        graph_info.append(np.array(line.split(",")[:-1]).astype(type_t[i]))

    return graph_info[0], graph_info[1], graph_info[2]


def get_target_locations(matrix_df_all, raw_order_pdf):
    target_locations = []

    matrix_df_all_cols = matrix_df_all.index.tolist()
    pickup = raw_order_pdf["source"].tolist()
    delivery = raw_order_pdf["sink"].tolist()
    orders = [depot] + pickup + delivery

    unique_locations = {}
    order_locations = []
    locations = []
    cnt = 0
    for order in orders:
        if unique_locations.get(order, -1) == -1:
            unique_locations[order] = cnt
            order_locations.append(cnt)
            locations.append(order)
            cnt = cnt + 1
        else:
            order_locations.append(unique_locations.get(order, -1))

    for i in range(cnt):
        target_locations.append(matrix_df_all_cols.index(locations[i]))

    return np.array(target_locations)


def get_date(date_time):
    d, t = date_time.split(" ")
    return d


def get_time(date_time, date, rt, t_type):
    d, t = date_time.split(" ")
    hh, mm, ss = t.split(":")
    add_day = 0
    if d != date:
        add_day = 24 * 60 * 60
    if ss == "00":
        mytime = int(hh) * 60 * 60 + int(mm) * 60 + add_day
    else:
        mytime = int(hh) * 60 * 60 + int(mm) * 60 + int(ss) + add_day
    if t_type == "earliest":
        mytime = mytime - rt
    else:
        mytime = mytime
    return mytime


def get_service_time(t):
    hh, mm, ss = t.split(":")
    return int(hh) * 60 * 60 + int(mm) * 60 + int(ss)


def check_matrix(
    waypoint_matrix,
    matrix_df,
    matrix_df_all,
    single_day_order_pdf,
    speed,
    weights,
):
    # Set target locations
    target_locations = get_target_locations(
        matrix_df_all, single_day_order_pdf
    )

    # Compute cost matrix
    computed_cost_matrix = waypoint_matrix.compute_cost_matrix(
        target_locations
    )
    computed_cost_matrix = computed_cost_matrix / speed

    # Compute time matrix with same weight
    computed_time_matrix = waypoint_matrix.compute_shortest_path_costs(
        target_locations, weights
    )
    computed_time_matrix = computed_time_matrix / speed

    # Check matrix
    assert np.isclose(
        computed_cost_matrix.to_pandas(), matrix_df.to_pandas()
    ).all()

    assert np.isclose(
        computed_cost_matrix.to_pandas(), computed_time_matrix.to_pandas()
    ).all()


def test_compute_cost_matrix():

    # Data loading
    matrix_df_all = pd.read_csv(
        utils.RAPIDS_DATASET_ROOT_DIR + "/distance_engine/traveltimes.csv",
        index_col=0,
    )
    order_pdf = pd.read_csv(
        utils.RAPIDS_DATASET_ROOT_DIR + "/distance_engine/order_sample.csv",
        encoding="ISO-8859-1",
    )
    matrix_pdf = pd.read_csv(
        utils.RAPIDS_DATASET_ROOT_DIR + "/distance_engine/ref_cost_matrix.csv",
        sep=";",
        header=0,
    )

    # Get orders delivered using direct transport
    output_pdf = pd.read_csv(
        utils.RAPIDS_DATASET_ROOT_DIR
        + "/distance_engine/direct_transport_sample.csv",
        index_col=0,
    )

    my_orders = output_pdf.Assignments.unique()
    order_pdf = order_pdf[order_pdf["#"].isin(my_orders)]

    # Rename demand column
    demand_column_name = "Resupply Quantity"
    order_pdf = order_pdf.rename(columns={demand_column_name: "demand"})
    order_pdf = order_pdf.rename(
        columns={
            "Stopping point at source stage": "source",
            "Stopping point at sink stage": "sink",
        }
    )

    # Set date
    date = "18.02.2021"
    order_pdf["earliest_date"] = order_pdf["Earliest delivery time"].apply(
        lambda x: get_date(x)
    )
    order_pdf["latest_date"] = order_pdf["Latest Provision Time"].apply(
        lambda x: get_date(x)
    )

    # Add service time from output file
    pickups = output_pdf[
        output_pdf["Activity"].isin(["Loading", "LoadingEmptyBoxes"])
    ][["Assignments", "Duration"]]
    pickups = pickups.rename(columns={"Duration": "pickup_service_time"})
    deliveries = output_pdf[
        output_pdf["Activity"].isin(["Unloading", "UnloadingEmptyBoxes"])
    ][["Assignments", "Duration"]]
    deliveries = deliveries.rename(
        columns={"Duration": "delivery_service_time"}
    )

    order_pdf = order_pdf.merge(
        pickups, how="left", left_on="#", right_on="Assignments"
    )
    order_pdf = order_pdf.merge(
        deliveries, how="left", left_on="#", right_on="Assignments"
    )
    order_pdf["pickup_service_time"] = order_pdf["pickup_service_time"].apply(
        lambda x: get_service_time(x)
    )
    order_pdf["delivery_service_time"] = order_pdf[
        "delivery_service_time"
    ].apply(lambda x: get_service_time(x))

    # Compute earliest and latest time
    # with earliest time relaxation of 60 minutes
    relaxation_time = 3600
    order_pdf["earliest_time"] = order_pdf["Earliest delivery time"].apply(
        lambda x: get_time(x, date, relaxation_time, "earliest")
    )
    order_pdf["latest_time"] = order_pdf["Latest Provision Time"].apply(
        lambda x: get_time(x, date, relaxation_time, "latest")
    )

    # Run in batches of 10mins
    my_batch = [600 * i for i in range(0, 289)]

    num_vehicles = 600
    vehicle_speed = 2.2
    vehicle_capacity = 1
    break_earliest = [
        21600,
        34200,
        47700,
        61200,
        72000,
        87300,
        108000,
        120600,
        134100,
        147600,
        158400,
    ]
    break_latest = [
        21900,
        35100,
        50400,
        61500,
        74700,
        88200,
        108300,
        121500,
        136800,
        147900,
        161100,
    ]

    # Set vehicle constraints
    vehicle_constraints = routing.add_vehicle_constraints(
        num_vehicles,
        vehicle_capacity,
        break_earliest,
        break_latest,
        vehicle_speed,
    )

    # Build waypoint matrix
    offsets, indices, weights = get_graph_data(
        utils.RAPIDS_DATASET_ROOT_DIR + "/distance_engine/waypoint_matrix.txt",
        ["int", "int", "float"],
    )
    waypoint_matrix = distance_engine.WaypointMatrix(offsets, indices, weights)

    # Run loop and check matrices are equal every time
    for i in range(0, 288):
        single_day_order_pdf = order_pdf[
            (order_pdf["latest_time"] > my_batch[i])
            & (order_pdf["latest_time"] <= my_batch[i + 1])
        ]

        if len(single_day_order_pdf) > 0:
            (
                matrix_df,
                order_data,
                vehicle_data,
            ) = routing.create_pickup_delivery_data(
                matrix_pdf, single_day_order_pdf, depot, vehicle_constraints
            )
            check_matrix(
                waypoint_matrix,
                matrix_df,
                matrix_df_all,
                single_day_order_pdf,
                vehicle_constraints.speed,
                weights,
            )


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


def start_compute_shortest_path_costs():
    offsets = np.array([0, 2, 3, 4, 6, 8, 9, 10])
    edges = np.array([1, 6, 4, 3, 2, 4, 2, 6, 4, 0])
    weights = np.array([2, 10, 3, 2, 2, 5, 1, 1, 2, 10])
    target_locations = np.array([0, 3, 6])
    custom_weights = np.array(
        [1, 10000000, 10, 1000, 1000, 10000, 100, 100000, 1000000, 10000000],
        dtype="float",
    )
    expected_custom_matrix = np.array(
        [[0, 1111, 100011], [10110000, 0, 110000], [10000000, 10001111, 0]],
        dtype="float",
    )

    w_matrix = distance_engine.WaypointMatrix(offsets, edges, weights)

    w_matrix.compute_cost_matrix(target_locations)

    custom_cost_matrix = w_matrix.compute_shortest_path_costs(
        target_locations, custom_weights
    )

    assert np.array_equal(
        custom_cost_matrix.to_numpy(), expected_custom_matrix
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


def test_compute_shortest_path_costs():
    start_compute_shortest_path_costs()


def test_waypoint_matrix_validity():
    start_waypoint_matrix_validity()


def test_target_locations_validity():
    start_target_locations_validity()


def test_locations_validity():
    start_locations_validity()
