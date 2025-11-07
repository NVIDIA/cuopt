# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd

import cudf

from cuopt import routing


# Function to extract order constraints, distance matrix
# and pickup/delivery indices data
def data_prep(order_pdf, matrix_pdf, depot):
    # Prepare order time window and demand constraints
    order_df = cudf.DataFrame(order_pdf[["earliest", "latest"]]).reset_index(
        drop=True
    )
    temp_df = cudf.DataFrame()
    temp_df["earliest"] = order_df["earliest"]
    temp_df["latest"] = order_df["latest"]
    order_df = cudf.concat([order_df, temp_df])
    order_df["service"] = [10] * len(order_df)
    order_df["demand"] = [-1] * len(temp_df) + [1] * len(temp_df)
    constraints = order_df.reset_index(drop=True)

    sources = order_pdf["From"].tolist()
    sinks = order_pdf["To"].tolist()

    orders = sources + sinks

    unique_locations = {}
    order_locations = []
    locations = []

    locations.append(depot)
    unique_locations[depot] = 0
    cnt = len(unique_locations)
    for order in orders:
        if unique_locations.get(order, -1) == -1:
            unique_locations[order] = cnt
            order_locations.append(cnt)
            locations.append(order)
            cnt = cnt + 1
        else:
            order_locations.append(unique_locations.get(order, -1))

    order_locations = cudf.Series(order_locations)

    num_locations = cnt

    # Prepare the distance matrix
    matrix = np.empty(shape=(num_locations, num_locations))
    matrix.fill(0.0)
    matrix_df_cols = matrix_pdf.columns.tolist()
    matrix_pdf = matrix_pdf.values.tolist()
    for i in range(num_locations):
        for j in range(num_locations):
            my_x = locations[i]
            my_y = locations[j]
            my_ix = matrix_df_cols.index(my_x)
            my_iy = matrix_df_cols.index(my_y)
            matrix[i][j] = matrix_pdf[my_ix][my_iy]

    pdf = pd.DataFrame(matrix)
    matrix_df = cudf.DataFrame.from_pandas(pdf).astype("float32")

    # Prepare pickup and delivery indices
    delivery_indices = cudf.Series(i for i in range(0, int(len(order_df) / 2)))
    pickup_indices = cudf.Series(
        i for i in range(int(len(order_df) / 2), len(order_df))
    )

    return (
        matrix_df,
        constraints,
        order_locations,
        pickup_indices,
        delivery_indices,
    )


# Get time in seconds
def get_time(t):
    hh, mm, ss = t.split(":")
    mytime = int(hh) * 60 * 60 + int(mm) * 60 + int(ss)
    return mytime


def run_cuopt(
    matrix_df,
    constraints,
    order_locations,
    num_vehicles,
    pickup_indices,
    delivery_indices,
):
    num_orders = len(constraints)
    num_locations = len(matrix_df)
    # Pass the distance matrix
    data_model = routing.DataModel(num_locations, num_vehicles, num_orders)
    data_model.add_cost_matrix(matrix_df)
    data_model.set_order_locations(order_locations)

    data_model.set_pickup_delivery_pairs(pickup_indices, delivery_indices)
    capacity_series = cudf.Series([1] * num_vehicles)
    data_model.add_capacity_dimension(
        "demand", constraints["demand"], capacity_series
    )
    data_model.set_order_time_windows(
        constraints["earliest"], constraints["latest"]
    )
    data_model.set_order_service_times(constraints["service"])

    # Composable solver settings for a simple CVRPTW
    solver_settings = routing.SolverSettings()

    routing_solution = routing.Solve(data_model, solver_settings)

    (
        ret_pickup_indices,
        ret_delivery_indices,
    ) = data_model.get_pickup_delivery_pairs()
    ret_order_locations = data_model.get_order_locations()
    ret_num_locations = data_model.get_num_locations()

    assert ret_num_locations == num_locations
    assert (ret_order_locations == order_locations).all()
    assert (ret_pickup_indices == pickup_indices).all()
    assert (ret_delivery_indices == delivery_indices).all()
    assert routing_solution.get_status() == 0
