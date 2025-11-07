# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pandas as pd

import cudf

from cuopt import routing
from cuopt.routing import utils


def create_from_file(file_path):
    node_list = {
        "vertex": [],
        "xcoord": [],
        "ycoord": [],
        "demand": [],
        "earliest_time": [],
        "latest_time": [],
        "service_time": [],
        "pickup_index": [],
        "drop_index": [],
        "total_demand": 0,
        "vehicle_capacity": 0,
        "fleet_size": 0,
    }
    with open(file_path, "rt") as f:
        count = 1
        current_index = 0
        for line in f:
            if count == 1:
                vehicle_num, vehicle_capacity, _ = line.split()
                node_list["fleet_size"] = int(vehicle_num)
                node_list["vehicle_capacity"] = int(vehicle_capacity)
            elif count > 1:
                vals = line.split()
                node_list["vertex"].append(int(vals[0]))
                node_list["xcoord"].append(int(vals[1]))
                node_list["ycoord"].append(int(vals[2]))
                demand = int(vals[3])
                node_list["demand"].append(demand)
                node_list["total_demand"] += int(demand)
                node_list["earliest_time"].append(int(vals[4]))
                node_list["latest_time"].append(int(vals[5]))
                node_list["service_time"].append(int(vals[6]))
                if count > 2:
                    pickup_index = int(vals[7])
                    drop_index = int(vals[8])
                    if pickup_index != 0:
                        drop_index = current_index
                    elif drop_index != 0:
                        pickup_index = current_index
                    if (
                        pickup_index not in node_list["pickup_index"]
                        and pickup_index not in node_list["drop_index"]
                    ):
                        node_list["pickup_index"].append(pickup_index)
                        node_list["drop_index"].append(drop_index)
                current_index += 1
            count += 1

    return node_list, int(vehicle_capacity), int(vehicle_num)


def run_cuopt(fname, node_list, vehicle_capacity, vehicle_num, time_limit):
    # Create Datamodel
    etl_start_time = time.time()
    n_locations = len(node_list["xcoord"])
    data_model = routing.DataModel(n_locations, node_list["fleet_size"])

    coords = cudf.DataFrame()
    coords["x"], coords["y"] = node_list["xcoord"], node_list["ycoord"]
    matrix = utils.build_matrix(coords)

    data_model.add_cost_matrix(matrix)

    data_model.set_pickup_delivery_pairs(
        cudf.Series(node_list["pickup_index"]),
        cudf.Series(node_list["drop_index"]),
    )
    data_model.set_order_time_windows(
        cudf.Series(node_list["earliest_time"]),
        cudf.Series(node_list["latest_time"]),
        cudf.Series(node_list["service_time"]),
    )
    data_model.add_capacity_dimension(
        "distance",
        cudf.Series(node_list["demand"]),
        cudf.Series([vehicle_capacity] * vehicle_num),
    )
    solver_settings = routing.SolverSettings()

    solver_settings.set_time_limit(time_limit)
    solve_start_time = time.time()
    sol = routing.Solve(data_model, solver_settings)
    solve_end_time = time.time()

    solver_settings_time = solve_end_time - solve_start_time
    total_time = solve_end_time - etl_start_time

    return sol, solver_settings_time, total_time


def run_cuopt_pdp(dir, time_limit, inter_folder):
    files = os.listdir(dir)

    for f in files:
        node_list, vehicle_capacity, vehicle_num = create_from_file(dir + f)
        time_limit = time_limit
        data = {
            "File Name": [],
            "Vehicles": [],
            "Total Distance": [],
            "SolverSettings Run Time": [],
            "Total Run Time": [],
            "SolverSettings Time Limit": [],
        }
        for t in range(2, time_limit, 2):
            sol, solver_settings_time, total_time = run_cuopt(
                f, node_list, vehicle_capacity, vehicle_num, t
            )
            data["File Name"].append(f)
            data["Vehicles"].append(sol.get_vehicle_count())
            data["Total Distance"].append(sol.get_total_objective())
            data["SolverSettings Run Time"].append(solver_settings_time)
            data["Total Run Time"].append(total_time)
            data["SolverSettings Time Limit"].append(t)
            pdf = pd.DataFrame(data)
            pdf.to_csv(inter_folder + f + ".csv")

            if len(data["File Name"]) >= 1:
                if (len(set(data["Vehicles"])) == 1) and (
                    len(set(data["Total Distance"])) == 1
                ):
                    break


# Add benchmark dir to this list to run cuopt on them
dirs = ["pdp_100/"]
inter_folder = "./inter/"
time_limit = 60

isExist = os.path.exists(inter_folder)

if not isExist:
    os.makedirs(inter_folder)

for dir in dirs:
    run_cuopt_pdp(dir, time_limit, inter_folder)
