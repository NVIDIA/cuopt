# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import benchmark_utils

import cudf

from cuopt import routing
from cuopt.routing import utils


def cuopt_cvrptw_data_preprocess(raw_instance_data, use_matrix):
    """
    Function to prepocess raw instance data for use with cuOpt CVRPTW.

    """

    time_window_cudf = cudf.DataFrame(
        list(
            zip(
                raw_instance_data["earliest_time"],
                raw_instance_data["latest_time"],
                raw_instance_data["service_time"],
            )
        ),
        columns=["earliest_time", "latest_time", "service_time"],
    )

    demand_cudf = cudf.DataFrame(
        raw_instance_data["demand"], columns=["demand"]
    )

    fleet_size = raw_instance_data["fleet_size"]
    capacity_cudf = cudf.DataFrame(
        [raw_instance_data["vehicle_capacity"]] * fleet_size,
        columns=["capacity"],
    )

    if use_matrix:
        cost_matrix_cudf = cudf.DataFrame(
            benchmark_utils.distance_matrix_from_point_list(
                list(
                    zip(
                        raw_instance_data["xcoord"],
                        raw_instance_data["ycoord"],
                    )
                ),
                1,
            )
        )

        return (cost_matrix_cudf, demand_cudf, capacity_cudf, time_window_cudf)

    else:
        point_list_cudf = cudf.DataFrame(
            list(
                zip(raw_instance_data["xcoord"], raw_instance_data["ycoord"])
            ),
            columns=["x_pos", "y_pos"],
        )

        return (point_list_cudf, demand_cudf, capacity_cudf, time_window_cudf)


def cuopt_solve(
    point_data,
    use_matrix=True,
    demand=None,
    capacity=None,
    time_windows=None,
    time_limit=None,
):
    """
    Function to setup and solve a given instance with cuOpt.

    """
    total_cuopt_run_time_start = time.time()
    fleet_size = capacity["capacity"].size
    data_model = routing.DataModel(len(point_data), fleet_size)

    if use_matrix:
        data_model.add_cost_matrix(point_data)
    else:
        df = cudf.DataFrame()
        df["xcord"] = point_data["x_pos"]
        df["ycord"] = point_data["y_pos"]
        matrix = utils.build_matrix(df)
        data_model.add_cost_matrix(matrix)

    if demand is not None and capacity is not None:
        data_model.add_capacity_dimension(
            "demand", demand["demand"], capacity["capacity"]
        )

    if time_windows is not None:
        data_model.set_order_time_windows(
            time_windows["earliest_time"], time_windows["latest_time"]
        )
        data_model.set_order_service_times(time_windows["service_time"])

    solver_settings = routing.SolverSettings()

    if time_limit is not None:
        solver_settings.set_time_limit(time_limit)

    cuopt_solver_start_time = time.time()
    routing_solution = routing.Solve(data_model, solver_settings)
    end_time = time.time()
    cuopt_solver_run_time = end_time - cuopt_solver_start_time
    total_cuopt_run_time = end_time - total_cuopt_run_time_start

    num_vehicles = routing_solution.get_vehicle_count()
    cost = routing_solution.get_total_objective()

    return (total_cuopt_run_time, cuopt_solver_run_time, num_vehicles, cost)


def run_cvrptw_benchmark_cuopt(
    cvrptw_instances,
    best_known_solutions,
    matrix_set,
    time_limit_coeff,
    params_explicit=False,
    verbose=True,
):
    """
    Function to run a list of cvrptw_instances and
    meassures cuopt against best known solutions.
    """

    instance_name = []
    total_demand = []
    capacity = []
    num_locations = []
    earliest_location_time = []
    latest_location_time = []

    solve_time = []
    cuopt_solver_run_times = []
    total_cuopt_run_times = []

    cuopt_vehicles = []
    cuopt_cost = []

    bks_vehicles = []
    bks_cost = []

    vehicle_error = []
    cost_error = []

    for instance in cvrptw_instances:
        raw_instance_data = benchmark_utils.get_homberger_instance_data(
            instance
        )

        (
            cuopt_point_data,
            cuopt_demand,
            cuopt_capacity,
            cuopt_time_windows,
        ) = cuopt_cvrptw_data_preprocess(raw_instance_data, matrix_set)

        problem_size = len(cuopt_point_data) - 1

        if params_explicit:
            time_limit = time_limit_coeff
        else:
            time_limit = time_limit_coeff * problem_size

        (
            total_cuopt_run_time,
            cuopt_solver_run_time,
            cuopt_num_vehicles,
            cuopt_solution_cost,
        ) = cuopt_solve(
            cuopt_point_data,
            use_matrix=matrix_set,
            demand=cuopt_demand,
            capacity=cuopt_capacity,
            time_windows=cuopt_time_windows,
            time_limit=time_limit,
        )

        reported_cuopt_cost = round(cuopt_solution_cost, 2)

        best_known_solution = best_known_solutions.loc[
            best_known_solutions["Instance"]
            == raw_instance_data["instance_name"]
        ]

        best_known_vehicles = best_known_solution.BKS_NumVehicles.values[0]
        best_known_cost = best_known_solution.BKS_Cost.values[0]

        instance_name.append(raw_instance_data["instance_name"])
        total_demand.append(raw_instance_data["total_demand"])
        capacity.append(raw_instance_data["vehicle_capacity"])
        num_locations.append(problem_size)

        earliest_location_time.append(
            cuopt_time_windows["earliest_time"][1::].min()
        )

        latest_location_time.append(
            cuopt_time_windows["latest_time"][1::].max()
        )

        solve_time.append(time_limit)
        cuopt_solver_run_times.append(cuopt_solver_run_time)
        total_cuopt_run_times.append(total_cuopt_run_time)

        cuopt_vehicles.append(cuopt_num_vehicles)
        cuopt_cost.append(reported_cuopt_cost)

        bks_vehicles.append(best_known_vehicles)
        bks_cost.append(best_known_cost)

        vehicle_error.append(cuopt_num_vehicles / best_known_vehicles - 1)
        cost_error.append(reported_cuopt_cost / best_known_cost - 1)

        if verbose:
            print("COMPLETED : {}".format(raw_instance_data["instance_name"]))

    return {
        "Instance_Name": instance_name,
        "Num_Locations": num_locations,
        "Total_Demand": total_demand,
        "Vehicle_Capacity": raw_instance_data["vehicle_capacity"],
        "Earliest_Location_Time": earliest_location_time,
        "Latest_Location_Time": latest_location_time,
        "GPU_Run_Time_Limit(s)": solve_time,
        "SolverSettings_Run_Time(s)": cuopt_solver_run_times,
        "Total_Run_Time(s)": total_cuopt_run_times,
        "cuOpt_Num_Vehicles": cuopt_vehicles,
        "cuOpt_Cost": cuopt_cost,
        "BKS_Vehicles": bks_vehicles,
        "BKS_Cost": bks_cost,
        "Vehicle_Error": vehicle_error,
        "Cost_Error": cost_error,
    }


def main():
    """
    Runs Homeberger benchmark instances contained within 'start_directory' or
    any of it's subdirectories.

    Homberger presents a x,y point list.  cuOpt can use the point list
    directly with use_cost_matrix=False or convert the point list to
    a distance matrix with use_cost_matrix=True

    time_limit_multiplier scales the cuOpt time_limit by the size of the
    problem instance.

    Solutions are compared against Best Known Solutions and saved out to .csv
    """

    start_directory = "datasets/cvrptw/Test"
    bks_homberger_filepath = "datasets/ref/bks_gehring_homberger.csv"

    use_cost_matrix = False
    set_params_explicit = True

    # used if set_params_explit = True
    time_limit = 10

    # used if set_params_explit = False
    # basic heuristic for the time_limit
    # as a function of problem size
    time_limit_function_weights = 0.2

    cvrptw_instances = sorted(
        benchmark_utils.get_problem_instance_paths(start_directory, ".TXT")
    )

    bks_homberger = cudf.read_csv(bks_homberger_filepath)

    # gpu_name = benchmark_utils.get_gpu_type()

    if set_params_explicit:
        cuopt_benchmark_data = run_cvrptw_benchmark_cuopt(
            cvrptw_instances,
            bks_homberger,
            use_cost_matrix,
            time_limit,
            params_explicit=set_params_explicit,
        )

    else:
        cuopt_benchmark_data = run_cvrptw_benchmark_cuopt(
            cvrptw_instances,
            bks_homberger,
            use_cost_matrix,
            time_limit_function_weights,
            params_explicit=set_params_explicit,
        )

    cuopt_benchmark_data_cudf = cudf.DataFrame(cuopt_benchmark_data)

    cuopt_benchmark_data_cudf.to_csv(
        "cuOpt_CVRPTW_Bechnmark.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
