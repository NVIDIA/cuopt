# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import benchmark_utils

import cudf

from cuopt import cuopt_benchmark_script

verbose = True

instance_path = "datasets/cvrptw/Test"
bks_homberger_filepath = "datasets/ref/bks_gehring_homberger.csv"

# time_array = [1, 5, 10, 30, 60, 120, 300, 600, 1200]
time_array = [1, 10]
restart_limit = 750

cvrptw_instances = sorted(
    benchmark_utils.get_problem_instance_paths(instance_path, ".TXT")
)

gpu_name = benchmark_utils.get_gpu_type()

if verbose:
    print("\nSelected Instances Running on {}:".format(gpu_name))
    for i in cvrptw_instances:
        print(i)
    print("\n")

bks_homberger = cudf.read_csv(bks_homberger_filepath)

instance_names = []
num_locations = []
gpu_time_limits = []
solver_run_times = []
total_run_times = []
restart_limits = []
cuopt_num_v = []
cuopt_costs = []
bks_v = []
bks_costs = []
vehicle_errors = []
cost_errors = []

for instance in cvrptw_instances:
    for time_limit in time_array:
        raw_instance_data = benchmark_utils.get_homberger_instance_data(
            instance
        )

        (
            cuopt_point_data,
            cuopt_demand,
            cuopt_capacity,
            cuopt_time_windows,
        ) = cuopt_benchmark_script.cuopt_cvrptw_data_preprocess(
            raw_instance_data, use_matrix=False
        )

        problem_size = len(cuopt_point_data) - 1

        (
            total_cuopt_run_time,
            cuopt_solver_run_time,
            cuopt_num_vehicles,
            cuopt_solution_cost,
        ) = cuopt_benchmark_script.cuopt_solve(
            cuopt_point_data,
            use_matrix=False,
            demand=cuopt_demand,
            capacity=cuopt_capacity,
            time_windows=cuopt_time_windows,
            time_limit=time_limit,
            restart_limit=restart_limit,
        )

        reported_cuopt_cost = round(cuopt_solution_cost, 2)

        best_known_solution = bks_homberger.loc[
            bks_homberger["Instance"] == raw_instance_data["instance_name"]
        ]

        best_known_vehicles = (
            best_known_solution.BKS_NumVehicles.to_arrow().to_pylist()[0]
        )

        best_known_cost = best_known_solution.BKS_Cost.to_arrow().to_pylist()[
            0
        ]

        instance_names.append(raw_instance_data["instance_name"])
        num_locations.append(problem_size)
        gpu_time_limits.append(time_limit)
        solver_run_times.append(cuopt_solver_run_time)
        total_run_times.append(total_cuopt_run_time)
        restart_limits.append(restart_limit)
        cuopt_num_v.append(cuopt_num_vehicles)
        cuopt_costs.append(reported_cuopt_cost)

        bks_v.append(best_known_vehicles)
        bks_costs.append(best_known_cost)

        vehicle_errors.append(cuopt_num_vehicles / best_known_vehicles - 1)
        cost_errors.append(reported_cuopt_cost / best_known_cost - 1)

        if verbose:
            print(
                "Finished {} with {} second time limit GPU".format(
                    raw_instance_data["instance_name"], time_limit
                )
            )

results = {
    "Instance_Name": instance_names,
    "Num_Locations": num_locations,
    "GPU_Run_Time_Limit(s)": gpu_time_limits,
    "SolverSettings_Run_Time(s)": solver_run_times,
    "Total_Run_Time(s)": total_run_times,
    "Restart_Limit": restart_limits,
    "cuOpt_Num_Vehicles": cuopt_num_v,
    "cuOpt_Cost": cuopt_costs,
    "BKS_Vehicles": bks_v,
    "BKS_Cost": bks_costs,
    "Vehicle_Error": vehicle_errors,
    "Cost_Error": cost_errors,
}

results_cudf = cudf.DataFrame(results)

results_cudf.to_csv("Homberger_Crossing_Benchmark_cuOpt.csv", index=False)
