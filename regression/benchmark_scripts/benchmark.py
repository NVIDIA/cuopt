# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from utils import get_configuration, LPMetrics, RoutingMetrics
from cuopt import linear_programming
from cuopt import routing
from cuopt import utilities
import rmm
import time
import pandas as pd
import glob
import logging as log
from datetime import datetime
import os
import argparse

def create_regression_markdown(data, regression_path, test_type_string):
    regression_md_file = regression_path + "/" + test_type_string + "_regressions.md" 

    md_data = "*No regressions*"
    # This to reduce size of slack message
    limit_no_of_regression_list = 5

    if len(data) > 0:
        status = "*!! Regressions found !!*"
        end_msg = "\n*Continues ...*" if len(data) > limit_no_of_regression_list else ""
        table = data[:limit_no_of_regression_list].to_string(index=False)
        md_data = status + f'\n```\n{table}\n```' + end_msg

    with open(regression_md_file, "w") as fp:
        fp.write(md_data)

def record_regressions(test_name, data, req_metrics, regression_path, test_type_string):

    regression_file = regression_path + "/" + test_type_string + "_regressions.csv"

    regression_df = pd.DataFrame({"Test Name":[], "Metric Name":[], "Value":[], "Avg Value":[], "Regression(%)":[]})
    for name in req_metrics:
        if name.startswith("bks_change_"):
            pchange = data[name].iloc[-1].item()
            metric_name = name.replace("bks_change_", "")
            limit = req_metrics[metric_name]["bks"].get("threshold", 5)
            prev_val_mean = pchange
            latest_val = pchange
        else:
            limit = req_metrics[name].get("threshold", 5)
            prev_val_mean = data[name][:-1][-30:].mean().item() if len(data) > 1 else data[name].iloc[-1].item()
            latest_val = data[name].iloc[-1].item()

            if prev_val_mean == 0:
                pchange = latest_val
            else:
                pchange = ((latest_val - prev_val_mean)/prev_val_mean) * 100

        if abs(pchange) >= limit:
            regression_df.loc[len(regression_df)] = [test_name, name, latest_val, prev_val_mean, pchange]

    regression_df.to_csv(regression_file)
    create_regression_markdown(regression_df, regression_path, test_type_string)

def get_bks_change(
    metrics, required_metrics
):
    bks_metrics = {}
    for metric, value in required_metrics.items():
        if "bks" in value.keys():
            bks = value["bks"]["value"]
            if bks == None:
                continue
            current = metrics[metric]
            if bks == 0:
                bks_metrics["bks_change_" + metric] = abs(current) * 100
            elif current == 0:
                bks_metrics["bks_change_" + metric] = abs(bks) * 100
            else:
                bks_metrics["bks_change_" + metric] = abs(((current - bks)/bks) * 100)

    return bks_metrics

def record_result(
    test_name, metrics, required_metrics, csv_path, test_type_string
):
    
    file_path = csv_path + "/" + test_name + ".csv"

    bks_metrics = get_bks_change(metrics, required_metrics)
    # Add default metrics to data
    required_metrics.update(bks_metrics)
    metrics.update(bks_metrics)

    req_metrics = list(required_metrics.keys()) + ["date_time", "git_commit"]

    current_data = pd.DataFrame({key : [metrics[key]] for key in sorted(req_metrics)})
    if os.path.isfile(file_path):
        previous_data = pd.read_csv(file_path, index_col=0)
        updated_data = pd.concat([previous_data, current_data], ignore_index=True)
    else:
        updated_data = current_data

    record_regressions(test_name, updated_data, required_metrics, csv_path, test_type_string)

    updated_data.to_csv(file_path)


def run_benchmark(
    test_name,
    data_model,
    solver_settings,
    required_metrics,
    csv_path,
    git_commit,
    test_status_file
):
    mr = rmm.mr.get_current_device_resource()

    start_time = time.time()
    if test_name.startswith("LP_") or test_name.startswith("MIP_"):
        metrics = LPMetrics()._asdict()
        solver_settings.set_parameter("infeasibility_detection", False)
        solver_settings.set_parameter("time_limit", 180)
        solution = linear_programming.Solve(data_model, solver_settings)
    else:
        metrics = RoutingMetrics()._asdict()
        solution = routing.Solve(data_model, solver_settings)
    end_time = time.time()

    metrics["gpu_memory_usage"] = int(mr.allocation_counts.peak_bytes/(1024*1024))
    metrics["date_time"] = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
    metrics["git_commit"] = git_commit

    success_status = False

    if test_name.startswith("LP_") or test_name.startswith("MIP_"):
        ## Optimal solution
        if solution.get_termination_reason() == 1:
            test_type_string = "lp" if test_name.startswith("LP_") else "mip"
            success_status = True
            metrics["solver_time"] = solution.get_solve_time()
            metrics["primal_objective_value"] = solution.get_primal_objective()
            if test_type_string == "lp":
                lp_stats = solution.get_lp_stats()
                metrics["nb_iterations"] = lp_stats["nb_iterations"]
            else:
                milp_stats = solution.get_milp_stats()
                metrics["mip_gap"] = milp_stats["mip_gap"]
                metrics["max_constraint_violation"] = milp_stats["max_constraint_violation"]
                metrics["max_int_violation"] = milp_stats["max_int_violation"]
                metrics["max_variable_bound_violation"] = milp_stats["max_variable_bound_violation"]
            record_result(test_name, metrics, required_metrics, csv_path, test_type_string)
    else:
        if solution.get_status() == 0:
            success_status = True
            metrics["solver_time"] = end_time - start_time
            metrics["total_objective_value"] = solution.get_total_objective()
            metrics["vehicle_count"] = solution.get_vehicle_count()

            objectives = solution.get_objective_values()
            if "prize" in required_metrics:
                metrics["prize"] = objectives[routing.Objective.PRIZE]
            if "cost" in required_metrics:
                metrics["cost"] = objectives[routing.Objective.COST]
            if "travel_time" in required_metrics:
                metrics["travel_time"] = objectives[routing.Objective.TRAVEL_TIME]

            record_result(test_name, metrics, required_metrics, csv_path, "routing")

    return "SUCCESS" if success_status is True else "FAILED"


def reinitialize_rmm():
    
    pool_size = 2**30
    rmm.reinitialize(pool_allocator=True, initial_pool_size=pool_size)

    base_mr = rmm.mr.get_current_device_resource()
    stats_mr = rmm.mr.StatisticsResourceAdaptor(base_mr)
    rmm.mr.set_current_device_resource(stats_mr) 

    return base_mr, stats_mr


def run(config_file_path, csv_path, git_commit, log_path, test_status_file):

    config_files = glob.glob(config_file_path + "/*_config.json")

    for config in config_files:

        mr, stats_mr = reinitialize_rmm()
        test_name = str(config)
        status = "FAILED"
        try:

            test_name, data_model, solver_settings, requested_metrics = get_configuration(config, config_file_path)

            log.basicConfig(level=log.INFO, filename=log_path+"/"+test_name+"_log.txt", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
            log.info(f"------------- Test Start : {test_name} -------------------")

            status = run_benchmark(
                test_name,
                data_model,
                solver_settings,
                requested_metrics,
                csv_path,
                git_commit,
                test_status_file
            )

        except Exception as e:
            log.error(str(e))

        with open(test_status_file, "a") as f:
            f.write("\n")
            f.write(test_name +": " + status)

        # Delete instance of rmm
        del mr
        del stats_mr

        log.info(f"------------- Test End : {test_name} -------------------")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-c", "--config-path", type=str, help="Path to all configuration file"
    )
    parser.add_argument(
        "-r", "--csv-path", type=str, help="Path to store result files, this would be for github where results gets stored"
    )
    parser.add_argument(
        "-g", "--git-commit", type=str, help="git commit sha to keep track of runs"
    )
    parser.add_argument(
        "-l", "--log-path", type=str, help="Path to log files"
    )
    parser.add_argument(
        "-s", "--test-status-file", type=str, help="All test status will be stored in this file"
    )

    args = parser.parse_args()
    run(args.config_path, args.csv_path, args.git_commit, args.log_path, args.test_status_file)
