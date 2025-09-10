#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
from pathlib import Path
import json
import cuopt_mps_parser

def _mps_parse(LP_problem_data, tolerances, time_limit, iteration_limit):

    if isinstance(LP_problem_data, cuopt_mps_parser.parser_wrapper.DataModel):
        model = LP_problem_data
    else:
        model = cuopt_mps_parser.ParseMps(LP_problem_data)

    problem_data = cuopt_mps_parser.toDict(model, json=True)
    #variable_names = problem_data.pop("variable_names")

    problem_data["solver_config"] = {}
    if tolerances is not None:
        problem_data["solver_config"]["tolerances"] = tolerances
    if time_limit is not None:
        problem_data["solver_config"]["time_limit"] = time_limit
    if iteration_limit is not None:
        problem_data["solver_config"]["iteration_limit"] = iteration_limit
    return problem_data


def create_config_and_data(input_directory, file_name, output_directory, prefix, time_limit=None, tolerances=None, iteration_limit=None):

    file_path = input_directory/file_name
    data = _mps_parse(file_path.as_posix(), tolerances, time_limit, iteration_limit)

    base_file_name = file_name.split(".")[0]

    config_file_name = prefix +"_" +base_file_name+"_config.json"
    data_file_name = prefix +"_" +base_file_name+"_data.json"

    config_data = {
        "test_name": prefix +"_" +base_file_name,
        "file_name": data_file_name,
        "metrics": {
            "primal_objective_value": {
                "threshold": 1,
                "unit": "primal_objective_value",
            },
            "solver_time": {
                "threshold": 1,
                "unit": "seconds"
            }
        },
        "details": base_file_name + " test"
    }

    with open(output_directory/config_file_name, "w") as fp:
        json.dump(config_data, fp, indent=4, sort_keys=True)

    with open(output_directory/data_file_name, "w") as fp:
        json.dump(data, fp, indent=4, sort_keys=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Solve a cuOpt problem using a managed service client."
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder path"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="Output folder path"
    )
    parser.add_argument(
        "-tl",
        "--time-limit",
        default=None,
        type=int,
        help="LP timit in milliseconds"
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="",
        help="Prefix for config and data"
    )

    args = parser.parse_args()
    input_directory = Path(args.folder)
    output_directory = Path(args.output)

    # List all files with .mps extension
    mps_files = [f.name for f in input_directory.glob('*.mps')]
    list_of_files = [
        "50v-10", "lotsize", "swath1", "nursesched-medium-hint03", "academictimetablesmall", "dano3_3",
        "neos-4338804-snowy", "istanbul-no-cutoff", "s100", "traininstance2"
    ]
    for mps_file in mps_files:
        if mps_file.split(".")[0] in list_of_files:
            create_config_and_data(input_directory, mps_file, output_directory, args.prefix, args.time_limit)
