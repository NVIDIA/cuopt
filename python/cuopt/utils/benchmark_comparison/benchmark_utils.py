# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess

from scipy.spatial.distance import pdist, squareform


def execute_bash_command(cmd):
    """
    Run a bash command and return the output
    """
    tenv = os.environ.copy()
    bash_command = cmd

    process = subprocess.Popen(
        bash_command.split(), stdout=subprocess.PIPE, env=tenv
    )

    return process.communicate()[0]


def get_gpu_type():
    """
    Get the type of GPU being used
    """
    bash_command = "nvidia-smi --query-gpu=name --format=csv,noheader"
    gpu_0 = execute_bash_command(bash_command).decode("ascii").split("\n")[0]
    return gpu_0


def get_cpu_type():
    """
    Get the type of CPU being used
    """
    bash_command = "lscpu"

    cpu_0 = (
        execute_bash_command(bash_command)
        .decode("ascii")
        .split("\n")[12]
        .split("          ")[-1]
        .replace(" ", "_")
    )

    return cpu_0


def distance_matrix_from_point_list(point_list, scale):
    """
    Create a distance matrix from a point list
    """
    return scale * squareform(pdist(point_list, metric="euclidean"))


def get_problem_instance_paths(start_dir, file_ext):
    """
    Get problem instances of a specific
    file type recusively from a starting directory
    """
    problem_instances = []
    for dirpath, dirnames, filenames in os.walk(start_dir):
        for filename in [f for f in filenames if f.endswith(file_ext)]:
            problem_instances.append(os.path.join(dirpath, filename))

    return problem_instances


def get_homberger_instance_data(instance_file_path):
    """
    Function to parse the raw instance data from .TXT.
    It is specific to the cvrptw Homberger format found here:

    https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/
    """
    num_vehicle_and_capacity_line_index = 4
    stop_data_line_index = 9

    with open(instance_file_path) as f:
        lines = f.readlines()

    instance_data = {}

    instance_data["instance_name"] = instance_file_path.split("/")[-1]

    instance_data["fleet_size"] = int(
        lines[num_vehicle_and_capacity_line_index].split()[0]
    )

    instance_data["vehicle_capacity"] = int(
        lines[num_vehicle_and_capacity_line_index].split()[1]
    )

    instance_data["xcoord"] = []
    instance_data["ycoord"] = []
    instance_data["demand"] = []
    instance_data["earliest_time"] = []
    instance_data["latest_time"] = []
    instance_data["service_time"] = []
    instance_data["total_demand"] = 0
    for i in range(stop_data_line_index, len(lines)):
        stop_data = lines[i].split()
        instance_data["xcoord"].append(int(stop_data[1]))
        instance_data["ycoord"].append(int(stop_data[2]))
        instance_data["demand"].append(int(stop_data[3]))
        instance_data["total_demand"] += int(stop_data[3])
        instance_data["earliest_time"].append(int(stop_data[4]))
        instance_data["latest_time"].append(int(stop_data[5]))
        instance_data["service_time"].append(int(stop_data[6]))

    return instance_data
