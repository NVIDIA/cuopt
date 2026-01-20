# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
# CONFIDENTIAL, provided under NDA.

from cuopt.routing import utils
import json

"""
This is an example of creating a modified test from Homberger dataset.
In this test, the RC2_10_5 test is modified so that the vehicle count is reduced to 12 and the order prizes are set.
The prizes are high enough so that prize always becomes the primary objective.
One can easily use an existing json file and modify the data as well by loading the json as a dictionary
"""
test_name = "prize_collection_vrp"
# test_name = "LC1_10_9"

# base_file_name = "/home/nfs/rgandham/git-repos/reopt/datasets/pdptw/LC1_10_9.pdptw"
base_file_name = (
    "/home/nfs/rgandham/git-repos/reopt/datasets/cvrptw/RC2_10_5.TXT"
)

# model_dict = utils.create_model_dictionary_from_file(base_file_name, is_pdp=True)
model_dict = utils.create_model_dictionary_from_file(base_file_name)


# Reduce the fleet size to 12
num_vehicles = 12
fleet_data = model_dict["fleet_data"]

vehicle_locations = fleet_data["vehicle_locations"]
vehicle_tw = fleet_data["vehicle_time_windows"]
capacities = fleet_data["capacities"]

new_locs = [vehicle_locations[i] for i in range(num_vehicles)]
new_tw = [vehicle_tw[i] for i in range(num_vehicles)]
new_cap = [[capacities[0][i] for i in range(num_vehicles)]] * 1

fleet_data["vehicle_locations"] = new_locs
fleet_data["vehicle_time_windows"] = new_tw
fleet_data["capacities"] = new_cap

# Add prizes
task_data = model_dict["task_data"]

n_tasks = len(task_data["demand"][0])

prizes = [10000.0] * n_tasks
task_data["prizes"] = prizes


# Set 10 min time limit
solver_config = {}
solver_config["time_limit"] = 600

model_dict["solver_config"] = solver_config

test_config_file_name = test_name + "_config.json"
model_data_file_name = test_name + "_data.json"

test_config = {}
test_config["test_name"] = test_name
test_config["file_name"] = model_data_file_name
test_config["metrics"] = [
    "vehicle_count",
    "total_cost",
    "prize",
    "memory_usage",
]

with open(test_config_file_name, "w") as fp:
    json.dump(test_config, fp)
    fp.close()

with open(model_data_file_name, "w") as fp:
    json.dump(model_dict, fp)
    fp.close()
