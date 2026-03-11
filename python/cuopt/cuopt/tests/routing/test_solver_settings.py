# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf

from cuopt import routing
from cuopt.routing import utils

filename = utils.RAPIDS_DATASET_ROOT_DIR + "/solomon/In/r107.txt"


def test_verbose_mode():
    """Solve with verbose mode on; assert solution status."""
    cost = cudf.DataFrame(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    dm = routing.DataModel(3, 1)
    dm.add_cost_matrix(cost)
    s = routing.SolverSettings()
    s.set_verbose_mode(True)
    s.set_time_limit(2)
    solution = routing.Solve(dm, s)
    assert solution.get_status() == 0


def test_dump_results():
    d = utils.create_data_model(filename, run_nodes=10)
    s = routing.SolverSettings()
    file_path = "best_results.txt"
    interval = 0.001
    s.dump_best_results(file_path, interval)
    s.set_time_limit(4)
    routing_solution = routing.Solve(d, s)
    assert routing_solution.get_status() == 0
    ret_file_path = s.get_best_results_file_path()
    ret_interval = s.get_best_results_interval()
    assert file_path == ret_file_path
    assert interval == ret_interval


def test_solver_settings_getters():
    s = routing.SolverSettings()
    time_limit = 10.5
    s.set_time_limit(time_limit)
    assert s.get_time_limit() == time_limit


def test_dump_config():
    """Test SolverSettings solve with config file"""
    s = routing.SolverSettings()
    config_file = "solver_cfg.yaml"
    s.dump_config_file(config_file)
    assert s.get_config_file_name() == config_file

    # Small example data model: 3 locations, 1 vehicle
    cost = cudf.DataFrame(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    dm = routing.DataModel(3, 1)
    dm.add_cost_matrix(cost)
    s.set_time_limit(2)
    routing_solution = routing.Solve(dm, s)
    assert routing_solution.get_status() == 0

    # Load from written solver_cfg.yaml and solve again
    dm_from_yaml, s_from_yaml = utils.create_data_model_from_yaml(config_file)
    solution_from_yaml = routing.Solve(dm_from_yaml, s_from_yaml)
    assert solution_from_yaml.get_status() == 0
