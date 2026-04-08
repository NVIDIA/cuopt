# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal LP demo for NVIDIA cuOpt gRPC remote execution.

Set CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT on the client before running to forward
the solve to cuopt_grpc_server; unset them to solve locally (GPU required locally).

The same LP is available as MPS in ``remote_lp_demo.mps`` for ``cuopt_cli``.
"""

import numpy as np
from cuopt import linear_programming

dm = linear_programming.DataModel()
A_values = np.array([3.0, 4.0, 2.7, 10.1], dtype=np.float64)
A_indices = np.array([0, 1, 0, 1], dtype=np.int32)
A_offsets = np.array([0, 2, 4], dtype=np.int32)
dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

b = np.array([5.4, 4.9], dtype=np.float64)
dm.set_constraint_bounds(b)

c = np.array([0.2, 0.1], dtype=np.float64)
dm.set_objective_coefficients(c)

dm.set_row_types(np.array(["L", "L"]))

dm.set_variable_lower_bounds(np.array([0.0, 0.0], dtype=np.float64))
dm.set_variable_upper_bounds(np.array([2.0, np.inf], dtype=np.float64))

settings = linear_programming.SolverSettings()
solution = linear_programming.Solve(dm, settings)

print("Termination:", solution.get_termination_reason())
print("Objective:  ", solution.get_primal_objective())
print("Primal x:   ", solution.get_primal_solution())
