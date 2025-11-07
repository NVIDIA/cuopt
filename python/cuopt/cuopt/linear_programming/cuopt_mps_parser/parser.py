# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from cuopt_mps_parser import parser_wrapper
from cuopt_mps_parser.utilities import catch_mps_parser_exception


@catch_mps_parser_exception
def ParseMps(mps_file_path, fixed_mps_format=False):
    """
    Reads the equation from the input text file which is MPS formatted

    Notes
    -----
    Read this link http://lpsolve.sourceforge.net/5.5/mps-format.htm for more
    details on both free and fixed MPS format.

    Parameters
    ----------
    mps_file_path : str
        Path to MPS formatted file
    fixed_mps_format : bool
        If MPS file should be parsed as fixed, false by default

    Returns
    -------
    data_model: DataModel
        A fully formed LP problem which represents the given MPS file

    Examples
    --------
    >>> from cuopt import linear_programming
    >>>
    >>> data_model = linear_programming.ParseMps(mps_file_path)
    >>>
    >>> # Build a solver setting object & lower the accuracy from 1e-4 to 1e-2
    >>> solver_settings = linear_programming.SolverSettings()
    >>> solver_settings.set_optimality_tolerance(1e-2)
    >>>
    >>> # Call solve
    >>> solution = linear_programming.Solve(data_model, solver_settings)
    >>>
    >>> # Print solution
    >>> print(solution.get_primal_solution())
    """

    return parser_wrapper.ParseMps(mps_file_path, fixed_mps_format)


def toDict(model, json=False):
    if not isinstance(model, parser_wrapper.DataModel):
        raise ValueError(
            "model must be a cuopt_mps_parser.parser_wrapper.Datamodel"
        )

    # Replace numpy objects in generated data so that it is JSON serializable
    def transform(data):
        for key, value in data.items():
            if isinstance(value, dict):
                transform(value)
            elif isinstance(value, list):
                if np.inf in data[key] or -np.inf in data[key]:
                    data[key] = [
                        "inf" if x == np.inf else "ninf" if x == -np.inf else x
                        for x in data[key]
                    ]

    if json is True:
        problem_data = {
            "csr_constraint_matrix": {
                "offsets": model.A_offsets.tolist(),
                "indices": model.A_indices.tolist(),
                "values": model.A_values.tolist(),
            },
            "constraint_bounds": {
                "bounds": model.b.tolist(),
                "upper_bounds": model.constraint_upper_bounds.tolist(),
                "lower_bounds": model.constraint_lower_bounds.tolist(),
                "types": model.host_row_types.tolist(),
            },
            "objective_data": {
                "coefficients": model.c.tolist(),
                "scalability_factor": model.objective_scaling_factor,
                "offset": model.objective_offset,
            },
            "variable_bounds": {
                "upper_bounds": model.variable_upper_bounds.tolist(),
                "lower_bounds": model.variable_lower_bounds.tolist(),
            },
            "maximize": model.maximize,
            "variable_types": model.variable_types.tolist(),
            "variable_names": model.variable_names.tolist(),
        }
        transform(problem_data)
    else:
        problem_data = {
            "csr_constraint_matrix": {
                "offsets": model.A_offsets,
                "indices": model.A_indices,
                "values": model.A_values,
            },
            "constraint_bounds": {
                "bounds": model.b,
                "upper_bounds": model.constraint_upper_bounds,
                "lower_bounds": model.constraint_lower_bounds,
                "types": model.host_row_types,
            },
            "objective_data": {
                "coefficients": model.c,
                "scalability_factor": model.objective_scaling_factor,
                "offset": model.objective_offset,
            },
            "variable_bounds": {
                "upper_bounds": model.variable_upper_bounds,
                "lower_bounds": model.variable_lower_bounds,
            },
            "maximize": model.maximize,
            "variable_types": model.variable_types,
            "variable_names": model.variable_names,
        }
    return problem_data
