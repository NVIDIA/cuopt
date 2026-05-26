# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.io import parser_wrapper
from cuopt.linear_programming.io.utilities import (
    catch_io_exception,
)


@catch_io_exception
def Read(file_path: str, fixed_mps_format: bool = False) -> DataModel:
    """Read an optimization problem from a file, dispatching on extension.

    Dispatches to the MPS/QPS or LP reader based on the filename suffix
    (case-insensitive), matching the C++ ``read`` entry point:

    - ``.mps``, ``.mps.gz``, ``.mps.bz2``, ``.qps``, ``.qps.gz``, ``.qps.bz2``
      → MPS/QPS reader
    - ``.lp``, ``.lp.gz``, ``.lp.bz2`` → LP reader

    Parameters
    ----------
    file_path : str
        Path to an MPS, QPS, or LP file (optionally ``.gz`` / ``.bz2``
        compressed).
    fixed_mps_format : bool
        If the MPS/QPS reader should parse as fixed MPS format. Ignored for
        LP inputs. False by default.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MILP/QP problem.

    Raises
    ------
    InputValidationError, InputRuntimeError, OutOfMemoryError
        Parser errors from the underlying C++ readers (via
        ``catch_io_exception``).
    RuntimeError
        If the file extension is not one of the supported suffixes (raised by
        the C++ ``read`` dispatch).
    """
    return parser_wrapper.Read(file_path, fixed_mps_format)


@catch_io_exception
def ParseMps(mps_file_path: str, fixed_mps_format: bool = False) -> DataModel:
    """Read an MPS or QPS file directly via the MPS/QPS reader.

    Unlike :func:`Read`, this function bypasses extension-based dispatch
    and always invokes the MPS/QPS reader (``read_mps`` on the C++ side),
    regardless of the filename suffix. Compressed inputs (``.mps.gz``,
    ``.mps.bz2``, ``.qps.gz``, ``.qps.bz2``) are still supported when
    zlib / libbz2 are available, because compression is detected from
    the file path inside the reader.

    Parameters
    ----------
    mps_file_path : str
        Path to an MPS or QPS file (optionally ``.gz`` / ``.bz2``
        compressed).
    fixed_mps_format : bool
        If the MPS/QPS reader should parse the file as fixed MPS format.
        False by default.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MILP/QP problem.

    Raises
    ------
    InputValidationError, InputRuntimeError, OutOfMemoryError
        Parser errors from the underlying C++ reader (via
        ``catch_io_exception``).
    """
    return parser_wrapper.ParseMps(mps_file_path, fixed_mps_format)


def toDict(model, json=False):
    if not isinstance(model, parser_wrapper.DataModel):
        raise ValueError(
            "model must be a cuopt.linear_programming.io.parser_wrapper.DataModel"
        )

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
