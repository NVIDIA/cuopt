# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from cuopt.linear_programming.data_model import DataModel
from cuopt.linear_programming.io import parser_wrapper
from cuopt.linear_programming.io.utilities import (
    catch_io_exception,
)


@catch_io_exception
def ParseMps(mps_file_path, fixed_mps_format=False):
    """
    Reads the equation from the input text file which is MPS formatted

    See Also
    --------
    ParseLp : parses LP format files (for users with .lp inputs).

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


@catch_io_exception
def ParseLp(lp_file_path: str) -> DataModel:
    """Read an optimization problem from a file in LP format.

    The LP format is a human-readable alternative to MPS and supports LP,
    MIP, and QP, plus semi-continuous variables (declared via a
    Semi-Continuous section; finite upper bound required) and
    quadratic constraints (QCQP; ``<=`` only).

    Quadratic terms live in ``[ ... ]`` blocks. The objective bracket must
    be followed by ``/ 2`` (the file states coefficients in the
    ``0.5 x^T Q x`` convention); a constraint bracket must NOT be followed
    by ``/ 2`` (coefficients are at face value, ``x^T Q x``). Only squared
    (``x^2``) and product (``x * y``) terms are allowed inside the
    bracket; bare linear terms must be written outside it.

    This function parses the dialect in which the objective and constraints
    are written as algebraic expressions over named variables (it does not
    implement the alternative tableau-style LP dialect used by some
    open-source readers).

    Parameters
    ----------
    lp_file_path : str
        Path to LP-formatted file. May end in ``.lp``, ``.lp.gz``, or
        ``.lp.bz2``; compressed inputs are decompressed at read time
        via zlib / libbz2 when those libraries are available.

    Returns
    -------
    data_model : DataModel
        A fully formed LP/MIP/QP problem representing the contents of
        ``lp_file_path``.

    Raises
    ------
    InputValidationError
        Raised when ``lp_file_path`` is malformed or uses unsupported
        syntax. Examples include unsupported sections (SOS, PWL
        objective, user cuts, general constraints), bare linear terms
        inside a quadratic ``[ ... ]`` bracket, an objective bracket
        not followed by ``/ 2``, a constraint bracket followed by
        ``/ 2``, a semi-continuous variable without a finite upper
        bound, and similar input-level errors raised by the underlying
        C++ parser. Exceptions propagated from
        :func:`parser_wrapper.ParseLp` are translated to this type by
        :func:`catch_io_exception`.
    InputRuntimeError
        Raised for non-validation runtime errors that the C++ parser
        flags during file I/O or parsing.
    OutOfMemoryError
        Raised when the parser cannot allocate memory for the
        resulting data model.

    Examples
    --------
    >>> from cuopt import linear_programming
    >>>
    >>> data_model = linear_programming.ParseLp(lp_file_path)
    >>> solver_settings = linear_programming.SolverSettings()
    >>> solution = linear_programming.Solve(data_model, solver_settings)
    """
    return parser_wrapper.ParseLp(lp_file_path)


def toDict(model, json=False):
    if not isinstance(model, parser_wrapper.DataModel):
        raise ValueError(
            "model must be a cuopt.linear_programming.io.parser_wrapper.DataModel"
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
