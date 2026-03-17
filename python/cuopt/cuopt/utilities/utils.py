# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import cudf
import pylibcudf as plc


def series_from_buf(buf, dtype):
    """Helper function to create a cudf series from a buffer.

    Parameters
    ----------
    buf : cudf.core.buffer.Buffer
        The buffer containing the data
    dtype : pyarrow.dtype or type
        The data type for the Series

    Returns
    -------
    cudf.Series
        A cudf Series built from the buffer
    """
    col = plc.column.Column.from_rmm_buffer(
        buf,
        dtype=plc.types.DataType.from_arrow(dtype),
        size=buf.size // dtype.byte_width,
        children=[],
    )

    return cudf.Series.from_pylibcudf(col)


def validate_variable_bounds(data, settings, solution):
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_MIP_INTEGRALITY_TOLERANCE,
    )

    integrality_tolerance = settings.get_parameter(
        CUOPT_MIP_INTEGRALITY_TOLERANCE
    )
    integrality_tolerance = (
        integrality_tolerance if integrality_tolerance else 1e-5
    )

    if len(data.get_variable_lower_bounds() > 0):
        assert len(solution) == len(data.get_variable_lower_bounds())
        assert np.all(
            solution
            >= (data.get_variable_lower_bounds() - integrality_tolerance)
        )
    if len(data.get_variable_upper_bounds() > 0):
        assert len(solution) == len(data.get_variable_upper_bounds())
        assert np.all(
            solution
            <= (data.get_variable_upper_bounds() + integrality_tolerance)
        )


def validate_constraint_sanity_per_row(
    data, solution, cost, abs_tolerance, rel_tolerance
):
    def combine_finite_abs_bounds(lower, upper):
        val = 0
        if np.isfinite(upper):
            val = max(val, abs(upper))
        if np.isfinite(lower):
            val = max(val, abs(lower))

        return val

    def get_violation(value, lower, upper):
        if value < lower:
            return lower - value
        elif value > upper:
            return value - upper
        else:
            return 0

    values = data.get_constraint_matrix_values()
    offsets = data.get_constraint_matrix_offsets()
    indices = data.get_constraint_matrix_indices()
    constraint_lower_bounds = data.get_constraint_lower_bounds()
    constraint_upper_bounds = data.get_constraint_upper_bounds()
    residual = np.zeros(len(constraint_lower_bounds))

    for i in range(len(offsets) - 1):
        for j in range(offsets[i], offsets[i + 1]):
            residual[i] += values[j] * solution[indices[j]]

    for i in range(len(residual)):
        tolerance = abs_tolerance + combine_finite_abs_bounds(
            constraint_lower_bounds[i],
            constraint_upper_bounds[i] * rel_tolerance,
        )
        violation = get_violation(
            residual[i], constraint_lower_bounds[i], constraint_upper_bounds[i]
        )

        assert violation <= tolerance


def validate_objective_sanity(data, solution, cost, tolerance):
    output = (data.get_objective_coefficients() * solution).sum()

    assert abs(output - cost) <= tolerance


def check_solution(data, setting, solution, cost):
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
        CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    )

    # check size of the solution matches variable size
    assert len(solution) == len(data.get_variable_types())

    validate_variable_bounds(data, setting, solution)

    abs_tolerance = setting.get_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE)
    abs_tolerance = abs_tolerance if abs_tolerance else 1e-4

    rel_tolerance = setting.get_parameter(CUOPT_RELATIVE_PRIMAL_TOLERANCE)
    rel_tolerance = rel_tolerance if rel_tolerance else 1e-6

    validate_constraint_sanity_per_row(
        data,
        solution,
        cost,
        abs_tolerance * 1e2,
        rel_tolerance,
    )

    validate_objective_sanity(data, solution, cost, 1e-4)
