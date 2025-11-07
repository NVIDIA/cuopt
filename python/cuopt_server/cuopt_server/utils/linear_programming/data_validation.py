# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from fastapi import HTTPException


def is_empty(value):
    if value is None or len(value) == 0:
        return True
    else:
        return False


def validate_csr_matrix(csr_data):
    if np.min(csr_data.indices) < 0:
        return (False, "indices values must be greater than or equal to 0")

    if np.min(csr_data.offsets) < 0:
        return (False, "offset values must be greater than or equal to 0")

    if len(csr_data.indices) != len(csr_data.values):
        return (
            False,
            "Length of values array must be equal to indices array",
        )

    return (True, "Valid CSR Matrix")


def validate_constraint_bounds(constraint_bounds):
    if is_empty(constraint_bounds.upper_bounds) or is_empty(
        constraint_bounds.lower_bounds
    ):
        if is_empty(constraint_bounds.types):
            return (
                False,
                "Either Row types or upper and lower bounds must be provided",
            )
        else:
            if any(
                [
                    row_type not in ["E", "G", "L"]
                    for row_type in constraint_bounds.types
                ]
            ):
                return (
                    False,
                    "Row types must be E, L or G",
                )
    elif not is_empty(constraint_bounds.types):
        return (
            False,
            "Both row types and upper and lower bounds can not be provided",
        )
    elif not len(constraint_bounds.upper_bounds) == len(
        constraint_bounds.lower_bounds
    ):
        return (
            False,
            "Size of constraint upper bounds must be same as constaint lower bounds",  # noqa
        )
    return (True, "Valid constraint bounds")


def validate_objective_data(objective_data):
    return (True, "Valid objective data")


def validate_variable_bounds(LP_data):
    variable_bounds = LP_data.variable_bounds
    coeff = LP_data.objective_data.coefficients
    if variable_bounds is not None:
        if (
            variable_bounds.upper_bounds is not None
            and variable_bounds.lower_bounds is not None
        ):
            if not len(variable_bounds.upper_bounds) == len(
                variable_bounds.lower_bounds
            ):
                return (
                    False,
                    "Size of variable upper bounds must be same as variable lower bounds",  # noqa
                )
        elif variable_bounds.upper_bounds is not None:
            if not len(variable_bounds.upper_bounds) == len(coeff):
                return (
                    False,
                    "Size of variable upper bounds must be same as size of objective coefficients",  # noqa
                )
        elif variable_bounds.lower_bounds is not None:
            if not len(variable_bounds.lower_bounds) == len(coeff):
                return (
                    False,
                    "Size of variable lower bounds must be same as size of objective coefficients",  # noqa
                )
    return (True, "Valid variable bounds")


def validate_initial_solution(LP_data):
    initial_solution = LP_data.initial_solution
    objective_data = LP_data.objective_data
    constraint_bounds = LP_data.constraint_bounds
    if initial_solution.primal is not None:
        if not len(initial_solution.primal) == len(
            objective_data.coefficients
        ):
            return (
                False,
                "Size of initial solution must be same as size of objective coefficients",  # noqa
            )
    if initial_solution.dual is not None:
        if not len(initial_solution.dual) == len(constraint_bounds.bounds):
            return (
                False,
                "Size of initial dual solution must be same as size of constraint bounds",  # noqa
            )
    return (True, "Valid initial solution")


def check_status(is_valid):
    if not is_valid[0]:
        raise HTTPException(status_code=400, detail=f"{is_valid[1]}")


def validate_LP_data(LP_data):
    check_status(validate_csr_matrix(LP_data.csr_constraint_matrix))
    check_status(validate_constraint_bounds(LP_data.constraint_bounds))
    check_status(validate_variable_bounds(LP_data))
    check_status(validate_objective_data(LP_data.objective_data))
    check_status(validate_initial_solution(LP_data))
