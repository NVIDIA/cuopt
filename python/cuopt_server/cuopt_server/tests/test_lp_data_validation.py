# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from cuopt_server.utils.linear_programming.data_validation import (
    validate_variable_bounds,
)


def _problem(upper, lower, coefficients):
    return SimpleNamespace(
        variable_bounds=SimpleNamespace(
            upper_bounds=upper,
            lower_bounds=lower,
        ),
        objective_data=SimpleNamespace(coefficients=coefficients),
    )


def test_both_bounds_must_match_the_objective():
    valid, message = validate_variable_bounds(
        _problem([1.0, 2.0], [0.0, 0.0], [1.0, 1.0, 1.0])
    )
    assert valid is False
    assert "objective coefficients" in message


def test_matching_bounds_stay_valid():
    valid, message = validate_variable_bounds(
        _problem([1.0, 2.0], [0.0, 0.0], [1.0, 1.0])
    )
    assert valid is True
    assert message == "Valid variable bounds"


def test_one_sided_upper_bound_still_matches_the_objective():
    valid, _message = validate_variable_bounds(
        _problem([1.0], None, [1.0, 1.0])
    )
    assert valid is False


def test_unequal_upper_and_lower_bounds_are_still_rejected():
    valid, message = validate_variable_bounds(
        _problem([1.0, 2.0], [0.0], [1.0, 1.0])
    )
    assert valid is False
    assert "lower bounds" in message
