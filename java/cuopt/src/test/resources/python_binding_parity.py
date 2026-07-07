#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python-side oracle for Java/Python cuOpt binding parity tests."""

import argparse
import json
import math
import sys

import numpy as np


PREFIX = "CUOPT_COMPARE "


def emit(name, value):
    print(f"{PREFIX}{name}={value}")


def emit_float(name, value):
    value = float(value)
    if math.isnan(value):
        emit(name, "nan")
    elif math.isinf(value):
        emit(name, "inf" if value > 0 else "-inf")
    else:
        emit(name, f"{value:.17g}")


def emit_float_array(name, values):
    emit(name, ",".join(f"{float(v):.17g}" for v in values))


def emit_int_array(name, values):
    emit(name, ",".join(str(int(v)) for v in values))


def emit_char_array(name, values):
    chars = []
    for value in values:
        if isinstance(value, bytes):
            chars.append(value.decode("ascii"))
        elif isinstance(value, np.bytes_):
            chars.append(bytes(value).decode("ascii"))
        else:
            chars.append(str(value))
    emit(name, ",".join(chars))


def as_float_array(values):
    return np.array(values, dtype=np.float64)


def as_int_array(values):
    return np.array(values, dtype=np.int32)


def as_char_array(values):
    return np.array(values, dtype="<U1")


def build_model(spec):
    from cuopt import linear_programming

    data_model = linear_programming.DataModel()
    data_model.set_csr_constraint_matrix(
        as_float_array(spec["csr_values"]),
        as_int_array(spec["csr_column_indices"]),
        as_int_array(spec["csr_row_offsets"]),
    )
    data_model.set_objective_coefficients(as_float_array(spec["objective_coefficients"]))
    data_model.set_objective_offset(spec["objective_offset"])
    if "objective_scaling_factor" in spec:
        data_model.set_objective_scaling_factor(spec["objective_scaling_factor"])
    data_model.set_maximize(spec["objective_sense"] == -1)
    if "quadratic_objective_values" in spec:
        data_model.set_quadratic_objective_matrix(
            as_float_array(spec["quadratic_objective_values"]),
            as_int_array(spec["quadratic_objective_column_indices"]),
            as_int_array(spec["quadratic_objective_row_offsets"]),
        )
    data_model.set_variable_lower_bounds(as_float_array(spec["variable_lower_bounds"]))
    data_model.set_variable_upper_bounds(as_float_array(spec["variable_upper_bounds"]))
    data_model.set_variable_types(as_char_array(spec["variable_types"]))
    if "variable_names" in spec:
        data_model.set_variable_names(np.array(spec["variable_names"], dtype=str))
    if "row_names" in spec:
        data_model.set_row_names(np.array(spec["row_names"], dtype=str))
    if "objective_name" in spec:
        data_model.set_objective_name(spec["objective_name"])
    if "problem_name" in spec:
        data_model.set_problem_name(spec["problem_name"])

    if "quadratic_constraint_values" in spec:
        data_model.add_quadratic_constraint(
            constraint_row_name=spec.get("quadratic_constraint_name", ""),
            linear_values=as_float_array(spec.get("quadratic_constraint_linear_values", [])),
            linear_indices=as_int_array(spec.get("quadratic_constraint_linear_indices", [])),
            rhs_value=spec["quadratic_constraint_rhs"],
            vals=as_float_array(spec["quadratic_constraint_values"]),
            rows=as_int_array(spec["quadratic_constraint_rows"]),
            cols=as_int_array(spec["quadratic_constraint_columns"]),
            sense=spec["quadratic_constraint_sense"],
        )

    if "constraint_lower_bounds" in spec:
        data_model.set_constraint_lower_bounds(as_float_array(spec["constraint_lower_bounds"]))
        data_model.set_constraint_upper_bounds(as_float_array(spec["constraint_upper_bounds"]))
    else:
        data_model.set_constraint_bounds(as_float_array(spec["rhs"]))
        data_model.set_row_types(as_char_array(spec["constraint_sense"]))

    return data_model


def build_settings(is_mip, is_qp):
    from cuopt import linear_programming

    settings = linear_programming.SolverSettings()
    settings.set_parameter("log_to_console", 0)
    settings.set_parameter("time_limit", 30.0)
    settings.set_parameter("random_seed", 1)
    if is_mip:
        settings.set_parameter("mip_determinism_mode", 1)
        settings.set_parameter("mip_absolute_gap", 1.0e-8)
        settings.set_parameter("mip_relative_gap", 1.0e-8)
    elif is_qp:
        settings.set_parameter("iteration_limit", 50)
    else:
        settings.set_parameter("method", 1)
        settings.set_parameter("pdlp_solver_mode", 0)
        settings.set_parameter("absolute_primal_tolerance", 1.0e-7)
        settings.set_parameter("relative_primal_tolerance", 1.0e-7)
        settings.set_parameter("absolute_dual_tolerance", 1.0e-7)
        settings.set_parameter("relative_dual_tolerance", 1.0e-7)
        settings.set_parameter("absolute_gap_tolerance", 1.0e-7)
        settings.set_parameter("relative_gap_tolerance", 1.0e-7)
    return settings


def emit_model(data_model, ranged):
    emit("model.num_variables", len(data_model.get_objective_coefficients()))
    emit("model.num_constraints", len(data_model.get_constraint_matrix_offsets()) - 1)
    emit("model.num_nonzeros", len(data_model.get_constraint_matrix_values()))
    emit("model.objective_sense", -1 if data_model.get_sense() else 1)
    emit_float("model.objective_offset", data_model.get_objective_offset())
    emit_float("model.objective_scaling_factor", data_model.get_objective_scaling_factor())
    emit_float_array("model.objective_coefficients", data_model.get_objective_coefficients())
    emit_float_array("model.csr_values", data_model.get_constraint_matrix_values())
    emit_int_array("model.csr_column_indices", data_model.get_constraint_matrix_indices())
    emit_int_array("model.csr_row_offsets", data_model.get_constraint_matrix_offsets())
    if not ranged:
        emit_char_array("model.constraint_sense", data_model.get_row_types())
        emit_float_array("model.rhs", data_model.get_constraint_bounds())
    else:
        emit_float_array(
            "model.constraint_lower_bounds",
            data_model.get_constraint_lower_bounds(),
        )
        emit_float_array(
            "model.constraint_upper_bounds",
            data_model.get_constraint_upper_bounds(),
        )
    emit_float_array("model.variable_lower_bounds", data_model.get_variable_lower_bounds())
    emit_float_array("model.variable_upper_bounds", data_model.get_variable_upper_bounds())
    emit_char_array("model.variable_types", data_model.get_variable_types())
    emit_char_array("model.variable_names", data_model.get_variable_names())
    emit_char_array("model.row_names", data_model.get_row_names())
    if len(data_model.get_quadratic_objective_offsets()) != 0:
        emit_float_array("model.quadratic_objective_values", data_model.get_quadratic_objective_values())
        emit_int_array("model.quadratic_objective_column_indices", data_model.get_quadratic_objective_indices())
        emit_int_array("model.quadratic_objective_row_offsets", data_model.get_quadratic_objective_offsets())
    emit("model.objective_name", data_model.get_objective_name())
    emit("model.problem_name", data_model.get_problem_name())
    quadratic_constraints = data_model.get_quadratic_constraints()
    emit("model.quadratic_constraint_count", len(quadratic_constraints))
    for index, constraint in enumerate(quadratic_constraints):
        prefix = f"model.quadratic_constraint.{index}"
        emit(prefix + ".row_index", constraint["constraint_row_index"])
        emit(prefix + ".row_name", constraint["constraint_row_name"])
        emit(prefix + ".sense", constraint["constraint_row_type"])
        emit_float_array(prefix + ".linear_values", constraint["linear_values"])
        emit_int_array(prefix + ".linear_indices", constraint["linear_indices"])
        emit_float(prefix + ".rhs", constraint["rhs_value"])
        emit_int_array(prefix + ".rows", constraint["rows"])
        emit_int_array(prefix + ".columns", constraint["cols"])
        emit_float_array(prefix + ".values", constraint["vals"])


def emit_solution(solution):
    problem_category = solution.get_problem_category()
    is_mip = problem_category.name != "LP"
    emit("solution.is_mip", str(is_mip).lower())
    emit("solution.problem_category", problem_category.name)
    emit("solution.status", int(solution.get_termination_status()))
    emit("solution.status_name", solution.get_termination_status().name)
    emit("solution.error_status", int(solution.get_error_status()))
    emit("solution.error_message", solution.get_error_message())
    emit_float("solution.solve_time", solution.get_solve_time())
    emit_float("solution.objective", solution.get_primal_objective())
    emit_float_array("solution.primal", solution.get_primal_solution())

    try:
        emit_float_array("solution.dual", solution.get_dual_solution())
        emit("solution.dual_unavailable", "false")
    except AttributeError:
        emit("solution.dual_unavailable", "true")

    try:
        emit_float("solution.dual_objective", solution.get_dual_objective())
        emit("solution.dual_objective_unavailable", "false")
    except AttributeError:
        emit("solution.dual_objective_unavailable", "true")

    try:
        emit_float_array("solution.reduced_cost", solution.get_reduced_cost())
        emit("solution.reduced_cost_unavailable", "false")
    except AttributeError:
        emit("solution.reduced_cost_unavailable", "true")

    try:
        milp_stats = solution.get_milp_stats()
        emit("solution.mip_stats_unavailable", "false")
        emit_float("solution.mip_gap", milp_stats["mip_gap"])
        emit_float("solution.solution_bound", milp_stats["solution_bound"])
        emit_float("solution.mip_presolve_time", milp_stats["presolve_time"])
        emit_float(
            "solution.max_constraint_violation",
            milp_stats["max_constraint_violation"],
        )
        emit_float("solution.max_int_violation", milp_stats["max_int_violation"])
        emit_float(
            "solution.max_variable_bound_violation",
            milp_stats["max_variable_bound_violation"],
        )
        emit("solution.num_nodes", int(milp_stats["num_nodes"]))
        emit(
            "solution.num_simplex_iterations",
            int(milp_stats["num_simplex_iterations"]),
        )
    except AttributeError:
        emit("solution.mip_stats_unavailable", "true")

    try:
        lp_stats = solution.get_lp_stats()
        emit("solution.lp_stats_unavailable", "false")
        emit_float("solution.lp_primal_residual", lp_stats["primal_residual"])
        emit_float("solution.lp_dual_residual", lp_stats["dual_residual"])
        emit_float("solution.lp_gap", lp_stats["gap"])
        emit("solution.lp_iterations", int(lp_stats["nb_iterations"]))
        emit("solution.solved_by", int(solution.get_solved_by()))
    except AttributeError:
        emit("solution.lp_stats_unavailable", "true")


def solve_case(spec):
    from cuopt import linear_programming

    data_model = build_model(spec)
    emit_model(data_model, "constraint_lower_bounds" in spec)
    is_mip = any(value in ("I", "S") for value in spec["variable_types"])
    is_qp = "quadratic_objective_values" in spec
    settings = build_settings(is_mip, is_qp)
    solution = linear_programming.Solve(data_model, settings)
    emit_solution(solution)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("case_file", nargs="?")
    args = parser.parse_args()

    if args.probe:
        import cuopt  # noqa: F401

        emit("probe", "ok")
        return 0

    if not args.case_file:
        parser.error("case_file is required unless --probe is set")

    with open(args.case_file, encoding="utf-8") as handle:
        solve_case(json.load(handle))
    return 0


if __name__ == "__main__":
    sys.exit(main())
