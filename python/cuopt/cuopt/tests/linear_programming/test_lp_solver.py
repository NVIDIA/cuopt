# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import os
from enum import IntEnum

from cuopt.linear_programming import Read
import numpy as np
import pytest

from cuopt.linear_programming import (
    data_model,
    solver,
    solver_settings,
)
from cuopt.linear_programming.solver.solver_parameters import (
    solver_params,
    CUOPT_ABSOLUTE_DUAL_TOLERANCE,
    CUOPT_ABSOLUTE_GAP_TOLERANCE,
    CUOPT_ABSOLUTE_PRIMAL_TOLERANCE,
    CUOPT_DUAL_INFEASIBLE_TOLERANCE,
    CUOPT_INFEASIBILITY_DETECTION,
    CUOPT_ITERATION_LIMIT,
    CUOPT_METHOD,
    CUOPT_MIP_HEURISTICS_ONLY,
    CUOPT_PDLP_SOLVER_MODE,
    CUOPT_PRIMAL_INFEASIBLE_TOLERANCE,
    CUOPT_RELATIVE_DUAL_TOLERANCE,
    CUOPT_RELATIVE_GAP_TOLERANCE,
    CUOPT_RELATIVE_PRIMAL_TOLERANCE,
    CUOPT_SOLUTION_FILE,
    CUOPT_TIME_LIMIT,
    CUOPT_USER_PROBLEM_FILE,
    CUOPT_PRESOLVE,
)
from cuopt.linear_programming.solver.solver_wrapper import (
    ErrorStatus,
    LPTerminationStatus,
)
from cuopt.linear_programming.solver_settings import (
    PDLPSolverMode,
    SolverMethod,
    SolverSettings,
)
from cuopt.linear_programming.problem import (
    Problem,
    CONTINUOUS,
    MINIMIZE,
)

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")


def test_solver():
    data_model_obj = data_model.DataModel()

    A_values = np.array([1.0, 1.0])
    A_indices = np.array([0, 0])
    A_offsets = np.array([0, 1, 2])
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    b = np.array([1.0, 1.0])
    data_model_obj.set_constraint_bounds(b)

    c = np.array([1.0])
    data_model_obj.set_objective_coefficients(c)

    row_types = np.array(["L", "L"])

    data_model_obj.set_row_types(row_types)

    settings = solver_settings.SolverSettings()
    settings.set_optimality_tolerance(1e-2)
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    # FIXME: Stable3 infinite-loops on this sample trivial problem
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    settings.set_parameter(CUOPT_PRESOLVE, 0)

    solution = solver.Solve(data_model_obj, settings)
    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_solution()[0] == pytest.approx(0.0)
    assert solution.get_lp_stats()["primal_residual"] == pytest.approx(0.0)
    assert solution.get_lp_stats()["dual_residual"] == pytest.approx(0.0)
    assert solution.get_primal_objective() == pytest.approx(0.0)
    assert solution.get_dual_objective() == pytest.approx(0.0)
    assert solution.get_lp_stats()["gap"] == pytest.approx(0.0)
    assert solution.get_solved_by() == SolverMethod.PDLP


def test_parser_and_solver():
    file_path = RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-1.mps"
    data_model_obj = Read(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_optimality_tolerance(1e-2)
    solution = solver.Solve(data_model_obj, settings)
    assert solution.get_termination_reason() == "Optimal"


def test_set_get_fields():
    data_model_obj = data_model.DataModel()

    A = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.int32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    c = np.array([7.0, 8.0, 9.0], dtype=np.float64)
    var_lb = np.array([0.0, 0.1, 0.2], dtype=np.float64)
    var_ub = np.array([1.0, 1.1, 1.2], dtype=np.float64)
    con_lb = np.array([0.5, 0.6, 0.7], dtype=np.float64)
    con_ub = np.array([1.5, 1.6, 1.7], dtype=np.float64)
    row_types = np.array(["L", "G", "E"])

    data_model_obj.set_csr_constraint_matrix(A, indices, indices)
    # Test A_value_
    assert 1.0 == data_model_obj.get_constraint_matrix_values()[0]
    assert 2.0 == data_model_obj.get_constraint_matrix_values()[1]
    assert 3.0 == data_model_obj.get_constraint_matrix_values()[2]

    # Test A_indices_
    assert 0 == data_model_obj.get_constraint_matrix_indices()[0]
    assert 1 == data_model_obj.get_constraint_matrix_indices()[1]
    assert 2 == data_model_obj.get_constraint_matrix_indices()[2]

    # Test A_offsets_
    assert 0 == data_model_obj.get_constraint_matrix_offsets()[0]
    assert 1 == data_model_obj.get_constraint_matrix_offsets()[1]
    assert 2 == data_model_obj.get_constraint_matrix_offsets()[2]

    # Test b_
    data_model_obj.set_constraint_bounds(b)
    assert 4.0 == data_model_obj.get_constraint_bounds()[0]
    assert 5.0 == data_model_obj.get_constraint_bounds()[1]
    assert 6.0 == data_model_obj.get_constraint_bounds()[2]

    # Test c_
    data_model_obj.set_objective_coefficients(c)
    assert 7.0 == data_model_obj.get_objective_coefficients()[0]
    assert 8.0 == data_model_obj.get_objective_coefficients()[1]
    assert 9.0 == data_model_obj.get_objective_coefficients()[2]

    # Test variable_lower_bounds_
    data_model_obj.set_variable_lower_bounds(var_lb)
    assert 0.0 == data_model_obj.get_variable_lower_bounds()[0]
    assert 0.1 == data_model_obj.get_variable_lower_bounds()[1]
    assert 0.2 == data_model_obj.get_variable_lower_bounds()[2]

    # Test variable_upper_bounds_
    data_model_obj.set_variable_upper_bounds(var_ub)
    assert 1.0 == data_model_obj.get_variable_upper_bounds()[0]
    assert 1.1 == data_model_obj.get_variable_upper_bounds()[1]
    assert 1.2 == data_model_obj.get_variable_upper_bounds()[2]

    # Test constraint_lower_bounds_
    data_model_obj.set_constraint_lower_bounds(con_lb)
    assert 0.5 == data_model_obj.get_constraint_lower_bounds()[0]
    assert 0.6 == data_model_obj.get_constraint_lower_bounds()[1]
    assert 0.7 == data_model_obj.get_constraint_lower_bounds()[2]

    # Test row_types
    data_model_obj.set_row_types(row_types)
    assert "L" == data_model_obj.get_row_types()[0]
    assert "G" == data_model_obj.get_row_types()[1]
    assert "E" == data_model_obj.get_row_types()[2]

    # Test constraint_upper_bounds_
    data_model_obj.set_constraint_upper_bounds(con_ub)
    assert 1.5 == data_model_obj.get_constraint_upper_bounds()[0]
    assert 1.6 == data_model_obj.get_constraint_upper_bounds()[1]
    assert 1.7 == data_model_obj.get_constraint_upper_bounds()[2]

    # Test objective_scaling_factor_
    data_model_obj.set_objective_scaling_factor(1.5)
    assert 1.5 == data_model_obj.get_objective_scaling_factor()

    # Test objective_offset_
    data_model_obj.set_objective_offset(0.5)
    assert 0.5 == data_model_obj.get_objective_offset()

    # Test initial_primal_solution
    data_model_obj.set_initial_primal_solution(con_ub)
    assert 1.5 == data_model_obj.get_initial_primal_solution()[0]
    assert 1.6 == data_model_obj.get_initial_primal_solution()[1]
    assert 1.7 == data_model_obj.get_initial_primal_solution()[2]

    # Test initial_dual_solution
    data_model_obj.set_initial_dual_solution(con_ub)
    assert 1.5 == data_model_obj.get_initial_dual_solution()[0]
    assert 1.6 == data_model_obj.get_initial_dual_solution()[1]
    assert 1.7 == data_model_obj.get_initial_dual_solution()[2]

    # Test set maximize
    data_model_obj.set_maximize(True)
    assert data_model_obj.get_sense()


def _cuopt_commented_dump_to_loadable(src_path, dst_path, param_names):
    """Turn ``dump_parameters_to_file`` output (commented ``# name = value``) into a loadable file.

    C++ ``load_parameters_from_file`` skips commented lines; the dump format is
    intentionally commented, so tests strip the leading ``#`` for assignment lines.
    """
    names_by_len = sorted(param_names, key=len, reverse=True)
    with (
        open(src_path, encoding="utf-8") as src,
        open(dst_path, "w", encoding="utf-8") as dst,
    ):
        for line in src:
            stripped = line.strip()
            for name in names_by_len:
                prefix = f"# {name} = "
                if stripped.startswith(prefix):
                    dst.write(f"{name} = {stripped[len(prefix) :]}\n")
                    break


def _assert_solver_param_equal(name, expected, got):
    """Compare get_parameter values across dump/load; enums and floats may differ in type."""
    exp = int(expected) if isinstance(expected, IntEnum) else expected
    g = int(got) if isinstance(got, IntEnum) else got
    if isinstance(exp, float) and isinstance(g, float):
        if math.isnan(exp) and math.isnan(g):
            return
        assert g == pytest.approx(exp), (name, got, expected)
    else:
        assert g == exp, (name, got, expected)


def _other_float_in_range(current, lo, hi):
    """Pick a float in [lo, hi] that differs from *current* (for bounded C++ params)."""
    cur = float(current)
    candidates = (
        lo,
        hi,
        lo + 0.25 * (hi - lo),
        hi - 0.25 * (hi - lo),
        min(hi, max(lo, cur * 0.5)),
        min(hi, max(lo, cur * 2.0)),
    )
    for candidate in candidates:
        if abs(candidate - cur) > max(1e-15, 1e-9 * max(abs(cur), 1.0)):
            return candidate
    return lo if abs(cur - hi) < abs(cur - lo) else hi


def _float_bounds_for_param(name):
    """Return (min, max) for float params with tight C++ validation, or (None, None)."""
    if name == "barrier_step_scale":
        return 0.5, 0.9999
    if name in ("mip_cut_min_orthogonality",):
        return 0.0, 1.0
    if name.endswith("_time_ratio") or name.endswith("_fix_rate"):
        return 0.0, 1.0
    if "tolerance" in name or name.endswith("tolerance"):
        return 0.0, 0.1
    return None, None


def _non_default_solver_param_value(name, current):
    """Pick a valid but non-default value for each registered solver parameter."""
    if name.startswith("mip_hyper"):
        return current
    if name in ("user_problem_file", "solution_file"):
        return ""
    if name == "num_gpus":
        return 2 if int(current) == 1 else 1
    if name == "time_limit":
        return 3600.0
    if name == "iteration_limit":
        return 9_999_999
    if name == "method":
        cur_i = int(current) if current is not None else -1
        return (
            SolverMethod.DualSimplex
            if cur_i != int(SolverMethod.DualSimplex)
            else SolverMethod.PDLP
        )
    if name == "pdlp_solver_mode":
        cur_i = int(current) if current is not None else -1
        return (
            PDLPSolverMode.Fast1
            if cur_i != int(PDLPSolverMode.Fast1)
            else PDLPSolverMode.Stable2
        )
    if name == "presolve":
        return 0 if int(current) == 1 else 1
    if name == "pdlp_precision":
        return 1 if int(current) == 0 else 0
    if name == "mip_objective_step":
        return 0 if int(current) == 1 else 1
    if isinstance(current, bool):
        return not current
    if isinstance(current, str):
        low = current.lower()
        if low == "true":
            return False
        if low == "false":
            return True
        return current + "_x" if current else "1"
    if isinstance(current, float):
        if not math.isfinite(current):
            return 7200.0
        if abs(current) < 1e-30:
            return 1e-2
        lo, hi = _float_bounds_for_param(name)
        if lo is not None and hi is not None:
            return _other_float_in_range(current, lo, hi)
        return (
            float(current) * 2.0
            if abs(current) < 1.0
            else float(current) * 0.5
        )
    if isinstance(current, int):
        if int(current) > 1:
            return int(current) - 1
        return int(current) + 1
    return current


def test_solver_settings_basic():
    settings = solver_settings.SolverSettings()

    tolerance_value = 1e-5

    # Setting tolerances
    settings.set_parameter(CUOPT_ABSOLUTE_DUAL_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_RELATIVE_DUAL_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_RELATIVE_PRIMAL_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_ABSOLUTE_GAP_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_RELATIVE_GAP_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_PRIMAL_INFEASIBLE_TOLERANCE, tolerance_value)
    settings.set_parameter(CUOPT_DUAL_INFEASIBLE_TOLERANCE, tolerance_value)

    # Getting and asserting tolerances
    assert settings.get_parameter(CUOPT_ABSOLUTE_DUAL_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_RELATIVE_DUAL_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_ABSOLUTE_PRIMAL_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_RELATIVE_PRIMAL_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_ABSOLUTE_GAP_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_RELATIVE_GAP_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_PRIMAL_INFEASIBLE_TOLERANCE) == 1e-5
    assert settings.get_parameter(CUOPT_DUAL_INFEASIBLE_TOLERANCE) == 1e-5

    assert settings.get_parameter(CUOPT_TIME_LIMIT) == float("inf")

    settings.set_parameter(CUOPT_ITERATION_LIMIT, 10)
    settings.set_parameter(CUOPT_TIME_LIMIT, 10.2)

    assert settings.get_parameter(CUOPT_ITERATION_LIMIT) == 10
    assert settings.get_parameter(CUOPT_TIME_LIMIT) == 10.2

    settings.set_parameter(CUOPT_INFEASIBILITY_DETECTION, False)
    assert not settings.get_parameter(CUOPT_INFEASIBILITY_DETECTION)

    assert settings.get_parameter(CUOPT_PDLP_SOLVER_MODE) == int(
        PDLPSolverMode.Stable3
    )

    with pytest.raises(ValueError):
        settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, 10)
        # Need to trigger a solver since solver_settings input checking is done
        # on the cpp side once Solve is called
        file_path = (
            RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-1.mps"
        )
        solver.Solve(Read(file_path), settings)

    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Methodical1)
    assert settings.get_parameter(CUOPT_PDLP_SOLVER_MODE) == int(
        PDLPSolverMode.Methodical1
    )


def test_solver_settings(tmp_path):
    """Push every registered parameter to the C++ layer via set_c_solver_settings."""
    settings = solver_settings.SolverSettings()
    for name in list(set(solver_params)):
        current = settings.get_parameter(name)
        new_val = _non_default_solver_param_value(name, current)
        settings.set_parameter(name, new_val)

    expected_by_name = {
        name: settings.get_parameter(name) for name in solver_params
    }

    dump_path = tmp_path / "solver_settings_dump.config"
    load_path = tmp_path / "solver_settings_load.config"
    assert settings.dump_parameters_to_file(
        str(dump_path), hyperparameters_only=False
    )
    _cuopt_commented_dump_to_loadable(
        str(dump_path), str(load_path), solver_params
    )
    reloaded = solver_settings.SolverSettings()
    reloaded.load_parameters_from_file(str(load_path))
    for name in solver_params:
        _assert_solver_param_equal(
            name,
            expected_by_name[name],
            reloaded.get_parameter(name),
        )

    reloaded.set_c_solver_settings()
    data_model_obj = data_model.DataModel()
    A_values = np.array([1.0, 1.0])
    A_indices = np.array([0, 0])
    A_offsets = np.array([0, 1, 2])
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
    data_model_obj.set_constraint_bounds(np.array([1.0, 1.0]))
    data_model_obj.set_objective_coefficients(np.array([1.0]))
    data_model_obj.set_row_types(np.array(["L", "L"]))

    solution = solver.Solve(data_model_obj, reloaded)
    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(0.0)
    assert solution.get_primal_solution()[0] == pytest.approx(0.0)


def test_check_data_model_validity():
    data_model_obj = data_model.DataModel()

    # Test if exception is thrown when A_CSR is not set
    solution = solver.Solve(data_model_obj)
    assert solution.get_error_status() == ErrorStatus.ValidationError

    # Set A_CSR_matrix with np.array
    A_values = np.array([1.0], dtype=np.float64)
    A_indices = np.array([0], dtype=np.int32)
    A_offsets = np.array([0, 1], dtype=np.int32)
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    # Test if exception is thrown when b is not set
    solution = solver.Solve(data_model_obj)
    assert solution.get_error_status() == ErrorStatus.ValidationError

    # Set b with np.array
    b = np.array([1.0], dtype=np.float64)
    data_model_obj.set_constraint_bounds(b)

    # Test if exception is thrown when c is not set
    solution = solver.Solve(data_model_obj)
    assert solution.get_error_status() == ErrorStatus.ValidationError

    # Set c with np.array
    c = np.array([1.0], dtype=np.float64)
    data_model_obj.set_objective_coefficients(c)

    # Set maximize
    data_model_obj.set_maximize(True)

    # Test if exception is thrown when maximize is set to true
    solution = solver.Solve(data_model_obj)
    assert solution.get_error_status() == ErrorStatus.ValidationError

    # Set maximize to correct value
    data_model_obj.set_maximize(False)

    # Test if exception is thrown when row_type is not set
    solution = solver.Solve(data_model_obj)
    assert solution.get_error_status() == ErrorStatus.ValidationError

    # Set row_type with np.array
    row_type = np.array(["E"])
    data_model_obj.set_row_types(row_type)

    # Test if no exception is thrown when row_type is set
    solver.Solve(data_model_obj)

    # Set constraint_lower_bounds with np.array
    constraint_lower_bounds = np.array([1.0], dtype=np.float64)
    data_model_obj.set_constraint_lower_bounds(constraint_lower_bounds)

    # Set constraint_upper_bounds with np.array
    constraint_upper_bounds = np.array([1.0], dtype=np.float64)
    data_model_obj.set_constraint_upper_bounds(constraint_upper_bounds)

    # Test if no exception is thrown when upper constraints bounds are not set
    solver.Solve(data_model_obj)


def test_parse_var_names():
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/afiro_original.mps"
    )
    data_model_obj = Read(file_path)

    expected_names = [
        "X01",
        "X02",
        "X03",
        "X04",
        "X06",
        "X07",
        "X08",
        "X09",
        "X10",
        "X11",
        "X12",
        "X13",
        "X14",
        "X15",
        "X16",
        "X22",
        "X23",
        "X24",
        "X25",
        "X26",
        "X28",
        "X29",
        "X30",
        "X31",
        "X32",
        "X33",
        "X34",
        "X35",
        "X36",
        "X37",
        "X38",
        "X39",
    ]

    for i, name in enumerate(data_model_obj.get_variable_names()):
        assert expected_names[i] == name

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    settings.set_parameter(CUOPT_PRESOLVE, 0)
    solution = solver.Solve(data_model_obj, settings)

    expected_dict = {
        "X01": 80.00603991232295,
        "X02": 25.52673622717911,
        "X03": 54.498387438550935,
        "X04": 0.0,
        "X06": 73.04802363832049,
        "X07": 0.0,
        "X08": 0.0,
        "X09": 0.0,
        "X10": 0.0,
        "X11": 0.0,
        "X12": 0.0,
        "X13": 0.0,
        "X14": 18.232656093528156,
        "X15": 0.0,
        "X16": 0.0,
        "X22": 499.9879512761402,
        "X23": 475.85273137206457,
        "X24": 24.097841116452646,
        "X25": 0.0,
        "X26": 0.0,
        "X28": 0.0,
        "X29": 0.0,
        "X30": 0.0,
        "X31": 0.0,
        "X32": 0.0,
        "X33": 0.0,
        "X34": 0.0,
        "X35": 0.0,
        "X36": 339.88604763129206,
        "X37": 25.615058891374325,
        "X38": 0.0,
        "X39": 0.0,
    }

    assert len(expected_dict) == len(solution.get_vars())

    for key in expected_dict:
        assert key in solution.get_vars()
        assert expected_dict[key] == pytest.approx(
            solution.get_vars()[key], rel=1e-4
        )


def test_parser_and_batch_solver():
    data_model_list = []
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/afiro_original.mps"
    )

    nb_solves = 5

    for i in range(nb_solves):
        data_model_list.append(Read(file_path))

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_optimality_tolerance(1e-4)

    # Call BatchSolve
    batch_solution, solve_time = solver.BatchSolve(data_model_list, settings)

    # Call Solve on each individual data model object
    individual_solutions = []
    for i in range(nb_solves):
        individual_solution = solver.Solve(Read(file_path), settings)
        individual_solutions.append(individual_solution)

    # Verify that the results are the same
    for i in range(nb_solves):
        assert (
            batch_solution[i].get_termination_status()
            == individual_solutions[i].get_termination_status()
        )


def test_warm_start():
    file_path = RAPIDS_DATASET_ROOT_DIR + "/linear_programming/a2864/a2864.mps"
    data_model_obj = Read(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_PDLP_SOLVER_MODE, PDLPSolverMode.Stable2)
    settings.set_optimality_tolerance(1e-3)
    settings.set_parameter(CUOPT_INFEASIBILITY_DETECTION, False)
    settings.set_parameter(CUOPT_PRESOLVE, 0)

    # Solving from scratch until 1e-3
    solution = solver.Solve(data_model_obj, settings)
    iterations_first_solve = solution.get_lp_stats()["nb_iterations"]

    # Solving until 1e-2 to use the result as a warm start
    settings.set_optimality_tolerance(1e-2)
    solution2 = solver.Solve(data_model_obj, settings)
    iterations_second_solve = solution2.get_lp_stats()["nb_iterations"]

    # Solving until 1e-3 using the previous state as a warm start
    settings.set_optimality_tolerance(1e-3)
    settings.set_pdlp_warm_start_data(solution2.get_pdlp_warm_start_data())

    solution3 = solver.Solve(data_model_obj, settings)
    iterations_third_solve = solution3.get_lp_stats()["nb_iterations"]

    assert (
        iterations_third_solve + iterations_second_solve
        == iterations_first_solve
    )

    # Should raise an exception as problems are different
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/afiro_original.mps"
    )

    data_model_obj_different = Read(file_path)
    with pytest.raises(Exception, match="Invalid PDLPWarmStart data"):
        solver.Solve(data_model_obj_different, settings)

    # Should raise an exception
    data_model_list = [data_model_obj, data_model_obj]
    with pytest.raises(
        Exception, match="Cannot use warmstart data with Batch Solve"
    ):
        solver.BatchSolve(data_model_list, settings)


def test_solved_by():
    # maximize   5*xs + 20*xl
    # subject to  1*xs +  3*xl <= 200
    #             3*xs +  2*xl <= 160

    data_model_obj = data_model.DataModel()

    A_values = np.array([1.0, 3.0, 3.0, 2.0])
    A_indices = np.array([0, 1, 0, 1])
    A_offsets = np.array([0, 2, 4])
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    b = np.array([200.0, 160.0])
    data_model_obj.set_constraint_bounds(b)

    c = np.array([5.0, 20.0])
    data_model_obj.set_objective_coefficients(c)

    row_types = np.array(["L", "L"])

    data_model_obj.set_row_types(row_types)
    data_model_obj.set_maximize(True)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    settings.set_parameter(CUOPT_PRESOLVE, 0)

    solution = solver.Solve(data_model_obj, settings)
    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1333.33, 2)
    assert solution.get_solved_by() == SolverMethod.Barrier


def test_heuristics_only():
    file_path = RAPIDS_DATASET_ROOT_DIR + "/mip/swath1.mps"
    data_model_obj = Read(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_MIP_HEURISTICS_ONLY, True)
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)

    solution = solver.Solve(data_model_obj, settings)

    lower_bound = solution.get_milp_stats()["solution_bound"]

    settings.set_parameter(CUOPT_MIP_HEURISTICS_ONLY, False)
    settings.set_parameter(CUOPT_TIME_LIMIT, 30)

    solution = solver.Solve(data_model_obj, settings)

    assert solution.get_milp_stats()["solution_bound"] > lower_bound


def test_bound_in_maximization():
    data_model_obj = data_model.DataModel()

    num_items = 8
    max_weight = 102.0
    value = [15, 100, 90, 60, 40, 15, 10, 1]
    weight = [2, 20, 20, 30, 40, 30, 60, 10]

    # maximize  sum_i value[i] * take[i]
    #           sum_i weight[i] * take[i] <= max_weight
    #           take[i] binary for all i

    c = np.array(value, dtype=np.float64)
    data_model_obj.set_objective_coefficients(c)
    data_model_obj.set_maximize(True)

    A_values = np.array(weight, dtype=np.float64)
    A_indices = np.array([j for j in range(num_items)], dtype=np.int32)
    A_offsets = np.array([0, num_items], dtype=np.int32)
    data_model_obj.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    b = np.array([max_weight])
    data_model_obj.set_constraint_bounds(b)

    row_types = np.array(["L"])
    data_model_obj.set_row_types(row_types)

    var_types = np.array(["I" for i in range(num_items)])
    data_model_obj.set_variable_types(var_types)

    lower_bounds = np.array([0.0 for i in range(num_items)], dtype=np.float64)
    upper_bounds = np.array([1.0 for i in range(num_items)], dtype=np.float64)
    data_model_obj.set_variable_lower_bounds(lower_bounds)
    data_model_obj.set_variable_upper_bounds(upper_bounds)

    settings = solver_settings.SolverSettings()
    settings.set_optimality_tolerance(1e-6)
    settings.set_parameter(CUOPT_TIME_LIMIT, 10)
    solution = solver.Solve(data_model_obj, settings)

    upper_bound = solution.get_milp_stats()["solution_bound"]
    assert upper_bound == pytest.approx(280, 1e-6)
    assert solution.get_primal_objective() == pytest.approx(280, 1e-6)


def test_write_files():
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/afiro_original.mps"
    )
    data_model_obj = Read(file_path)

    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.DualSimplex)
    settings.set_parameter(CUOPT_USER_PROBLEM_FILE, "afiro_out.mps")
    settings.set_parameter(CUOPT_SOLUTION_FILE, "afiro.sol")

    solver.Solve(data_model_obj, settings)

    assert os.path.isfile("afiro_out.mps")

    afiro = Read("afiro_out.mps")
    os.remove("afiro_out.mps")

    settings.set_parameter(CUOPT_USER_PROBLEM_FILE, "")
    settings.set_parameter(CUOPT_SOLUTION_FILE, "afiro.sol")

    solution = solver.Solve(afiro, settings)

    assert solution.get_termination_status() == LPTerminationStatus.Optimal
    assert solution.get_primal_objective() == pytest.approx(-464.7531)

    assert os.path.isfile("afiro.sol")

    with open("afiro.sol") as f:
        for line in f:
            if "X01" in line:
                assert float(line.split()[-1]) == pytest.approx(80)

    os.remove("afiro.sol")


def test_unbounded_problem():
    problem = Problem("unbounded")
    x = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name="x")
    y = problem.addVariable(lb=0.0, vtype=CONTINUOUS, name="y")

    problem.addConstraint(-1 * x + 2 * y <= 0, name="c1")

    problem.setObjective(-1 * x - 1 * y, sense=MINIMIZE)

    settings = SolverSettings()

    problem.solve(settings)

    assert problem.Status.name == "UnboundedOrInfeasible"
