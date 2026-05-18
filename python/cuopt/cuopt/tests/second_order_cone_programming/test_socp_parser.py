# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for SOCP / QCMATRIX MPS I/O: writeMPS, ParseMps, and external dataset parsing.
"""

import os

import numpy as np
import pytest

from cuopt.linear_programming import data_model, mps_parser, solver, solver_settings
from cuopt.linear_programming.solver.solver_parameters import (
    CUOPT_METHOD,
    CUOPT_PRESOLVE,
)
from cuopt.linear_programming.solver_settings import SolverMethod

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.path.join(os.getcwd(), "datasets")


def _barrier_settings():
    settings = solver_settings.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.Barrier)
    settings.set_parameter(CUOPT_PRESOLVE, 0)
    return settings


def _solve_socp(dm, settings=None):
    if settings is None:
        settings = _barrier_settings()
    return solver.Solve(dm, settings)


def _lorentz_qcoo(variable_indices):
    """COO triplets for sum_{i>0} x_i^2 - x_0^2 <= 0 on variable_indices."""
    indices = [int(i) for i in variable_indices]
    q_values = [-1.0] + [1.0] * (len(indices) - 1)
    return (
        np.array(q_values, dtype=np.float64),
        np.array(indices, dtype=np.int32),
        np.array(indices, dtype=np.int32),
    )


def _build_socp_min_x0_model():
    """min x0 s.t. x1=1 and Lorentz cone on (x0,x1,x2); optimal obj=1."""
    dm = data_model.DataModel()

    A_values = np.array([1.0], dtype=np.float64)
    A_indices = np.array([1], dtype=np.int32)
    A_offsets = np.array([0, 1], dtype=np.int32)
    dm.set_csr_constraint_matrix(A_values, A_indices, A_offsets)

    dm.set_constraint_lower_bounds(np.array([1.0], dtype=np.float64))
    dm.set_constraint_upper_bounds(np.array([1.0], dtype=np.float64))

    dm.set_objective_coefficients(np.array([1.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_lower_bounds(np.array([0.0, 0.0, 0.0], dtype=np.float64))
    dm.set_variable_upper_bounds(
        np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    )

    qv, qr, qc = _lorentz_qcoo([0, 1, 2])
    dm.add_quadratic_constraint(
        constraint_row_index=1,
        constraint_row_name="soc",
        quadratic_values=qv,
        quadratic_row_indices=qr,
        quadratic_col_indices=qc,
    )
    return dm


def test_socp_mps_write_parse_solve(tmp_path):
    """writeMPS -> ParseMps preserves QCMATRIX and barrier solve."""
    dm = _build_socp_min_x0_model()
    mps_file = tmp_path / "socp_lorentz.mps"
    dm.writeMPS(str(mps_file))

    dm2 = mps_parser.ParseMps(str(mps_file))
    qcs = dm2.get_quadratic_constraints()
    assert len(qcs) == 1
    assert qcs[0]["constraint_row_name"] == "soc"
    assert qcs[0]["constraint_row_type"] == "L"

    solution = _solve_socp(dm2)
    assert solution.get_termination_reason() == "Optimal"
    assert solution.get_primal_objective() == pytest.approx(1.0, rel=1e-3, abs=1e-3)


def test_parse_qcmatrix_dataset_mps():
    """Parse external QCQP MPS with QCMATRIX when dataset is available."""
    mps_path = os.path.join(RAPIDS_DATASET_ROOT_DIR, "qcqp", "QC_Test_1.mps")
    if not os.path.exists(mps_path):
        pytest.skip(f"{mps_path} not in dataset root")

    dm = mps_parser.ParseMps(mps_path)
    qcs = dm.get_quadratic_constraints()
    assert len(qcs) == 2
    assert qcs[0]["constraint_row_name"] == "QC0"
    assert qcs[1]["constraint_row_name"] == "QC1"
    for qc in qcs:
        assert qc["constraint_row_type"] == "L"
        assert qc["quadratic_row_indices"].shape == qc["quadratic_col_indices"].shape
        assert qc["quadratic_values"].shape == qc["quadratic_row_indices"].shape
