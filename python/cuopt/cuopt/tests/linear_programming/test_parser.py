# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile

from cuopt.linear_programming import Read
import numpy as np
import pytest
from cuopt.linear_programming.io.utilities import InputValidationError

RAPIDS_DATASET_ROOT_DIR = os.getenv("RAPIDS_DATASET_ROOT_DIR")
if RAPIDS_DATASET_ROOT_DIR is None:
    RAPIDS_DATASET_ROOT_DIR = os.getcwd()
    RAPIDS_DATASET_ROOT_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "datasets")

GOOD_MPS_1_DIR = os.path.join(RAPIDS_DATASET_ROOT_DIR, "linear_programming")

# Plain and compressed encodings of the same tiny LP (see good-mps-1-README.md).
GOOD_MPS_1_VARIANTS = (
    "good-mps-1.mps",
    "good-mps-1.lp",
    "good-mps-1.mps.gz",
    "good-mps-1.mps.bz2",
    "good-mps-1.lp.gz",
    "good-mps-1.lp.bz2",
)


def _assert_good_mps_1_model(data_model):
    """Same checks as C++ good_mps_1_test::check_model."""
    assert not data_model.get_sense()
    assert data_model.get_variable_names().tolist() == ["VAR1", "VAR2"]
    assert data_model.get_row_names().tolist() == ["ROW1", "ROW2"]
    assert data_model.get_objective_coefficients().tolist() == pytest.approx(
        [0.2, 0.1]
    )
    assert data_model.get_variable_lower_bounds().tolist() == pytest.approx(
        [0.0, 0.0]
    )
    assert np.isinf(data_model.get_variable_upper_bounds()[0])
    assert np.isinf(data_model.get_variable_upper_bounds()[1])
    assert data_model.get_constraint_upper_bounds().tolist() == pytest.approx(
        [5.4, 4.9]
    )
    assert data_model.get_constraint_matrix_values().tolist() == pytest.approx(
        [3.0, 4.0, 2.7, 10.1]
    )


@pytest.mark.parametrize("filename", GOOD_MPS_1_VARIANTS)
def test_read_good_mps_1_variants(filename):
    path = os.path.join(GOOD_MPS_1_DIR, filename)
    if not os.path.isfile(path):
        pytest.skip(f"missing dataset {path}")
    _assert_good_mps_1_model(Read(path))


def test_bad_mps_files():
    NumMpsFiles = 13
    for i in range(1, NumMpsFiles + 1):
        file_path = (
            RAPIDS_DATASET_ROOT_DIR + f"/linear_programming/bad-mps-{i}.mps"
        )
        if os.path.exists(file_path):
            with pytest.raises(InputValidationError):
                Read(file_path, fixed_mps_format=True)


def test_good_mps_file():
    file_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-free-var.mps"
    )
    data_model = Read(file_path)

    assert not data_model.get_sense()

    assert 3.0 == data_model.get_constraint_matrix_values()[0]
    assert 4.0 == data_model.get_constraint_matrix_values()[1]
    assert 2.7 == data_model.get_constraint_matrix_values()[2]
    assert 10.1 == data_model.get_constraint_matrix_values()[3]

    assert 0 == data_model.get_constraint_matrix_indices()[0]
    assert 1 == data_model.get_constraint_matrix_indices()[1]
    assert 0 == data_model.get_constraint_matrix_indices()[2]
    assert 1 == data_model.get_constraint_matrix_indices()[3]

    assert 0 == data_model.get_constraint_matrix_offsets()[0]
    assert 2 == data_model.get_constraint_matrix_offsets()[1]
    assert 4 == data_model.get_constraint_matrix_offsets()[2]

    assert 5.4 == data_model.get_constraint_bounds()[0]
    assert 4.9 == data_model.get_constraint_bounds()[1]

    assert 0.2 == data_model.get_objective_coefficients()[0]
    assert 0.1 == data_model.get_objective_coefficients()[1]

    assert 1.0 == data_model.get_objective_scaling_factor()
    assert 0.0 == data_model.get_objective_offset()

    assert -np.inf == data_model.get_variable_lower_bounds()[0]
    assert 0.0 == data_model.get_variable_lower_bounds()[1]

    assert np.inf == data_model.get_variable_upper_bounds()[0]
    assert np.inf == data_model.get_variable_upper_bounds()[1]

    assert -np.inf == data_model.get_constraint_lower_bounds()[0]
    assert -np.inf == data_model.get_constraint_lower_bounds()[1]

    assert 5.4 == data_model.get_constraint_upper_bounds()[0]
    assert 4.9 == data_model.get_constraint_upper_bounds()[1]


# Minimal LP content that should parse identically regardless of whether it's
# routed through Read() or the server's extension-based dispatch path.
_MINIMAL_LP = """
Minimize
  x
Subject To
 c1: x >= 2.5
Bounds
 x <= 10
End
"""


def test_parse_lp_basic():
    with tempfile.NamedTemporaryFile(
        suffix=".lp", mode="w", delete=False
    ) as f:
        f.write(_MINIMAL_LP)
        path = f.name
    try:
        data_model = Read(path)
    finally:
        os.unlink(path)

    # Minimize ⇒ sense is False.
    assert not data_model.get_sense()
    # Single variable with default lb=0, explicit ub=10.
    assert data_model.get_variable_names().tolist() == ["x"]
    assert data_model.get_variable_lower_bounds()[0] == 0.0
    assert data_model.get_variable_upper_bounds()[0] == 10.0
    # Objective is just "x" ⇒ c = [1.0].
    assert data_model.get_objective_coefficients()[0] == 1.0
    # Single >= constraint c1: x >= 2.5.
    assert data_model.get_row_names().tolist() == ["c1"]
    assert data_model.get_constraint_lower_bounds()[0] == 2.5
    assert np.isinf(data_model.get_constraint_upper_bounds()[0])
    assert data_model.get_constraint_matrix_values().tolist() == [1.0]


def test_parse_lp_rejects_unsupported_section():
    # SOS is explicitly out of scope; the parser should raise.
    bad_lp = """
Minimize
  x
Subject To
 c1: x >= 1
SOS
 s1: S1 :: x : 1
End
"""
    with tempfile.NamedTemporaryFile(
        suffix=".lp", mode="w", delete=False
    ) as f:
        f.write(bad_lp)
        path = f.name
    try:
        with pytest.raises(InputValidationError):
            Read(path)
    finally:
        os.unlink(path)


def test_parse_lp_and_parse_mps_agree_on_trivial_problem():
    # Same problem written in LP and MPS — both parsers should produce the
    # same data model (modulo variable/constraint ordering, but this problem
    # has exactly one of each).
    mps_text = (
        "NAME trivial\n"
        "ROWS\n"
        " N OBJ\n"
        " G c1\n"
        "COLUMNS\n"
        " x OBJ 1\n"
        " x c1 1\n"
        "RHS\n"
        " RHS1 c1 2.5\n"
        "BOUNDS\n"
        " UP BND1 x 10\n"
        "ENDATA\n"
    )
    with tempfile.NamedTemporaryFile(
        suffix=".mps", mode="w", delete=False
    ) as f:
        f.write(mps_text)
        mps_path = f.name
    with tempfile.NamedTemporaryFile(
        suffix=".lp", mode="w", delete=False
    ) as f:
        f.write(_MINIMAL_LP)
        lp_path = f.name
    try:
        lp_model = Read(lp_path)
        mps_model = Read(mps_path)
    finally:
        os.unlink(mps_path)
        os.unlink(lp_path)

    assert lp_model.get_sense() == mps_model.get_sense()
    assert (
        lp_model.get_variable_names().tolist()
        == mps_model.get_variable_names().tolist()
    )
    assert (
        lp_model.get_objective_coefficients().tolist()
        == mps_model.get_objective_coefficients().tolist()
    )
    assert (
        lp_model.get_variable_upper_bounds().tolist()
        == mps_model.get_variable_upper_bounds().tolist()
    )
    assert (
        lp_model.get_constraint_lower_bounds().tolist()
        == mps_model.get_constraint_lower_bounds().tolist()
    )


def test_read_dispatches_mps_and_lp():
    mps_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-free-var.mps"
    )
    lp_path = (
        RAPIDS_DATASET_ROOT_DIR + "/linear_programming/good-mps-free-var.lp"
    )
    mps_model = Read(mps_path)
    lp_model = Read(lp_path)
    assert mps_model.get_sense() == lp_model.get_sense()
    assert (
        mps_model.get_variable_names().tolist()
        == lp_model.get_variable_names().tolist()
    )


def test_read_unrecognized_extension():
    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        f.write(b"x\n")
        path = f.name
    try:
        with pytest.raises(
            RuntimeError, match="unrecognized input file extension"
        ):
            Read(path)
    finally:
        os.unlink(path)


def test_write_lp_exact_output():
    # Continuous + integer variables; assert the precise LP text written.
    from cuopt.linear_programming.problem import (
        INTEGER,
        MINIMIZE,
        Problem,
    )

    problem = Problem("exact_lp")
    x = problem.addVariable(lb=0.0, ub=8.0, obj=3.0, name="x")
    y = problem.addVariable(lb=1.0, ub=float("inf"), obj=2.0, name="y")
    z = problem.addVariable(lb=0.0, ub=10.0, obj=1.0, vtype=INTEGER, name="z")
    b = problem.addVariable(lb=0.0, ub=1.0, obj=2.0, vtype=INTEGER, name="b")
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x + z >= 1, name="c2")
    problem.addConstraint(2 * x + y == 6, name="c3")
    problem.setObjective(3 * x + 2 * y + z + 2 * b, sense=MINIMIZE)

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        out_path = f.name
    try:
        problem.write(out_path)
        with open(out_path) as f:
            written = f.read()
    finally:
        os.unlink(out_path)

    assert written == (
        "Minimize\n"
        " obj: + 3 x + 2 y + 1 z + 2 b\n"
        "Subject To\n"
        " c1: + 1 x + 1 y <= 10\n"
        " c2: + 1 x + 1 z >= 1\n"
        " c3: + 2 x + 1 y = 6\n"
        "Bounds\n"
        " x <= 8\n"
        " y >= 1\n"
        " z <= 10\n"
        "Generals\n"
        " z\n"
        "Binaries\n"
        " b\n"
        "End\n"
    )


def test_write_read_write():
    # Build a Problem, write it as LP, read that LP back, write it again, and
    # assert the two emitted LP files are byte-for-byte identical. A stable
    # writer must reach a fixed point after the first write/read cycle.
    from cuopt.linear_programming.problem import (
        INTEGER,
        MINIMIZE,
        Problem,
    )

    problem = Problem("idempotent_lp")
    x = problem.addVariable(lb=0.0, ub=8.0, obj=3.0, name="x")
    y = problem.addVariable(lb=1.0, ub=float("inf"), obj=2.0, name="y")
    z = problem.addVariable(lb=0.0, ub=1.0, obj=-1.0, vtype=INTEGER, name="z")
    problem.addConstraint(x + y <= 10, name="c1")
    problem.addConstraint(x - z >= -4, name="c2")
    problem.addConstraint(2 * x + y == 6, name="c3")
    problem.setObjective(3 * x + 2 * y - z, sense=MINIMIZE)

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        first_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        second_path = f.name
    try:
        # First write straight from the Python model.
        problem.write(first_path)
        with open(first_path) as f:
            first_lp = f.read()

        # Read the emitted LP back and write it out a second time.
        reparsed = Read(first_path)
        reparsed.write(second_path)
        with open(second_path) as f:
            second_lp = f.read()
    finally:
        os.unlink(first_path)
        os.unlink(second_path)

    assert first_lp == second_lp


def test_write_dispatches_on_extension():
    from cuopt.linear_programming.problem import MINIMIZE, Problem

    problem = Problem()
    x = problem.addVariable(lb=0.0, ub=10.0, obj=1.0, name="x")
    problem.addConstraint(x <= 5, name="c1")
    problem.setObjective(x, sense=MINIMIZE)

    with tempfile.NamedTemporaryFile(suffix=".lp", delete=False) as f:
        lp_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".mps", delete=False) as f:
        mps_path = f.name
    try:
        problem.write(lp_path)
        problem.write(mps_path)
        with open(lp_path) as f:
            lp_content = f.read()
        with open(mps_path) as f:
            mps_content = f.read()
    finally:
        os.unlink(lp_path)
        os.unlink(mps_path)

    assert "Subject To" in lp_content
    assert "ENDATA" in mps_content


def test_write_unrecognized_extension():
    from cuopt.linear_programming.problem import MINIMIZE, Problem

    problem = Problem()
    x = problem.addVariable(lb=0.0, ub=10.0, obj=1.0, name="x")
    problem.setObjective(x, sense=MINIMIZE)

    with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
        out_path = f.name
    try:
        with pytest.raises(
            RuntimeError, match="unrecognized output file extension"
        ):
            problem.write(out_path)
    finally:
        os.unlink(out_path)
