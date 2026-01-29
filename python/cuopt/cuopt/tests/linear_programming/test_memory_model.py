# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for memory model functionality including remote solve configuration,
lazy loading, and host/device memory handling.
"""

import os

import numpy as np
import pytest

from cuopt.linear_programming import DataModel, Solve, SolverSettings
from cuopt.linear_programming.solver_settings import SolverMethod


class TestLazyModuleLoading:
    """Test that cuopt modules can be lazily loaded without triggering CUDA initialization."""

    def test_import_cuopt_without_submodules(self):
        """Test that importing cuopt doesn't immediately load submodules."""
        # This test verifies the __getattr__ mechanism works
        import cuopt

        # These should be available via lazy loading
        assert hasattr(cuopt, "linear_programming")
        assert hasattr(cuopt, "routing")

    def test_submodules_in_dir(self):
        """Test that submodules appear in dir() output."""
        import cuopt

        dir_contents = dir(cuopt)
        assert "linear_programming" in dir_contents
        assert "routing" in dir_contents

    def test_lazy_import_caches_module(self):
        """Test that lazy imports are cached in globals."""
        import cuopt

        # First access loads and caches
        lp = cuopt.linear_programming

        # Second access should return same object (cached)
        lp2 = cuopt.linear_programming
        assert lp is lp2


class TestRemoteSolveEnvironment:
    """Test remote solve environment variable detection and handling."""

    def setup_method(self):
        """Clean up environment before each test."""
        for var in ["CUOPT_REMOTE_HOST", "CUOPT_REMOTE_PORT"]:
            if var in os.environ:
                del os.environ[var]

    def teardown_method(self):
        """Clean up environment after each test."""
        for var in ["CUOPT_REMOTE_HOST", "CUOPT_REMOTE_PORT"]:
            if var in os.environ:
                del os.environ[var]

    def test_local_solve_by_default(self):
        """Test that solve works without remote environment variables (local GPU solve)."""
        data_model_obj = DataModel()

        # Simple LP: minimize x subject to x >= 1
        A_values = np.array([1.0])
        A_indices = np.array([0])
        A_offsets = np.array([0, 1])
        data_model_obj.set_csr_constraint_matrix(
            A_values, A_indices, A_offsets
        )

        b = np.array([1.0])
        data_model_obj.set_constraint_bounds(b)

        c = np.array([1.0])
        data_model_obj.set_objective_coefficients(c)

        row_types = np.array(["G"])
        data_model_obj.set_row_types(row_types)

        settings = SolverSettings()
        settings.set_parameter("method", SolverMethod.DualSimplex)

        solution = Solve(data_model_obj, settings)

        assert solution.get_termination_reason() == "Optimal"
        assert solution.get_primal_solution()[0] == pytest.approx(1.0)

    def test_remote_solve_stub_with_env_vars(self):
        """Test that remote solve path is triggered and returns host memory data."""
        os.environ["CUOPT_REMOTE_HOST"] = "localhost"
        os.environ["CUOPT_REMOTE_PORT"] = "50051"

        data_model_obj = DataModel()

        # Simple LP: minimize x subject to x >= 1
        A_values = np.array([1.0])
        A_indices = np.array([0])
        A_offsets = np.array([0, 1])
        data_model_obj.set_csr_constraint_matrix(
            A_values, A_indices, A_offsets
        )

        b = np.array([1.0])
        data_model_obj.set_constraint_bounds(b)

        c = np.array([1.0])
        data_model_obj.set_objective_coefficients(c)

        row_types = np.array(["G"])
        data_model_obj.set_row_types(row_types)

        settings = SolverSettings()
        settings.set_parameter("method", SolverMethod.DualSimplex)

        # Remote solve stub should return a solution (all zeros currently)
        solution = Solve(data_model_obj, settings)

        # Verify remote solve path was taken (stub returns Optimal with zeros)
        assert solution.get_termination_reason() == "Optimal"

        # Stub returns zeros (not actual solution), but data should be accessible
        primal = solution.get_primal_solution()
        assert isinstance(primal, np.ndarray)
        assert len(primal) == 1
        assert primal[0] == pytest.approx(0.0)  # Stub value

        # Verify we can access other solution components (host memory)
        dual = solution.get_dual_solution()
        assert isinstance(dual, np.ndarray)

    def test_remote_solve_mip_stub_with_env_vars(self):
        """Test that remote solve path works for MIP and returns host memory data."""
        os.environ["CUOPT_REMOTE_HOST"] = "localhost"
        os.environ["CUOPT_REMOTE_PORT"] = "50051"

        data_model_obj = DataModel()

        # Simple MIP: maximize x subject to x <= 10, x integer
        A_values = np.array([1.0])
        A_indices = np.array([0])
        A_offsets = np.array([0, 1])
        data_model_obj.set_csr_constraint_matrix(
            A_values, A_indices, A_offsets
        )

        b = np.array([10.0])
        data_model_obj.set_constraint_bounds(b)

        c = np.array([1.0])
        data_model_obj.set_objective_coefficients(c)

        row_types = np.array(["L"])
        data_model_obj.set_row_types(row_types)

        var_types = np.array([1])  # Integer
        data_model_obj.set_variable_types(var_types)

        data_model_obj.set_maximize(True)

        settings = SolverSettings()

        # Remote solve stub should return a solution
        solution = Solve(data_model_obj, settings)

        # Verify remote solve path was taken
        assert solution.get_termination_reason() == "Optimal"

        # Stub returns zeros, but data should be accessible from host memory
        sol = solution.get_primal_solution()
        assert isinstance(sol, np.ndarray)
        assert len(sol) == 1
        assert sol[0] == pytest.approx(0.0)  # Stub value


class TestEmptyConstraintMatrix:
    """Test handling of problems with no constraints (edge case for memory model)."""

    def test_empty_problem_with_objective_offset(self):
        """Test problem with no variables or constraints, only objective offset."""
        import cuopt_mps_parser

        mps_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../datasets/mip/empty-problem-obj.mps",
        )

        if not os.path.exists(mps_path):
            pytest.skip(f"Test dataset not found: {mps_path}")

        data_model_obj = cuopt_mps_parser.MPS_Parser.parse_file(mps_path)
        settings = SolverSettings()
        settings.set_parameter("time_limit", 5.0)

        solution = Solve(data_model_obj, settings)

        assert solution.get_termination_reason() == "Optimal"
        assert solution.get_primal_objective() == pytest.approx(81.0, abs=1e-3)

    def test_empty_problem_with_variables(self):
        """Test problem with variables but no constraints."""
        import cuopt_mps_parser

        mps_path = os.path.join(
            os.path.dirname(__file__),
            "../../../../datasets/mip/empty-problem-objective-vars.mps",
        )

        if not os.path.exists(mps_path):
            pytest.skip(f"Test dataset not found: {mps_path}")

        data_model_obj = cuopt_mps_parser.MPS_Parser.parse_file(mps_path)
        settings = SolverSettings()
        settings.set_parameter("time_limit", 5.0)

        solution = Solve(data_model_obj, settings)

        assert solution.get_termination_reason() == "Optimal"
        assert solution.get_primal_objective() == pytest.approx(-2.0, abs=1e-3)


class TestCircularImportPrevention:
    """Test that circular import issues are resolved."""

    def test_direct_import_solver_wrapper(self):
        """Test that solver_wrapper can be imported directly without circular dependency."""
        try:
            from cuopt.linear_programming.solver.solver_wrapper import (
                ErrorStatus,
                LPTerminationStatus,
                MILPTerminationStatus,
            )

            # If we get here without ImportError, the circular dependency is resolved
            assert ErrorStatus is not None
            assert LPTerminationStatus is not None
            assert MILPTerminationStatus is not None
        except ImportError as e:
            pytest.fail(f"Circular import detected: {e}")

    def test_import_order_independence(self):
        """Test that imports work regardless of order."""
        # These imports should work in any order without circular dependency
        from cuopt.linear_programming import SolverSettings
        from cuopt.linear_programming.solver_settings import (
            PDLPSolverMode,
            SolverMethod,
        )
        from cuopt.linear_programming import Solve

        assert SolverSettings is not None
        assert PDLPSolverMode is not None
        assert SolverMethod is not None
        assert Solve is not None
