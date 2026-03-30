# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for CPU-only execution mode and solution interface polymorphism.

TestCPUOnlyExecution / TestCuoptCliCPUOnly:
    Run in subprocesses with CUDA_VISIBLE_DEVICES="" so the CUDA driver
    never initializes.  Subprocess isolation is required because the
    driver reads that variable once at init time.

TestSolutionInterfacePolymorphism:
    Run in-process on real GPU hardware and assert correctness of
    solution values against known optima.
"""

import os
import subprocess
import sys

import cuopt_mps_parser
import pytest
from cuopt import linear_programming
from cuopt.linear_programming.solver.solver_parameters import CUOPT_TIME_LIMIT

RAPIDS_DATASET_ROOT_DIR = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cpu_only_env():
    """Return an env dict that hides all GPUs and enables remote mode."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUOPT_REMOTE_HOST"] = "localhost"
    env["CUOPT_REMOTE_PORT"] = "12345"
    return env


def _run_in_subprocess(func, env=None, timeout=120):
    """Run *func* (a top-level function in this module) in a fresh subprocess."""
    result = subprocess.run(
        [sys.executable, os.path.abspath(__file__), func.__name__],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    return result


# ---------------------------------------------------------------------------
# CPU-only subprocess test implementations
#
# Each function below runs in an isolated subprocess with
# CUDA_VISIBLE_DEVICES="" so the CUDA driver is never initialized.
# Imports are local so the subprocess doesn't need GPU-dependent modules
# at the top level.
# ---------------------------------------------------------------------------


def _impl_lp_solve_cpu_only():
    """LP solve returns correctly-sized solution vectors."""
    from cuopt import linear_programming
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())

    solution = linear_programming.Solve(
        dm, linear_programming.SolverSettings()
    )

    primal = solution.get_primal_solution()
    assert len(primal) == n_vars, (
        f"primal size {len(primal)} != n_vars {n_vars}"
    )

    obj = solution.get_primal_objective()
    assert obj is not None, "objective is None"


def _impl_lp_dual_solution_cpu_only():
    """Dual solution and reduced costs are correctly sized."""
    from cuopt import linear_programming
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())
    n_cons = len(dm.get_constraint_bounds())

    solution = linear_programming.Solve(
        dm, linear_programming.SolverSettings()
    )

    dual = solution.get_dual_solution()
    assert len(dual) == n_cons, f"dual size {len(dual)} != n_cons {n_cons}"

    rc = solution.get_reduced_cost()
    assert len(rc) == n_vars, f"reduced_cost size {len(rc)} != n_vars {n_vars}"


def _impl_mip_solve_cpu_only():
    """MIP solve returns correctly-sized solution vector."""
    from cuopt import linear_programming
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_TIME_LIMIT,
    )
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/mip/bb_optimality.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)
    n_vars = len(dm.get_objective_coefficients())

    settings = linear_programming.SolverSettings()
    settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

    solution = linear_programming.Solve(dm, settings)
    vals = solution.get_primal_solution()
    assert len(vals) == n_vars, f"solution size {len(vals)} != n_vars {n_vars}"


def _impl_warmstart_cpu_only():
    """Warmstart round-trip works without touching CUDA."""
    from cuopt import linear_programming
    from cuopt.linear_programming.solver.solver_parameters import (
        CUOPT_METHOD,
        CUOPT_ITERATION_LIMIT,
    )
    from cuopt.linear_programming.solver_settings import SolverMethod
    import cuopt_mps_parser

    dataset_root = os.environ.get("RAPIDS_DATASET_ROOT_DIR", "./")
    mps_file = f"{dataset_root}/linear_programming/afiro_original.mps"
    dm = cuopt_mps_parser.ParseMps(mps_file)

    settings = linear_programming.SolverSettings()
    settings.set_parameter(CUOPT_METHOD, SolverMethod.PDLP)
    settings.set_parameter(CUOPT_ITERATION_LIMIT, 100)

    sol1 = linear_programming.Solve(dm, settings)
    ws = sol1.get_pdlp_warm_start_data()

    if ws is not None:
        settings.set_pdlp_warm_start_data(ws)
        settings.set_parameter(CUOPT_ITERATION_LIMIT, 200)
        sol2 = linear_programming.Solve(dm, settings)
        assert sol2.get_primal_solution() is not None


# ---------------------------------------------------------------------------
# CPU-only Python tests  (subprocess required)
# ---------------------------------------------------------------------------


class TestCPUOnlyExecution:
    """Tests that run with CUDA_VISIBLE_DEVICES='' to simulate CPU-only hosts."""

    @pytest.fixture
    def env(self):
        return _cpu_only_env()

    def test_lp_solve_cpu_only(self, env):
        """LP solve returns correctly-sized solution vectors."""
        result = _run_in_subprocess(_impl_lp_solve_cpu_only, env=env)
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_lp_dual_solution_cpu_only(self, env):
        """Dual solution and reduced costs are correctly sized."""
        result = _run_in_subprocess(_impl_lp_dual_solution_cpu_only, env=env)
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_mip_solve_cpu_only(self, env):
        """MIP solve returns correctly-sized solution vector."""
        result = _run_in_subprocess(_impl_mip_solve_cpu_only, env=env)
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"

    def test_warmstart_cpu_only(self, env):
        """Warmstart round-trip works without touching CUDA."""
        result = _run_in_subprocess(_impl_warmstart_cpu_only, env=env)
        assert result.returncode == 0, f"Test failed:\n{result.stderr}"


# ---------------------------------------------------------------------------
# CPU-only CLI tests  (subprocess inherently needed)
# ---------------------------------------------------------------------------


class TestCuoptCliCPUOnly:
    """Test that cuopt_cli runs without CUDA in remote-execution mode."""

    @pytest.fixture
    def env(self):
        return _cpu_only_env()

    @staticmethod
    def _find_cuopt_cli():
        import shutil

        for loc in [
            shutil.which("cuopt_cli"),
            "./cuopt_cli",
            "../cpp/build/cuopt_cli",
            "../../cpp/build/cuopt_cli",
        ]:
            if loc and os.path.isfile(loc) and os.access(loc, os.X_OK):
                return os.path.abspath(loc)

        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            p = os.path.join(conda_prefix, "bin", "cuopt_cli")
            if os.path.isfile(p):
                return p
        return None

    _CUDA_ERRORS = [
        "CUDA error",
        "cudaErrorNoDevice",
        "no CUDA-capable device",
        "CUDA driver version is insufficient",
        "CUDA initialization failed",
    ]

    def _run_cli(self, mps_file, env):
        cli = self._find_cuopt_cli()
        if cli is None:
            pytest.skip("cuopt_cli not found")
        if not os.path.exists(mps_file):
            pytest.skip(f"Test file not found: {mps_file}")

        result = subprocess.run(
            [cli, mps_file, "--time-limit", "60"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(result.stdout)
        print(result.stderr)

        combined = result.stdout + result.stderr
        for err in self._CUDA_ERRORS:
            assert err not in combined, (
                f"CUDA error '{err}' in output -- "
                "cuopt_cli must not require CUDA in remote mode"
            )
        assert result.returncode == 0, (
            f"cuopt_cli exited with {result.returncode}"
        )

    def test_cuopt_cli_lp_cpu_only(self, env):
        self._run_cli(
            f"{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps",
            env,
        )

    def test_cuopt_cli_mip_cpu_only(self, env):
        self._run_cli(f"{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps", env)


# ---------------------------------------------------------------------------
# Solution-interface tests  (in-process, real GPU, assert correctness)
# ---------------------------------------------------------------------------

_AFIRO_OBJ = -464.7531428571


class TestSolutionInterfacePolymorphism:
    """Verify solution accessors return correct values on real GPU solves."""

    def test_lp_solution_values(self):
        """LP solve of afiro.mps returns correct objective and sizes."""
        mps_file = (
            f"{RAPIDS_DATASET_ROOT_DIR}/linear_programming/afiro_original.mps"
        )
        dm = cuopt_mps_parser.ParseMps(mps_file)
        n_vars = len(dm.get_objective_coefficients())
        n_cons = len(dm.get_constraint_bounds())

        solution = linear_programming.Solve(
            dm, linear_programming.SolverSettings()
        )

        primal = solution.get_primal_solution()
        assert len(primal) == n_vars

        obj = solution.get_primal_objective()
        assert abs(obj - _AFIRO_OBJ) / abs(_AFIRO_OBJ) < 0.01, (
            f"Objective {obj} too far from expected {_AFIRO_OBJ}"
        )

        dual = solution.get_dual_solution()
        assert len(dual) == n_cons

        rc = solution.get_reduced_cost()
        assert len(rc) == n_vars

        dual_obj = solution.get_dual_objective()
        if obj != 0:
            assert abs(dual_obj - obj) / abs(obj) < 0.05

    def test_mip_solution_values(self):
        """MIP solve of bb_optimality.mps returns valid stats."""
        mps_file = f"{RAPIDS_DATASET_ROOT_DIR}/mip/bb_optimality.mps"
        dm = cuopt_mps_parser.ParseMps(mps_file)
        n_vars = len(dm.get_objective_coefficients())

        settings = linear_programming.SolverSettings()
        settings.set_parameter(CUOPT_TIME_LIMIT, 60.0)

        solution = linear_programming.Solve(dm, settings)

        vals = solution.get_primal_solution()
        assert len(vals) == n_vars

        stats = solution.get_milp_stats()
        assert "mip_gap" in stats
        assert "solution_bound" in stats
        assert stats["mip_gap"] >= 0, f"Negative MIP gap: {stats['mip_gap']}"


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    globals()[sys.argv[1]]()
