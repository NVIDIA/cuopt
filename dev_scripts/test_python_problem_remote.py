#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Python Problem class with remote solve.

This test uses the high-level Python Problem class (not raw protobuf) and
verifies it works with both local and remote solve.

Usage:
  # Test local solve (requires GPU)
  python test_python_problem_remote.py local

  # Test remote solve (requires cuopt_remote_server running)
  CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=8765 python test_python_problem_remote.py remote
"""

import os
import sys


def test_problem_class():
    """Test the Python Problem class API."""
    from cuopt import linear_programming
    from cuopt.linear_programming.problem import Problem, MINIMIZE

    # Check if remote solve is configured
    remote_host = os.environ.get("CUOPT_REMOTE_HOST", "")
    remote_port = os.environ.get("CUOPT_REMOTE_PORT", "")
    is_remote = bool(remote_host and remote_port)

    mode = "REMOTE" if is_remote else "LOCAL"
    print(f"=== Testing Python Problem class ({mode} solve) ===")
    if is_remote:
        print(f"    Server: {remote_host}:{remote_port}")

    # Create a simple LP:
    #   minimize: x + 2y
    #   subject to:
    #     x + y >= 1
    #     x, y >= 0

    problem = Problem("SimpleLP")

    # Add variables
    x = problem.addVariable(name="x", lb=0.0)
    y = problem.addVariable(name="y", lb=0.0)

    # Add constraint: x + y >= 1
    problem.addConstraint(x + y >= 1, name="c1")

    # Set objective: minimize x + 2y
    problem.setObjective(x + 2 * y, sense=MINIMIZE)

    print(f"    Variables: {problem.NumVariables}")
    print(f"    Constraints: {problem.NumConstraints}")

    # Solve
    settings = linear_programming.SolverSettings()
    settings.log_to_console = True

    print("    Solving...")
    problem.solve(settings)

    # Check results
    print(f"    Status: {problem.Status}")
    print(f"    Objective value: {problem.ObjValue}")
    print(f"    Solve time: {problem.SolveTime:.4f}s")
    print(f"    x = {x.Value}")
    print(f"    y = {y.Value}")

    # Verify solution
    # Optimal solution should be x=1, y=0 with objective=1
    # (since y has coefficient 2 and we minimize)
    expected_obj = 1.0
    tolerance = 0.01

    if abs(problem.ObjValue - expected_obj) < tolerance:
        print(
            f"\n=== SUCCESS: Objective {problem.ObjValue:.4f} matches expected {expected_obj} ==="
        )
        return True
    else:
        print(
            f"\n=== FAILED: Objective {problem.ObjValue:.4f} != expected {expected_obj} ==="
        )
        return False


def test_mip_problem():
    """Test a MIP problem."""
    from cuopt import linear_programming
    from cuopt.linear_programming.problem import Problem, INTEGER, MINIMIZE

    remote_host = os.environ.get("CUOPT_REMOTE_HOST", "")
    remote_port = os.environ.get("CUOPT_REMOTE_PORT", "")
    is_remote = bool(remote_host and remote_port)
    mode = "REMOTE" if is_remote else "LOCAL"

    print(f"\n=== Testing Python MIP Problem class ({mode} solve) ===")

    # Simple MIP:
    #   minimize: x + y
    #   subject to:
    #     x + y >= 2.5
    #     x, y >= 0, x integer

    problem = Problem("SimpleMIP")

    # Add variables
    x = problem.addVariable(name="x", lb=0.0, vtype=INTEGER)
    y = problem.addVariable(name="y", lb=0.0)

    # Add constraint
    problem.addConstraint(x + y >= 2.5, name="c1")

    # Set objective
    problem.setObjective(x + y, sense=MINIMIZE)

    print(f"    Variables: {problem.NumVariables} (1 integer)")
    print(f"    Constraints: {problem.NumConstraints}")

    # Solve with MIP settings
    settings = linear_programming.SolverSettings()
    settings.log_to_console = True
    settings.time_limit = 60.0

    print("    Solving...")
    problem.solve(settings)

    print(f"    Status: {problem.Status}")
    print(f"    Objective value: {problem.ObjValue}")
    print(f"    x = {x.Value} (integer)")
    print(f"    y = {y.Value}")

    # Since x must be integer, optimal is x=1 or x=2 + y to make sum >= 2.5
    # x=1, y=1.5 gives obj=2.5
    # x=2, y=0.5 gives obj=2.5
    # x=3, y=0 gives obj=3
    # So optimal is 2.5
    expected_obj = 2.5
    tolerance = 0.1

    if abs(problem.ObjValue - expected_obj) < tolerance:
        print(
            f"\n=== SUCCESS: MIP objective {problem.ObjValue:.4f} matches expected {expected_obj} ==="
        )
        return True
    else:
        print(
            f"\n=== FAILED: MIP objective {problem.ObjValue:.4f} != expected {expected_obj} ==="
        )
        return False


if __name__ == "__main__":
    # Show usage hint
    if len(sys.argv) > 1 and sys.argv[1] in ("--help", "-h"):
        print(__doc__)
        sys.exit(0)

    # Run tests
    try:
        lp_ok = test_problem_class()
        mip_ok = test_mip_problem()

        if lp_ok and mip_ok:
            print("\n=== All tests PASSED ===")
            sys.exit(0)
        else:
            print("\n=== Some tests FAILED ===")
            sys.exit(1)
    except Exception as e:
        print(f"\n=== ERROR: {e} ===")
        import traceback

        traceback.print_exc()
        sys.exit(1)
