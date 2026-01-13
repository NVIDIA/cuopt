#!/usr/bin/env python3
"""
Test script for cuOpt Python API - works with both local and remote solve.

Usage:
    # Local solve (default):
    python test_python_api.py /path/to/problem.mps

    # Remote solve (set environment variables first):
    CUOPT_REMOTE_HOST=localhost CUOPT_REMOTE_PORT=9090 python test_python_api.py /path/to/problem.mps

Example:
    python test_python_api.py /home/tmckay/repos/HiGHS/check/instances/afiro.mps
"""

import os
import sys

import cuopt_mps_parser
from cuopt.linear_programming import solver, solver_settings


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <mps_file_path>")
        sys.exit(1)

    mps_file = sys.argv[1]
    if not os.path.exists(mps_file):
        print(f"Error: File not found: {mps_file}")
        sys.exit(1)

    # Check if remote solve is configured
    remote_host = os.environ.get("CUOPT_REMOTE_HOST")
    remote_port = os.environ.get("CUOPT_REMOTE_PORT")
    if remote_host and remote_port:
        print(f"Remote solve enabled: {remote_host}:{remote_port}")
    else:
        print("Local solve (no CUOPT_REMOTE_HOST/PORT set)")

    print(f"Reading MPS file: {mps_file}")

    # Parse MPS file into DataModel
    data_model = cuopt_mps_parser.ParseMps(mps_file)

    # Create solver settings
    settings = solver_settings.SolverSettings()
    settings.set_optimality_tolerance(1e-6)

    # Solve
    print("Solving...")
    solution = solver.Solve(data_model, settings)

    # Print results
    print("\nResults:")
    print("-" * 40)
    print(f"Status: {solution.get_termination_reason()}")
    print(f"Objective: {solution.get_primal_objective():.6e}")
    print(f"Solve time: {solution.get_solve_time():.3f} seconds")

    # Print first few solution values
    primal = solution.get_primal_solution()
    if len(primal) > 0:
        print(f"\nPrimal solution (first 10 of {len(primal)} variables):")
        for i, val in enumerate(primal[:10]):
            print(f"  x{i+1} = {val:.6f}")
        if len(primal) > 10:
            print(f"  ... ({len(primal) - 10} more variables)")

    print("\nDone!")


if __name__ == "__main__":
    main()
