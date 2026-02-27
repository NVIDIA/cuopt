"""
MPS File Solver using cuOpt Python API

Read and solve LP/MILP problems from standard MPS files using
cuOpt's built-in readMPS method.

Default benchmark: air05.mps (airline crew scheduling from MIPLIB)
- Best known optimal: 26,374
"""

import os
import gzip
import urllib.request

from cuopt.linear_programming.problem import Problem
from cuopt.linear_programming.solver_settings import SolverSettings


# MIPLIB benchmark URL
AIR05_URL = "https://miplib.zib.de/WebData/instances/air05.mps.gz"
AIR05_OPTIMAL = 26374  # Best known optimal solution


def download_air05(data_dir: str) -> str:
    """Download air05.mps from MIPLIB if not present."""
    mps_file = os.path.join(data_dir, "air05.mps")
    
    if os.path.exists(mps_file):
        return mps_file
    
    os.makedirs(data_dir, exist_ok=True)
    gz_file = os.path.join(data_dir, "air05.mps.gz")
    
    print(f"Downloading air05.mps from MIPLIB...")
    urllib.request.urlretrieve(AIR05_URL, gz_file)
    
    # Decompress
    print("Decompressing...")
    with gzip.open(gz_file, 'rb') as f_in:
        with open(mps_file, 'wb') as f_out:
            f_out.write(f_in.read())
    
    # Clean up
    os.remove(gz_file)
    print(f"Downloaded: {mps_file}")
    
    return mps_file


def solve_mps(
    filepath: str,
    time_limit: float = 60.0,
    mip_gap: float = 0.01,
    verbose: bool = True
) -> tuple:
    """
    Solve an LP/MILP problem from an MPS file.
    
    Parameters
    ----------
    filepath : str
        Path to the MPS file
    time_limit : float
        Solver time limit in seconds
    mip_gap : float
        MIP relative gap tolerance
    verbose : bool
        Print solver output
    
    Returns
    -------
    tuple
        (problem, solution_dict) or (problem, None) if no solution
    """
    
    # Read MPS file directly (static method returns Problem object)
    problem = Problem.readMPS(filepath)
    
    print(f"Loaded MPS file: {filepath}")
    print(f"Variables: {problem.NumVariables}")
    print(f"Constraints: {problem.NumConstraints}")
    print(f"Is MIP: {problem.IsMIP}")
    
    # Solver settings
    settings = SolverSettings()
    settings.set_parameter("time_limit", time_limit)
    settings.set_parameter("log_to_console", verbose)
    settings.set_parameter("mip_relative_gap", mip_gap)
    
    # Solve
    print("\nSolving...")
    problem.solve(settings)
    
    # Extract solution
    status = problem.Status.name
    print(f"\nStatus: {status}")
    
    if status in ["Optimal", "FeasibleFound", "PrimalFeasible"]:
        solution = {
            'status': status,
            'objective': problem.ObjValue,
            'num_variables': problem.NumVariables,
            'num_constraints': problem.NumConstraints,
            'is_mip': problem.IsMIP,
            'mip_gap': mip_gap,
        }
        
        # Get variable values (use getVariables() for MPS-loaded problems)
        var_values = {}
        try:
            variables = problem.getVariables()
            for var in variables:
                val = var.getValue()
                if abs(val) > 1e-6:  # Only include non-zero values
                    var_values[var.Name] = val
        except (AttributeError, Exception):
            # For MPS problems, variable access may be limited
            pass
        
        solution['variables'] = var_values
        return problem, solution
    else:
        return problem, None


def compare_gaps(filepath: str, time_limit: float = 120.0) -> dict:
    """
    Compare solutions at different MIP gap tolerances.
    
    Parameters
    ----------
    filepath : str
        Path to the MPS file
    time_limit : float
        Solver time limit per run
    
    Returns
    -------
    dict
        Results for each gap tolerance
    """
    gaps = [0.01, 0.001]  # 1% and 0.1%
    results = {}
    
    for gap in gaps:
        print(f"\n{'='*60}")
        print(f"Solving with MIP gap = {gap*100}%")
        print(f"{'='*60}")
        
        problem, solution = solve_mps(
            filepath=filepath,
            time_limit=time_limit,
            mip_gap=gap,
            verbose=True
        )
        
        if solution:
            results[gap] = {
                'objective': solution['objective'],
                'status': solution['status'],
                'gap_to_optimal': (solution['objective'] - AIR05_OPTIMAL) / AIR05_OPTIMAL * 100
            }
        else:
            results[gap] = {'objective': None, 'status': 'No solution'}
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Solve LP/MILP from MPS file")
    parser.add_argument("--file", type=str, default=None, help="Path to MPS file")
    parser.add_argument("--time-limit", type=float, default=60.0, help="Solver time limit")
    parser.add_argument("--mip-gap", type=float, default=0.01, help="MIP gap tolerance")
    parser.add_argument("--compare", action="store_true", help="Compare 1% vs 0.1% gap")
    args = parser.parse_args()
    
    print("=" * 60)
    print("MPS File Solver using cuOpt")
    print("=" * 60)
    
    # Determine MPS file to use
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    if args.file:
        mps_file = args.file
    else:
        # Download air05.mps if not present
        mps_file = download_air05(data_dir)
    
    if args.compare:
        # Compare different gap tolerances
        print(f"\nComparing MIP gap tolerances on: {mps_file}")
        print(f"Best known optimal: {AIR05_OPTIMAL}")
        
        results = compare_gaps(mps_file, time_limit=args.time_limit)
        
        print()
        print("=" * 60)
        print("COMPARISON SUMMARY")
        print("=" * 60)
        print(f"Best known optimal: {AIR05_OPTIMAL}")
        print()
        print(f"{'Gap Tolerance':<15} {'Objective':<15} {'Gap to Optimal':<15}")
        print("-" * 45)
        
        for gap, result in sorted(results.items()):
            if result['objective']:
                print(f"{gap*100:.1f}%{'':<12} {result['objective']:<15.0f} {result['gap_to_optimal']:.2f}%")
            else:
                print(f"{gap*100:.1f}%{'':<12} {'No solution':<15}")
    else:
        # Single solve
        print(f"\nMPS File: {mps_file}")
        print(f"Time Limit: {args.time_limit}s")
        print(f"MIP Gap: {args.mip_gap * 100}%")
        print()
        
        problem, solution = solve_mps(
            filepath=mps_file,
            time_limit=args.time_limit,
            mip_gap=args.mip_gap,
            verbose=True
        )
        
        if solution:
            print()
            print("=" * 60)
            print("SOLUTION")
            print("=" * 60)
            print(f"Status: {solution['status']}")
            print(f"Objective Value: {solution['objective']:.0f}")
            print(f"Best Known Optimal: {AIR05_OPTIMAL}")
            print(f"Gap to Optimal: {(solution['objective'] - AIR05_OPTIMAL) / AIR05_OPTIMAL * 100:.2f}%")
        else:
            print("\nNo feasible solution found.")
