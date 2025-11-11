#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple client test for remote solve - uses environment variables from shell"""

import os
import sys

print("=" * 60)
print("Remote Solve Client Test")
print("=" * 60)

# Check environment variables
remote_host = os.environ.get("CUOPT_REMOTE_HOST")
remote_port = os.environ.get("CUOPT_REMOTE_PORT")

if not remote_host or not remote_port:
    print("❌ Error: CUOPT_REMOTE_HOST and CUOPT_REMOTE_PORT must be set")
    sys.exit(1)

print(f"\n✅ CUOPT_REMOTE_HOST={remote_host}")
print(f"✅ CUOPT_REMOTE_PORT={remote_port}")

print("\nLoading cuopt library...")
from cuopt.linear_programming import Problem

print("✅ Library loaded")

# Read and solve problem
mps_file = "datasets/linear_programming/afiro_original.mps"

if not os.path.exists(mps_file):
    print(f"❌ MPS file not found: {mps_file}")
    sys.exit(1)

print(f"\nReading MPS file: {mps_file}")
problem = Problem.readMPS(mps_file)

print("\n" + "=" * 60)
print("Calling solve() - watch for remote connection...")
print("=" * 60)

try:
    # Note: problem.solve() modifies the problem object in place and returns None
    problem.solve()

    print("\n✅ Solve completed!")
    print(f"Status: {problem.Status}")
    print(f"Solve Time: {problem.SolveTime:.3f}s")

    # Show solution stats
    if hasattr(problem, 'SolutionStats'):
        stats = problem.SolutionStats
        if hasattr(stats, 'primal_objective'):
            print(f"Objective: {stats.primal_objective:.6f}")
        if hasattr(stats, 'dual_objective'):
            print(f"Dual Objective: {stats.dual_objective:.6f}")
        if hasattr(stats, 'nb_iterations'):
            print(f"Iterations: {stats.nb_iterations}")

    # Show variable count and sample values
    if problem.vars and len(problem.vars) > 0:
        print(f"\nVariables: {len(problem.vars)}")
        non_zero_count = sum(1 for var in problem.vars if var.Value is not None and var.Value != 0)
        print(f"Non-zero values: {non_zero_count}")

    # Verify warm start data was received
    if hasattr(problem, 'warmstart_data') and problem.warmstart_data:
        print(f"\n✅ Warm start data received and populated")
    else:
        print(f"\n⚠️  No warm start data")

except Exception as e:
    print(f"\n❌ Exception during solve: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Client test completed")
print("=" * 60)
