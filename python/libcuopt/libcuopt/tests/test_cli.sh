#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Add cuopt_cli path to PATH variable
if command -v pyenv &> /dev/null; then
    PATH="$(pyenv root)/versions/$(pyenv version-name)/bin:$PATH"
    export PATH
fi

# Test the CLI

# Add a test for the help command
cuopt_cli --help | grep "Usage: cuopt_cli" > /dev/null || (echo "Expected usage information not found" && exit 1)

# Add a test with a simple linear programming problem

# Run solver and check for optimal status - fail if not found

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.mps | grep -q "Status: " || (echo "Expected solution not found" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp | grep -q "Status: " || (echo "Expected solution not found for .lp" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp.gz | grep -q "Status: " || (echo "Expected solution not found for .lp.gz" && exit 1)

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/linear_programming/good-mps-1.lp.bz2 | grep -q "Status: " || (echo "Expected solution not found for .lp.bz2" && exit 1)

# Add a for mixed integer programming test with options

cuopt_cli "${RAPIDS_DATASET_ROOT_DIR}"/mip/sample.mps --mip-absolute-gap 0.01 --time-limit 10 | grep -q "Solution objective" || (echo "Expected solution objective not found" && exit 1)

# Regression for opportunistic concurrent root LP solve on an infeasible MILP.
# Default opportunistic mode should return Infeasible instead of aborting.
tmp_mps="$(mktemp "${TMPDIR:-/tmp}/cuopt-infeasible-milp.XXXXXX.mps")"
trap 'rm -f "${tmp_mps}"' EXIT
python3 - "${tmp_mps}" <<'PY'
import sys

path = sys.argv[1]
n = 120
a_vals = [1.0 + ((i * 37) % 90) / 10.0 for i in range(n)]
b_vals = [a + (((i * 17) % 11) - 5.0) / 20.0 for i, a in enumerate(a_vals)]
need_a = 0.8 * sum(a_vals)
cap_b = 0.3 * sum(b_vals)

with open(path, "w", encoding="utf-8") as f:
    f.write("NAME          INFEAS_MILP\n")
    f.write("ROWS\n")
    f.write(" N  COST\n")
    f.write(" G  NEED_A\n")
    f.write(" L  CAP_B\n")
    f.write("COLUMNS\n")
    f.write("    M1        'MARKER'                 'INTORG'\n")
    for i, (a, b) in enumerate(zip(a_vals, b_vals)):
        f.write(f"    x{i:<8} COST      1                      NEED_A    {a:.17g}\n")
        f.write(f"    x{i:<8} CAP_B     {b:.17g}\n")
    f.write("    M2        'MARKER'                 'INTEND'\n")
    f.write("RHS\n")
    f.write(f"    RHS       NEED_A    {need_a:.17g}       CAP_B     {cap_b:.17g}\n")
    f.write("BOUNDS\n")
    for i in range(n):
        f.write(f" UP BND       x{i:<8} 1\n")
    f.write("ENDATA\n")
PY
cuopt_cli "${tmp_mps}" --time-limit 30 --mip-determinism-mode 0 | grep -q "Termination Status: Infeasible" || (echo "Expected infeasible status for opportunistic MILP" && exit 1)
