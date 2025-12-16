#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script downloads and benchmarks the LP instances from the Mittlemann collection using cuOpt LP Solver.
# Instances will not be downloaded if they already exist.


# Get absolute path to script directory and navigate up to cuopt root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUOPT_HOME="$( cd "$SCRIPT_DIR/../../../" && pwd )"

python $CUOPT_HOME/benchmarks/linear_programming/utils/get_datasets.py -LPfeasible -instance-download-path $CUOPT_HOME/benchmarks/linear_programming/datasets

echo "Download done"

# EAGER module loading to simulate real-life condition
export CUDA_MODULE_LOADING=EAGER

# Benchmark all instances (cuOpt needs to be compiled first, you can compile in LP only mode and you should turn on BUILD_LP_BENCHMARKS)
for instance in ${CUOPT_HOME}/benchmarks/linear_programming/datasets/*/ ; do
    # Will generate the solver log for each instance. Could addtionally generate the solution file by uncommenting the --solution-path
    instance_name=$(basename $instance)
    echo "Parsing ${instance_name}.mps then solving"
    ${CUOPT_HOME}/cpp/build/solve_LP --path ${CUOPT_HOME}/benchmarks/linear_programming/datasets/${instance_name}/${instance_name}.mps --time-limit 3600 # --solution-path $CUOPT_HOME/benchmarks/linear_programming/datasets/$instance.sol
done

echo "Benchmark done"
