#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)}
source ${PROJECT_DIR}/config.sh
source ${PROJECT_DIR}/functions.sh

################################################################################

# Extract the build meta-data from either the conda environment or the
# cugraph source dir and write out a file which can be read by other
# scripts.  If the cugraph conda packages are present, those take
# precedence, otherwise meta-data will be extracted from the sources.

GIT_COMMIT=$(</opt/cuopt/COMMIT_SHA)
LOG_PATH=${RESULTS_DIR}/benchmarks/

nvidia-smi

mkdir -p ${RESULTS_DIR}/benchmarks/results/csvs/

logger "Running mip tests ........"
python ${CUOPT_SCRIPTS_DIR}/benchmark_scripts/benchmark.py -c ${MIP_DATASETS_PATH} -r ${RESULTS_DIR}/benchmarks/results/csvs/ -g ${GIT_COMMIT} -l ${LOG_PATH} -s ${RESULTS_DIR}/benchmarks/results/mip_tests_status.txt -n ${GPUS_PER_NODE} -t mip
logger "Completed mip tests ........"
