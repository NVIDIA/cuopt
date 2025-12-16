#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Abort script on first error
set -e

DELAY=30

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
source ${PROJECT_DIR}/config.sh
source ${PROJECT_DIR}/functions.sh

################################################################################

# Extract the build meta-data from either the conda environment or the
# cugraph source dir and write out a file which can be read by other
# scripts.  If the cugraph conda packages are present, those take
# precedence, otherwise meta-data will be extracted from the sources.

GIT_COMMIT="abc" #$(cd ${WORKSPACE}/${REPO_DIR_NAME}; git rev-parse HEAD)
LOG_PATH=${RESULTS_DIR}/benchmarks/

nvidia-smi

mkdir -p ${RESULTS_DIR}/benchmarks/results/csvs/
#rm -rf ${WORKSPACE}/${RESULT_DIR_NAME}/data/regressions.csv


logger "Running routing tests ........"
python ${CUOPT_SCRIPTS_DIR}/benchmark_scripts/benchmark.py -c ${ROUTING_CONFIGS_PATH} -r ${RESULTS_DIR}/benchmarks/results/csvs/ -g ${GIT_COMMIT} -l ${LOG_PATH} -s ${RESULTS_DIR}/benchmarks/results/routing_tests_status.txt -n ${GPUS_PER_NODE} -t routing
logger "Completed routing tests ........"


#cp ${WORKSPACE}/${RESULT_DIR_NAME}/data/* ${RESULTS_DIR}/benchmarks/results/csvs/
