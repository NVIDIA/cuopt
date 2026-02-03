#!/bin/bash
# shellcheck disable=SC1090
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Abort script on first error to ensure script-env.sh is sourced.
set -e

if [[ -v SLURM_NODEID ]]; then
    echo "Detected the env var SLURM_NODEID is set. Is this script running on a compute node?"
    echo "This script must be run *outside* of a slurm job (this script starts slurm jobs, but is not a job itself)."
    exit 1
fi

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)}

source ${PROJECT_DIR}/config.sh
source ${PROJECT_DIR}/functions.sh

RUN_BENCHMARKS=0

if hasArg --benchmark; then
    RUN_BENCHMARKS=1
fi

if (! hasArg --test) && (! hasArg --benchmark); then
    RUN_BENCHMARKS=1
fi

################################################################################

# Create a results dir unique for this run
setupResultsDir

# Switch to allowing errors from commands, since test failures will
# result in non-zero return codes and this script should attempt to
# run all tests.
set +e

################################################################################
logger "Testing cuOpt in container..."
srun \
    --account $ACCOUNT \
    --partition $PARTITION \
    --job-name=test-container.testing \
    --nodes 1 \
    --gpus-per-node 1 \
    --time=120 \
    --export=ALL \
    --container-mounts=${CUOPT_SCRIPTS_DIR}:${CUOPT_SCRIPTS_DIR},${OUTPUT_DIR}:${OUTPUT_DIR} \
    --container-image=$IMAGE \
    --output=$BUILD_LOG_FILE \
    bash ${PROJECT_DIR}/test-container.sh
TESTING_FAILED=$?
logger "done testing container, return code was $TESTING_FAILED"


if [[ $TESTING_FAILED == 0 ]]; then

    ############################################################################
    # Setup and run tests
    if [[ $RUN_BENCHMARKS == 1 ]]; then
        logger "Running benchmarks..."
        logger "GPUs per node : $GPUS_PER_NODE"
        # SNMG tests - run in parallel
        srun \
            --account $ACCOUNT \
            --partition $PARTITION \
            --job-name=run-nightly-benchmarks \
            --nodes 1 \
            --gpus-per-node $GPUS_PER_NODE \
            --time=4:00:00 \
            --export=ALL \
            --exclusive \
            -K \
            --container-mounts ${ROUTING_CONFIGS_PATH}:${ROUTING_CONFIGS_PATH},${CUOPT_SCRIPTS_DIR}:${CUOPT_SCRIPTS_DIR},${OUTPUT_DIR}:${OUTPUT_DIR} \
            --container-image=$IMAGE \
            --output=${BENCHMARK_RESULTS_DIR}/benchmark_routing_log.txt \
            bash ${CUOPT_SCRIPTS_DIR}/routing_regression_test.sh &
        PID_1=$!
        logger "Process ID $PID_1 in background"

        srun \
            --account $ACCOUNT \
            --partition $PARTITION \
            --job-name=run-nightly-benchmarks \
            --nodes 1 \
            --gpus-per-node $GPUS_PER_NODE \
            --time=4:00:00 \
            --export=ALL \
            --exclusive \
            -K \
            --container-mounts ${LP_DATASETS_PATH}:${LP_DATASETS_PATH},${CUOPT_SCRIPTS_DIR}:${CUOPT_SCRIPTS_DIR},${OUTPUT_DIR}:${OUTPUT_DIR} \
            --container-image=$IMAGE \
            --output=${BENCHMARK_RESULTS_DIR}/benchmark_lp_log.txt \
            bash ${CUOPT_SCRIPTS_DIR}/lp_regression_test.sh &
        PID_2=$!

        srun \
            --account $ACCOUNT \
            --partition $PARTITION \
            --job-name=run-nightly-benchmarks \
            --nodes 1 \
            --gpus-per-node $GPUS_PER_NODE \
            --time=4:00:00 \
            --export=ALL \
            --exclusive \
            -K \
            --container-mounts ${MIP_DATASETS_PATH}:${MIP_DATASETS_PATH},${CUOPT_SCRIPTS_DIR}:${CUOPT_SCRIPTS_DIR},${OUTPUT_DIR}:${OUTPUT_DIR} \
            --container-image=$IMAGE \
            --output=${BENCHMARK_RESULTS_DIR}/benchmark_mip_log.txt \
            bash ${CUOPT_SCRIPTS_DIR}/mip_regression_test.sh &
        PID_3=$!

        wait $PID_1 $PID_2 $PID_3
    fi

else   # if [[ $TESTING_FAILED == 0 ]]
    logger "Container testing Failed!"
fi

################################################################################
# Send report based on contents of $RESULTS_DIR
# These steps do not require a worker node.

# When running both testing and benchmark and if some benchmarks fail,
# the entire nightly will fail. The benchmark logs reported on Slack
# contains information about the failures.
logger "Generating report"

if [ -f $METADATA_FILE ]; then
    source $METADATA_FILE
fi

# Copy all config files to one folder
cp $ROUTING_CONFIGS_PATH/*config.json $LP_DATASETS_PATH/*config.json $MIP_DATASETS_PATH/*config.json $ALL_CONFIGS_PATH/

RUN_ASV_OPTION=""
if hasArg --skip-asv; then
    logger "Skipping running ASV"
else
    # Only create/update the asv database if there is both a commit Hash and a branch otherwise
    # asv will return an error. If there is $PROJECT_BUILD, that implies there is Neither the
    # git commit hash nor the branch which are required to create/update the asv db
    if [[ "$PROJECT_BUILD" == "" ]]; then
        # Update/create the ASV database
        logger "Updating ASV database"
        python $PROJECT_DIR/update_asv_database.py --commitHash=$PROJECT_VERSION --repo-url=$PROJECT_REPO_URL --branch=$PROJECT_REPO_BRANCH --commitTime=$PROJECT_REPO_TIME --results-dir=$RESULTS_DIR --machine-name=$MACHINE --gpu-type=$GPU_TYPE --configs=$ALL_CONFIGS_PATH
        RUN_ASV_OPTION=--run-asv
        logger "Updated ASV database"
    else
        logger "Detected a conda install, cannot run ASV since a commit hash/time is needed."
    fi
fi

# The cuopt pull has missing .git folder which causes subsequent runs, lets delete and pull it fresh everytime.
rm -rf $RESULTS_DIR/benchmarks/results/asv/cuopt/
rm -rf $RESULTS_DIR/tests

${SCRIPTS_DIR}/create-html-reports.sh $RUN_ASV_OPTION

if hasArg --skip-sending-report; then
    logger "Skipping sending report."
else
    logger "Uploading to S3, posting to Slack"
    ${PROJECT_DIR}/send-slack-report.sh
fi

logger "cronjob.sh done."
