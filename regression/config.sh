#!/bin/bash
# shellcheck disable=all
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


THIS_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)

CUOPT_SCRIPTS_DIR=$THIS_DIR
OUTPUT_DIR=$SCRATCH_DIR/benchmark_runs/

ACCOUNT=datascience_rapids_testing
PARTITION="batch"
GPUS_PER_NODE=8

# Path to the squashs file containing the container image
IMAGE="nvidia/cuopt:26.2.0a-cuda12.9-py3.13"

ALL_CONFIGS_PATH=$SCRATCH_DIR/configs/
ROUTING_CONFIGS_PATH=$SCRATCH_DIR/routing_configs/
ROUTING_DATASETS_PATH=$SCRATCH_DIR/routing_datasets/
LP_DATASETS_PATH=$SCRATCH_DIR/lp_datasets/
MIP_DATASETS_PATH=$SCRATCH_DIR/mip_datasets/

STATUS_FILE=$OUTPUT_DIR/status.txt
WORKER_RMM_POOL_SIZE=${WORKER_RMM_POOL_SIZE:-24G}

DATASETS_DIR=$SCRATCH_DIR/datasets

# Assume CUOPT_SLACK_APP_ID is defined!
CUOPT_SLACK_APP_ID="MY_SLACK_APP_ID"
WEBHOOK_URL=${WEBHOOK_URL:-https://hooks.slack.com/services/${CUOPT_SLACK_APP_ID}}
S3_FILE_PREFIX="MY_S3_FILE_PREFIX"
S3_URL_PREFIX="MY_S3_URL_PREFIX"

# Most are defined using the bash := or :- syntax, which means they
# will be set only if they were previously unset. The project config
# is loaded first, which gives it the opportunity to override anything
# in this file that uses that syntax.  If there are variables in this
# file that should not be overridded by a project, then they will
# simply not use that syntax and override, since these variables are
# read last.
RESULTS_ARCHIVE_DIR=$OUTPUT_DIR/results
RESULTS_DIR=$RESULTS_ARCHIVE_DIR/latest
METADATA_FILE=$RESULTS_DIR/metadata.sh
WORKSPACE=$OUTPUT_DIR/workspace
TESTING_DIR=$WORKSPACE/testing
BENCHMARK_DIR=$WORKSPACE/benchmark
SCRIPTS_DIR=$THIS_DIR

BUILD_LOG_FILE=$RESULTS_DIR/build_log.txt
DATE=${DATE:-$(date --utc "+%Y-%m-%d_%H:%M:%S")_UTC}

# vars that are not overridden by the project.

# These must remain relative to $RESULTS_DIR since some scripts assume
# that, and also assume the names "tests" and "benchmarks", and
# therefore cannot be overridden by a project.
TESTING_RESULTS_DIR=${RESULTS_DIR}/tests
BENCHMARK_RESULTS_DIR=${RESULTS_DIR}/benchmarks
