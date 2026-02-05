#!/bin/bash
# shellcheck disable=SC1090
# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Abort script on first error
set -e


# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)}
if [ -n "$RAPIDS_MG_TOOLS_DIR" ]; then
    source ${RAPIDS_MG_TOOLS_DIR}/script-env.sh
elif [ -n "$(which script-env.sh)" ]; then
    source "$(which script-env.sh)"
else
    echo "Error: \$RAPIDS_MG_TOOLS_DIR/script-env.sh could not be read nor was script-env.sh in PATH."
    exit 1
fi

################################################################################
# Need to activate a specific conda env to use AWS CLI tools.
# NOTE: the AWS CLI tools are also available to install directly:
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html

activateCondaEnv

# FIXME : These env variables should already be exported like RESULTS_DIR but
# verify that before removing them
TESTING_RESULTS_DIR=${RESULTS_DIR}/tests
BENCHMARK_RESULTS_DIR=${RESULTS_DIR}/benchmarks

# Get the overall status.
ALL_REPORTS=$(find -L $RESULTS_DIR -maxdepth 2 -name "*-results-*.txt")
BENCHMARK_REPORT=$(find -L $BENCHMARK_RESULTS_DIR -maxdepth 1 -name "*-results-*.txt")
TEST_REPORT=$(find -L $TESTING_RESULTS_DIR -maxdepth 1 -name "*-results-*.txt")

STATUS='FAILED'
STATUS_IMG='https://img.icons8.com/cotton/80/000000/cancel--v1.png'
if [ "$ALL_REPORTS" != "" ]; then
    if ! (grep -w FAILED $ALL_REPORTS > /dev/null); then
        STATUS='PASSED'
        STATUS_IMG='https://img.icons8.com/bubbles/100/000000/approval.png'
    fi

fi

# Generate a one-line summary based on existance of certain reports, etc.
if [[ "$ALL_REPORTS" == "" ]]; then
    ONE_LINE_SUMMARY="*Build failed*"
elif [[ "$STATUS" == "FAILED" ]]; then
    if (grep -w FAILED $BENCHMARK_REPORT > /dev/null); then
        ONE_LINE_SUMMARY="*One or more benchmarks failed*"
    fi
    if (grep -w FAILED $TEST_REPORT > /dev/null); then
        ONE_LINE_SUMMARY="*One or more tests failed*"
    fi
    if (grep -w FAILED $TEST_REPORT > /dev/null) && (grep -w FAILED $BENCHMARK_REPORT > /dev/null); then
        ONE_LINE_SUMMARY="*One or more tests and benchmarks failed*"
    fi
else
    ONE_LINE_SUMMARY="Build succeeded, all tests and benchmarks passed"
fi

RESULTS_DIR_NAME=$(basename "$(getNonLinkedFileName $RESULTS_DIR)")

# Upload everything
logger "Uploading all files in $RESULTS_DIR ..."
logger "Uploading all files in $RESULTS_DIR_NAME ..."
aws s3 cp --follow-symlinks --acl public-read --recursive ${RESULTS_DIR} ${S3_FILE_PREFIX}/${RESULTS_DIR_NAME}
logger "done uploading all files in $RESULTS_DIR"

# Set vars used in the report
PROJECT_VERSION_STRING=""
PROJECT_VERSION=""
PROJECT_BUILD=""
PROJECT_CHANNEL=""
PROJECT_REPO_URL=""
PROJECT_REPO_BRANCH=""
if [ -f $METADATA_FILE ]; then
    source $METADATA_FILE
fi
# Assume if PROJECT_BUILD is set then a conda version string should be
# created, else a git version string.
if [[ "$PROJECT_BUILD" != "" ]]; then
    PROJECT_VERSION_STRING="    cuOpt ver.: $PROJECT_VERSION
    build:        $PROJECT_BUILD
    channel:      $PROJECT_CHANNEL"
else
    PROJECT_VERSION_STRING="    cuOpt ver.: $PROJECT_VERSION
    repo:         $PROJECT_REPO_URL
    branch:       $PROJECT_REPO_BRANCH"
fi

export STATUS
export STATUS_IMG
export PROJECT_VERSION_STRING
HUMAN_READABLE_DATE="$(date '+`%D`, `%H:%M` (PT)')"
export HUMAN_READABLE_DATE
# These files should be created by create-html-reports.sh
export REPORT_URL="${S3_URL_PREFIX}/${RESULTS_DIR_NAME}/report.html"
export ASV_URL="${S3_URL_PREFIX}/${RESULTS_DIR_NAME}/benchmarks/asv/html/index.html"
export LOGS_URL="${S3_URL_PREFIX}/${RESULTS_DIR_NAME}/index.html"
# export SPREADSHEET_URL=$SPREADSHEET_URL
export ONE_LINE_SUMMARY

echo
echo "REPORT_URL: ${REPORT_URL}"
# echo "SPREADSHEET_URL: ${SPREADSHEET_URL}"

if hasArg --skip-sending-report; then
    logger "Skipping sending Slack report."
else
    echo "$(envsubst < ${PROJECT_DIR}/slack_msg.json)"
    curl -X POST \
         -H 'Content-type: application/json' \
         --data "$(envsubst < ${PROJECT_DIR}/slack_msg.json)" \
         ${WEBHOOK_URL}
fi
