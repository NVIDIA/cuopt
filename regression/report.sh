#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.

# Creates a conda environment to be used for cuopt benchmarking.

# Abort script on first error
set -e
# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
if [ -n "$RAPIDS_MG_TOOLS_DIR" ]; then
    source ${RAPIDS_MG_TOOLS_DIR}/script-env.sh
elif [ -n "$(which script-env.sh)" ]; then
    source $(which script-env.sh)
else
    echo "Error: \$RAPIDS_MG_TOOLS_DIR/script-env.sh could not be read nor was script-env.sh in PATH."
    exit 1
fi

activateCondaEnv

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
        python $PROJECT_DIR/update_asv_database.py --commitHash=$PROJECT_VERSION --repo-url=$PROJECT_REPO_URL --branch=$PROJECT_REPO_BRANCH --commitTime=$PROJECT_REPO_TIME --results-dir=$RESULTS_DIR --machine-name=$MACHINE --gpu-type=$GPU_TYPE
        RUN_ASV_OPTION=--run-asv
    else
        logger "Detected a conda install, cannot run ASV since a commit hash/time is needed."
    fi
fi

if hasArg --spreadsheet; then
    logger "Generating spreadsheet"
    export SPREADSHEET_URL=$(python $PROJECT_DIR/gsheet-report.py --results-dir=$RESULTS_DIR |grep "spreadsheet url is"|cut -d ' ' -f4)
    #python $PROJECT_DIR/gsheet-report.py --results-dir=$RESULTS_DIR

fi

${SCRIPTS_DIR}/create-html-reports.sh $RUN_ASV_OPTION

if hasArg --skip-sending-report; then
    logger "Skipping sending report."
else
    logger "Uploading to S3, posting to Slack"
    ${PROJECT_DIR}/send-slack-report.sh
fi

logger "cronjob.sh done."
