#!/bin/bash
# Copyright (c) 2021, NVIDIA CORPORATION.

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

################################################################################



if [ ! -d ${WORKSPACE}/${REPO_DIR_NAME} ]; then
    cloneRepo "$CUGRAPH_REPO_URL" $REPO_DIR_NAME $WORKSPACE
fi


rm -rf ${BENCHMARK_DIR}
mkdir -p ${BENCHMARK_DIR}
cp -r ${WORKSPACE}/${REPO_DIR_NAME}/benchmarks/python_e2e ${BENCHMARK_DIR}