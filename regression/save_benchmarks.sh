#!/bin/bash
# shellcheck disable=SC1090
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
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

# Extract the build meta-data from either the conda environment or the
# cugraph source dir and write out a file which can be read by other
# scripts.  If the cugraph conda packages are present, those take
# precedence, otherwise meta-data will be extracted from the sources.

#module load cuda/11.0.3
activateCondaEnv

echo "Saving benchmarks ........"

echo $1
echo ${RESULTS_DIR}
echo ${PROJECT_VERSION}

python ${CUOPT_SCRIPTS_DIR}/save_benchmark_results.py -b ${RESULTS_DIR} -o ${CUOPT_SCRIPTS_DIR} -c $1

cd ${CUOPT_SCRIPTS_DIR}; git add benchmarks/*; git commit -m "update benchmarks"; git push; cd -
