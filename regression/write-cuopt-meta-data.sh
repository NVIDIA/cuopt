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

# Extract the build meta-data from either the conda environment or the
# cugraph source dir and write out a file which can be read by other
# scripts.  If the cugraph conda packages are present, those take
# precedence, otherwise meta-data will be extracted from the sources.

#module load cuda/11.0.3
activateCondaEnv

nvidia-smi


# auto-detect based on if the libcugraph conda pacakge is installed
# (a from-source build does not have a libcugraph package registered
# in the conda env since it is installed directly via the build).
if (conda list | grep -q libcuopt); then
    ${SCRIPTS_DIR}/write-meta-data.sh --from-conda
else
    ${SCRIPTS_DIR}/write-meta-data.sh --from-source
fi
