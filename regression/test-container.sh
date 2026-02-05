#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Creates a conda environment to be used for cuopt benchmarking.

# Abort script on first error
set -e

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)}
source ${PROJECT_DIR}/config.sh

################################################################################

# Test
logger "Testing container image $IMAGE"
python -c "import cuopt; print(cuopt)"

trap '${SCRIPTS_DIR}/write-meta-data.sh' EXIT

# Other scripts look for this to be the last line to determine if this
# script completed successfully. This is only possible because of the
# "set -e" above.
echo "done."
logger "done."
