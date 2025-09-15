#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2019-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations.
#
# This script executes Jupyter notebooks directly using nbconvert.

set +e           # do not abort the script on error
set -o pipefail  # piped commands propagate their error
set -E           # ERR traps are inherited by subcommands
trap "EXITCODE=1" ERR

# Save the original directory
ORIGINAL_DIR=$(pwd)

EXITCODE=0

for nb in "$@"; do
    NBFILENAME=$nb
    NBNAME=${NBFILENAME%.*}
    NBNAME=${NBNAME##*/}

    # Get the directory where the notebook is located
    NBDIR=$(dirname "$NBFILENAME")

    cd "${NBDIR}" || exit 1

    # Output the executed notebook in the same folder
    EXECUTED_NOTEBOOK="${NBNAME}-executed.ipynb"

    echo --------------------------------------------------------------------------------
    echo STARTING: "${NBNAME}"
    echo --------------------------------------------------------------------------------

    # Skip notebooks that are not yet supported
    SKIP_NOTEBOOKS=(
        "trnsport_cuopt"
        "Production_Planning_Example_Pulp"
        "Simple_LP_pulp"
        "Simple_MIP_pulp"
        "Sudoku_pulp"
    )

    for skip in "${SKIP_NOTEBOOKS[@]}"; do
        if [[ "$NBNAME" == "$skip"* ]]; then
            echo "Skipping notebook '${NBNAME}' as it matches skip pattern '${skip}'"
            cd "$ORIGINAL_DIR" || exit 1
            continue 2
        fi
    done

    rapids-logger "Running commands from notebook: ${NBNAME}.ipynb"

    python3 "$ORIGINAL_DIR/../ci/utils/notebook_command_extractor.py" "$NBNAME.ipynb" --verbose

    rapids-logger "Executing notebook: ${NBNAME}.ipynb"
    # Execute notebook with default kernel
    jupyter nbconvert --execute "${NBNAME}.ipynb" --to notebook --output "${EXECUTED_NOTEBOOK}" --ExecutePreprocessor.kernel_name="python3"

    if [ $? -eq 0 ]; then
        echo "Notebook executed successfully: ${EXECUTED_NOTEBOOK}"
    else
        echo "ERROR: Failed to execute notebook: ${NBFILENAME}"
        EXITCODE=1
    fi

    cd "${ORIGINAL_DIR}" || exit 1
done

exit ${EXITCODE}
