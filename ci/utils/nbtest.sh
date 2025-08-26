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

    echo "Changing to directory: ${NBDIR}"
    cd "${NBDIR}" || exit 1

    # Output the executed notebook in the same folder
    EXECUTED_NOTEBOOK="${NBNAME}-executed.ipynb"

    echo --------------------------------------------------------------------------------
    echo STARTING: "${NBNAME}"
    echo --------------------------------------------------------------------------------

    echo "Executing notebook: ${NBFILENAME}"
    echo "Output will be saved to: ${EXECUTED_NOTEBOOK}"

    # Extract and execute pip install commands from notebook
    echo "Checking for pip install commands in notebook..."
    PIP_COMMANDS=$(grep -h "pip install" "$NBNAME.ipynb" 2>/dev/null | grep -v "#" | sed 's/^[[:space:]]*//' | sed 's/^["'"'"']*//' | sed 's/["'"'"']*$//' || true)

    if [ -n "$PIP_COMMANDS" ]; then
        echo "Found pip install commands:"
        echo "$PIP_COMMANDS"
        echo "Executing pip install commands..."
        echo "$PIP_COMMANDS" | while read -r cmd; do
            echo "Processing command: '$cmd'"
            if [[ "$cmd" =~ ^!?pip[[:space:]]+install ]]; then
                echo "Running: $cmd"
                # Remove the ! prefix if present for execution
                EXEC_CMD="${cmd#!}"
                # Clean up escaped quotes, extra quotes, and newlines
                EXEC_CMD=$(echo "$EXEC_CMD" | sed 's/\\"/"/g' | sed 's/^"//' | sed 's/"$//' | tr -d '\n\r')
                echo "Executing: $EXEC_CMD"
                eval "$EXEC_CMD"
                if [ $? -eq 0 ]; then
                    echo "✓ Successfully executed: $cmd"
                else
                    echo "✗ Failed to execute: $cmd"
                fi
            else
                echo "Command '$cmd' did not match pip install pattern"
            fi
        done
    fi

    # Execute notebook with default kernel
    jupyter nbconvert --execute "${NBNAME}.ipynb" --to notebook --output "${EXECUTED_NOTEBOOK}" --ExecutePreprocessor.kernel_name="python3"

    if [ $? -eq 0 ]; then
        echo "Notebook executed successfully: ${EXECUTED_NOTEBOOK}"
    else
        echo "ERROR: Failed to execute notebook: ${NBFILENAME}"
        EXITCODE=1
    fi

    echo "Returning to original directory: ${ORIGINAL_DIR}"
    cd "${ORIGINAL_DIR}" || exit 1
done

exit ${EXITCODE}
