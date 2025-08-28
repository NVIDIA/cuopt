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
    echo "DEBUG: NBFILENAME=$NBFILENAME"
    echo "DEBUG: NBNAME=$NBNAME"

    # Get the directory where the notebook is located
    NBDIR=$(dirname "$NBFILENAME")

    echo "Changing to directory: ${NBDIR}"
    echo "Original directory was: ${ORIGINAL_DIR}"
    echo "Target directory: ${NBDIR}"
    echo "Full notebook path: ${NBFILENAME}"
    cd "${NBDIR}" || exit 1
    echo "Current directory after cd: $(pwd)"

    # Output the executed notebook in the same folder
    EXECUTED_NOTEBOOK="${NBNAME}-executed.ipynb"

    echo --------------------------------------------------------------------------------
    echo STARTING: "${NBNAME}"
    echo --------------------------------------------------------------------------------

    echo "Executing notebook: ${NBFILENAME}"
    echo "Output will be saved to: ${EXECUTED_NOTEBOOK}"

    echo "Checking for pip install commands in notebook..."
    echo "Notebook file: $NBNAME.ipynb"
    echo "Current directory: $(pwd)"
    if [ -f "$NBNAME.ipynb" ]; then
        echo "Notebook file exists and is readable"
        echo "First few lines of notebook:"
        head -20 "$NBNAME.ipynb" | grep -E "(pip install|!pip)" || echo "No pip install lines found in first 20 lines"
    else
        echo "ERROR: Notebook file not found or not readable"
    fi

    if [[ "$NBNAME" = *trnsport* ]] || [[ "$NBNAME" = *Pulp* ]]; then
        echo "Skipping notebook '${NBNAME}' as it does not contain '01_optimization' in the name."
        cd "$ORIGINAL_DIR"
        continue
    fi


    # Extract pip install lines by parsing the notebook JSON properly
    echo "Extracting pip install commands from notebook JSON..."
    
    # Use python to properly parse the notebook JSON and extract pip commands
    PIP_COMMANDS=$(python3 -c "
import json
import sys

try:
    with open('$NBNAME.ipynb', 'r') as f:
        notebook = json.load(f)
    
    pip_commands = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            lines = source.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('!pip install') or line.startswith('pip install'):
                    # Clean up the line but preserve quotes
                    clean_line = line.strip()
                    if clean_line:
                        pip_commands.append(clean_line)
    
    # Print each command on a separate line for processing
    for cmd in pip_commands:
        print(cmd)
        
except Exception as e:
    print(f'Error parsing notebook: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || true)

    if [ -n "$PIP_COMMANDS" ]; then
        echo "Found pip install commands:"
        echo "Raw commands: '$PIP_COMMANDS'"
        echo "Number of commands found: $(echo "$PIP_COMMANDS" | wc -l)"
        echo "Executing pip install commands..."

        # Process each pip install command
        echo "$PIP_COMMANDS" | while IFS= read -r cmd; do
            if [ -n "$cmd" ] && [[ "$cmd" =~ ^!?pip[[:space:]]+install ]]; then
                echo "Processing command: '$cmd'"
                echo "Running: $cmd"
                # Remove the ! prefix if present for execution
                EXEC_CMD="${cmd#!}"

                echo "DEBUG: Original command: '$cmd'"
                echo "DEBUG: Cleaned command: '$EXEC_CMD'"
                echo "DEBUG: Command length: ${#EXEC_CMD}"
                echo "DEBUG: Command contains 'numpy': $([[ "$EXEC_CMD" == *numpy* ]] && echo "YES" || echo "NO")"
                echo "Executing: $EXEC_CMD"

                # Add --pre to EXEC_CMD if not already present
                if [[ "$EXEC_CMD" =~ ^pip[[:space:]]+install ]] && [[ ! "$EXEC_CMD" =~ [[:space:]]--pre([[:space:]]|$) ]]; then
                    # Check if --extra-index-url is already present
                    if [[ "$EXEC_CMD" =~ [[:space:]]--extra-index-url ]]; then
                        EXEC_CMD="$EXEC_CMD --pre"
                    else
                        EXEC_CMD="$EXEC_CMD --pre --extra-index-url https://pypi.anaconda.org/rapidsai-nightly/simple"
                    fi
                fi

                # Execute pip install commands using eval to properly handle quoted arguments
                if [[ "$EXEC_CMD" =~ ^pip[[:space:]]+install ]]; then
                    echo "Executing pip install command with eval to handle quoted arguments..."
                    if eval "$EXEC_CMD"; then
                        echo "✓ Successfully executed: $cmd"
                    else
                        echo "✗ Failed to execute: $cmd"
                    fi
                else
                    echo "✗ Invalid pip install command format: $EXEC_CMD"
                fi
            elif [ -n "$cmd" ]; then
                echo "Command '$cmd' did not match pip install pattern"
            fi
        done
    fi

    # Extract and execute other shell commands (wget, curl, git, etc.) using proper JSON parsing
    echo "Checking for other shell commands in notebook..."
    OTHER_COMMANDS=$(python3 -c "
import json
import sys

try:
    with open('$NBNAME.ipynb', 'r') as f:
        notebook = json.load(f)
    
    shell_commands = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = ''.join(cell.get('source', []))
            lines = source.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('!'):
                    # Check if it's a shell command we want to execute
                    cmd = line[1:].strip().split()[0] if line[1:].strip() else ''
                    if cmd in ['wget', 'curl', 'git', 'python', 'jupyter', 'ls', 'pwd', 'cd', 'mkdir', 'rm', 'cp', 'mv', 'chmod', 'unzip', 'tar', 'apt', 'yum', 'brew', 'conda', 'npm', 'yarn', 'docker', 'kubectl', 'helm', 'aws', 'gcloud', 'az']:
                        shell_commands.append(line)
    
    # Print each command on a separate line for processing
    for cmd in shell_commands:
        print(cmd)
        
except Exception as e:
    print(f'Error parsing notebook: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null || true)

    if [ -n "$OTHER_COMMANDS" ]; then
        echo "Found other shell commands:"
        echo "Raw commands: '$OTHER_COMMANDS'"
        echo "Executing other shell commands..."

        # Process each shell command
        echo "$OTHER_COMMANDS" | while IFS= read -r cmd; do
            if [ -n "$cmd" ] && [[ "$cmd" =~ ^! ]]; then
                echo "Processing command: '$cmd'"
                echo "Running: $cmd"
                # Remove the ! prefix for execution
                EXEC_CMD="${cmd#!}"

                echo "DEBUG: Original command: '$cmd'"
                echo "DEBUG: Cleaned command: '$EXEC_CMD'"
                echo "DEBUG: Command length: ${#EXEC_CMD}"
                echo "Executing: $EXEC_CMD"

                # Use eval for better argument handling, but with safety checks
                if [ -n "$EXEC_CMD" ]; then
                    # Skip potentially dangerous commands
                    if [[ "$EXEC_CMD" =~ ^(chmod|chown|sudo|su) ]]; then
                        echo "⚠ Skipping potentially dangerous command: $cmd"
                        continue
                    fi
                    
                    echo "Executing shell command with eval to handle quoted arguments..."
                    if $EXEC_CMD; then
                        echo "✓ Successfully executed: $cmd"
                    else
                        echo "✗ Failed to execute: $cmd"
                    fi
                else
                    echo "✗ Invalid command format: $EXEC_CMD"
                fi
            elif [ -n "$cmd" ]; then
                echo "Command '$cmd' did not match shell command pattern"
            fi
        done
    fi

    # Summary of executed commands
    if [ -n "$PIP_COMMANDS" ] || [ -n "$OTHER_COMMANDS" ]; then
        echo "------------------------------------------------------------------------"
        echo "SUMMARY: Commands executed for notebook ${NBNAME}:"
        if [ -n "$PIP_COMMANDS" ]; then
            echo "  Pip install commands: $(echo "$PIP_COMMANDS" | wc -l)"
        fi
        if [ -n "$OTHER_COMMANDS" ]; then
            echo "  Other shell commands: $(echo "$OTHER_COMMANDS" | wc -l)"
        fi
        echo "------------------------------------------------------------------------"
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
