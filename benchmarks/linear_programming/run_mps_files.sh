#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# See the License for the specific language governing permissions and
# limitations under the License.


# This script runs MPS (Mathematical Programming System) files in parallel across multiple GPUs
# It supports running linear programming (LP) and mixed-integer programming (MIP) problems
#
# Usage:
#   ./run_mps_files.sh --path /path/to/mps/files --ngpus 4 --time-limit 3600 --output-dir results
#
# Arguments:
#   --path        : Directory containing .mps files (required)
#   --ngpus       : Number of GPUs to use (default: 1)
#   --time-limit  : Time limit in seconds for each problem (default: 360)
#   --output-dir  : Directory to store output log files (default: current directory)
#   --relaxation  : Run relaxation instead of solving the MIP
#   --mip-heuristics-only : Run mip heuristics only
#   --write-log-file : Write log file
#   --num-cpu-threads : Number of CPU threads to use
#   --presolve : Enable presolve (default: true for MIP problems, false for LP problems)
#   --batch-num : Batch number.  This allows to split the work across multiple batches uniformly when resources are limited.
#   --n-batches : Number of batches
#   --log-to-console : Log to console
#
# Examples:
#   # Run all MPS files in /data/lp using 2 GPUs with 1 hour time limit
#   ./run_mps_files.sh --path /data/lp --ngpus 2 --time-limit 3600
#
#   # Run with specific GPU devices using CUDA_VISIBLE_DEVICES
#   CUDA_VISIBLE_DEVICES=0,2,3 ./run_mps_files.sh --path /data/mip
#
#   # Run with custom output directory
#   ./run_mps_files.sh --path /data/lp --output-dir ~/results/lp_benchmark
#
#   # Run with batch number
#   ./run_mps_files.sh --path /data/lp --ngpus 2 --time-limit 3600 --batch-num 0 --n-batches 2
#
# Notes:
#   - Files are distributed dynamically across available GPUs for load balancing
#   - Each problem's results are logged to a separate file in the output directory
#   - The script uses file locking to safely distribute work across parallel processes
#   - If CUDA_VISIBLE_DEVICES is set, it can override the --ngpus argument

# Help function
print_help() {
    cat << EOF
This script runs MPS (Mathematical Programming System) files in parallel across multiple GPUs.
It supports running linear programming (LP) and mixed-integer programming (MIP) problems.

Usage:
    $(basename "$0") --path /path/to/mps/files [options]

Required Arguments:
    --path PATH          Directory containing .mps files

Optional Arguments:
    --ngpus N           Number of GPUs to use (default: 1)
    --time-limit N      Time limit in seconds for each problem (default: 360)
    --output-dir PATH   Directory to store output log files (default: current directory)
    --relaxation       Run relaxation instead of solving the MIP
    --mip-heuristics-only  Run mip heuristics only
    --write-log-file   Write log file
    --num-cpu-threads  Number of CPU threads to use
    --presolve         Enable presolve (default: true for MIP problems, false for LP problems)
    --batch-num        Batch number
    --n-batches        Number of batches
    --log-to-console   Log to console
    --model-list       File containing a list of models to run
    -h, --help         Show this help message and exit

Examples:
    # Run all MPS files in /data/lp using 2 GPUs with 1 hour time limit
    $(basename "$0") --path /data/lp --ngpus 2 --time-limit 3600

    # Run with specific GPU devices using CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,2,3 $(basename "$0") --path /data/mip

    # Run with custom output directory
    $(basename "$0") --path /data/lp --output-dir ~/results/lp_benchmark

Notes:
    - Files are distributed dynamically across available GPUs for load balancing
    - Each problem's results are logged to a separate file in the output directory
    - The script uses file locking to safely distribute work across parallel processes
    - If CUDA_VISIBLE_DEVICES is set, it can override the --ngpus argument
EOF
    exit 0
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    print_help
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --path)
            echo "MPS_DIR: $2"
            MPS_DIR="$2"
            shift 2
            ;;
        --ngpus)
            echo "GPU_COUNT: $2"
            GPU_COUNT="$2"
            shift 2
            ;;
        --time-limit)
            echo "TIME_LIMIT: $2"
            TIME_LIMIT="$2"
            shift 2
            ;;
        --output-dir)
            echo "OUTPUT_DIR: $2"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --relaxation)
            echo "RELAXATION: true"
            RELAXATION=true
            shift
            ;;
        --mip-heuristics-only)
            echo "MIP_HEURISTICS_ONLY: true"
            MIP_HEURISTICS_ONLY=true
            shift
            ;;
        --write-log-file)
            echo "WRITE_LOG_FILE: true"
            WRITE_LOG_FILE=true
            shift
            ;;
        --num-cpu-threads)
            echo "NUM_CPU_THREADS: $2"
            NUM_CPU_THREADS="$2"
            shift 2
            ;;
        --presolve)
            echo "PRESOLVE: $2"
            PRESOLVE="$2"
            shift 2
            ;;
        --batch-num)
            echo "BATCH_NUM: $2"
            BATCH_NUM="$2"
            shift 2
            ;;
        --n-batches)
            echo "N_BATCHES: $2"
            N_BATCHES="$2"
            shift 2
            ;;
        --log-to-console)
            echo "LOG_TO_CONSOLE: $2"
            LOG_TO_CONSOLE="$2"
            shift 2
            ;;
        --model-list)
            echo "MODEL_LIST: $2"
            MODEL_LIST="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            print_help
            exit 1
            ;;
    esac
done
# Set defaults and validate required arguments
if [[ -z "$MPS_DIR" ]]; then
    echo "Missing required argument: --path MPS_DIR"
    print_help
    exit 1
fi

# Set defaults if not provided
GPU_COUNT=${GPU_COUNT:-1}
TIME_LIMIT=${TIME_LIMIT:-360}
OUTPUT_DIR=${OUTPUT_DIR:-.}
RELAXATION=${RELAXATION:-false}
MIP_HEURISTICS_ONLY=${MIP_HEURISTICS_ONLY:-false}
WRITE_LOG_FILE=${WRITE_LOG_FILE:-false}
NUM_CPU_THREADS=${NUM_CPU_THREADS:-1}
PRESOLVE=${PRESOLVE:-true}
BATCH_NUM=${BATCH_NUM:-0}
N_BATCHES=${N_BATCHES:-1}
LOG_TO_CONSOLE=${LOG_TO_CONSOLE:-true}
MODEL_LIST=${MODEL_LIST:-}
# Determine GPU list
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_LIST+=("$i")
    done
fi
GPU_COUNT=${#GPU_LIST[@]}

# Ensure all entries in MODEL_LIST have .mps extension
if [[ -n "$MODEL_LIST" && -f "$MODEL_LIST" ]]; then
    # Create a temporary file to store the updated model list
    TMP_MODEL_LIST=$(mktemp)
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines
        [[ -z "$line" ]] && continue
        # If the line does not end with .mps, append it
        if [[ "$line" != *.mps ]]; then
            echo "${line}.mps" >> "$TMP_MODEL_LIST"
        else
            echo "$line" >> "$TMP_MODEL_LIST"
        fi
    done < "$MODEL_LIST"
    # Overwrite the original MODEL_LIST with the updated one
    mv "$TMP_MODEL_LIST" "$MODEL_LIST"
fi


# Gather all mps files into an array, either from the model list or from the directory
if [[ -n "$MODEL_LIST" ]]; then
    if [[ ! -f "$MODEL_LIST" ]]; then
        echo "Model list file not found: $MODEL_LIST"
        exit 1
    fi
    mapfile -t mps_files < <(grep -v '^\s*$' "$MODEL_LIST" | sed "s|^|$MPS_DIR/|")
    # Optionally, check that all files exist
    missing_files=()
    for f in "${mps_files[@]}"; do
        if [[ ! -f "$f" ]]; then
            missing_files+=("$f")
        fi
    done
    if (( ${#missing_files[@]} > 0 )); then
        echo "The following files from the model list do not exist in $MPS_DIR:"
        for f in "${missing_files[@]}"; do
            echo "  $f"
        done
        exit 1
    fi
else
    mapfile -t mps_files < <(ls "$MPS_DIR"/*.mps)
fi

# Calculate batch size and start/end indices
batch_size=$(( (${#mps_files[@]} + N_BATCHES - 1) / N_BATCHES ))
start_idx=$((BATCH_NUM * batch_size))
end_idx=$((start_idx + batch_size))

# Ensure end_idx doesn't exceed array length
if ((end_idx > ${#mps_files[@]})); then
    end_idx=${#mps_files[@]}
fi

# Extract subset of files for this batch
mps_files=("${mps_files[@]:$start_idx:$((end_idx-start_idx))}")


file_count=${#mps_files[@]}

# Initialize the index file for locking mechanism
INDEX_FILE="/tmp/mps_file_index.$$"

# Remove the index file if it exists
rm -f "$INDEX_FILE"

# Initialize the index file
echo 0 > "$INDEX_FILE"

# Adjust GPU_COUNT if there are fewer files than GPUs
if ((GPU_COUNT > file_count)); then
    echo "Reducing number of GPUs from $GPU_COUNT to $file_count since there are fewer files than GPUs"
    GPU_COUNT=$file_count
    # Trim GPU_LIST to match file_count
    GPU_LIST=("${GPU_LIST[@]:0:$file_count}")
fi

# Function for each worker (GPU)
worker() {
    local gpu_id=$1
    # echo "GPU $gpu_id Using index file $INDEX_FILE"
    while :; do
        # Atomically get and increment the index
        my_index=$(flock "$INDEX_FILE" bash -c '
            idx=$(<"$0")
            if (( idx >= '"$file_count"' )); then
                echo -1
            else
                echo $((idx+1)) > "$0"
                echo $idx
            fi
        ' "$INDEX_FILE")

        if (( my_index == -1 )); then
            return
        fi

        mps_file="${mps_files[my_index]}"
        echo "GPU $gpu_id processing $my_index"

        # Build arguments string
        args=""
        if [ -n "$NUM_CPU_THREADS" ]; then
            args="$args --num-cpu-threads $NUM_CPU_THREADS"
        fi
        if [ "$MIP_HEURISTICS_ONLY" = true ]; then
            args="$args --mip-heuristics-only true"
        fi
        if [ "$WRITE_LOG_FILE" = true ]; then
            args="$args --log-file $OUTPUT_DIR/$(basename "${mps_file%.mps}").log"
        fi
        if [ "$RELAXATION" = true ]; then
            args="$args --relaxation"
        fi
        args="$args --log-to-console $LOG_TO_CONSOLE"
        args="$args --presolve $PRESOLVE"

        CUDA_VISIBLE_DEVICES=$gpu_id cuopt_cli "$mps_file" --time-limit $TIME_LIMIT $args
    done
}

# Start one worker per GPU in the list
for gpu_id in "${GPU_LIST[@]}"; do
    worker "$gpu_id" &
done

wait

# Remove the index file
rm -f "$INDEX_FILE"
