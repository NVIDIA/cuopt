#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# This script runs MPS (Mathematical Programming System) files in parallel across multiple GPUs
# It supports running linear programming (LP) and mixed-integer programming (MIP) problems
#
# Usage:
#   ./run_mps_files.sh --path /path/to/mps/files --ngpus 4 --time-limit 3600 --output-dir results
#
# Arguments:
#   --path        : Directory containing .mps files (required)
#   --ngpus       : Number of GPUs to use (default: 1)
#   --gpus-per-instance : Number of GPUs per benchmark instance (default: 1, set to 2 for multi-GPU)
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
#   # Run with multi-GPU mode using 4 GPUs (2 GPUs per instance, 2 parallel instances)
#   ./run_mps_files.sh --path /data/lp --ngpus 4 --gpus-per-instance 2 --time-limit 3600
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
#   - With --gpus-per-instance 2, benchmarks use 2 GPUs each (e.g., 4 GPUs = 2 parallel instances)

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
    --gpus-per-instance N  Number of GPUs per benchmark instance (default: 1, set to 2 for multi-GPU)
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
    --pdlp-tolerances  Tolerances for PDLP solver (default: 1e-4)
    -h, --help         Show this help message and exit

Examples:
    # Run all MPS files in /data/lp using 2 GPUs with 1 hour time limit
    $(basename "$0") --path /data/lp --ngpus 2 --time-limit 3600

    # Run with multi-GPU mode using 4 GPUs (2 GPUs per instance, 2 parallel instances)
    $(basename "$0") --path /data/lp --ngpus 4 --gpus-per-instance 2 --time-limit 3600

    # Run with specific GPU devices using CUDA_VISIBLE_DEVICES
    CUDA_VISIBLE_DEVICES=0,2,3 $(basename "$0") --path /data/mip

    # Run with custom output directory
    $(basename "$0") --path /data/lp --output-dir ~/results/lp_benchmark

Notes:
    - Files are distributed dynamically across available GPUs for load balancing
    - Each problem's results are logged to a separate file in the output directory
    - The script uses file locking to safely distribute work across parallel processes
    - If CUDA_VISIBLE_DEVICES is set, it can override the --ngpus argument
    - With --gpus-per-instance 2, benchmarks use 2 GPUs each (e.g., 4 GPUs = 2 parallel instances)
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
        --gpus-per-instance)
            echo "GPUS_PER_INSTANCE: $2"
            GPUS_PER_INSTANCE="$2"
            shift 2
            ;;
        --time-limit)
            echo "TIME_LIMIT: $2"
            TIME_LIMIT="$2"
            shift 2
            ;;
        --iteration-limit)
            echo "ITERATION_LIMIT: $2"
            ITERATION_LIMIT="$2"
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
        --method)
            echo "METHOD: $2"
            METHOD="$2"
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
        --pdlp-tolerances)
            echo "PDLP_TOLERANCES: $2"
            PDLP_TOLERANCES="$2"
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
GPUS_PER_INSTANCE=${GPUS_PER_INSTANCE:-1}
TIME_LIMIT=${TIME_LIMIT:-360}
OUTPUT_DIR=${OUTPUT_DIR:-.}
RELAXATION=${RELAXATION:-false}
MIP_HEURISTICS_ONLY=${MIP_HEURISTICS_ONLY:-false}
WRITE_LOG_FILE=${WRITE_LOG_FILE:-false}
NUM_CPU_THREADS=${NUM_CPU_THREADS:--1}
BATCH_NUM=${BATCH_NUM:-0}
N_BATCHES=${N_BATCHES:-1}
LOG_TO_CONSOLE=${LOG_TO_CONSOLE:-true}
MODEL_LIST=${MODEL_LIST:-}
PDLP_TOLERANCES=${PDLP_TOLERANCES:-1e-4}

# Validate GPUS_PER_INSTANCE
if [[ "$GPUS_PER_INSTANCE" != "1" && "$GPUS_PER_INSTANCE" != "2" ]]; then
    echo "Error: --gpus-per-instance must be 1 or 2"
    exit 1
fi

# Determine GPU list
if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
    IFS=',' read -ra GPU_LIST <<< "$CUDA_VISIBLE_DEVICES"
else
    GPU_LIST=()
    for ((i=0; i<GPU_COUNT; i++)); do
        GPU_LIST+=("$i")
    done
fi
# Check that requested number of GPUs does not exceed available GPUs (if nvidia-smi is present)
if command -v nvidia-smi &> /dev/null; then
    AVAILABLE_GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [[ "$GPU_COUNT" -gt "$AVAILABLE_GPU_COUNT" ]]; then
        echo "Error: Requested --ngpus $GPU_COUNT, but only $AVAILABLE_GPU_COUNT GPU(s) available according to nvidia-smi."
        exit 1
    fi
fi

# Group GPUs into sets based on GPUS_PER_INSTANCE
GPU_GROUPS=()
if [[ "$GPUS_PER_INSTANCE" == "2" ]]; then
    # Group GPUs into pairs
    echo "Grouping ${#GPU_LIST[@]} GPUs into pairs..."
    for ((i=0; i<${#GPU_LIST[@]}; i+=2)); do
        if (( i+1 < ${#GPU_LIST[@]} )); then
            pair="${GPU_LIST[$i]},${GPU_LIST[$((i+1))]}"
            echo "  Creating pair: $pair"
            GPU_GROUPS+=("$pair")
        else
            echo "Error: Odd number of GPUs (${#GPU_LIST[@]}) for multi-GPU mode (2 GPUs per instance). Exiting."
            exit 1
        fi
    done
else
    # Single GPU per instance
    echo "Using single GPU per instance..."
    GPU_GROUPS=("${GPU_LIST[@]}")
fi

GPU_COUNT=${#GPU_GROUPS[@]}
echo "Total GPU groups created: $GPU_COUNT"

# Print configuration information
echo "=========================================="
echo "Benchmark Configuration:"
echo "  GPUs per instance: $GPUS_PER_INSTANCE"
echo "  Total GPU groups: $GPU_COUNT"
if [[ "$GPUS_PER_INSTANCE" == "2" ]]; then
    echo "  GPU groups (pairs):"
    for ((i=0; i<${#GPU_GROUPS[@]}; i++)); do
        echo "    Group $((i+1)): ${GPU_GROUPS[$i]}"
    done
else
    echo "  GPU groups (single):"
    for ((i=0; i<${#GPU_GROUPS[@]}; i++)); do
        echo "    Group $((i+1)): ${GPU_GROUPS[$i]}"
    done
fi
echo "=========================================="
echo ""

# Ensure all entries in MODEL_LIST have .mps extension
if [[ -n "$MODEL_LIST" && -f "$MODEL_LIST" ]]; then
    # Create a temporary file to store the updated model list
    TMP_MODEL_LIST=$(mktemp)
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip empty lines
        [[ -z "$line" ]] && continue
        # If the line does not end with .mps, append it
        # ignore if it ends with .SIF
        if [[ "$line" != *.mps && "$line" != *.SIF ]]; then
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
    # Gather both .mps and .SIF files in the directory
    mapfile -t mps_files < <(ls "$MPS_DIR"/*.mps "$MPS_DIR"/*.SIF 2>/dev/null)

    echo "Found ${#mps_files[@]} .mps and .SIF files in $MPS_DIR"
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
    echo "Reducing number of GPU groups from $GPU_COUNT to $file_count since there are fewer files than GPU groups"
    GPU_COUNT=$file_count
    # Trim GPU_GROUPS to match file_count
    GPU_GROUPS=("${GPU_GROUPS[@]:0:$file_count}")
fi

# Function for each worker (GPU group)
worker() {
    local gpu_devices=$1
    # echo "GPU(s) $gpu_devices Using index file $INDEX_FILE"
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
        echo "GPU(s) $gpu_devices processing $my_index"

        # Build arguments string
        args=""
        if [ -n "$ITERATION_LIMIT" ]; then
            args="$args --iteration-limit $ITERATION_LIMIT"
        fi
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
        if [ "$GPUS_PER_INSTANCE" -gt 1 ]; then
            args="$args --num-gpus $GPUS_PER_INSTANCE"
        fi
        args="$args --log-to-console $LOG_TO_CONSOLE"
        if [ -n "$PRESOLVE" ]; then
            args="$args --presolve $PRESOLVE"
        fi
        if [ -n "$METHOD" ]; then
            args="$args --method $METHOD"
        fi
        if [ -n "$PDLP_TOLERANCES" ]; then
            args="$args --absolute-primal-tolerance $PDLP_TOLERANCES --absolute-dual-tolerance $PDLP_TOLERANCES --relative-primal-tolerance $PDLP_TOLERANCES --relative-dual-tolerance $PDLP_TOLERANCES --absolute-gap-tolerance $PDLP_TOLERANCES --relative-gap-tolerance $PDLP_TOLERANCES"
        fi

        CUDA_VISIBLE_DEVICES=$gpu_devices cuopt_cli "$mps_file" --time-limit $TIME_LIMIT $args
    done
}

# Start one worker per GPU group
echo "Starting workers for ${#GPU_GROUPS[@]} GPU groups..."
for gpu_devices in "${GPU_GROUPS[@]}"; do
    echo "  Starting worker for GPU(s): $gpu_devices"
    worker "$gpu_devices" &
done
echo "All workers started."

wait

# Remove the index file
rm -f "$INDEX_FILE"
