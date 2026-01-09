#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# 10 easiest Mittleman instances
datasets=(
    "graph40-40"
    "ex10"
    "datt256_lp"
    "woodlands09"
    "savsched1"
    "nug08-3rd"
    "qap15"
    "scpm1"
    "neos3"
    "a2864"
    "ns1687037"
    "square41"
)

BASEDIR=$(dirname "$0")

################################################################################
# S3 Download Support
################################################################################
# Set CUOPT_DATASET_S3_URI to base S3 path
# Use CUOPT_AWS_ACCESS_KEY_ID and CUOPT_AWS_SECRET_ACCESS_KEY
# or standard AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY

function try_download_from_s3() {
    if [ -z "${CUOPT_DATASET_S3_URI:-}" ]; then
        return 1
    fi

    if ! command -v aws &> /dev/null; then
        echo "AWS CLI not found, skipping S3 download..."
        return 1
    fi

    # Append linear_programming/pdlp subdirectory to base S3 URI
    local s3_uri="${CUOPT_DATASET_S3_URI}linear_programming/pdlp/"
    echo "Attempting to download PDLP datasets from S3: $s3_uri"

    # Support custom credential variable names
    local access_key="${CUOPT_AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
    local secret_key="${CUOPT_AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
    local region="${CUOPT_AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"

    # Temporarily export for AWS CLI if custom variables are used
    if [ -n "$CUOPT_AWS_ACCESS_KEY_ID" ]; then
        echo "Using custom CUOPT_AWS_ACCESS_KEY_ID credentials"
        export AWS_ACCESS_KEY_ID="$access_key"
        export AWS_SECRET_ACCESS_KEY="$secret_key"
        # Unset session token to avoid mixing credentials
        unset AWS_SESSION_TOKEN
        export AWS_DEFAULT_REGION="$region"
    fi

    # Test AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        echo "AWS credentials not configured, skipping S3 download..."
        return 1
    fi

    # Try to sync from S3 (downloads from pdlp/ subdirectory)
    local success=true
    for dataset in "${datasets[@]}"; do
        echo "Downloading ${dataset} from S3..."
        if ! aws s3 sync "${s3_uri}${dataset}/" "$BASEDIR/${dataset}/" --exclude "*.sh"; then
            echo "Warning: Failed to download ${dataset}"
            success=false
        fi
    done

    if $success; then
        echo "Successfully downloaded PDLP datasets from S3!"
        return 0
    else
        echo "Some downloads failed, falling back to HTTP download..."
        return 1
    fi
}

# Try S3 first
if try_download_from_s3; then
    exit 0
fi

# HTTP fallback using Python script
echo "Downloading PDLP datasets using Python script..."
for dataset in "${datasets[@]}"; do
    python benchmarks/linear_programming/utils/get_datasets.py -d "$dataset"
done
