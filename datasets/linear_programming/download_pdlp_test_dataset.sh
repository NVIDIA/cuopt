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
# Requires explicit CUOPT credentials to avoid using unintended AWS credentials:
#   - CUOPT_S3_URI: Base S3 bucket root (e.g., s3://cuopt-datasets/)
#   - CUOPT_AWS_ACCESS_KEY_ID: AWS access key
#   - CUOPT_AWS_SECRET_ACCESS_KEY: AWS secret key
#   - CUOPT_AWS_REGION (optional): AWS region, defaults to us-east-1

function try_download_from_s3() {
    if [ -z "${CUOPT_S3_URI:-}" ]; then
        echo "WARNING: CUOPT_S3_URI not set — S3 dataset download disabled, using HTTP fallback." >&2
        echo "WARNING: HTTP fallback requires gcc for nug08-3rd dataset (may fail in wheel containers)." >&2
        return 1
    fi

    # Require explicit CUOPT credentials to avoid accidentally using generic AWS credentials
    if [ -z "${CUOPT_AWS_ACCESS_KEY_ID:-}" ]; then
        echo "WARNING: CUOPT_AWS_ACCESS_KEY_ID not set — cannot download datasets from S3." >&2
        return 1
    fi

    if [ -z "${CUOPT_AWS_SECRET_ACCESS_KEY:-}" ]; then
        echo "WARNING: CUOPT_AWS_SECRET_ACCESS_KEY not set — cannot download datasets from S3." >&2
        return 1
    fi

    if ! command -v aws &> /dev/null; then
        echo "WARNING: AWS CLI not found — cannot download datasets from S3." >&2
        return 1
    fi

    # Append ci_datasets/linear_programming/pdlp subdirectory to base S3 URI
    local s3_uri="${CUOPT_S3_URI}ci_datasets/linear_programming/pdlp/"
    echo "Downloading PDLP datasets from S3..."

    # Use CUOPT-specific credentials only
    local region="${CUOPT_AWS_REGION:-us-east-1}"

    # Export credentials for AWS CLI
    export AWS_ACCESS_KEY_ID="$CUOPT_AWS_ACCESS_KEY_ID"
    export AWS_SECRET_ACCESS_KEY="$CUOPT_AWS_SECRET_ACCESS_KEY"
    # Unset session token to avoid mixing credentials
    unset AWS_SESSION_TOKEN
    export AWS_DEFAULT_REGION="$region"

    # Test AWS credentials
    if ! aws sts get-caller-identity &> /dev/null 2>&1; then
        echo "AWS credentials invalid, skipping S3 download..."
        return 1
    fi

    # Try to sync from S3 (downloads from pdlp/ subdirectory)
    local success=true
    local total=${#datasets[@]}
    local count=0
    for dataset in "${datasets[@]}"; do
        count=$((count + 1))
        if ! aws s3 sync "${s3_uri}${dataset}/" "$BASEDIR/${dataset}/" --exclude "*.sh" --only-show-errors; then
            success=false
        fi
        printf "\rProgress: %d/%d" "$count" "$total"
    done
    echo ""

    if $success; then
        echo "✓ Downloaded PDLP datasets from S3"
        return 0
    else
        echo "S3 download failed, falling back to HTTP..."
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
