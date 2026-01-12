#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

INSTANCES=(
    "50v-10"
    "fiball"
    "gen-ip054"
    "sct2"
    "uccase9"
    "drayage-25-23"
    "tr12-30"
    "neos-3004026-krka"
    "ns1208400"
    "gmu-35-50"
    "n2seq36q"
    "seymour1"
    "rmatr200-p5"
    "cvs16r128-89"
    "thor50dday"
    "stein9inf"
    "neos5"
    "swath1"
    "enlight_hard"
    "enlight11"
    "supportcase22"
)

BASE_URL="https://miplib.zib.de/WebData/instances"
BASEDIR=$(dirname "$0")

################################################################################
# S3 Download Support
################################################################################
# Requires explicit CUOPT credentials to avoid using unintended AWS credentials:
#   - CUOPT_DATASET_S3_URI: Base S3 path
#   - CUOPT_AWS_ACCESS_KEY_ID: AWS access key
#   - CUOPT_AWS_SECRET_ACCESS_KEY: AWS secret key
#   - CUOPT_AWS_REGION (optional): AWS region, defaults to us-east-1

function try_download_from_s3() {
    if [ -z "${CUOPT_DATASET_S3_URI:-}" ]; then
        return 1
    fi

    # Require explicit CUOPT credentials to avoid accidentally using generic AWS credentials
    if [ -z "${CUOPT_AWS_ACCESS_KEY_ID:-}" ]; then
        echo "CUOPT_AWS_ACCESS_KEY_ID not set, skipping S3 download..."
        return 1
    fi

    if [ -z "${CUOPT_AWS_SECRET_ACCESS_KEY:-}" ]; then
        echo "CUOPT_AWS_SECRET_ACCESS_KEY not set, skipping S3 download..."
        return 1
    fi

    if ! command -v aws &> /dev/null; then
        echo "AWS CLI not found, skipping S3 download..."
        return 1
    fi

    # Append linear_programming/miplib subdirectory to base S3 URI
    local s3_uri="${CUOPT_DATASET_S3_URI}linear_programming/miplib/"
    echo "Downloading MIPLIB datasets from S3..."

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

    # Try to sync from S3 (downloads from miplib/ subdirectory)
    local success=true
    local total=${#INSTANCES[@]}
    local count=0
    for instance in "${INSTANCES[@]}"; do
        count=$((count + 1))
        if ! aws s3 cp "${s3_uri}${instance}.mps" "$BASEDIR/${instance}.mps" --only-show-errors; then
            success=false
        fi
        printf "\rProgress: %d/%d" "$count" "$total"
    done
    echo ""

    if $success; then
        echo "✓ Downloaded MIPLIB datasets from S3"
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

# HTTP fallback
echo "Downloading MIPLIB datasets from HTTP..."
for INSTANCE in "${INSTANCES[@]}"; do
    URL="${BASE_URL}/${INSTANCE}.mps.gz"
    OUTFILE="${BASEDIR}/${INSTANCE}.mps.gz"

    wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${URL}" -O "${OUTFILE}" || {
        echo "Failed to download: ${URL}"
        continue
    }
    gunzip -f "${OUTFILE}"
done
