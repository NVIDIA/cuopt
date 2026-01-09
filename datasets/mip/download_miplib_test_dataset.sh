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

    # Append linear_programming/miplib subdirectory to base S3 URI
    local s3_uri="${CUOPT_DATASET_S3_URI}linear_programming/miplib/"
    echo "Attempting to download MIPLIB datasets from S3: $s3_uri"

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

    # Try to sync from S3 (downloads from miplib/ subdirectory)
    local success=true
    for instance in "${INSTANCES[@]}"; do
        echo "Downloading ${instance}.mps from S3..."
        if ! aws s3 cp "${s3_uri}${instance}.mps" "$BASEDIR/${instance}.mps"; then
            echo "Warning: Failed to download ${instance}"
            success=false
        fi
    done

    if $success; then
        echo "Successfully downloaded MIPLIB datasets from S3!"
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
