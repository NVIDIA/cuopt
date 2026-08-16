#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

INSTANCES=(
    "QPLIB_8515"
)

BASE_URL="https://qplib.zib.de/lp"
BASEDIR=$(dirname "$0")
OUTDIR="${BASEDIR}/qplib"

mkdir -p "${OUTDIR}"

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

    # Append ci_datasets/quadratic_programming/qplib subdirectory to base S3 URI
    local s3_uri="${CUOPT_S3_URI}ci_datasets/quadratic_programming/qplib/"
    echo "Downloading QPLIB datasets from S3..."

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

    local success=true
    local total=${#INSTANCES[@]}
    local count=0
    for instance in "${INSTANCES[@]}"; do
        count=$((count + 1))
        if ! aws s3 cp "${s3_uri}${instance}.lp" "${OUTDIR}/${instance}.lp" --only-show-errors; then
            success=false
        fi
        printf "\rProgress: %d/%d" "$count" "$total"
    done
    echo ""

    if $success; then
        echo "✓ Downloaded QPLIB datasets from S3"
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
echo "Downloading QPLIB datasets from HTTP..."
for INSTANCE in "${INSTANCES[@]}"; do
    URL="${BASE_URL}/${INSTANCE}.lp"
    OUTFILE="${OUTDIR}/${INSTANCE}.lp"

    wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${URL}" -O "${OUTFILE}" || {
        echo "Failed to download: ${URL}"
        continue
    }
done
