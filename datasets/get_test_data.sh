#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -o pipefail

################################################################################
# S3 Dataset Download Support
################################################################################
# Set CUOPT_DATASET_S3_URI to base S3 path
# AWS credentials should be configured via:
#   - Environment variables (CUOPT_AWS_ACCESS_KEY_ID, CUOPT_AWS_SECRET_ACCESS_KEY)
#   - Standard AWS variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
#   - AWS CLI configuration (~/.aws/credentials)
#   - IAM role (for EC2 instances)

function try_download_from_s3() {
    local s3_dirs=("$@")  # Array of directories to sync from S3

    if [ -z "${CUOPT_DATASET_S3_URI:-}" ]; then
        echo "CUOPT_DATASET_S3_URI not set, skipping S3 download..."
        return 1
    fi

    # Require explicit CUOPT credentials to avoid accidentally using generic AWS credentials
    if [ -z "${CUOPT_AWS_ACCESS_KEY_ID:-}" ]; then
        echo "CUOPT_AWS_ACCESS_KEY_ID not set, skipping S3 download..."
        return 1
    fi

    if ! command -v aws &> /dev/null; then
        echo "AWS CLI not found, skipping S3 download..."
        return 1
    fi

    # Append routing subdirectory to base S3 URI
    local s3_uri="${CUOPT_DATASET_S3_URI}routing/"
    echo "Downloading datasets from S3..."

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

    # Try to sync from S3
    local success=true
    if [ ${#s3_dirs[@]} -eq 0 ]; then
        # No specific directories - download everything
        if ! aws s3 sync "$s3_uri" . --exclude "tmp/*" --exclude "get_test_data.sh" --exclude "*.sh" --exclude "*.md" --only-show-errors; then
            success=false
        fi
    else
        # Download specific directories only
        for dir in "${s3_dirs[@]}"; do
            if ! aws s3 sync "${s3_uri}${dir}/" "${dir}/" --exclude "*.sh" --exclude "*.md" --only-show-errors; then
                success=false
            fi
        done
    fi

    if $success; then
        echo "✓ Downloaded datasets from S3"
        return 0
    else
        echo "S3 download failed, falling back to HTTP..."
        return 1
    fi
}

################################################################################
# HTTP Dataset Download Configuration
################################################################################
# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator

#TSP_DATASET_DATA="
# # 0.1s
# http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
# tsp
# "

CVRP_DATASET_DATA="
# 0.01s
http://vrp.atd-lab.inf.puc-rio.br/media/com_vrp/instances/Vrp-Set-X.tgz
cvrp
"

ACVRP_DATASET_DATA="
# 0.01s
http://www.vrp-rep.org/datasets/download/ftv1994.zip
acvrp
"

CVRPTW_DATASET_DATA=""
for i in {200..1000..200}
do
  data="
  # 0.1s
  https://www.sintef.no/globalassets/project/top/vrptw/homberger/${i}/homberger_${i}_customer_instances.zip
  cvrptw
  "
  CVRPTW_DATASET_DATA="${CVRPTW_DATASET_DATA}${data}"
done

PDPTW_DATASET_DATA=""
for i in {200..600..200}
do
  data="
  # 0.1s
  https://www.sintef.no/contentassets/1338af68996841d3922bc8e87adc430c/pdp_${i}.zip
  pdptw
  "
  PDPTW_DATASET_DATA="${PDPTW_DATASET_DATA}${data}"
done
for i in {800..1000..200}
do
  data="
  # 0.1s
  https://www.sintef.no/contentassets/1338af68996841d3922bc8e87adc430c/pdptw${i}.zip
  pdptw
  "
  PDPTW_DATASET_DATA="${PDPTW_DATASET_DATA}${data}"
done

SOLOMON_DATASET_DATA="
# 0.1s
https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip
solomon
"

# Add back ${TSP_DATASET_DATA} when issue #609 is fixed
ALL_DATASET_DATA="${CVRP_DATASET_DATA} ${ACVRP_DATASET_DATA} ${CVRPTW_DATASET_DATA} ${SOLOMON_DATASET_DATA} ${PDPTW_DATASET_DATA}"

################################################################################
# Do not change the script below this line if only adding/updating a dataset

NUMARGS=$#
ARGS=$*
function hasArg {
    (( NUMARGS != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

if hasArg -h || hasArg --help; then
    echo "$0 [--tsplib]"
    exit 0
fi

# Select the datasets to install
# Add back --tsp when issue #609 is fixed
# if hasArg "--tsp"; then
#   DATASET_DATA="${TSP_DATASET_DATA}"
if hasArg "--cvrp"; then
  DATASET_DATA="${CVRP_DATASET_DATA}"
elif hasArg "--acvrp"; then
  DATASET_DATA="${ACVRP_DATASET_DATA}"
elif hasArg "--cvrptw"; then
  DATASET_DATA="${CVRPTW_DATASET_DATA} ${SOLOMON_DATASET_DATA}"
elif hasArg "--solomon"; then
  DATASET_DATA="${SOLOMON_DATASET_DATA}"
elif hasArg "--pdptw"; then
  DATASET_DATA="${PDPTW_DATASET_DATA}"
else
  DATASET_DATA="${ALL_DATASET_DATA}"
fi

# shellcheck disable=SC2207
URLS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 3) print $0}'))  # extract 3rd fields to a bash array
# shellcheck disable=SC2207
DESTDIRS=($(echo "$DATASET_DATA"|awk '{if (NR%4 == 0) print $0}'))  # extract 4th fields to a bash array

# Try S3 download first with selected directories
if try_download_from_s3 "${DESTDIRS[@]}"; then
    echo "Datasets successfully retrieved from S3, skipping HTTP download."
    exit 0
fi

echo "Downloading from HTTP sources..."

# Download all tarfiles to a tmp dir
rm -rf tmp
mkdir tmp
cd tmp
# Loop through URLs with error handling
for url in "${URLS[@]}"; do
    # Try up to 3 times with continue option
    wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${url}" || {
        echo "Failed to download: ${url}"
        continue
    }
done
cd ..

# Setup the destination dirs, removing any existing ones first!
for index in ${!DESTDIRS[*]}; do
    rm -rf "${DESTDIRS[$index]}"
done
for index in ${!DESTDIRS[*]}; do
    mkdir -p "${DESTDIRS[$index]}"
done

# Iterate over the arrays and untar the nth tarfile to the nth dest directory.
# The tarfile name is derived from the download url.
echo Decompressing ...
set +e  # Disable exit on error for the entire script

for index in ${!DESTDIRS[*]}; do
    tfname=$(basename "${URLS[$index]}")

    if file --mime-type "tmp/${tfname}" | grep -q gzip$; then
        tar xvzf tmp/"${tfname}" -C "${DESTDIRS[$index]}" || true
    elif file --mime-type "tmp/${tfname}" | grep -q zip$; then
        unzip tmp/"${tfname}" -d "${DESTDIRS[$index]}" || true
    else
        tar xvf tmp/"${tfname}" -C "${DESTDIRS[$index]}" || true
    fi

    if ls "${DESTDIRS[$index]}"/*.gz >/dev/null 2>&1; then
        gzip -d "${DESTDIRS[$index]}"/* || true
    fi
done

rm -rf tmp

if [ -d "./pdptw" ]; then
  cd pdptw
  # change file extensions
  for f in $(shopt -s nullglob; echo *.txt); do
    mv -- "$f" "${f%.txt}.pdptw"
  done
fi
