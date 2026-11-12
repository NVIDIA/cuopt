#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -e
set -o pipefail

################################################################################
# S3 Dataset Download Support
################################################################################
# Set CUOPT_S3_URI to S3 bucket root (e.g., s3://cuopt-datasets/)
# AWS credentials should be configured via:
#   - Environment variables (CUOPT_AWS_ACCESS_KEY_ID, CUOPT_AWS_SECRET_ACCESS_KEY)
#   - Standard AWS variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
#   - AWS CLI configuration (~/.aws/credentials)
#   - IAM role (for EC2 instances)

function try_download_from_s3() {
    local s3_dirs=("$@")  # Array of directories to sync from S3

    if [ -z "${CUOPT_S3_URI:-}" ]; then
        echo "CUOPT_S3_URI not set, skipping S3 download..."
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
    local s3_uri="${CUOPT_S3_URI}ci_datasets/routing/"
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

    if aws s3 cp "${CUOPT_S3_URI}ci_datasets/routing_tar/routing.tar.gz" - --only-show-errors | tar xzf - ; then
        echo "✓ Downloaded datasets from S3"
        return 0
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

# Stage all downloads and extraction in a unique sibling directory. The existing
# datasets must remain untouched until every selected dataset is ready.
work_dir=$(mktemp -d "${PWD}/.cuopt-datasets.XXXXXXXX")
backup_dir="${work_dir}/backup"
mkdir -p "${backup_dir}"
unique_destinations=()
backed_up=()
installed=()
committed=false

for dir in "${DESTDIRS[@]}"; do
    seen=false
    for previous in "${unique_destinations[@]}"; do
        if [[ "${dir}" == "${previous}" ]]; then
            seen=true
            break
        fi
    done
    if ! $seen; then unique_destinations+=("${dir}"); fi
done

cleanup() {
    local status=$?
    local preserve=false
    trap - EXIT
    if ! $committed; then
        if [[ ${#installed[@]} -gt 0 ]]; then
            mkdir -p "${work_dir}/uncommitted" || preserve=true
        fi
        for dir in "${installed[@]}"; do
            if [[ -e "${dir}" || -L "${dir}" ]]; then
                mv -- "${dir}" "${work_dir}/uncommitted/${dir}" || preserve=true
            fi
        done
        for dir in "${backed_up[@]}"; do
            if [[ -e "${backup_dir}/${dir}" || -L "${backup_dir}/${dir}" ]]; then
                if [[ -e "${dir}" || -L "${dir}" ]]; then
                    preserve=true
                else
                    mv -- "${backup_dir}/${dir}" "${dir}" || preserve=true
                fi
            fi
        done
    fi
    if $preserve; then
        echo "Could not restore all dataset directories; recovery data retained at ${work_dir}" >&2
    else
        rm -rf -- "${work_dir}"
    fi
    exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

s3_dir="${work_dir}/s3"
mkdir -p "${s3_dir}"
data_dir=""
# The S3 attempt may partially extract or sync data before failing; isolate it
# from both the existing destinations and the separate HTTP fallback.
if (cd "${s3_dir}" && try_download_from_s3 "${DESTDIRS[@]}"); then
    s3_complete=true
    for dir in "${unique_destinations[@]}"; do
        if [[ ! -d "${s3_dir}/${dir}" ]] || [[ -z "$(find "${s3_dir}/${dir}" -type f -print -quit)" ]]; then
            echo "S3 dataset is empty or missing: ${dir}; falling back to HTTP" >&2
            s3_complete=false
            break
        fi
    done
    if $s3_complete; then
        # S3 sync previously merged into existing datasets. Preserve that behavior
        # on a staged copy, without following destination symlinks on overwrite.
        merged_dir="${work_dir}/merged"
        mkdir -p "${merged_dir}"
        for dir in "${unique_destinations[@]}"; do
            if [[ -L "${dir}" || ( -e "${dir}" && ! -d "${dir}" ) ]]; then
                echo "Cannot safely merge S3 data into non-directory destination: ${dir}" >&2
                exit 1
            fi
            if [[ -d "${dir}" ]]; then
                cp -a --reflink=auto "${dir}" "${merged_dir}/${dir}"
            else
                mkdir -p "${merged_dir}/${dir}"
            fi
            cp -a --remove-destination "${s3_dir}/${dir}/." "${merged_dir}/${dir}/"
        done
        data_dir="${merged_dir}"
    fi
fi

if [[ -z "${data_dir}" ]]; then
    echo "Downloading from HTTP sources..."
    downloads_dir="${work_dir}/downloads"
    data_dir="${work_dir}/data"
    mkdir -p "${downloads_dir}" "${data_dir}"

    for url in "${URLS[@]}"; do
        if ! (cd "${downloads_dir}" && wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${url}"); then
            echo "Failed to download: ${url}" >&2
            exit 1
        fi
    done

    echo "Decompressing ..."
    for index in "${!DESTDIRS[@]}"; do
        dir="${DESTDIRS[$index]}"
        archive="${downloads_dir}/$(basename "${URLS[$index]}")"
        mkdir -p "${data_dir}/${dir}"
        if [[ ! -s "${archive}" ]]; then
            echo "Downloaded archive is empty or missing: ${archive}" >&2
            exit 1
        fi
        mime_type=$(file --mime-type -b "${archive}")
        case "${mime_type}" in
            application/gzip|application/x-gzip) tar xzf "${archive}" -C "${data_dir}/${dir}" ;;
            application/zip|application/x-zip-compressed) unzip -oq "${archive}" -d "${data_dir}/${dir}" ;;
            *) tar xf "${archive}" -C "${data_dir}/${dir}" ;;
        esac
        for compressed in "${data_dir}/${dir}"/*.gz; do
            if [[ -f "${compressed}" ]]; then gzip -d -- "${compressed}"; fi
        done
    done

    if [[ -d "${data_dir}/pdptw" ]]; then
        for dataset in "${data_dir}/pdptw"/*.txt; do
            if [[ -f "${dataset}" ]]; then mv -- "${dataset}" "${dataset%.txt}.pdptw"; fi
        done
    fi
fi

# Validate every selected destination before moving any existing dataset.
for dir in "${unique_destinations[@]}"; do
    if [[ ! -d "${data_dir}/${dir}" ]] || [[ -z "$(find "${data_dir}/${dir}" -type f -print -quit)" ]]; then
        echo "Dataset is empty or missing after download: ${dir}" >&2
        exit 1
    fi
done

for dir in "${unique_destinations[@]}"; do
    if [[ -e "${dir}" || -L "${dir}" ]]; then
        mv -- "${dir}" "${backup_dir}/${dir}"
        backed_up+=("${dir}")
    fi
    mv -- "${data_dir}/${dir}" "${dir}"
    installed+=("${dir}")
done
committed=true
