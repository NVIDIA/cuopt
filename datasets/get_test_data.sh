#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -e
set -o pipefail

# Update this to add/remove/change a dataset, using the following format:
#
#  comment about the dataset
#  dataset download URL
#  destination dir to untar to
#  blank line separator

TSP_DATASET_DATA="
# 0.1s
http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz
tsp
"

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

ALL_DATASET_DATA="${TSP_DATASET_DATA} ${CVRP_DATASET_DATA} ${ACVRP_DATASET_DATA} ${CVRPTW_DATASET_DATA} ${SOLOMON_DATASET_DATA} ${PDPTW_DATASET_DATA}"

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
if hasArg "--tsp"; then
  DATASET_DATA="${TSP_DATASET_DATA}"
elif hasArg "--cvrp"; then
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

echo Downloading ...

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
