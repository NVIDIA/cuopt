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
    "supportcase22"
)

BASE_URL="https://miplib.zib.de/WebData/instances"
BASEDIR=$(dirname "$0")

for INSTANCE in "${INSTANCES[@]}"; do
    URL="${BASE_URL}/${INSTANCE}.mps.gz"
    OUTFILE="${BASEDIR}/${INSTANCE}.mps.gz"

    wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${URL}" -O "${OUTFILE}" || {
        echo "Failed to download: ${URL}"
        continue
    }
    gunzip -f "${OUTFILE}"
done
