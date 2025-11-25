#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

for INSTANCE in "${INSTANCES[@]}"; do
    URL="${BASE_URL}/${INSTANCE}.mps.gz"
    OUTFILE="${BASEDIR}/${INSTANCE}.mps.gz"

    wget -4 --tries=3 --continue --progress=dot:mega --retry-connrefused "${URL}" -O "${OUTFILE}" || {
        echo "Failed to download: ${URL}"
        continue
    }
    gunzip -f "${OUTFILE}"
done
