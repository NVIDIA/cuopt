#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Abort script on first error
set -e

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd "$(dirname ${BASH_SOURCE[0]})" && pwd)}
source ${PROJECT_DIR}/config.sh
source ${PROJECT_DIR}/functions.sh

PROJECT_VERSION=$(</opt/cuopt/COMMIT_SHA)
PROJECT_REPO_URL="https://github.com/NVIDIA/cuopt.git"
max_ver=$(cat VERSION | cut -d "." -f 1)
min_ver=$(cat VERSION | cut -d "." -f 2)
PROJECT_REPO_BRANCH="branch-$max_ver.$min_ver"
PROJECT_REPO_TIME=$(</opt/cuopt/COMMIT_TIME)

echo "# source this file for project meta-data" >> $METADATA_FILE
echo "PROJECT_VERSION=\"$PROJECT_VERSION\"" >> $METADATA_FILE
echo "PROJECT_BUILD=\"$PROJECT_BUILD\"" >> $METADATA_FILE
echo "PROJECT_CHANNEL=\"$PROJECT_CHANNEL\"" >> $METADATA_FILE
echo "PROJECT_REPO_URL=\"$PROJECT_REPO_URL\"" >> $METADATA_FILE
echo "PROJECT_REPO_BRANCH=\"$PROJECT_REPO_BRANCH\"" >> $METADATA_FILE
echo "PROJECT_REPO_TIME=\"$PROJECT_REPO_TIME\"" >> $METADATA_FILE
