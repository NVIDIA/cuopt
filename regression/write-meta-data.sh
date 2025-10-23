#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2021-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Abort script on first error
set -e

DELAY=30

# Must ensure PROJECT_DIR is exported first then load rapids-mg-tools env
export PROJECT_DIR=${PROJECT_DIR:-$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)}
source ${PROJECT_DIR}/config.sh
source ${PROJECT_DIR}/functions.sh

PROJECT_VERSION=$(</opt/cuopt/COMMIT_SHA)
PROJECT_REPO_URL="https://github.com/NVIDIA/cuopt.git"
PROJECT_REPO_BRANCH="branch-25.10"
PROJECT_REPO_TIME=1757541732 #$(cd ${WORKSPACE}/${REPO_DIR_NAME}; git log -n1 --pretty='%ct' ${PROJECT_VERSION})

echo "# source this file for project meta-data" >> $METADATA_FILE
echo "PROJECT_VERSION=\"$PROJECT_VERSION\"" >> $METADATA_FILE
echo "PROJECT_BUILD=\"$PROJECT_BUILD\"" >> $METADATA_FILE
echo "PROJECT_CHANNEL=\"$PROJECT_CHANNEL\"" >> $METADATA_FILE
echo "PROJECT_REPO_URL=\"$PROJECT_REPO_URL\"" >> $METADATA_FILE
echo "PROJECT_REPO_BRANCH=\"$PROJECT_REPO_BRANCH\"" >> $METADATA_FILE
echo "PROJECT_REPO_TIME=\"$PROJECT_REPO_TIME\"" >> $METADATA_FILE
