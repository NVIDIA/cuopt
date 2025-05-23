#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

NEXT_FULL_TAG=$1

# Get current version
CURRENT_TAG=$(git tag --merged HEAD | grep -xE '^v.*' | sort --version-sort | tail -n 1 | tr -d 'v')

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}
PY_NEXT_SHORT_TAG=${NEXT_MAJOR}.$(echo "$NEXT_MINOR" | tr -d '0')


echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > VERSION

dependencies='cuopt-cu12 cuopt-mps-parser'
for FILE in conda/environments/*.yaml dependencies.yaml; do
    for dependency in ${dependencies}; do
        sed_runner "s/- ${dependency}==.*/- ${dependency}==${NEXT_SHORT_TAG}\.*/g" "${FILE}";
    done
done

dependencies='cuopt cuopt-mps-parser'
for FILE in conda/environments/*.yaml dependencies.yaml; do
    for dependency in ${dependencies}; do
        sed_runner "s/- \(&[^ ]* \)\?${dependency}==.*/- \1${dependency}==${PY_NEXT_SHORT_TAG}\.*/g" "${FILE}";
    done
done

# Update nightly image tag
sed_runner 's/'cuopt\:[0-9][0-9].[0-9][0-9]'/'"cuopt:${NEXT_SHORT_TAG}"'/g' .github/workflows/managed_service_nightly.yaml
sed_runner 's/'cuopt\:[0-9][0-9].[0-9][0-9].arm'/'"cuopt:${NEXT_SHORT_TAG}.arm"'/g' .github/workflows/managed_service_nightly.yaml

# CMakeLists update
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/CMakeLists.txt
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/libmps_parser/CMakeLists.txt

# Server version update
sed_runner 's/'"\"version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/routing/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/linear_programming/data_definition.py

# Doc update
sed_runner 's/'"version = \"[0-9][0-9].[0-9][0-9]\""'/'"version = \"${NEXT_SHORT_TAG}\""'/g' docs/cuopt/repo.toml

# RTD update
sed_runner "/^set(cuopt_version/ s/[0-9][0-9].[0-9][0-9].[0-9][0-9]/${NEXT_FULL_TAG}/g" python/cuopt/CMakeLists.txt
sed_runner "/^set(cuopt_version/ s/[0-9][0-9].[0-9][0-9].[0-9][0-9]/${NEXT_FULL_TAG}/g" python/cuopt/cuopt/linear_programming/CMakeLists.txt

# Update nightly
sed_runner 's/'"cuopt_version: \"[0-9][0-9].[0-9][0-9]\""'/'"cuopt_version: \"${NEXT_SHORT_TAG}\""'/g' .github/workflows/nightly.yaml

# Update branch usage
sed_runner 's/'"branch-[0-9][0-9].[0-9][0-9]"'/'"branch-${NEXT_SHORT_TAG}"'/g' docs/cuopt/docs/resources.rst
sed_runner 's/'"branch-[0-9][0-9].[0-9][0-9]"'/'"branch-${NEXT_SHORT_TAG}"'/g' utilities/cuopt_user_onboarding/run_onboarding.sh
sed_runner 's/'"cuopt-self-hosted:[0-9][0-9].[0-9][0-9]"'/'"cuopt-self-hosted:${NEXT_SHORT_TAG}"'/g' .github/workflows/managed_service_nightly.yaml
sed_runner 's/'"cuopt-managed:[0-9][0-9].[0-9][0-9]"'/'"cuopt-managed:${NEXT_SHORT_TAG}"'/g' .github/workflows/managed_service_nightly.yaml


# Fixing dependencies and pyproject.toml

DEPENDENCIES=(
  cuopt
  cuopt-mps-parser
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in python/*/pyproject.toml; do
    sed_runner "/\"${DEP}==/ s/==.*\"/==${PY_NEXT_SHORT_TAG}.*\"/g" "${FILE}"
  done
done
