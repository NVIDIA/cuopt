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
DOCKER_TAG=$(echo "$NEXT_FULL_TAG" | sed -E 's/^([0-9]{2})\.0*([0-9]+)\.0*([0-9]+).*/\1.\2.\3/')

# Need to distutils-normalize the versions for some use cases (RAPIDS)
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

echo "Preparing release $CURRENT_TAG => $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > ./VERSION
echo "${NEXT_FULL_TAG}" > ./RAPIDS_VERSION

DEPENDENCIES=(
  cudf
  cuvs
  cuopt
  cuopt-mps-parser
  cuopt-server
  cuopt-sh-client
  libcuopt
  libraft-headers
  librmm
  pylibraft
  raft-dask
  rapids-dask-dependency
  rmm
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml python/*/pyproject.toml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "/-.* ${DEP}\(-cu[[:digit:]]\{2\}\)\{0,1\}\(\[.*\]\)\{0,1\}==/ s/==.*/==${NEXT_SHORT_TAG_PEP440}.*,>=0.0.0a0/g" "${FILE}"
  done
  for FILE in docs/cuopt/source/*/quick-start.rst README.md; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" "${FILE}"
    sed_runner "s/\(${DEP}=\)[0-9]\+\.[0-9]\+\(\.[0-9]\+\)\?[^ ]*/\1${NEXT_SHORT_TAG}.*/g" "${FILE}"
    sed_runner "s/\(${DEP}:\)[0-9]\{2\}\.[0-9]\{1,2\}\.[0-9]\+\(-cuda[0-9]\+\.[0-9]\+-\)\(py[0-9]\+\)/\1${DOCKER_TAG}\2\3/g" "${FILE}"
  done
done

# CMakeLists update
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/CMakeLists.txt
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/libmps_parser/CMakeLists.txt
sed_runner 's/'"DEPENDENT_LIB_MAJOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MAJOR_VERSION \"${NEXT_MAJOR}\""'/g' cpp/CMakeLists.txt
sed_runner 's/'"DEPENDENT_LIB_MINOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MINOR_VERSION \"${NEXT_MINOR}\""'/g' cpp/CMakeLists.txt

# Server version update
sed_runner 's/'"\"version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/routing/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/linear_programming/data_definition.py

# Doc update
sed_runner 's/'"version = \"[0-9][0-9].[0-9][0-9]\""'/'"version = \"${NEXT_SHORT_TAG}\""'/g' docs/cuopt/source/conf.py
sed_runner 's/'"PROJECT_NUMBER         = [0-9][0-9].[0-9][0-9]"'/'"PROJECT_NUMBER         = ${NEXT_SHORT_TAG}"'/g' cpp/doxygen/Doxyfile

# Update project.json
PROJECT_FILE="docs/cuopt/source/project.json"
sed_runner 's/\("version": "\)[0-9][0-9]\.[0-9][0-9]\.[0-9][0-9]"/\1'${NEXT_FULL_TAG}'"/g' "${PROJECT_FILE}"

# Update VERSIONS.json
VERSIONS_FILE="docs/cuopt/source/versions1.json"
# Only update if NEXT_FULL_TAG is not already present
if ! grep -q "\"version\": \"${NEXT_FULL_TAG}\"" "${VERSIONS_FILE}"; then
  # Remove preferred and latest flags, but keep the version entry and URL
  sed_runner '/"name": "latest",/d' "${VERSIONS_FILE}"
  sed_runner '/"preferred": true,\?/d' "${VERSIONS_FILE}"
  # Remove trailing comma after "url": ... in all version entries
  sed_runner 's/\("url": "[^"]*"\),/\1/' "${VERSIONS_FILE}"
  # Add new version entry with both preferred and latest flags
  NEW_VERSION_ENTRY='    {\n      "version": "'${NEXT_FULL_TAG}'",\n      "url": "../'${NEXT_FULL_TAG}'/",\n      "name": "latest",\n      "preferred": true\n    },'
  sed_runner "/\[/a\\${NEW_VERSION_ENTRY}" "${VERSIONS_FILE}"
fi

# RTD update
sed_runner "/^set(cuopt_version/ s/[0-9][0-9].[0-9][0-9].[0-9][0-9]/${NEXT_FULL_TAG}/g" python/cuopt/CMakeLists.txt
sed_runner "/^set(cuopt_version/ s/[0-9][0-9].[0-9][0-9].[0-9][0-9]/${NEXT_FULL_TAG}/g" python/cuopt/cuopt/linear_programming/CMakeLists.txt
sed_runner "/^set(cuopt_version/ s/[0-9][0-9].[0-9][0-9].[0-9][0-9]/${NEXT_FULL_TAG}/g" python/libcuopt/CMakeLists.txt

# Update nightly
sed_runner 's/'"cuopt_version: \"[0-9][0-9].[0-9][0-9]\""'/'"cuopt_version: \"${NEXT_SHORT_TAG}\""'/g' .github/workflows/nightly.yaml

# Update Helm chart files
sed_runner 's/\(tag: "\)[0-9][0-9]\.[0-9]\+\.[0-9]\+\(-cuda12\.9-py3\.12"\)/\1'${DOCKER_TAG}'\2/g' helmchart/cuopt-server/values.yaml
sed_runner 's/\(appVersion: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml
sed_runner 's/\(version: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml

for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # CI image tags of the form {rapids_version}-{something}
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# PYTHON for RAPIDS
sed_runner "/DOWNLOAD.*rapids-cmake/ s/branch-[0-9][0-9].[0-9][0-9]/branch-${NEXT_SHORT_TAG}/g" python/cuopt/CMakeLists.txt
