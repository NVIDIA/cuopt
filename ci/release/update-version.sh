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
echo "${NEXT_FULL_TAG}" > VERSION
echo "${NEXT_FULL_TAG}" > RAPIDS_VERSION

dependencies='libcuopt libcuopt-cu12 cuopt cuopt-cu12 cuopt-mps-parser'
for FILE in conda/environments/*.yaml dependencies.yaml; do
    for dependency in ${dependencies}; do
        sed_runner "s/\(${dependency}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
    done
done

# CMakeLists update
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/CMakeLists.txt
sed_runner 's/'"VERSION [0-9][0-9].[0-9][0-9].[0-9][0-9]"'/'"VERSION ${NEXT_FULL_TAG}"'/g' cpp/libmps_parser/CMakeLists.txt

# RAPIDS CMakeLists update
sed_runner 's/'"DEPENDENT_LIB_MAJOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MAJOR_VERSION \"${NEXT_MAJOR}\""'/g' cpp/CMakeLists.txt
sed_runner 's/'"DEPENDENT_LIB_MINOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MINOR_VERSION \"${NEXT_MINOR}\""'/g' cpp/CMakeLists.txt

# Server version update
sed_runner 's/'"\"version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/routing/data_definition.py
sed_runner 's/'"\"client_version\": \"[0-9][0-9].[0-9][0-9]\""'/'"\"client_version\": \"${NEXT_SHORT_TAG}\""'/g' python/cuopt_server/cuopt_server/utils/linear_programming/data_definition.py

# Doc update

sed_runner 's/'"version = \"[0-9][0-9].[0-9][0-9]\""'/'"version = \"${NEXT_SHORT_TAG}\""'/g' docs/cuopt/source/conf.py
sed_runner 's/'"PROJECT_NUMBER         = [0-9][0-9].[0-9][0-9]"'/'"PROJECT_NUMBER         = ${NEXT_SHORT_TAG}"'/g' cpp/doxygen/Doxyfile
# Update quick-start docs for conda
sed_runner 's/cuopt=[0-9][0-9].[0-9][0-9].[^ ]* python=[0-9].[0-9][0-9] cuda-version=[0-9][0-9].[0-9]/cuopt='${NEXT_SHORT_TAG}'.* python=3.12 cuda-version=12.8/g' docs/cuopt/source/cuopt-python/quick-start.rst
sed_runner 's/libcuopt=[0-9][0-9].[0-9][0-9].[^ ]* python=[0-9].[0-9][0-9] cuda-version=[0-9][0-9].[0-9]/libcuopt='${NEXT_SHORT_TAG}'.* python=3.12 cuda-version=12.8/g' docs/cuopt/source/cuopt-c/quick-start.rst
sed_runner 's/cuopt-server=[0-9][0-9].[0-9][0-9].[^ ]* cuopt-sh-client=[0-9][0-9].[0-9][0-9].[^ ]* python=[0-9].[0-9][0-9]/cuopt-server='${NEXT_SHORT_TAG}'.* cuopt-sh-client='${NEXT_SHORT_TAG}'.* python=3.12/g' docs/cuopt/source/cuopt-server/quick-start.rst
# Update quick-start docs for pip
sed_runner "s/\(cuopt-cu12==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" docs/cuopt/source/cuopt-python/quick-start.rst
sed_runner "s/\(libcuopt-cu12==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" docs/cuopt/source/cuopt-c/quick-start.rst
sed_runner "s/\(cuopt-server-cu12==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" docs/cuopt/source/cuopt-server/quick-start.rst
sed_runner "s/\(cuopt-sh-client==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" docs/cuopt/source/cuopt-server/quick-start.rst

# Update docker image tags in docs
sed_runner 's|cuopt:[0-9]\{2\}\.[0-9]\{1,2\}\.[0-9]\+\(-cuda12\.8-\)\(py[0-9]\+\)|cuopt:'"${DOCKER_TAG}"'\1\2|g' docs/cuopt/source/cuopt-python/quick-start.rst
sed_runner 's|cuopt:[0-9]\{2\}\.[0-9]\{1,2\}\.[0-9]\+\(-cuda12\.8-\)\(py[0-9]\+\)|cuopt:'"${DOCKER_TAG}"'\1\2|g' docs/cuopt/source/cuopt-server/quick-start.rst

# Update doc version

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
  # Clean up trailing commas and empty lines
  sed_runner 's/,\s*}/}/g' "${VERSIONS_FILE}"  # Remove trailing commas before closing braces
  sed_runner 's/,\s*$//g' "${VERSIONS_FILE}"   # Remove trailing commas at end of lines
  sed_runner '/^$/d' "${VERSIONS_FILE}"
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

# Update README.md
sed_runner "s/\(cuopt-server-cu12==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" README.md
sed_runner "s/\(cuopt-sh-client==\)[0-9]\+\.[0-9]\+\.\\*/\1${NEXT_SHORT_TAG_PEP440}.\*/g" README.md
sed_runner 's/cuopt-server=[0-9][0-9].[0-9][0-9] cuopt-sh-client=[0-9][0-9].[0-9][0-9] python=[0-9].[0-9][0-9] cuda-version=[0-9][0-9].[0-9]/cuopt-server='${NEXT_SHORT_TAG}' cuopt-sh-client='${NEXT_SHORT_TAG}' python=3.12 cuda-version=12.8/g' README.md
sed_runner 's|cuopt:[0-9]\{2\}\.[0-9]\{1,2\}\.[0-9]\+\(-cuda12\.8-\)\(py[0-9]\+\)|cuopt:'"${DOCKER_TAG}"'\1\2|g' README.md

# Update Helm chart files
sed_runner 's/\(tag: "\)[0-9][0-9]\.[0-9]\+\.[0-9]\+\(-cuda12\.8-py3\.12"\)/\1'${DOCKER_TAG}'\2/g' helmchart/cuopt-server/values.yaml
sed_runner 's/\(appVersion: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml
sed_runner 's/\(version: \)[0-9][0-9]\.[0-9]\+\.[0-9]\+/\1'${DOCKER_TAG}'/g' helmchart/cuopt-server/Chart.yaml

DEPENDENCIES=(
  libcuopt
  cuopt
  cuopt-mps-parser
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in python/*/pyproject.toml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
done

# RAPIDS-specific updates
# RTD update for RAPIDS dependencies
rapids_dependencies='cudf cudf-cu12 cuvs cuvs-cu12 libcudf librmm librmm-cu12 rmm rmm-cu12 librmm libraft-headers pylibraft pylibraft-cu12 raft-dask raft-dask-cu12 rapids-dask-dependency'
for FILE in conda/environments/*.yaml dependencies.yaml; do
    for dependency in ${rapids_dependencies}; do
        sed_runner "s/\(${dependency}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
    done
done

# WORKFLOWS for RAPIDS
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # CI image tags of the form {rapids_version}-{something}
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# CI for RAPIDS
sed_runner 's/'"DEPENDENT_PACKAGE_VERSION=\"[0-9][0-9].[0-9][0-9]\""'/'"DEPENDENT_PACKAGE_VERSION=\"${NEXT_SHORT_TAG}\""'/g' ci/build_cpp.sh
sed_runner 's/'"DEPENDENT_PACKAGE_VERSION=\"[0-9][0-9].[0-9][0-9]\""'/'"DEPENDENT_PACKAGE_VERSION=\"${NEXT_SHORT_TAG}\""'/g' ci/build_python.sh

# PYTHON for RAPIDS
sed_runner "/DOWNLOAD.*rapids-cmake/ s/branch-[0-9][0-9].[0-9][0-9]/branch-${NEXT_SHORT_TAG}/g" python/cuopt/CMakeLists.txt

# Fixing RAPIDS dependencies and pyproject.toml
RAPIDS_DEPENDENCIES=(
  cudf
  cuvs
  librmm
  rmm
  pylibraft
  raft-dask
  rapids-dask-dependency
)

for DEP in "${RAPIDS_DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
done
