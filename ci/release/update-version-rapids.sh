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

#Get <major>.<minor> for next version
NEXT_MAJOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[1]}')
NEXT_MINOR=$(echo "$NEXT_FULL_TAG" | awk '{split($0, a, "."); print a[2]}')
NEXT_SHORT_TAG=${NEXT_MAJOR}.${NEXT_MINOR}


# Need to distutils-normalize the versions for some use cases
NEXT_SHORT_TAG_PEP440=$(python -c "from packaging.version import Version; print(Version('${NEXT_SHORT_TAG}'))")

echo "Next is ${NEXT_SHORT_TAG_PEP440}"

echo "Preparing release $NEXT_FULL_TAG"

# Inplace sed replace; workaround for Linux and Mac
function sed_runner() {
    sed -i.bak ''"$1"'' "$2" && rm -f "${2}".bak
}

# Centralized version file update
echo "${NEXT_FULL_TAG}" > RAPIDS_VERSION

# CMakeLists update
sed_runner 's/'"DEPENDENT_LIB_MAJOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MAJOR_VERSION \"${NEXT_MAJOR}\""'/g' cpp/CMakeLists.txt
sed_runner 's/'"DEPENDENT_LIB_MINOR_VERSION \"[0-9][0-9]\""'/'"DEPENDENT_LIB_MINOR_VERSION \"${NEXT_MINOR}\""'/g' cpp/CMakeLists.txt

# RTD update
dependencies='cudf cudf-cu12 cuvs cuvs-cu12 libcudf librmm librmm-cu12 rmm rmm-cu12 librmm libraft-headers pylibraft pylibraft-cu12 raft-dask raft-dask-cu12 rapids-dask-dependency'
for FILE in conda/environments/*.yaml dependencies.yaml; do
    for dependency in ${dependencies}; do
        sed_runner "s/\(${dependency}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
    done
done

# WORKFLOWS
for FILE in .github/workflows/*.yaml; do
  sed_runner "/shared-workflows/ s/@.*/@branch-${NEXT_SHORT_TAG}/g" "${FILE}"
  # CI image tags of the form {rapids_version}-{something}
  sed_runner "s/:[0-9]*\\.[0-9]*-/:${NEXT_SHORT_TAG}-/g" "${FILE}"
done

# CI
sed_runner 's/'"DEPENDENT_PACKAGE_VERSION=\"[0-9][0-9].[0-9][0-9]\""'/'"DEPENDENT_PACKAGE_VERSION=\"${NEXT_SHORT_TAG}\""'/g' ci/build_cpp.sh
sed_runner 's/'"DEPENDENT_PACKAGE_VERSION=\"[0-9][0-9].[0-9][0-9]\""'/'"DEPENDENT_PACKAGE_VERSION=\"${NEXT_SHORT_TAG}\""'/g' ci/build_python.sh


# PYTHON
sed_runner "/DOWNLOAD.*rapids-cmake/ s/branch-[0-9][0-9].[0-9][0-9]/branch-${NEXT_SHORT_TAG}/g" python/cuopt/CMakeLists.txt

# Fixing dependencies and pyproject.toml

DEPENDENCIES=(
  cudf
  cuvs
  librmm
  rmm
  pylibraft
  raft-dask
  rapids-dask-dependency
)

for DEP in "${DEPENDENCIES[@]}"; do
  for FILE in dependencies.yaml conda/environments/*.yaml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
  for FILE in python/*/pyproject.toml; do
    sed_runner "s/\(${DEP}==\)[0-9]\+\.[0-9]\+/\1${NEXT_SHORT_TAG_PEP440}/" "${FILE}"
  done
done
