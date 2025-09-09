#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -euo pipefail

. /opt/conda/etc/profile.d/conda.sh

CUOPT_VERSION="$(rapids-version)"

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-logger "Generate notebook testing dependencies"

ENV_YAML_DIR="$(mktemp -d)"

rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --prepend-channel "${CPP_CHANNEL}" \
  --prepend-channel "${PYTHON_CHANNEL}" \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee "${ENV_YAML_DIR}/env.yaml"

rapids-mamba-retry env create --yes -f "${ENV_YAML_DIR}/env.yaml" -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

EXAMPLES_BRANCH="branch-${CUOPT_VERSION%.*}"

# Remove any existing cuopt-examples directory

rapids-logger "Cloning cuopt-examples repository for branch: ${EXAMPLES_BRANCH}"
rm -rf cuopt-examples
git clone --single-branch --branch "${EXAMPLES_BRANCH}" https://github.com/NVIDIA/cuopt-examples.git

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
NBLIST_PATH="$(realpath "$(dirname "$0")/utils/notebook_list.py")"

pushd cuopt-examples

NBLIST=$(python "${NBLIST_PATH}")

EXITCODE=0
trap "EXITCODE=1" ERR

rapids-logger "Start cuopt-server"

set +e

rapids-logger "Start notebooks tests"
for nb in ${NBLIST}; do
  nvidia-smi
  ${NBTEST} "${nb}"
  if [ $? -ne 0 ]; then
    echo "Notebook ${nb} failed to execute. Exiting."
    exit 1
  fi
done

rapids-logger "Notebook test script exiting with value: $EXITCODE"
exit ${EXITCODE}
