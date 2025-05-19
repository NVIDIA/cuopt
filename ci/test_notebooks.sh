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

rapids-logger "Generate notebook testing dependencies"
rapids-dependency-file-generator \
  --output conda \
  --file-key test_notebooks \
  --matrix "cuda=${RAPIDS_CUDA_VERSION%.*};arch=$(arch);py=${RAPIDS_PY_VERSION}" | tee env.yaml

rapids-mamba-retry env create --yes -f env.yaml -n test

# Temporarily allow unbound variables for conda activation.
set +u
conda activate test
set -u

rapids-print-env

rapids-logger "Downloading artifacts from previous jobs"
CPP_CHANNEL=$(rapids-download-conda-from-github cpp)
PYTHON_CHANNEL=$(rapids-download-conda-from-github python)

rapids-mamba-retry install \
  --channel "${CPP_CHANNEL}" \
  --channel "${PYTHON_CHANNEL}" \
  "libcuopt=${CUOPT_VERSION}" \
  "cuopt=${CUOPT_VERSION}" \
  "cuopt-server=${CUOPT_VERSION}"

pip install python/cuopt_self_hosted/

NBTEST="$(realpath "$(dirname "$0")/utils/nbtest.sh")"
NBLIST_PATH="$(realpath "$(dirname "$0")/utils/notebook_list.py")"
NBLIST=$(python "${NBLIST_PATH}")
SERVER_WAIT_DELAY=10

pushd notebooks

EXITCODE=0
trap "EXITCODE=1" ERR

rapids-logger "Start cuopt-server"

set +e
#python -c "from cuopt_server.cuopt_service import run_server; run_server()" &

python -m cuopt_server.cuopt_service &
export SERVER_PID=$!
sleep "${SERVER_WAIT_DELAY}"
curl http://0.0.0.0:5000/cuopt/health

rapids-logger "Start notebooks tests"
for nb in ${NBLIST}; do
  nvidia-smi
  ${NBTEST} "${nb}"
done

rapids-logger "Notebook test script exiting with value: $EXITCODE"
kill -s SIGTERM $SERVER_PID
wait $SERVER_PID
exit ${EXITCODE}
