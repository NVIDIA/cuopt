#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For each of client / mathopt / routing: install ONLY that component's freshly built conda package
# in a fresh env, build a small gtest against it and run it. Depends only on conda-cpp-build, not
# on any test job.

set -euo pipefail

# shellcheck disable=SC1091
. /opt/conda/etc/profile.d/conda.sh

rapids-logger "Configuring conda strict channel priority"
conda config --set channel_priority strict

CPP_CHANNEL=$(rapids-download-from-github "$(rapids-artifact-name conda_cpp libcuopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

HERE="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
FAILED=()

for component in client mathopt routing; do
  rapids-logger "Isolated install (conda): ${component}"
  # Subshell so one component's failure cannot leak environment into the next.
  if ! (
    export COMPONENT="${component}" CUDA_MAJOR LOCAL_CHANNEL="${CPP_CHANNEL}"
    bash "${HERE}/isolated_install/container_conda.sh"
  ); then
    FAILED+=("conda/${component}")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  rapids-logger "Isolated install FAILED: ${FAILED[*]}"
  exit 1
fi
rapids-logger "Isolated install passed for all components"
