#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For each of client / mathopt / routing: install ONLY that component's freshly built wheel (plus
# the client wheel it depends on) in a clean venv, build a small gtest against it and run it.
# Depends only on the libcuopt_{client,mathopt,routing} wheel builds, not on any test job.

set -euo pipefail

# Download the packages built in the previous step
CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"
LIBCUOPT_CLIENT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_client cuopt)")
LIBCUOPT_MATHOPT_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_mathopt cuopt --cuda "$RAPIDS_CUDA_VERSION")")
LIBCUOPT_ROUTING_WHEELHOUSE=$(rapids-download-from-github "$(rapids-artifact-name wheel_cpp libcuopt_routing cuopt --cuda "$RAPIDS_CUDA_VERSION")")

HERE="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
FAILED=()

for component in client mathopt routing; do
  rapids-logger "Isolated install (wheel): ${component}"
  case "${component}" in
    client)  dirs="${LIBCUOPT_CLIENT_WHEELHOUSE}" ;;
    mathopt) dirs="${LIBCUOPT_CLIENT_WHEELHOUSE}:${LIBCUOPT_MATHOPT_WHEELHOUSE}" ;;
    routing) dirs="${LIBCUOPT_CLIENT_WHEELHOUSE}:${LIBCUOPT_ROUTING_WHEELHOUSE}" ;;
  esac
  # Subshell so one component's failure cannot leak environment into the next.
  if ! (
    export COMPONENT="${component}" CUDA_MAJOR WHEEL_DIRS="${dirs}"
    bash "${HERE}/isolated_install/container_pip.sh"
  ); then
    FAILED+=("wheel/${component}")
  fi
done

if [[ ${#FAILED[@]} -gt 0 ]]; then
  rapids-logger "Isolated install FAILED: ${FAILED[*]}"
  exit 1
fi
rapids-logger "Isolated install passed for all components"
