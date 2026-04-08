#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Downloads test datasets based on CUOPT_TEST_COMPONENTS.
# Sources derive_test_components.sh to set CUOPT_TEST_COMPONENTS if not already set.
#
# Usage:  source ./ci/download_test_datasets.sh
#
# Optional env vars:
#   CUOPT_ROUTING_DATASET_ARGS  — extra args for routing get_test_data.sh (e.g. "--solomon --tsp")
#                                  Defaults to no args (downloads all routing data).

set -euo pipefail

SCRIPT_DIR="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

# Derive CUOPT_TEST_COMPONENTS from changed-files env vars (no-op if already set)
if [[ -z "${CUOPT_TEST_COMPONENTS:-}" ]]; then
    source "${SCRIPT_DIR}/../derive_test_components.sh"
fi

RAPIDS_DATASET_ROOT_DIR="$(realpath datasets)"
export RAPIDS_DATASET_ROOT_DIR

_log() {
    if command -v rapids-logger &>/dev/null; then
        rapids-logger "$1"
    else
        echo "$1"
    fi
}

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"lp"* ]]; then
    ./datasets/linear_programming/download_pdlp_test_dataset.sh
else
    _log "Skipping LP dataset download (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"mip"* ]]; then
    ./datasets/mip/download_miplib_test_dataset.sh
else
    _log "Skipping MIP dataset download (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi

if [[ "${CUOPT_TEST_COMPONENTS}" == "all" || "${CUOPT_TEST_COMPONENTS}" == *"routing"* ]]; then
    pushd "${RAPIDS_DATASET_ROOT_DIR}"
    # shellcheck disable=SC2086
    ./get_test_data.sh ${CUOPT_ROUTING_DATASET_ARGS:-}
    popd
else
    _log "Skipping routing dataset downloads (not needed for components: ${CUOPT_TEST_COMPONENTS})"
fi
