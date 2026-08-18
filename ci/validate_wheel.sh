#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_dir=$1
wheel_dir_relative_path=$2

RAPIDS_CUDA_MAJOR="${RAPIDS_CUDA_VERSION%%.*}"

cd "${package_dir}"

rapids-logger "validate packages with 'pydistcheck'"

PYDISTCHECK_ARGS=(
    --inspect
)

# PyPI hard limit is 1GiB, but try to keep these as small as possible
if [[ "${package_dir}" == "python/libcuopt" ]]; then
    if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '690Mi'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '550Mi'
        )
    fi
elif [[ "${package_dir}" != "python/cuopt" ]] && \
     [[ "${package_dir}" != "python/cuopt/cuopt/linear_programming" ]] && \
     [[ "${package_dir}" != "python/cuopt_server" ]] && \
     [[ "${package_dir}" != "python/cuopt_self_hosted" ]]; then
    rapids-echo-stderr "unrecognized package_dir: '${package_dir}'"
    exit 1
fi

pydistcheck \
    "${PYDISTCHECK_ARGS[@]}" \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'twine'"

twine check \
    --strict \
    "$(echo "${wheel_dir_relative_path}"/*.whl)"

rapids-logger "validate packages with 'abi3audit'"

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag, so only the abi3 wheels
# are audited. Of the packages sharing this script, only 'cuopt' builds one; the rest
# are 'py3-none'.
abi3_wheels=()
while IFS= read -r -d '' wheel; do
    abi3_wheels+=("${wheel}")
done < <(find "${wheel_dir_relative_path}" -type f -name '*-abi3-*.whl' -print0)

# Guard against 'cuopt' silently losing its abi3 tag: without this, dropping
# 'wheel.py-api' would skip the audit entirely and leave CI green.
if [[ "${package_dir}" == "python/cuopt" ]] && [[ "${#abi3_wheels[@]}" -eq 0 ]]; then
    rapids-echo-stderr "expected an abi3 wheel in '${wheel_dir_relative_path}', found none"
    exit 1
fi

if [[ "${#abi3_wheels[@]}" -gt 0 ]]; then
    abi3audit --strict --summary --verbose "${abi3_wheels[@]}"
fi
