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
            --max-allowed-size-compressed '725Mi'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '550Mi'
        )
    fi
elif [[ "${package_dir}" == "python/libcuopt_mathopt" ]] || \
     [[ "${package_dir}" == "python/libcuopt_routing" ]]; then
    # Each is strictly smaller than the combined libcuopt wheel was, so its limits are a
    # safe ceiling. Worth tightening once CI reports the real compressed sizes.
    if [[ "${RAPIDS_CUDA_MAJOR}" == "12" ]]; then
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '725Mi'
        )
    else
        PYDISTCHECK_ARGS+=(
            --max-allowed-size-compressed '550Mi'
        )
    fi
elif [[ "${package_dir}" == "python/libcuopt_client" ]]; then
    # No CUDA kernels: libcuopt_client.so is single-digit MB, so anything near the other
    # limits would mean something got vendored into it that should not have been.
    PYDISTCHECK_ARGS+=(
        --max-allowed-size-compressed '50Mi'
    )
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

# 'abi3audit' fails on wheels with DSOs that lack an ABI tag (e.g. 'lib*' wheels).
# Filtering by '*abi*' avoids those.
find \
    "${wheel_dir_relative_path}" \
    -type f \
    -name '*abi*' \
    -exec abi3audit --strict --summary --verbose '{}' \+
