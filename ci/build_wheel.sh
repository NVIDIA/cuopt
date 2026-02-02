#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

package_name=$1
package_dir=$2

source rapids-configure-sccache
source rapids-date-string
RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX="true"
export RAPIDS_INIT_PIP_REMOVE_NVIDIA_INDEX
source rapids-init-pip

# Update the version to accomdate nightly and release changes for the wheel name
rapids-generate-version > ./VERSION

cd "${package_dir}"

sccache --zero-stats

rapids-logger "Building '${package_name}' wheel"

RAPIDS_PIP_WHEEL_ARGS=(
  -w dist
  -v
  --no-deps
  --disable-pip-version-check
)

# Only use --build-constraint when build isolation is enabled.
#
# Passing '--build-constraint' and '--no-build-isolation` together results in an error from 'pip',
# but we want to keep environment variable PIP_CONSTRAINT set unconditionally.
# PIP_NO_BUILD_ISOLATION=0 means "add --no-build-isolation" (ref: https://github.com/pypa/pip/issues/5735)
if [[ "${PIP_NO_BUILD_ISOLATION:-}" != "0" ]]; then
    RAPIDS_PIP_WHEEL_ARGS+=(--build-constraint="${PIP_CONSTRAINT}")
fi

# unset PIP_CONSTRAINT (set by rapids-init-pip)... it doesn't affect builds as of pip 25.3, and
# results in an error from 'pip wheel' when set and --build-constraint is also passed
unset PIP_CONSTRAINT
rapids-pip-retry wheel \
    "${RAPIDS_PIP_WHEEL_ARGS[@]}" \
    .

sccache --show-adv-stats
