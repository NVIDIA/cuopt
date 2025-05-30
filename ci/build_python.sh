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

source rapids-configure-sccache

source rapids-date-string

export CMAKE_GENERATOR=Ninja

rapids-print-env

CPP_CHANNEL=$(rapids-download-conda-from-github cpp)

version=$(rapids-generate-version)
export RAPIDS_PACKAGE_VERSION=${version}
export DEPENDENT_PACKAGE_VERSION="25.04"
echo "${version}" > VERSION

git_commit=$(git rev-parse HEAD)
package_dir="python"
for package_name in cuopt cuopt_server; do
  sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/${package_name}/${package_name}/_version.py"
done

sed -i "/^__git_commit__/ s/= .*/= \"${git_commit}\"/g" "${package_dir}/cuopt/cuopt/linear_programming/cuopt_mps_parser/_version.py"

# populates `RATTLER_CHANNELS` array and `RATTLER_ARGS` array
source rapids-rattler-channel-string

# Override `rapids-rattler-channelstring` while cuOpt is not on the standard RAPIDS release cycle
RATTLER_ARGS=("--experimental" "--no-build-id" "--channel-priority" "disabled" "--output-dir" "$RAPIDS_CONDA_BLD_OUTPUT_DIR")
# Prepending `rapidsai` channel so cuOpt can grab release builds of dependencies
# that have been cleared from `rapidsai-nightly`
RATTLER_CHANNELS=("--channel" "rapidsai" "${RATTLER_CHANNELS[@]}")

rapids-logger "Prepending channel ${CPP_CHANNEL} to RATTLER_CHANNELS"

RATTLER_CHANNELS=("--channel" "${CPP_CHANNEL}" "${RATTLER_CHANNELS[@]}")

sccache --zero-stats

rapids-logger "Building mps-parser"

# --no-build-id allows for caching with `sccache`
# more info is available at
# https://rattler.build/latest/tips_and_tricks/#using-sccache-or-ccache-with-rattler-build
rattler-build build --recipe conda/recipes/mps-parser \
                    --test skip \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats
sccache --zero-stats

rapids-logger "Building cuopt"

rattler-build build --recipe conda/recipes/cuopt \
                    --test skip \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

sccache --show-adv-stats

rattler-build build --recipe conda/recipes/cuopt-server \
                    --test skip \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

rattler-build build --recipe conda/recipes/cuopt-sh-client \
                    --test skip \
                    "${RATTLER_ARGS[@]}" \
                    "${RATTLER_CHANNELS[@]}"

# remove build_cache directory to avoid uploading the entire source tree
# tracked in https://github.com/prefix-dev/rattler-build/issues/1424
rm -rf "$RAPIDS_CONDA_BLD_OUTPUT_DIR"/build_cache
