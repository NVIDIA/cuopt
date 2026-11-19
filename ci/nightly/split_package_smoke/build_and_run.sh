#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Usage: build_and_run.sh <client|mathopt|routing> <CMAKE_PREFIX_PATH>
#
# Builds the consumer for one component against whatever is on CMAKE_PREFIX_PATH, then runs it.

set -euo pipefail

component=$1
prefix_path=$2
src="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="$(mktemp -d)/${component}"

cmake -S "${src}" -B "${build}" -GNinja \
  -DCOMPONENT="${component}" \
  -DCMAKE_PREFIX_PATH="${prefix_path}"
cmake --build "${build}"

cd "${build}"
"./smoke_${component}"
