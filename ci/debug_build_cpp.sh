#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Temporary diagnostic wrapper around ci/build_cpp.sh for #1976. Delete this
# file, ci/debug_build_wheel_libcuopt_mathopt.sh, ci/utils/debug_resource_watchdog.sh,
# and .github/workflows/debug_cpp_hang.yaml once the root cause is confirmed and fixed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils/debug_resource_watchdog.sh"

debug_print_resource_snapshot
debug_start_watchdog 20

"${SCRIPT_DIR}/build_cpp.sh"
