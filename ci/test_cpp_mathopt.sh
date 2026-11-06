#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export CUOPT_CPP_TEST_GROUP=mathopt
exec "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/test_cpp.sh" "$@"
