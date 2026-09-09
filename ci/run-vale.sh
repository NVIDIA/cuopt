#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if ! command -v vale &>/dev/null; then
    if [[ "${CI:-}" != "true" ]]; then
        echo "WARNING: vale not found — skipping prose lint. Install 'vale' to run this check locally." >&2
        exit 0
    fi
    echo "ERROR: vale not found and CI=true." >&2
    exit 1
fi

exec vale "$@"
