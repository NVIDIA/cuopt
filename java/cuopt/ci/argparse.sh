#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Argument-handling helpers shared by the Java CI scripts, so a missing or empty flag fails the
# same way everywhere rather than surfacing later as an unbound variable.

# require_value <flag> <value> — the flag was given but its value is missing.
require_value() {
  local flag=$1
  local value=${2:-}
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} requires a value" >&2
    exit 1
  fi
}

# require_arg <flag> <value> — the flag itself is mandatory.
require_arg() {
  local flag=$1
  local value=${2:-}
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} is required." >&2
    if declare -F print_help > /dev/null; then
      print_help >&2
    fi
    exit 1
  fi
}
