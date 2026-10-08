#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared argparse helpers for the ci/maven-publish/ host orchestrator scripts.
# Meant to be sourced, not executed:
#   . "${SCRIPT_DIR}/argparse.sh"

# require_value <flag> <value>: exit 1 if <value> is empty.
require_value() {
  local flag=$1
  local value=$2
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} requires a value" >&2
    exit 1
  fi
}

# require_arg <flag> <value>: exit 1 if <value> empty, printing help if defined.
require_arg() {
  local flag=$1
  local value=$2
  if [[ -z ${value} ]]; then
    echo "Error: ${flag} is required." >&2
    if declare -F print_help > /dev/null; then
      print_help
    fi
    exit 1
  fi
}
