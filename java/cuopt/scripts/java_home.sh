#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cuopt_java_setup_home() {
  local required_binary="${1:-javac}"

  if [[ -z "${JAVA_HOME:-}" ]]; then
    local javac_path
    javac_path="$(command -v javac || true)"
    if [[ -n "${javac_path}" ]]; then
      JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${javac_path}")")")"
      export JAVA_HOME
    fi
  fi

  if [[ ! -x "${JAVA_HOME:-}/bin/${required_binary}" ]]; then
    echo "JAVA_HOME must point to a JDK containing bin/${required_binary} (Java 11 is required)." >&2
    exit 1
  fi
}
