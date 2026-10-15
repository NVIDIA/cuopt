#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cuopt_java_setup_home() {
  local required_binary="${1:-javac}"
  local required_major_version=17

  if [[ -z "${JAVA_HOME:-}" ]]; then
    local javac_path
    javac_path="$(command -v javac || true)"
    if [[ -n "${javac_path}" ]]; then
      JAVA_HOME="$(dirname "$(dirname "$(readlink -f "${javac_path}")")")"
      export JAVA_HOME
    fi
  fi

  if [[ ! -x "${JAVA_HOME:-}/bin/${required_binary}" ]]; then
    echo "JAVA_HOME must point to a JDK containing bin/${required_binary} (Java ${required_major_version} or newer is required)." >&2
    exit 1
  fi

  # java -version prints "17.0.20" or the older "1.8.0_292" scheme; normalize
  # both to a bare major version before comparing.
  local version_line version_string major_version
  version_line="$("${JAVA_HOME}/bin/java" -version 2>&1 | head -n1)"
  version_string="$(printf '%s\n' "${version_line}" | sed -n 's/.*version "\([^"]*\)".*/\1/p')"
  if [[ "${version_string}" == 1.* ]]; then
    major_version="${version_string#1.}"
    major_version="${major_version%%.*}"
  else
    major_version="${version_string%%.*}"
    major_version="${major_version%%-*}"
  fi

  if [[ ! "${major_version}" =~ ^[0-9]+$ ]] || (( major_version < required_major_version )); then
    echo "JAVA_HOME (${JAVA_HOME}) reports '${version_line}', but Java ${required_major_version} or newer is required." >&2
    exit 1
  fi
}
