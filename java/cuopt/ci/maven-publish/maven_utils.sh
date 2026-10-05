#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the ci/maven-publish/ pipeline scripts (host and
# in-container). Sourced, not executed:
#   . "${SCRIPT_DIR}/maven_utils.sh"

fatal() {
  echo "Error: $*" >&2
  exit 1
}

require_cmds() {
  local cmd
  for cmd in "$@"; do
    if ! command -v "${cmd}" >/dev/null; then
      fatal "'${cmd}' not on PATH"
    fi
  done
}

require_release_version() {
  local version=$1
  if [[ ${version} == *-SNAPSHOT ]]; then
    fatal "release-shaped version required, got '${version}'"
  fi
}

require_maven_coordinates() {
  local group_id=$1 artifact_id=$2 version=$3
  local coordinate_pattern='^[A-Za-z0-9][A-Za-z0-9_.+-]*$'
  if [[ ! ${group_id} =~ ${coordinate_pattern} ]]; then
    fatal "invalid Maven groupId '${group_id}'"
  fi
  if [[ ! ${artifact_id} =~ ${coordinate_pattern} ]]; then
    fatal "invalid Maven artifactId '${artifact_id}'"
  fi
  if [[ ! ${version} =~ ${coordinate_pattern} ]]; then
    fatal "invalid Maven version '${version}'"
  fi
}

# Maven groupId (a.b.c) -> filesystem path (a/b/c).
maven_group_path() {
  printf '%s\n' "$1" | tr '.' '/'
}
