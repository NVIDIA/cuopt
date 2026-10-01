#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check that the JNI layer and the Java `native` declarations agree.
#
# The bindings are hand-written: every `static native` method in Java needs a matching
# Java_com_nvidia_..._name function in cuopt_jni.cpp. Nothing in the compiler enforces that
# pairing, and a mismatch is invisible until a test (or a user) touches the method and gets an
# UnsatisfiedLinkError. This compares what javac says the symbols must be against what the built
# library actually exports.

set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${MODULE_DIR}/../.." && pwd)"
NATIVE_BUILD_DIR="${CUOPT_JAVA_NATIVE_BUILD_DIR:-${MODULE_DIR}/build/native}"
LIBRARY="${NATIVE_BUILD_DIR}/libcuopt_jni.so"

source "${MODULE_DIR}/scripts/java_home.sh"
cuopt_java_setup_home javac

if [[ ! -f "${LIBRARY}" ]]; then
  echo "libcuopt_jni.so was not found at ${LIBRARY}; build it first." >&2
  exit 1
fi
if ! command -v nm > /dev/null; then
  echo "nm was not found; skipping the JNI symbol check." >&2
  exit 0
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

# Regenerate the constants so javac sees the same sources the real build compiles.
bash "${MODULE_DIR}/scripts/generate_constants.sh" \
  "${REPO_ROOT}/cpp/include/cuopt/mathematical_optimization/constants.h" \
  "${WORK_DIR}/generated" > /dev/null

mapfile -t SOURCES < <(find "${MODULE_DIR}/src/main/java" "${WORK_DIR}/generated" -name '*.java')
"${JAVA_HOME}/bin/javac" -nowarn -d "${WORK_DIR}/classes" -h "${WORK_DIR}/headers" "${SOURCES[@]}"

# javac emits one prototype per native method; the built library must define exactly those.
grep -ho 'Java_com_nvidia[A-Za-z0-9_]*' "${WORK_DIR}"/headers/*.h | sort -u > "${WORK_DIR}/declared"
nm -D --defined-only "${LIBRARY}" \
  | grep -o 'Java_com_nvidia[A-Za-z0-9_]*' | sort -u > "${WORK_DIR}/defined"

missing="$(comm -23 "${WORK_DIR}/declared" "${WORK_DIR}/defined")"
orphaned="$(comm -13 "${WORK_DIR}/declared" "${WORK_DIR}/defined")"

status=0
if [[ -n "${missing}" ]]; then
  echo "Java declares native methods that libcuopt_jni.so does not define." >&2
  echo "Calling any of these throws UnsatisfiedLinkError at run time:" >&2
  echo "${missing}" | sed 's/^/  /' >&2
  status=1
fi
if [[ -n "${orphaned}" ]]; then
  echo "libcuopt_jni.so defines JNI entry points with no Java declaration." >&2
  echo "These are dead code, or a rename left one side behind:" >&2
  echo "${orphaned}" | sed 's/^/  /' >&2
  status=1
fi

if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi

echo "JNI symbols match: $(wc -l < "${WORK_DIR}/declared") native methods declared and defined."
