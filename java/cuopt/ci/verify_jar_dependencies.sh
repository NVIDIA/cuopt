#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Checks that a classifier JAR can satisfy its own native dependencies.
#
# A JAR is only self-contained if every library it needs is either inside it, part of the CUDA
# toolkit the classifier names, or part of the base system. Anything else resolves on a build
# machine, because the build environment happens to have it, and fails for a consumer who
# installed only the JAR. Linking libcuopt statically surfaced four such libraries one at a time
# (rmm, TBB, NCCL, cuDSS), each as an UnsatisfiedLinkError at run time; this catches that class
# of gap at build time instead.
#
# The check reads DT_NEEDED rather than resolving against a directory, because a build
# environment's lib directory contains every dependency by construction and would make any JAR
# look self-contained.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=java/cuopt/ci/argparse.sh
source "${SCRIPT_DIR}/argparse.sh"

JAR=""

# Supplied by the CUDA toolkit a consumer installs for the classifier's CUDA major version.
ALLOWED_CUDA_LIBRARIES=(
  libcublas.so libcublasLt.so libcusparse.so libcusolver.so libcudart.so libcurand.so
  libnvJitLink.so libnvrtc.so libcuda.so
)

# Present on any Linux that can run a JVM.
ALLOWED_SYSTEM_LIBRARIES=(
  libc.so libm.so libdl.so librt.so libpthread.so libstdc++.so libgcc_s.so libgomp.so
  ld-linux-x86-64.so ld-linux-aarch64.so libresolv.so
)

print_help() {
  cat << 'EOF'
Usage: verify_jar_dependencies.sh --jar <path>

Fails if the JAR's native libraries need anything that is neither packaged inside it, nor part
of the CUDA toolkit, nor part of the base system.

REQUIRED:
    -j, --jar    Classifier JAR to check.
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help) print_help; exit 0 ;;
    -j | --jar) require_value "$1" "${2:-}"; JAR=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; print_help >&2; exit 2 ;;
  esac
done

require_arg --jar "${JAR}"
if [[ ! -f "${JAR}" ]]; then
  echo "JAR not found: ${JAR}" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
unzip -q "${JAR}" '*/Linux/*.so*' -d "${WORK}" 2>/dev/null || true

JNI_LIB="$(find "${WORK}" -name 'libcuopt_jni.so' -print -quit)"
if [[ -z "${JNI_LIB}" ]]; then
  echo "ERROR: ${JAR} contains no libcuopt_jni.so" >&2
  exit 1
fi
NATIVE_DIR="$(dirname "${JNI_LIB}")"

echo "Packaged libraries:"
while read -r lib; do
  printf '  %6s MB  %s\n' "$(( $(stat -c%s "${NATIVE_DIR}/${lib}") / 1048576 ))" "${lib}"
done < <(cd "${NATIVE_DIR}" && ls -S ./*.so* | sed 's|^\./||')

# Strip the version suffix so libnccl.so.2 matches an allowlist entry of libnccl.so.
soname_stem() { sed -E 's/\.so\.[0-9.]+$/.so/' <<< "$1"; }

allowed_external=("${ALLOWED_CUDA_LIBRARIES[@]}" "${ALLOWED_SYSTEM_LIBRARIES[@]}")
unsatisfied=()

echo
echo "Checking DT_NEEDED of every packaged library"
for lib in "${NATIVE_DIR}"/*.so*; do
  while read -r needed; do
    [[ -z "${needed}" ]] && continue
    # Packaged beside it, so the $ORIGIN RPATH resolves it.
    if [[ -e "${NATIVE_DIR}/${needed}" ]]; then
      continue
    fi
    stem="$(soname_stem "${needed}")"
    permitted=false
    for allowed in "${allowed_external[@]}"; do
      if [[ "${stem}" == "${allowed}" ]]; then
        permitted=true
        break
      fi
    done
    if [[ "${permitted}" == false ]]; then
      unsatisfied+=("$(basename "${lib}") needs ${needed}")
    fi
  done < <(readelf -d "${lib}" 2>/dev/null | sed -n 's/.*NEEDED.*\[\(.*\)\]/\1/p')
done

if [[ ${#unsatisfied[@]} -gt 0 ]]; then
  echo >&2
  echo "ERROR: the JAR is not self-contained. Unsatisfied dependencies:" >&2
  printf '  %s\n' "${unsatisfied[@]}" | sort -u >&2
  echo >&2
  echo "Each must be linked into the JNI library or packaged beside it; see" >&2
  echo "build_cuopt_java_jar.sh. A consumer installs nothing but this JAR." >&2
  exit 1
fi

# libcuopt.so reappearing means the static link silently fell back to the shared library.
if readelf -d "${JNI_LIB}" | grep -q 'NEEDED.*libcuopt\.so'; then
  echo "ERROR: libcuopt_jni.so depends on libcuopt.so; it was not linked statically." >&2
  exit 1
fi

echo
echo "Self-contained: every dependency is packaged, CUDA toolkit, or base system."
