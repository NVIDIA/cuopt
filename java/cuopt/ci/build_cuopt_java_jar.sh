#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Packages one classifier JAR: the Java classes plus the native library for a single
# CUDA-major/architecture pair, laid out where NativeLibraryLoader looks for it.
#
# The library placed here must be self-contained, because the JAR is the only thing a consumer
# installs. Build it with build_static_libcuopt.sh; see #1817.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODULE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=java/cuopt/ci/java_classifier.sh
source "${SCRIPT_DIR}/java_classifier.sh"

NATIVE_LIB=""
CUDA_VERSION=""
OUTPUT_DIR=""
ARCH="$(uname -m)"

print_help() {
  cat << 'EOF'
Usage: build_cuopt_java_jar.sh --native-lib <path> --cuda-version <ver> --output-dir <dir>

Packages a single self-contained cuOpt Java classifier JAR.

REQUIRED:
    -n, --native-lib     Path to the built libcuopt_jni.so to embed.
    -c, --cuda-version   CUDA version the library was built against, e.g. 13.0.3 or 13.
                         Its major version becomes part of the classifier.
    -o, --output-dir     Directory to receive <classifier>/ with the JAR and its POM.

OPTIONS:
    -a, --arch           Target architecture (default: uname -m).
    -h, --help           Show this message.
EOF
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -h | --help) print_help; exit 0 ;;
    -n | --native-lib) NATIVE_LIB="${2:?--native-lib needs a value}"; shift 2 ;;
    -c | --cuda-version) CUDA_VERSION="${2:?--cuda-version needs a value}"; shift 2 ;;
    -o | --output-dir) OUTPUT_DIR="${2:?--output-dir needs a value}"; shift 2 ;;
    -a | --arch) ARCH="${2:?--arch needs a value}"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; print_help >&2; exit 2 ;;
  esac
done

: "${NATIVE_LIB:?--native-lib is required}"
: "${CUDA_VERSION:?--cuda-version is required}"
: "${OUTPUT_DIR:?--output-dir is required}"

if [[ ! -f "${NATIVE_LIB}" ]]; then
  echo "native library not found: ${NATIVE_LIB}" >&2
  exit 1
fi

CLASSIFIER="$(cuopt_java_classifier "${CUDA_VERSION}" "${ARCH}")"
RESOURCE_DIR="$(cuopt_java_native_resource_dir "${ARCH}")"

# A library that still needs libcuopt.so alongside it would load on the build machine and fail
# for a consumer who installed nothing else, so refuse to ship one.
if readelf -d "${NATIVE_LIB}" 2>/dev/null | grep -q 'NEEDED.*libcuopt\.so'; then
  echo "ERROR: ${NATIVE_LIB} still has a DT_NEEDED on libcuopt.so." >&2
  echo "       A classifier JAR must embed a self-contained library; link the static" >&2
  echo "       archive from build_static_libcuopt.sh instead. See #1817." >&2
  exit 1
fi

STAGING="$(mktemp -d)"
trap 'rm -rf "${STAGING}"' EXIT
mkdir -p "${STAGING}/${RESOURCE_DIR}"
cp "${NATIVE_LIB}" "${STAGING}/${RESOURCE_DIR}/libcuopt_jni.so"

echo "Packaging classifier ${CLASSIFIER}"
echo "  native library -> ${RESOURCE_DIR}/libcuopt_jni.so"

# rmm and rapids_logger define the exception types cuOpt throws and have no static build, so
# they ship beside the JNI library, which finds them through its $ORIGIN RPATH.
for companion in librmm.so librapids_logger.so libtbb.so.12 libnccl.so.2 libcudss.so.0; do
  companion_path="${CUOPT_PREFIX:-}/lib/${companion}"
  if [[ ! -f "${companion_path}" ]]; then
    echo "ERROR: ${companion} not found at ${companion_path}; set CUOPT_PREFIX" >&2
    exit 1
  fi
  # Dereference, since the conda entries are symlinks into a versioned file.
  cp -L "${companion_path}" "${STAGING}/${RESOURCE_DIR}/${companion}"
  echo "  companion      -> ${RESOURCE_DIR}/${companion}"
done

mkdir -p "${OUTPUT_DIR}/${CLASSIFIER}"
mvn -f "${MODULE_DIR}/pom.xml" -B \
  -DskipTests \
  -Dcuopt.jar.classifier="${CLASSIFIER}" \
  -Dcuopt.native.resources="${STAGING}" \
  package

VERSION="$(mvn -f "${MODULE_DIR}/pom.xml" -B -q \
  -Dexec.executable=echo -Dexec.args='${project.version}' \
  --non-recursive exec:exec 2>/dev/null | tail -1)"

cp "${MODULE_DIR}/target/cuopt-${VERSION}-${CLASSIFIER}.jar" "${OUTPUT_DIR}/${CLASSIFIER}/"
cp "${MODULE_DIR}/pom.xml" "${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}.pom"

jar_mb=$(( $(stat -c%s "${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}-${CLASSIFIER}.jar") / 1048576 ))
echo "  wrote ${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}-${CLASSIFIER}.jar (${jar_mb} MB)"
