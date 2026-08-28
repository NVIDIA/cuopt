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
# shellcheck source=java/cuopt/ci/argparse.sh
source "${SCRIPT_DIR}/argparse.sh"
# shellcheck source=java/cuopt/scripts/maven.sh
source "${SCRIPT_DIR}/../scripts/maven.sh"
cuopt_maven_args
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
    -n | --native-lib) require_value "$1" "${2:-}"; NATIVE_LIB=$2; shift 2 ;;
    -c | --cuda-version) require_value "$1" "${2:-}"; CUDA_VERSION=$2; shift 2 ;;
    -o | --output-dir) require_value "$1" "${2:-}"; OUTPUT_DIR=$2; shift 2 ;;
    -a | --arch) require_value "$1" "${2:-}"; ARCH=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; print_help >&2; exit 2 ;;
  esac
done

require_arg --native-lib "${NATIVE_LIB}"
require_arg --cuda-version "${CUDA_VERSION}"
require_arg --output-dir "${OUTPUT_DIR}"

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
cuopt_mvn -f "${MODULE_DIR}/pom.xml" -B \
  -DskipTests \
  -Dcuopt.jar.classifier="${CLASSIFIER}" \
  -Dcuopt.native.resources="${STAGING}" \
  package

# Read straight from the POM rather than asking Maven: this needs no network, and
# ci/release/update-version.sh keeps the marker in step with the version.
VERSION="$(sed -n 's/.*VERSION_UPDATE_MARKER_START--><version>\([^<]*\)<\/version>.*/\1/p' \
  "${MODULE_DIR}/pom.xml")"
if [[ -z "${VERSION}" ]]; then
  echo "could not read the version from ${MODULE_DIR}/pom.xml" >&2
  exit 1
fi

# Each classifier directory carries everything Maven Central needs for the artifact, so the
# gather step can work from the classifier directories alone.
cp "${MODULE_DIR}/target/cuopt-${VERSION}-${CLASSIFIER}.jar" "${OUTPUT_DIR}/${CLASSIFIER}/"
cp "${MODULE_DIR}/pom.xml" "${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}.pom"
for kind in sources javadoc; do
  if [[ -f "${MODULE_DIR}/target/cuopt-${VERSION}-${kind}.jar" ]]; then
    cp "${MODULE_DIR}/target/cuopt-${VERSION}-${kind}.jar" "${OUTPUT_DIR}/${CLASSIFIER}/"
  else
    echo "WARNING: no ${kind} JAR in ${MODULE_DIR}/target; Maven Central requires one" >&2
  fi
done

jar_mb=$(( $(stat -c%s "${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}-${CLASSIFIER}.jar") / 1048576 ))
echo "  wrote ${OUTPUT_DIR}/${CLASSIFIER}/cuopt-${VERSION}-${CLASSIFIER}.jar (${jar_mb} MB)"
