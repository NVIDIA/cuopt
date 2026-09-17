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

# The companion set is read off the linked library's own DT_NEEDED entries rather than
# hardcoded, because their exact SONAMEs are build-environment-dependent: e.g. conda-forge's
# TBB is libtbb.so.12, but Rocky 8's dnf tbb-devel package is the much older libtbb.so.2 --
# hardcoding either one breaks on the other environment. Reading DT_NEEDED means this always
# matches whatever this particular library actually links against.
#
# Two categories are deliberately excluded from bundling:
#   - the baseline libraries every Linux system with a working dynamic linker already has
#     (libc, libm, the dynamic linker itself, etc.);
#   - the CUDA math libraries (libcublas*, libcusparse*), which NativeLibraryLoader's
#     preloadCudaLibraries() resolves from the CUDA toolkit layout at runtime instead (they are
#     large, and assumed already present on any CUDA-capable system -- see its own comment).
# Everything else DT_NEEDED names -- TBB (KaMinPar throws through it), NCCL (PDLP's distributed
# path references it unconditionally), cuDSS (kept dynamic deliberately, matching
# CMakeLists.txt's CUOPT_CUDSS_LIBRARY comment), rmm and rapids_logger (prebuilt RAPIDS wheels,
# not a static build -- see CMakeLists.txt and setup_java_static_env.sh), and the build host's
# own GCC runtime (libgomp, libstdc++, libgcc_s, whose consumer-side copies can be too old) --
# travels beside the JNI library, which finds it all through its $ORIGIN RPATH.
#
# libcudss_mtlayer_cuopt.so is the one exception: cuOpt's own cuDSS threading-layer plugin
# (built by build_static_libcuopt.sh against a modern libgomp -- see cpp/CMakeLists.txt's
# CUOPT_BUILD_CUSTOM_CUDSS_MTLAYER and setup_java_static_env.sh), which cudssSetThreadingLayer
# dlopen()s at runtime rather than linking directly, so it never appears in DT_NEEDED at all.
# Without it that call fails and cuDSS writes the failure straight to the process's native
# stdout, corrupting Maven Surefire's forked-JVM protocol, so it is always added explicitly.
BASELINE_SYSTEM_LIBRARY_PATTERN='^(ld-linux|linux-vdso)|^lib(c|m|dl|rt|pthread|resolv|util)\.so'
CUDA_RUNTIME_LIBRARY_PATTERN='^lib(cublas|cublasLt|cusparse|cusolver|nvJitLink)\.so'

# Searched by filename rather than a single prefix, since the build environment may be a conda
# environment (CUOPT_PREFIX/lib), a dnf/system install (e.g. /usr/lib64, cuDSS's own versioned
# directory), a pip-installed wheel's site-packages directory, or the static build tree itself
# (CUOPT_STATIC_BUILD_DIR, for libcudss_mtlayer_cuopt.so), depending on which of
# ci/build_java_static.sh's paths produced this library.
#
# preferred_dir, when given, is searched first and exclusively -- no falling through to the
# broad search below even on a miss. It exists for two cases where the broad search could match
# the wrong file even though a name-only match succeeds:
#   - libcudss_mtlayer_gomp.so.0 (when cuDSS's prebuilt plugin is used instead of the custom
#     one): dnf's cuDSS package registers /usr/lib64/libcudss_mtlayer_gomp.so.0 as an
#     `alternatives` symlink, which can point at a *different* cuDSS version's copy than the one
#     actually pinned and linked. The versioned directory libcudss.so.0 itself was found in is
#     unambiguous, so once that is known, use it for anything else that must be its exact version
#     match.
#   - libgomp.so.1: Rocky 8's own libgomp (including gcc-toolset-14's copy) lacks the OpenMP 5.0
#     symbol cuDSS's threading layer needs (see setup_java_static_env.sh), so the modern one
#     fetched there has to be the one that actually ships, not whichever libgomp.so.1 a broad
#     /usr/lib64 search happens to find first.
find_companion() {
  local name="$1"
  local preferred_dir="${2:-}"
  local found
  if [[ -n "${preferred_dir}" ]]; then
    found="$(find "${preferred_dir}" -maxdepth 1 -name "${name}" -print -quit 2>/dev/null)"
    if [[ -n "${found}" ]]; then
      printf '%s\n' "${found}"
      return
    fi
    echo "ERROR: ${name} not found under ${preferred_dir}" >&2
    exit 1
  fi
  local -a site_packages_dirs=()
  if command -v python3 &> /dev/null; then
    mapfile -t site_packages_dirs < <(python3 -c \
      'import site; print("\n".join(site.getsitepackages()))' 2> /dev/null)
  fi
  found="$(find "${CUOPT_PREFIX:-}/lib" /usr/lib64 /usr/lib /usr/local/cuda*/lib64 \
    "${CUOPT_STATIC_BUILD_DIR:-}" "${site_packages_dirs[@]}" \
    -maxdepth 4 -name "${name}" -print -quit 2>/dev/null)"
  if [[ -z "${found}" ]]; then
    echo "ERROR: ${name} not found under ${CUOPT_PREFIX:-<unset>}/lib, /usr/lib64, /usr/lib, /usr/local/cuda*/lib64, ${CUOPT_STATIC_BUILD_DIR:-<unset>}, or the active Python's site-packages" >&2
    exit 1
  fi
  printf '%s\n' "${found}"
}

declare -a COMPANIONS=()
while IFS= read -r needed; do
  [[ -z "${needed}" ]] && continue
  if [[ "${needed}" =~ ${BASELINE_SYSTEM_LIBRARY_PATTERN} || "${needed}" =~ ${CUDA_RUNTIME_LIBRARY_PATTERN} ]]; then
    continue
  fi
  COMPANIONS+=("${needed}")
done < <(readelf -d "${NATIVE_LIB}" 2>/dev/null \
  | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
COMPANIONS+=(libcudss_mtlayer_cuopt.so)

MANIFEST="${STAGING}/${RESOURCE_DIR}/companions.txt"
: > "${MANIFEST}"
CUDSS_LIBRARY_DIR=""
for companion in "${COMPANIONS[@]}"; do
  preferred_dir=""
  if [[ "${companion}" == "libcudss_mtlayer_gomp.so.0" && -n "${CUDSS_LIBRARY_DIR}" ]]; then
    preferred_dir="${CUDSS_LIBRARY_DIR}"
  elif [[ "${companion}" == "libgomp.so.1" && -n "${CUOPT_MODERN_LIBGOMP_DIR:-}" ]]; then
    preferred_dir="${CUOPT_MODERN_LIBGOMP_DIR}"
  fi
  companion_path="$(find_companion "${companion}" "${preferred_dir}")"
  if [[ "${companion}" == "libcudss.so.0" ]]; then
    CUDSS_LIBRARY_DIR="$(dirname "${companion_path}")"
  fi
  # Dereference, since these are commonly symlinks into a versioned file.
  cp -L "${companion_path}" "${STAGING}/${RESOURCE_DIR}/${companion}"
  echo "${companion}" >> "${MANIFEST}"
  echo "  companion      -> ${RESOURCE_DIR}/${companion} (from ${companion_path})"
done

mkdir -p "${OUTPUT_DIR}/${CLASSIFIER}"
# -Pattach-source-javadoc: this publishes to a Maven repository, which requires sources and
# javadoc jars. Most mvn invocations (test, verify) don't activate it, since they don't
# package anything -- see the profile's own comment in pom.xml for why that distinction exists.
cuopt_mvn -f "${MODULE_DIR}/pom.xml" -B \
  -Pattach-source-javadoc \
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
