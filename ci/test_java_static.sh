#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Runs the Java test suite against an already-packaged classifier JAR, on a GPU, with no
# libcuopt installed. See #1817.
#
# ci/build_java_static.sh checks the JAR's dependencies statically; this is the other half,
# that the libraries it carries actually load and produce correct answers.
#
# Activates -Ppackaged-jar-tests so main compilation is skipped and the JAR supplies the classes
# and the native libraries. PackagedJarOriginCheck then asserts that is genuinely where they came
# from, so a stray target/classes cannot make this pass while testing the wrong thing.
#
# CUOPT_JAVA_JAR may be set to a classifier JAR to skip the download and test it directly.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=java/cuopt/ci/java_classifier.sh
. "${REPO_ROOT}/java/cuopt/ci/java_classifier.sh"
# shellcheck source=java/cuopt/scripts/maven.sh
. "${REPO_ROOT}/java/cuopt/scripts/maven.sh"
cuopt_maven_args

if [[ -z "${CUOPT_JAVA_JAR:-}" ]]; then
  case "$(arch)" in
    x86_64) JOB_ARCH=amd64 ;;
    aarch64) JOB_ARCH=arm64 ;;
    *) echo "unsupported architecture $(arch)" >&2; exit 1 ;;
  esac
  ARTIFACT="cuopt_java_${JOB_ARCH}_cu${RAPIDS_CUDA_VERSION%%.*}"
  rapids-logger "Downloading ${ARTIFACT}"
  JAVA_PKG="$(rapids-download-from-github "${ARTIFACT}")"
  CUOPT_JAVA_JAR="$(cuopt_java_resolve_artifact_jar "${JAVA_PKG}")"
fi
rapids-logger "Testing $(basename "${CUOPT_JAVA_JAR}")"

# A JDK and Maven only -- no conda, no cuOpt package. The container image (rapidsai/ci-wheel)
# already ships the CUDA runtime (libcublas/libcusparse) that the JAR dynamically links against;
# installing libcuopt itself would defeat the test, since the JAR is supposed to carry its own
# copy of everything else it needs. See #1817 and the java-static-classifiers PR discussion for
# why this moved off a fresh `conda create`: that env-solve was slow and consistently synced up
# concurrent matrix jobs' cold Maven Central resolution, which is what triggered repeated 429s.
rapids-logger "Installing a JDK (dnf's own maven package is too old; see MAVEN_VERSION below)"
MAVEN_VERSION="3.9.16"
# Matches maven.compiler.release in java/cuopt/pom.xml (bumped to 17 in #1865); an older JDK
# cannot target a newer --release version.
dnf install -y java-17-openjdk-devel
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk
MAVEN_HOME="$(mktemp -d)"
MAVEN_TARBALL="$(mktemp)"
# dlcdn.apache.org (not archive.apache.org) is Apache's actual mirror-network-backed download
# endpoint -- archive.apache.org is documented as permanent, unmirrored storage for old releases,
# not meant for routine/automated downloads, and has been observed intermittently unreachable
# (connection timeouts) from GPU test runners as a result. dlcdn only serves the current release
# per branch, so MAVEN_VERSION has to track that (currently 3.9.16, the latest 3.9.x).
MAVEN_TARBALL_URL="https://dlcdn.apache.org/maven/maven-3/${MAVEN_VERSION}/binaries/apache-maven-${MAVEN_VERSION}-bin.tar.gz"
# Retry on top of the mirror switch: even a mirror-backed CDN can have a transient blip, and
# retrying is cheap insurance against failing an otherwise-passing job over it.
CURL_RETRY_ARGS=(--retry 5 --retry-delay 5 --retry-connrefused)
curl -fsSL "${CURL_RETRY_ARGS[@]}" "${MAVEN_TARBALL_URL}" -o "${MAVEN_TARBALL}"
# Verify against Apache's published SHA-512 rather than trusting transport security alone.
echo "$(curl -fsSL "${CURL_RETRY_ARGS[@]}" "${MAVEN_TARBALL_URL}.sha512")  ${MAVEN_TARBALL}" | sha512sum --check --status
tar xz -C "${MAVEN_HOME}" --strip-components=1 -f "${MAVEN_TARBALL}"
rm -f "${MAVEN_TARBALL}"
export PATH="${MAVEN_HOME}/bin:${JAVA_HOME}/bin:${PATH}"

if command -v ldconfig >/dev/null 2>&1 && ldconfig -p | grep -q libcuopt.so; then
  echo "ERROR: libcuopt.so is present in the test environment, so passing here would not show" >&2
  echo "       that the JAR is self-contained." >&2
  exit 1
fi

# The JAR dynamically links against the image's own CUDA runtime (libcublas/libcusparse), on
# the assumption that the right /usr/local/cuda*/targets/*/lib is already on the dynamic
# linker's search path via ldconfig. Observed missing on at least one arm64 runner
# (UnsatisfiedLinkError: libcublas.so.13) despite being present and ldconfig-registered in the
# published image itself, so don't rely on that implicit setup -- find and export the path
# explicitly instead. arm64 images ship both a targets/aarch64-linux and a targets/sbsa-linux
# directory; only the latter actually has the libraries, so match on libcublas.so being present
# rather than just the first target directory found (aarch64-linux sorts first and is empty).
CUDA_LIB_DIR=""
for candidate in /usr/local/cuda*/targets/*/lib; do
  if [[ -e "${candidate}/libcublas.so" ]]; then
    CUDA_LIB_DIR="${candidate}"
    break
  fi
done 2>/dev/null || true
if [[ -n "${CUDA_LIB_DIR}" ]]; then
  rapids-logger "Adding ${CUDA_LIB_DIR} to LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH="${CUDA_LIB_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
  echo "WARNING: no CUDA targets lib dir with libcublas.so found; relying on the image's own" >&2
  echo "         search path" >&2
fi

java -version
mvn -version
nvidia-smi

rapids-logger "Running the suite against the packaged JAR"
if ! cuopt_mvn -B -f "${REPO_ROOT}/java/cuopt/pom.xml" test \
  -Ppackaged-jar-tests \
  "-Dcuopt.jar.path=${CUOPT_JAVA_JAR}"; then
  # Surefire's forked-JVM crash diagnostics (e.g. a raw native write to stdout corrupting its
  # fork-communication channel) land in target/surefire-reports/*.dumpstream and any
  # hs_err_pid*.log a real JVM crash leaves behind. Neither is printed to the console or
  # uploaded as an artifact by this job, so a failure here is otherwise a dead end without
  # reproducing it locally. Print them inline instead.
  rapids-logger "Test failure -- dumping Surefire fork-crash diagnostics"
  find "${REPO_ROOT}/java/cuopt/target/surefire-reports" -type f \
    \( -name '*.dumpstream' -o -name 'hs_err_pid*.log' \) -print0 2>/dev/null |
    while IFS= read -r -d '' f; do
      echo "----- ${f} -----"
      cat "${f}"
    done

  # A crash while loading/using an extracted native library (rather than a plain assertion
  # failure) is otherwise a dead end: the extraction directory is private and torn down with the
  # runner, so nothing about what actually got bundled survives past this job. Dump each
  # library's own dependency resolution -- an unresolved symbol/SONAME here, not visible from the
  # packaging step alone, points straight at the mismatch (e.g. a companion resolved from a
  # different install than the one actually linked).
  NATIVE_EXTRACT_DIR="$(find /tmp -maxdepth 1 -name 'cuopt-native-*' -print -quit 2>/dev/null)"
  if [[ -n "${NATIVE_EXTRACT_DIR}" ]]; then
    rapids-logger "Test failure -- dumping native library dependency resolution"
    for lib in "${NATIVE_EXTRACT_DIR}"/*.so*; do
      [[ -f "${lib}" ]] || continue
      echo "----- ldd ${lib} -----"
      ldd "${lib}" 2>&1
    done
  fi
  exit 1
fi

rapids-logger "Classifier JAR verified end to end"
