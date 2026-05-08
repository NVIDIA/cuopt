#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build entry point for cuopt-java.
#
# Pipeline:
#   1. Verify JDK 22 + Maven 3.9.6+ are available.
#   2. Verify libcuopt.so is reachable.
#   3. Run panama-bindings/generate-bindings.sh (regenerates jextract output).
#   4. Run drift gate: 'git diff --exit-code' on the generated panama dir
#      forces commit of any changes.
#   5. Run 'mvn clean verify'.
#
# Environment variables (all optional):
#   JEXTRACT          Path to jextract binary. Default: search PATH.
#   CUOPT_INCLUDE     Path to cuopt include dir. Default: ../cpp/include.
#   CUDA_INCLUDE      Path to CUDA include dir. Default: /usr/local/cuda/include.
#   CMAKE_PREFIX_PATH Prepended to LD_LIBRARY_PATH so libcuopt.so is found.
#                     Default: ../cpp/build.
#   SKIP_DRIFT_CHECK  If set to 'true', skips the panama drift gate.
#                     Useful in initial-bootstrap commits before the
#                     bindings are first committed.
#   SKIP_TESTS        If set to 'true', runs 'mvn package' instead of
#                     'mvn verify'.

set -euo pipefail

CURDIR="$(cd "$(dirname "$0")" && pwd)"
REPODIR="$(cd "${CURDIR}/.." && pwd)"

JEXTRACT="${JEXTRACT:-jextract}"
CUOPT_INCLUDE="${CUOPT_INCLUDE:-${REPODIR}/cpp/include}"
CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-${REPODIR}/cpp/build}"

echo "==> cuopt-java build"
echo "    REPODIR=${REPODIR}"
echo "    JEXTRACT=${JEXTRACT}"
echo "    CUOPT_INCLUDE=${CUOPT_INCLUDE}"
echo "    CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}"

# 1. Toolchain checks
if ! command -v java >/dev/null 2>&1; then
    echo "ERROR: java not found in PATH." >&2
    exit 1
fi
JAVA_VERSION="$(java -version 2>&1 | head -1 | sed -E 's/.*"([0-9]+).*/\1/')"
if [[ -z "${JAVA_VERSION}" || "${JAVA_VERSION}" -lt 22 ]]; then
    echo "ERROR: cuopt-java requires JDK 22 or higher (got: $(java -version 2>&1 | head -1))." >&2
    echo "       Install with: conda install -c conda-forge openjdk=22" >&2
    exit 1
fi

if ! command -v mvn >/dev/null 2>&1; then
    echo "ERROR: mvn not found in PATH." >&2
    echo "       Install with: conda install -c conda-forge maven" >&2
    exit 1
fi

if ! command -v "${JEXTRACT}" >/dev/null 2>&1; then
    echo "ERROR: jextract not found at '${JEXTRACT}'." >&2
    echo "       Download jextract for JDK 22 from:" >&2
    echo "         https://jdk.java.net/jextract/" >&2
    echo "       Then set JEXTRACT to the binary path, or add it to PATH." >&2
    exit 1
fi

# 2. libcuopt.so check
if [[ ! -f "${CMAKE_PREFIX_PATH}/libcuopt.so" ]]; then
    echo "WARNING: libcuopt.so not found at ${CMAKE_PREFIX_PATH}/libcuopt.so" >&2
    echo "         Tests will fail unless libcuopt is on java.library.path." >&2
fi
export LD_LIBRARY_PATH="${CMAKE_PREFIX_PATH}:${LD_LIBRARY_PATH:-}"

# 3. Regenerate panama bindings
echo "==> Regenerating panama bindings"
"${CURDIR}/panama-bindings/generate-bindings.sh"

# 4. Drift gate
if [[ "${SKIP_DRIFT_CHECK:-false}" != "true" ]]; then
    echo "==> Checking for panama bindings drift"
    PANAMA_REL="java/cuopt-java/src/main/java22/com/nvidia/cuopt/internal/panama"
    if ! git -C "${REPODIR}" diff --quiet -- "${PANAMA_REL}" 2>/dev/null; then
        echo "ERROR: Panama bindings drifted from committed state." >&2
        echo "       Run './java/build.sh' and commit the regenerated files at:" >&2
        echo "       ${PANAMA_REL}/" >&2
        echo "       (Use SKIP_DRIFT_CHECK=true to bypass during initial bootstrap.)" >&2
        git -C "${REPODIR}" diff --stat -- "${PANAMA_REL}" >&2
        exit 1
    fi
fi

# 5. Maven build
echo "==> Running Maven"
cd "${CURDIR}/cuopt-java"
if [[ "${SKIP_TESTS:-false}" == "true" ]]; then
    mvn clean package -DskipTests
else
    mvn clean verify -Djava.library.path="${CMAKE_PREFIX_PATH}"
fi

echo "==> cuopt-java build complete"
