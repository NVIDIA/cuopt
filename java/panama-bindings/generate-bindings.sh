#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates Panama bindings for cuOpt's C API into
# cuopt-java/src/main/java22/com/nvidia/cuopt/internal/panama/.
#
# Auto-downloads jextract for JDK 22 into ./jextract-22/ on first run
# (cuvs pattern). Subsequent runs reuse the local copy. Pass JEXTRACT
# explicitly to use an externally-installed jextract.
#
# Inputs:
#   JEXTRACT         Path to a jextract binary. Default: search PATH (and
#                    the local ./jextract-22/bin/).
#   CUOPT_INCLUDE    Path to cpp/include (defaults to ../../cpp/include).
#   CUDA_INCLUDE_DIR Path to CUDA include dir. Default: $CONDA_PREFIX/
#                    targets/x86_64-linux/include if present, else
#                    /usr/local/cuda/include.

set -euo pipefail

CURDIR="$(cd "$(dirname "$0")" && pwd)"
REPODIR="$(cd "${CURDIR}/../.." && pwd)"

CUOPT_INCLUDE="${CUOPT_INCLUDE:-${REPODIR}/cpp/include}"
OUTPUT_DIR="${REPODIR}/java/cuopt-java/src/main/java22"
TARGET_PACKAGE="com.nvidia.cuopt.internal.panama"

# CUDA include detection — prefer the conda env's CUDA toolkit if active.
TARGET_DIR="targets/x86_64-linux/include"
if [[ -z "${CUDA_INCLUDE_DIR:-}" ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]] && [[ -d "${CONDA_PREFIX}/${TARGET_DIR}" ]]; then
        CUDA_INCLUDE_DIR="${CONDA_PREFIX}/${TARGET_DIR}"
    elif [[ -d "/usr/local/cuda/${TARGET_DIR}" ]]; then
        CUDA_INCLUDE_DIR="/usr/local/cuda/${TARGET_DIR}"
    elif [[ -d "/usr/local/cuda/include" ]]; then
        CUDA_INCLUDE_DIR="/usr/local/cuda/include"
    else
        echo "ERROR: Could not locate a CUDA include directory." >&2
        echo "       Set CUDA_INCLUDE_DIR explicitly." >&2
        exit 1
    fi
fi

# Auto-download jextract (cuvs pattern): prepend local ./jextract-22/bin/
# to PATH, then check if jextract is reachable. If not, download.
PATH="${CURDIR}/jextract-22/bin:${PATH}"
export PATH

JEXTRACT="${JEXTRACT:-jextract}"
if ! command -v "${JEXTRACT}" >/dev/null 2>&1; then
    JEXTRACT_FILENAME="openjdk-22-jextract+6-47_linux-x64_bin.tar.gz"
    JEXTRACT_DOWNLOAD_URL="https://download.java.net/java/early_access/jextract/22/6/${JEXTRACT_FILENAME}"
    echo "jextract not found. Downloading from ${JEXTRACT_DOWNLOAD_URL} ..."
    (
        cd "${CURDIR}"
        wget -c "${JEXTRACT_DOWNLOAD_URL}"
        tar -xzf "./${JEXTRACT_FILENAME}"
        echo "jextract installed to ${CURDIR}/jextract-22"
    )
    # PATH was already updated above to include ${CURDIR}/jextract-22/bin
fi

# Clean previous output so removed symbols actually disappear.
PANAMA_DIR="${OUTPUT_DIR}/com/nvidia/cuopt/internal/panama"
mkdir -p "${PANAMA_DIR}"
find "${PANAMA_DIR}" -name '*.java' -delete 2>/dev/null || true

echo "Running jextract ..."
jextract \
    --include-dir "${CUOPT_INCLUDE}" \
    --include-dir "${CUDA_INCLUDE_DIR}" \
    --output "${OUTPUT_DIR}" \
    --target-package "${TARGET_PACKAGE}" \
    --library cuopt \
    "${CURDIR}/headers.h"

echo "Panama bindings regenerated at:"
echo "  ${PANAMA_DIR}"
