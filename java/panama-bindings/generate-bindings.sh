#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerates Panama bindings for cuOpt's C API into
# cuopt-java/src/main/java22/com/nvidia/cuopt/internal/panama/.
#
# Inputs:
#   $JEXTRACT       Path to the jextract binary (set by build.sh, or in PATH).
#   $CUOPT_INCLUDE  Path to cpp/include (defaults to ../../cpp/include).
#   $CUDA_INCLUDE   Path to CUDA include dir (defaults to /usr/local/cuda/include).
#
# Output: regenerates files under
#   ../cuopt-java/src/main/java22/com/nvidia/cuopt/internal/panama/

set -euo pipefail

CURDIR="$(cd "$(dirname "$0")" && pwd)"
REPODIR="$(cd "${CURDIR}/../.." && pwd)"

JEXTRACT="${JEXTRACT:-jextract}"
CUOPT_INCLUDE="${CUOPT_INCLUDE:-${REPODIR}/cpp/include}"
CUDA_INCLUDE="${CUDA_INCLUDE:-/usr/local/cuda/include}"

OUTPUT_DIR="${REPODIR}/java/cuopt-java/src/main/java22"
TARGET_PACKAGE="com.nvidia.cuopt.internal.panama"

# Sanity checks
if ! command -v "${JEXTRACT}" >/dev/null 2>&1; then
    echo "ERROR: jextract not found at '${JEXTRACT}'." >&2
    echo "       Run build.sh from the java/ directory to auto-download it," >&2
    echo "       or set JEXTRACT to the path of an installed jextract binary." >&2
    exit 1
fi

if [[ ! -d "${CUOPT_INCLUDE}" ]]; then
    echo "ERROR: cuopt include dir not found at '${CUOPT_INCLUDE}'." >&2
    exit 1
fi

# Clean previous output to ensure removed symbols are actually removed
PANAMA_DIR="${OUTPUT_DIR}/com/nvidia/cuopt/internal/panama"
mkdir -p "${PANAMA_DIR}"
find "${PANAMA_DIR}" -name '*.java' -delete 2>/dev/null || true

# Run jextract
# --library cuopt: the generated code dlopen's libcuopt.so
# --target-package: where the generated classes land
# --output: source directory for generated files
# --include-dir: search paths for #include resolution
"${JEXTRACT}" \
    --include-dir "${CUOPT_INCLUDE}" \
    --include-dir "${CUDA_INCLUDE}" \
    --output "${OUTPUT_DIR}" \
    --target-package "${TARGET_PACKAGE}" \
    --library cuopt \
    "${CURDIR}/headers.h"

echo "Panama bindings regenerated at:"
echo "  ${PANAMA_DIR}"
