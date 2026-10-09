#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install ONE nightly cuOpt component in a clean container and run its sample gtest.
# Usage: ci/isolated_install/run.sh <client|mathopt|routing> <pip|conda> [image]
set -euo pipefail

COMPONENT="${1:?usage: run.sh <client|mathopt|routing> <pip|conda> [image]}"
SOURCE="${2:?usage: run.sh <client|mathopt|routing> <pip|conda> [image]}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${SOURCE}" in
  pip)   IMAGE="${3:-nvidia/cuda:${CUDA_IMAGE_TAG:-13.0.1-devel-ubuntu24.04}}" ;;
  conda) IMAGE="${3:-condaforge/miniforge3:latest}" ;;
  *) echo "source must be pip or conda" >&2; exit 2 ;;
esac

GPU_ARGS=(--gpus all)
[[ "${COMPONENT}" == "client" ]] && GPU_ARGS=()   # client needs no GPU

# ${arr[@]+...}: an empty array is "unbound" under set -u on bash < 4.4 (e.g. macOS bash 3.2).
docker run --rm ${GPU_ARGS[@]+"${GPU_ARGS[@]}"} -e COMPONENT="${COMPONENT}" \
  -e CUDA_MAJOR="${CUDA_MAJOR:-13}" \
  -v "${HERE}:/work:ro" "${IMAGE}" bash "/work/container_${SOURCE}.sh"
