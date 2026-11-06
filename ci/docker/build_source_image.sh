#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build a GPU-ready cuOpt image from this checkout. Set PUSH=1 after logging in
# to Docker Hub to publish the resulting tag.

set -euo pipefail

: "${IMAGE:?Set IMAGE to the complete image tag, e.g. docker.io/<user>/cuopt:break-load-windows}"

platform="${PLATFORM:-linux/amd64}"
cuda_env="${CUDA_ENV:-133}"
cuda_devel_image="${CUDA_DEVEL_IMAGE:-nvidia/cuda:13.3.0-devel-ubuntu24.04}"
cuda_runtime_image="${CUDA_RUNTIME_IMAGE:-nvidia/cuda:13.3.0-runtime-ubuntu24.04}"
parallel_level="${PARALLEL_LEVEL:-1}"
cuda_architectures="${CUDA_ARCHITECTURES:-90}"
source_context="${SOURCE_CONTEXT:-.}"

build_args=(
    --file ci/docker/Dockerfile.source
    --platform "${platform}"
    --tag "${IMAGE}"
    --build-arg "CUDA_ENV=${cuda_env}"
    --build-arg "CUDA_DEVEL_IMAGE=${cuda_devel_image}"
    --build-arg "CUDA_RUNTIME_IMAGE=${cuda_runtime_image}"
    --build-arg "PARALLEL_LEVEL=${parallel_level}"
    --build-arg "CUDA_ARCHITECTURES=${cuda_architectures}"
    --build-context "source=${source_context}"
)

# A normal Docker build makes a locally runnable image. `--push` is used only
# when explicitly requested, avoiding accidental publication of development
# code. BuildKit cache mounts in Dockerfile.source require BuildKit.
if [[ "${PUSH:-0}" == "1" ]]; then
    docker buildx build --progress=plain --push "${build_args[@]}" .
else
    docker buildx build --progress=plain --load "${build_args[@]}" .
fi
