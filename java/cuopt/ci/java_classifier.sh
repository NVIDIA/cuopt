#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Derives the Maven classifier for a self-contained cuOpt Java JAR.
#
# A classifier names the one combination of CUDA major version and CPU architecture that the
# JAR's embedded native library will run on. x86_64 carries no architecture suffix, matching
# the scheme cuDF publishes under (cuda12, cuda12-arm64, cuda13, cuda13-arm64).

# cuopt_java_classifier <cuda-version> [arch]
#   cuda-version  full or major-only, e.g. "13.0.3" or "13"
#   arch          defaults to the host's uname -m
cuopt_java_classifier() {
  local cuda_version="${1:?missing cuda version}"
  local arch="${2:-$(uname -m)}"
  local cuda_major="${cuda_version%%.*}"

  case "${arch}" in
    x86_64 | amd64) printf 'cuda%s\n' "${cuda_major}" ;;
    aarch64 | arm64) printf 'cuda%s-arm64\n' "${cuda_major}" ;;
    *)
      echo "unsupported architecture '${arch}'; expected x86_64 or aarch64" >&2
      return 1
      ;;
  esac
}

# The directory a JAR for this classifier expects its native library under, which is also the
# resource path the loader searches at run time.
cuopt_java_native_resource_dir() {
  local arch="${1:-$(uname -m)}"
  case "${arch}" in
    x86_64 | amd64) printf 'amd64/Linux\n' ;;
    aarch64 | arm64) printf 'aarch64/Linux\n' ;;
    *)
      echo "unsupported architecture '${arch}'; expected x86_64 or aarch64" >&2
      return 1
      ;;
  esac
}
