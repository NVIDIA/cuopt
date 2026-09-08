#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

HEADER=${1:?missing constants.h path}
OUT_DIR=${2:?missing output directory}
PACKAGE_DIR="${OUT_DIR}/com/nvidia/cuopt/mathematicaloptimization"
OUT_FILE="${PACKAGE_DIR}/CuOptConstants.java"

mkdir -p "${PACKAGE_DIR}"

{
  echo "/*"
  echo " * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved."
  echo " * SPDX-License-Identifier: Apache-2.0"
  echo " *"
  echo " * AUTO-GENERATED FILE. DO NOT EDIT."
  echo " * Generated from cpp/include/cuopt/mathematical_optimization/constants.h."
  echo " * Run 'mvn generate-sources' from java/cuopt to regenerate."
  echo " */"
  echo "package com.nvidia.cuopt.mathematicaloptimization;"
  echo
  echo "public final class CuOptConstants {"
  echo "  private CuOptConstants() {}"
  echo
  awk '
    /^#define CUOPT_/ {
      name = $2
      value = $3
      if (name ~ /CUOPT_INFINITY/) next
      if (value ~ /^[-]?[0-9]+$/) {
        printf("  public static final int %s = %s;%s", name, value, "\n")
      } else if (value ~ /^'\''.'\''$/) {
        printf("  public static final byte %s = %s;%s", name, value, "\n")
      } else if (value ~ /^".*"$/) {
        printf("  public static final String %s = %s;%s", name, value, "\n")
      }
    }
  ' "${HEADER}"
  echo "}"
} > "${OUT_FILE}"
