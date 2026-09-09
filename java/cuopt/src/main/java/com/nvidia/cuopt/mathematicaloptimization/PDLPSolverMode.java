/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/** PDLP solver modes backed by constants generated from the C++ public header. */
public enum PDLPSolverMode {
  STABLE1(CuOptConstants.CUOPT_PDLP_SOLVER_MODE_STABLE1),
  STABLE2(CuOptConstants.CUOPT_PDLP_SOLVER_MODE_STABLE2),
  METHODICAL1(CuOptConstants.CUOPT_PDLP_SOLVER_MODE_METHODICAL1),
  FAST1(CuOptConstants.CUOPT_PDLP_SOLVER_MODE_FAST1),
  STABLE3(CuOptConstants.CUOPT_PDLP_SOLVER_MODE_STABLE3);

  private final int nativeValue;

  PDLPSolverMode(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }
}
