/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum PDLPSolverMode {
  STABLE1(0),
  STABLE2(1),
  METHODICAL1(2),
  FAST1(3),
  STABLE3(4);

  private final int nativeValue;

  PDLPSolverMode(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }
}
