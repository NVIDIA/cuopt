/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum SolverMethod {
  CONCURRENT(0),
  PDLP(1),
  DUAL_SIMPLEX(2),
  BARRIER(3),
  UNSET(4);

  private final int nativeValue;

  SolverMethod(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }

  static SolverMethod fromNative(int value) {
    for (SolverMethod method : values()) {
      if (method.nativeValue == value) {
        return method;
      }
    }
    return UNSET;
  }
}
