/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/** Solver methods backed by constants generated from the C++ public header. */
public enum SolverMethod {
  CONCURRENT(CuOptConstants.CUOPT_METHOD_CONCURRENT),
  PDLP(CuOptConstants.CUOPT_METHOD_PDLP),
  DUAL_SIMPLEX(CuOptConstants.CUOPT_METHOD_DUAL_SIMPLEX),
  BARRIER(CuOptConstants.CUOPT_METHOD_BARRIER),
  UNSET(CuOptConstants.CUOPT_METHOD_UNSET);

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
