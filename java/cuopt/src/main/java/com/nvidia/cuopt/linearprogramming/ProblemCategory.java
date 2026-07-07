/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

/** Problem categories returned by cuOpt solutions. */
public enum ProblemCategory {
  LP(0),
  MIP(1),
  IP(2);

  private final int nativeValue;

  ProblemCategory(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }

  static ProblemCategory fromNative(int value) {
    for (ProblemCategory category : values()) {
      if (category.nativeValue == value) {
        return category;
      }
    }
    throw new IllegalArgumentException("Unknown problem category: " + value);
  }
}
