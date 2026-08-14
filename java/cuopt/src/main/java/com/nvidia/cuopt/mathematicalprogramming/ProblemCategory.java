/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

/** Problem categories returned by cuOpt solutions. */
public enum ProblemCategory {
  LP(0),
  MIP(1);

  /**
   * The engine still reports a third category for a problem whose variables are all discrete (see
   * problem_category_t::IP). It is not surfaced here, because the distinction carries no meaning
   * for a caller that has already been told the problem is not an LP, and it is folded into
   * {@link #MIP} instead.
   */
  private static final int NATIVE_ALL_INTEGER = 2;

  private final int nativeValue;

  ProblemCategory(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }

  static ProblemCategory fromNative(int value) {
    if (value == NATIVE_ALL_INTEGER) {
      return MIP;
    }
    for (ProblemCategory category : values()) {
      if (category.nativeValue == value) {
        return category;
      }
    }
    throw new IllegalArgumentException("Unknown problem category: " + value);
  }
}
