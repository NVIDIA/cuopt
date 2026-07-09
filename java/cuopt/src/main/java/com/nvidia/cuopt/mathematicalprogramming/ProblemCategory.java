/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicalprogramming;

/** Problem categories returned by cuOpt solutions. */
public enum ProblemCategory {
  LP(0),
  MIP(1),
  /** @deprecated Integer problems are categorized as {@link #MIP}. */
  @Deprecated(since = "26.08")
  IP(2);

  private final int nativeValue;

  ProblemCategory(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }

  static ProblemCategory fromNative(int value) {
    if (value == IP.nativeValue) {
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
