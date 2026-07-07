/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum ObjectiveSense {
  MAXIMIZE(-1),
  MINIMIZE(1);

  private final int nativeValue;

  ObjectiveSense(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  int nativeValue() {
    return nativeValue;
  }
}
