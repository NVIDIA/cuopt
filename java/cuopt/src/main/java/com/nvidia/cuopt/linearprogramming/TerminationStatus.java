/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public enum TerminationStatus {
  NO_TERMINATION(0),
  OPTIMAL(1),
  INFEASIBLE(2),
  UNBOUNDED(3),
  ITERATION_LIMIT(4),
  TIME_LIMIT(5),
  NUMERICAL_ERROR(6),
  PRIMAL_FEASIBLE(7),
  FEASIBLE_FOUND(8),
  CONCURRENT_LIMIT(9),
  WORK_LIMIT(10),
  UNBOUNDED_OR_INFEASIBLE(11),
  UNKNOWN(Integer.MIN_VALUE);

  private final int nativeValue;

  TerminationStatus(int nativeValue) {
    this.nativeValue = nativeValue;
  }

  public int nativeValue() {
    return nativeValue;
  }

  static TerminationStatus fromNative(int value) {
    for (TerminationStatus status : values()) {
      if (status.nativeValue == value) {
        return status;
      }
    }
    return UNKNOWN;
  }
}
