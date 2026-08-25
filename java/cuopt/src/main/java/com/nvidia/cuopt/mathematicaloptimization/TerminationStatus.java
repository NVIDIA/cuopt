/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

/** Termination statuses backed by constants generated from the C++ public header. */
public enum TerminationStatus {
  NO_TERMINATION(CuOptConstants.CUOPT_TERMINATION_STATUS_NO_TERMINATION),
  OPTIMAL(CuOptConstants.CUOPT_TERMINATION_STATUS_OPTIMAL),
  INFEASIBLE(CuOptConstants.CUOPT_TERMINATION_STATUS_INFEASIBLE),
  UNBOUNDED(CuOptConstants.CUOPT_TERMINATION_STATUS_UNBOUNDED),
  ITERATION_LIMIT(CuOptConstants.CUOPT_TERMINATION_STATUS_ITERATION_LIMIT),
  TIME_LIMIT(CuOptConstants.CUOPT_TERMINATION_STATUS_TIME_LIMIT),
  NUMERICAL_ERROR(CuOptConstants.CUOPT_TERMINATION_STATUS_NUMERICAL_ERROR),
  PRIMAL_FEASIBLE(CuOptConstants.CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE),
  FEASIBLE_FOUND(CuOptConstants.CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND),
  CONCURRENT_LIMIT(CuOptConstants.CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT),
  WORK_LIMIT(CuOptConstants.CUOPT_TERMINATION_STATUS_WORK_LIMIT),
  UNBOUNDED_OR_INFEASIBLE(
      CuOptConstants.CUOPT_TERMINATION_STATUS_UNBOUNDED_OR_INFEASIBLE),
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
