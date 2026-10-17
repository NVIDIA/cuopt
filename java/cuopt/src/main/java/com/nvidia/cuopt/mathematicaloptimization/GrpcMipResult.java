/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.util.Arrays;

/** Unpacked view of {@link NativeGrpcClient#getMipResult}'s packed double array. */
public final class GrpcMipResult {
  private static final int HEADER_SIZE = 5;

  private final int terminationStatus;
  private final double objective;
  private final double mipGap;
  private final double solutionBound;
  private final double totalSolveTime;
  private final double[] solution;

  private GrpcMipResult(
      int terminationStatus,
      double objective,
      double mipGap,
      double solutionBound,
      double totalSolveTime,
      double[] solution) {
    this.terminationStatus = terminationStatus;
    this.objective = objective;
    this.mipGap = mipGap;
    this.solutionBound = solutionBound;
    this.totalSolveTime = totalSolveTime;
    this.solution = solution;
  }

  static GrpcMipResult fromPacked(double[] packed) {
    if (packed.length < HEADER_SIZE) {
      throw new IllegalArgumentException("packed MIP result is too short: " + packed.length);
    }
    return new GrpcMipResult(
        (int) packed[0],
        packed[1],
        packed[2],
        packed[3],
        packed[4],
        Arrays.copyOfRange(packed, HEADER_SIZE, packed.length));
  }

  /** Matches cuopt::mathematical_optimization::mip_termination_status_t's underlying values. */
  public int getTerminationStatus() {
    return terminationStatus;
  }

  public double getObjective() {
    return objective;
  }

  public double getMipGap() {
    return mipGap;
  }

  public double getSolutionBound() {
    return solutionBound;
  }

  public double getTotalSolveTime() {
    return totalSolveTime;
  }

  public double[] getSolution() {
    return solution.clone();
  }
}
