/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.mathematicaloptimization;

import java.util.Arrays;

/** Unpacked view of {@link NativeGrpcClient#getLpResult}'s packed double array. */
public final class GrpcLpResult {
  private static final int HEADER_SIZE = 7;

  private final int terminationStatus;
  private final double primalObjective;
  private final double dualObjective;
  private final double gap;
  private final double solveTime;
  private final double[] primalSolution;
  private final double[] dualSolution;
  private final double[] reducedCost;

  private GrpcLpResult(
      int terminationStatus,
      double primalObjective,
      double dualObjective,
      double gap,
      double solveTime,
      double[] primalSolution,
      double[] dualSolution,
      double[] reducedCost) {
    this.terminationStatus = terminationStatus;
    this.primalObjective = primalObjective;
    this.dualObjective = dualObjective;
    this.gap = gap;
    this.solveTime = solveTime;
    this.primalSolution = primalSolution;
    this.dualSolution = dualSolution;
    this.reducedCost = reducedCost;
  }

  static GrpcLpResult fromPacked(double[] packed) {
    if (packed.length < HEADER_SIZE) {
      throw new IllegalArgumentException("packed LP result is too short: " + packed.length);
    }
    int numVariables = (int) packed[5];
    int numConstraints = (int) packed[6];
    int primalStart = HEADER_SIZE;
    int dualStart = primalStart + numVariables;
    int reducedCostStart = dualStart + numConstraints;
    int end = reducedCostStart + numVariables;
    if (packed.length < end) {
      throw new IllegalArgumentException(
          "packed LP result is shorter than its own header claims: " + packed.length + " < " + end);
    }
    return new GrpcLpResult(
        (int) packed[0],
        packed[1],
        packed[2],
        packed[3],
        packed[4],
        Arrays.copyOfRange(packed, primalStart, dualStart),
        Arrays.copyOfRange(packed, dualStart, reducedCostStart),
        Arrays.copyOfRange(packed, reducedCostStart, end));
  }

  /** Matches cuopt::mathematical_optimization::pdlp_termination_status_t's underlying values. */
  public int getTerminationStatus() {
    return terminationStatus;
  }

  public double getPrimalObjective() {
    return primalObjective;
  }

  public double getDualObjective() {
    return dualObjective;
  }

  public double getGap() {
    return gap;
  }

  public double getSolveTime() {
    return solveTime;
  }

  public double[] getPrimalSolution() {
    return primalSolution.clone();
  }

  public double[] getDualSolution() {
    return dualSolution.clone();
  }

  public double[] getReducedCost() {
    return reducedCost.clone();
  }
}
