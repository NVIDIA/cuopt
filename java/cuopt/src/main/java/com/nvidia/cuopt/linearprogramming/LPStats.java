/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public final class LPStats {
  private final double primalResidual;
  private final double dualResidual;
  private final double gap;
  private final int numIterations;
  private final SolverMethod solvedBy;

  LPStats(double[] values) {
    this.primalResidual = values[0];
    this.dualResidual = values[1];
    this.gap = values[2];
    this.numIterations = (int) values[3];
    this.solvedBy = SolverMethod.fromNative((int) values[4]);
  }

  public double getPrimalResidual() {
    return primalResidual;
  }

  public double getDualResidual() {
    return dualResidual;
  }

  public double getGap() {
    return gap;
  }

  public int getNumIterations() {
    return numIterations;
  }

  public SolverMethod getSolvedBy() {
    return solvedBy;
  }
}
