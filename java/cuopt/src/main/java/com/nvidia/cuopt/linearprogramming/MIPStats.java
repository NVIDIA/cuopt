/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

public final class MIPStats {
  private final double presolveTime;
  private final double maxConstraintViolation;
  private final double maxIntViolation;
  private final double maxVariableBoundViolation;
  private final int numNodes;
  private final int numSimplexIterations;

  MIPStats(double[] values) {
    this.presolveTime = values[0];
    this.maxConstraintViolation = values[1];
    this.maxIntViolation = values[2];
    this.maxVariableBoundViolation = values[3];
    this.numNodes = (int) values[4];
    this.numSimplexIterations = (int) values[5];
  }

  public double getPresolveTime() {
    return presolveTime;
  }

  public double getMaxConstraintViolation() {
    return maxConstraintViolation;
  }

  public double getMaxIntViolation() {
    return maxIntViolation;
  }

  public double getMaxVariableBoundViolation() {
    return maxVariableBoundViolation;
  }

  public int getNumNodes() {
    return numNodes;
  }

  public int getNumSimplexIterations() {
    return numSimplexIterations;
  }
}
