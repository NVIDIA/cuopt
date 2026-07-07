/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.List;

/** Result of the Java compatibility implementation of Python BatchSolve. */
public final class BatchSolveResult implements AutoCloseable {
  private final List<Solution> solutions;
  private final double solveTime;

  BatchSolveResult(List<Solution> solutions, double solveTime) {
    this.solutions = List.copyOf(solutions);
    this.solveTime = solveTime;
  }

  public List<Solution> getSolutions() {
    return solutions;
  }

  public double getSolveTime() {
    return solveTime;
  }

  @Override
  public void close() {
    solutions.forEach(Solution::close);
  }
}
