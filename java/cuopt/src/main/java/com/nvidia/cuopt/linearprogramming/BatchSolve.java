/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package com.nvidia.cuopt.linearprogramming;

import java.util.ArrayList;
import java.util.List;

/** Compatibility entry point for Python's deprecated LP BatchSolve API. */
public final class BatchSolve {
  private BatchSolve() {}

  /**
   * Solve each model and return the solutions plus aggregate elapsed time.
   *
   * <p>The Python API marks batch solving deprecated; Java deliberately uses sequential solves so
   * callers can control their own concurrency without adding another native batch ABI.
   */
  public static BatchSolveResult solve(List<DataModel> dataModels, SolverSettings settings) {
    List<Solution> solutions = new ArrayList<>();
    long start = System.nanoTime();
    try {
      for (DataModel dataModel : dataModels) {
        solutions.add(dataModel.solve(settings));
      }
      return new BatchSolveResult(solutions, (System.nanoTime() - start) / 1.0e9);
    } catch (RuntimeException e) {
      solutions.forEach(Solution::close);
      throw e;
    }
  }

  public static BatchSolveResult solve(List<DataModel> dataModels) {
    try (SolverSettings settings = new SolverSettings()) {
      BatchSolveResult result = solve(dataModels, settings);
      // The result owns the solutions; only the temporary settings are closed here.
      return result;
    }
  }
}
